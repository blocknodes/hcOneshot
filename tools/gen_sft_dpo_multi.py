#!/usr/bin/env python3
"""多意图（bench_serial_*）金标 × 一把出日志 → SFT / DPO。

金标：bench_serial_train_10k.csv（seg 用 # 分子句、gold 用 + 分中文域、tools 用 + 分工具、
      tool_param 为各步参数 JSON 数组，null=该步未标）。
日志：hcSmith/<dir>/_raw/*.jsonl 里的一把出首轮计划调用（system/user 原样取用，含动态 few-shot）。

chosen = 各步都是可立即执行条目 [{"q","d","vt","params"}, ...]（bench 的 serial 是顺序执行、无指代依赖，
         与日志里模型绝大多数输出一致，不写 dep）。
         q：模型输出与金标逐步工具一致时用模型改写，否则用金标 seg 子句；params 去系统代填参数。
可用性（不满足则该样本不进 SFT/DPO，原因记报告）：
  · tools_ok=False（金标工具未通过校验）→ 跳过
  · 某步参数为 null 且该工具有参数 → 跳过（无法给出完整 chosen）；无参工具（fuzzy/fan/recommend…）null 视为 {}
  · 金标工具不在目录 → 跳过
DPO rejected = 判错的模型原始输出；判错口径：步数不同 / 工具链不同（parallel 按多重集合，serial 按顺序）/
  任一步参数不同（参数比较同 eval_stats.canon，另把仅大小写/空格标点不同视为一致，避免金标写法差异进 DPO）。
用法：python3 tools/gen_sft_dpo_multi.py --log '../hcSmith/train3/_raw/*.jsonl' --prefix train3_multi
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from app import catalog  # noqa: E402
from app.parser import parse_json_loose, parse_oneshot  # noqa: E402
from eval_stats import canon  # noqa: E402
from gen_sft_dpo import TOOL_ALIAS  # noqa: E402
from clean_testset import _strip_k12  # noqa: E402

CN_DOMAIN = {"设备": "device", "影视": "vod", "音乐": "music", "教育": "education", "少儿": "children",
             "体育": "sports", "有声": "audio"}
PLAN_SYS = "你是电视语音助手。读用户请求"
_PUNCT = re.compile(r"[\s·・.,，。!！?？'\"《》“”:：\-—_]")


def nq(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def compact(o) -> str:
    return json.dumps(o, ensure_ascii=False, separators=(",", ":"))


def _loose(c: str) -> str:
    return _PUNCT.sub("", c).lower()


def load_gold(path: Path, stats: collections.Counter, drops: list) -> dict[str, dict]:
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        q = nq(r["query"])
        segs = [re.sub(r"[\s，。！？!?,.~～、]+$", "", s.strip()) for s in r["seg"].split("#")]
        doms = [CN_DOMAIN.get(x.strip(), "") for x in r["gold"].split("+")]
        tools = [TOOL_ALIAS.get(t.strip(), t.strip()) for t in r["tools"].split("+")]
        ps = parse_json_loose(r["tool_param"] or "")
        ps = ps if isinstance(ps, list) else [None] * len(tools)
        why = ""
        if r.get("tools_ok") == "False":
            why = "tools_ok=False"
        elif not (len(segs) == len(doms) == len(tools) == len(ps)):
            why = "列长度不一致"
        steps = []
        for i in range(len(tools)) if not why else []:
            d, t, p = doms[i], tools[i], ps[i]
            if t not in catalog.valid_tools_for(d):
                d2 = catalog.domain_of_tool(t)
                if not d2:
                    why = f"工具不在目录:{d}:{t}"
                    break
                d = d2
            if p is None:
                if catalog.param_names_for(d, t) - catalog.system_filled(t):
                    why = "某步参数未标"
                    break
                p = {}
            if not isinstance(p, dict):
                why = "参数非对象"
                break
            p = {k: v for k, v in p.items() if k not in catalog.system_filled(t)}
            p, _ = _strip_k12(p, r["query"])          # 与 all_eval 同口径（clean_testset 规则 8）
            p = {k: v for k, v in p.items() if v not in ("", None, [], {})}
            steps.append({"seg": segs[i], "d": d, "tool": t, "params": p})
        if why:
            stats[f"skip_{why.split(':')[0]}"] += 1
            drops.append({"id": r["id"], "query": r["query"], "reason": why})
            continue
        out[q] = {"row": r, "steps": steps, "parallel": r["exec_mode"] == "parallel"}
    return out


def iter_calls(pattern: str):
    for f in sorted(glob.glob(pattern)):
        for line in open(f, encoding="utf-8"):
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            msgs = (o.get("request") or {}).get("messages") or []
            if len(msgs) < 2 or not msgs[0].get("content", "").startswith(PLAN_SYS):
                continue
            try:
                out = o["response"]["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                continue
            m = re.search(r"用户请求：(.*)$", msgs[-1]["content"], re.S)
            if out is None or not m:
                continue
            q = nq(re.sub(r"\n【判定提示】.*$", "", m.group(1), flags=re.S))
            yield q, [{"role": "system", "content": msgs[0]["content"]},
                      {"role": "user", "content": msgs[-1]["content"]}], out


def judge(raw: str, g: dict) -> tuple[str, list]:
    its, err = parse_oneshot(raw or "")
    if err and not its:
        return "parse_fail", []
    gs = g["steps"]
    if len(its) != len(gs):
        return "step_count", its
    gt = [s["tool"] for s in gs]
    pt = [TOOL_ALIAS.get(i.tool, i.tool) for i in its]
    if g["parallel"]:
        if collections.Counter(pt) != collections.Counter(gt):
            return "tool_wrong", its
        used, pairs = set(), []
        for s in gs:
            j = next(j for j, t in enumerate(pt) if t == s["tool"] and j not in used)
            used.add(j)
            pairs.append((s, its[j]))
    else:
        if pt != gt:
            return "tool_wrong", its
        pairs = list(zip(gs, its))
    for s, it in pairs:
        a, b = canon(s["params"], s["tool"]), canon(it.params, s["tool"])
        if a != b and _loose(a) != _loose(b):
            return "param_diff", its
    return "match", its


def build_chosen(g: dict, its) -> str:
    use_model_q = len(its) == len(g["steps"]) and \
        [TOOL_ALIAS.get(i.tool, i.tool) for i in its] == [s["tool"] for s in g["steps"]]
    plan = []
    for k, s in enumerate(g["steps"]):
        q = its[k].q if use_model_q and its[k].q else s["seg"]
        plan.append({"q": q, "d": s["d"], "vt": f"{s['d']}:{s['tool']}", "params": s["params"]})
    return compact(plan)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(ROOT / "testset" / "feishu" / "bench_serial_train_10k.csv"))
    ap.add_argument("--log", required=True)
    ap.add_argument("--outdir", default=str(ROOT / "testset" / "sftdpo"))
    ap.add_argument("--prefix", default="train3_multi")
    args = ap.parse_args()
    stats: collections.Counter = collections.Counter()
    drops: list = []
    gold = load_gold(Path(args.gold), stats, drops)
    calls: dict[str, dict] = {}
    for q, msgs, out in iter_calls(args.log):
        if q not in gold:
            stats["log_no_gold"] += 1
            continue
        c = calls.setdefault(q, {"msgs": msgs, "outs": []})
        c["outs"].append(out)

    sft, dpo, audit = [], [], []
    for q, g in gold.items():
        c = calls.get(q)
        if not c:
            stats["no_log"] += 1
            continue
        vs = [(raw, *judge(raw, g)) for raw in c["outs"]]
        ref = next((v for v in vs if v[1] == "match"), vs[0])
        chosen = build_chosen(g, ref[2])
        sft.append({"messages": c["msgs"] + [{"role": "assistant", "content": chosen}]})
        seen = set()
        for raw, vd, its in vs:
            stats[vd] += 1
            rec = {"id": g["row"]["id"], "query": g["row"]["query"], "exec_mode": g["row"]["exec_mode"],
                   "verdict": vd, "chosen": chosen, "model_output": raw, "is_dpo": "0"}
            if vd != "match":
                key = compact([[i.vt, i.params] for i in its]) if its else raw
                if key not in seen:
                    seen.add(key)
                    dpo.append({"messages": c["msgs"] + [{"role": "assistant", "content": chosen}],
                                "rejected_response": raw})
                    rec["is_dpo"] = "1"
            audit.append(rec)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in ((f"{args.prefix}.sft.jsonl", sft), (f"{args.prefix}.dpo.jsonl", dpo),
                       (f"{args.prefix}.dropped.jsonl", drops)):
        with open(out / name, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / f"{args.prefix}.audit.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "query", "exec_mode", "verdict", "is_dpo", "chosen", "model_output"])
        w.writeheader()
        w.writerows(audit)
    v = {k: stats[k] for k in ("match", "param_diff", "tool_wrong", "step_count", "parse_fail")}
    rep = {"gold_usable": len(gold), "sft": len(sft), "dpo": len(dpo), "verdicts": v,
           "accuracy": round(v["match"] / max(1, sum(v.values())), 4),
           "counts": dict(sorted(stats.items()))}
    (out / f"{args.prefix}.report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
