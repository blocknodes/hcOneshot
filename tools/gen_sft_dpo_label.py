#!/usr/bin/env python3
"""飞书 8 份评测标注（7 个单域 + 多意图）× 一把出日志 → SFT / DPO（ms-swift 格式）。

标注  testset/feishu/<日期>/{vod,educ,edu,music,audio,sports,tv_control}_eval.csv  单步
      testset/feishu/<日期>/multi_intent.csv                                       多步（有序）
日志  hcSmith/train_new/_raw/*.jsonl（system/user 原样取自日志，含动态 few-shot 与「当前时间」行）

判对错（只看工具和参数，q 不参与）
  · 步数不同                    → step_count
  · 某步没给工具（含依赖条）    → tool_missing / dep_wrong   标注里该步参数已静态可定，模型不该留空
  · 某步 (域, 工具) 不同        → tool_wrong
  · 某步参数语义不同            → param_diff   （检索树归一，忽略 retext / audio.query / fan.messages，
                                                  仅大小写/空格/标点不同视为一致）
  全部一致                      → match

样本
  SFT  每个有日志的标注 query 一条。
       match：assistant = 模型原始输出（原样）；若输出带系统代填参数或不是纯 JSON 数组，则按标注重建。
       非 match：assistant = 标注计划（q 沿用模型改写，步数不同时用标注分句），参数去掉系统代填项。
  DPO  只有 非 match 才产出：chosen = 同一条 SFT 的 assistant，rejected = 模型原始输出（同 query 同错去重）。

标注可用性
  多意图仅取 打标状态=完整 且 需人工核对 为空（--include-review 可放开后者）；其余（部分/待核对/无法打标）
  标注不全或不可靠，不能拿来判模型错。教育域默认补的 k12_category 按清洗规则 8 去掉（query 含“同步”则保留）。

虚拟 schema 模式（--device-virtual，对应服务端 HC_DEVICE_VIRTUAL=1）
  · 只取 system 里含当前 device_schema.render_block() 的日志调用（旧版/真实 schema 的 prompt 与现行不一致，不能混用）。
  · 判对错：模型设备步的虚拟参数经 compile_tolerant 还原成真实参数，再与标注比较
    （播放控制 object 为空视同「播放控制」，两边同样归一）。还原失败判 param_diff。
  · chosen：设备步参数写成虚拟写法（标注经 decompile）；match 的原始输出若设备步仍是真实写法（有 operation/object）
    则不原样采用，改按标注重建。标注设备步无法用虚拟 schema 表示（约 0.3%）的样本丢弃。

用法：python3 tools/gen_sft_dpo_label.py [--prefix train_new_20261116]
      python3 tools/gen_sft_dpo_label.py --gold-csv testset/train_manual/train.csv \\
          --log '../hcSmith/train_pattern/_raw/*.jsonl' --prefix train_pattern_manual
      python3 tools/gen_sft_dpo_label.py --device-virtual --log '../hcSmith/train_vir/_raw/*.jsonl' --prefix train_vir_20261116
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import logging
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from app import catalog, device_schema  # noqa: E402
from app.parser import parse_json_loose, parse_oneshot  # noqa: E402
from clean_testset import _strip_k12  # noqa: E402
from gen_sft_dpo import (DOMAIN_MAP, HINT, PLAN_SYS, TOOL_ALIAS, canon_params,  # noqa: E402
                         compact, nq, prune)

logging.getLogger("app.parser").setLevel(logging.WARNING)

WS = ROOT.parent
DEF_GOLD_DIR = ROOT / "testset" / "feishu" / "20261116"
DEF_LOG = str(WS / "hcSmith/train_new/_raw/*.jsonl")
DEF_OUT = ROOT / "testset" / "sftdpo"
_PUNCT = re.compile(r"[\s·・.,，。!！?？'\"《》“”:：\-—_]")


def _loose(s: str) -> str:
    return _PUNCT.sub("", s).lower()


def _norm(tool: str, params: dict) -> dict:
    """标注写法不一致的归一：playback_control 的 object 为空视同「播放控制」。"""
    p = {k: v for k, v in params.items() if v not in ("", None, [], {})}
    if (tool == "playback_control" and not p.get("object")
            and p.get("operation") in device_schema._PB_OPS):
        p["object"] = "播放控制"
    return p


def _sort_str_lists(o):
    """纯字符串数组（如 music 的 tag/singer、检索树的 values）顺序无语义：比较前排序。"""
    if isinstance(o, dict):
        return {k: _sort_str_lists(v) for k, v in o.items()}
    if isinstance(o, list):
        xs = [_sort_str_lists(x) for x in o]
        return sorted(xs) if xs and all(isinstance(x, str) for x in xs) else xs
    return o


def same_params(tool: str, gold_p: dict, model_p: dict) -> bool:
    a, b = canon_params(_norm(tool, gold_p), tool), canon_params(_norm(tool, model_p), tool)
    return a == b or _loose(a) == _loose(b)


def _clean_params(d: str, tool: str, p: dict, query: str) -> dict:
    """标注参数 → 可作 chosen 的参数：去系统代填项、去默认补的 k12、去空值。
    fan_knowledge_agent 在目录里只有系统代填的 messages，标注里的 sceneType 等契约字段由引擎重建，故置空。"""
    if tool == "fan_knowledge_agent":
        return {}
    p = {k: v for k, v in p.items() if k not in catalog.system_filled(tool)}
    p, _ = _strip_k12(p, query)
    return prune(p)


# ---------------------------------------------------------------------------
# 标注载入 → {归一 query: {"query", "steps":[{"seg","d","tool","params"}], "kind"}}
# ---------------------------------------------------------------------------
def _needs_params(d: str, tool: str) -> bool:
    return bool(catalog.param_names_for(d, tool) - catalog.system_filled(tool))


def load_single(gold_dir: Path, stats, drops) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for prefix, d in DOMAIN_MAP.items():
        path = gold_dir / f"{prefix}_eval.csv"
        if not path.exists():
            stats[f"missing_file_{prefix}"] += 1
            continue
        for r in csv.DictReader(open(path, encoding="utf-8-sig")):
            query = r["query"].strip()
            q = nq(query)
            tool = TOOL_ALIAS.get(r["期望工具"].strip(), r["期望工具"].strip())
            why = ""
            if not q or not tool:
                why = "bad_label"
            elif tool not in catalog.valid_tools_for(d):
                why = "tool_not_in_catalog"
            params = {}
            if not why:
                raw = r["期望参数"].strip()
                if raw:
                    params = parse_json_loose(raw)
                    if not isinstance(params, dict):
                        why = "param_unparseable"
            if not why:
                params = _clean_params(d, tool, params, query)
                if not params and _needs_params(d, tool):
                    why = "param_missing"
            if why:
                stats[f"drop_single_{why}"] += 1
                drops.append({"src": prefix, "query": query, "tool": tool, "reason": why})
                continue
            if q in out:
                stats["single_dup_query"] += 1
                continue
            out[q] = {"kind": "single", "query": query, "id": prefix,
                      "steps": [{"seg": query, "d": d, "tool": tool, "params": params}]}
    return out


def load_csv(path: Path, stats, drops) -> dict[str, dict]:
    """单步金标 CSV（列 domain,query,tool,tool_param；domain 为 vod/educ/edu/... 前缀），
    如 testset/train_manual/train.csv。清洗口径同 load_single。"""
    out: dict[str, dict] = {}
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        query = r["query"].strip()
        q = nq(query)
        prefix = r["domain"].strip()
        d = DOMAIN_MAP.get(prefix, "")
        tool = TOOL_ALIAS.get(r["tool"].strip(), r["tool"].strip())
        why = ""
        if not q or not tool or not d:
            why = "bad_label"
        elif tool not in catalog.valid_tools_for(d):
            why = "tool_not_in_catalog"
        params = {}
        if not why:
            raw = (r.get("tool_param") or "").strip()
            if raw:
                params = parse_json_loose(raw)
                if not isinstance(params, dict):
                    why = "param_unparseable"
        if not why:
            params = _clean_params(d, tool, params, query)
            if not params and _needs_params(d, tool):
                why = "param_missing"
        if why:
            stats[f"drop_single_{why}"] += 1
            drops.append({"src": prefix, "query": query, "tool": tool, "reason": why})
            continue
        if q in out:
            stats["single_dup_query"] += 1
            continue
        out[q] = {"kind": "single", "query": query, "id": prefix,
                  "steps": [{"seg": query, "d": d, "tool": tool, "params": params}]}
    return out


def load_multi(path: Path, include_review: bool, stats, drops) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.exists():
        stats["missing_file_multi"] += 1
        return out
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        query = r["query"].strip()
        q = nq(query)
        why = ""
        if r["打标状态"] != "完整":
            why = f"status_{r['打标状态']}"
        elif r["需人工核对"].strip() and not include_review:
            why = "need_review"
        segs = [s.strip() for s in r["分句结果"].split("#")]
        steps = []
        n = int(r["工具数"] or 0)
        for i in range(1, 9):
            if why:
                break
            vt = r.get(f"tool{i}", "").strip()
            if not vt:
                continue
            d, _, tool = vt.partition(":")
            tool = TOOL_ALIAS.get(tool, tool)
            if tool not in catalog.valid_tools_for(d):
                d2 = catalog.domain_of_tool(tool)
                if not d2:
                    why = "tool_not_in_catalog"
                    break
                d = d2
            raw = r.get(f"tool{i}_param", "").strip()
            p = parse_json_loose(raw) if raw else {}
            if not isinstance(p, dict):
                why = "param_unparseable"
                break
            p = _clean_params(d, tool, p, query)
            if not p and _needs_params(d, tool):
                why = "param_missing"
                break
            seg = r.get(f"tool{i}_q", "").strip() or (segs[len(steps)] if len(steps) < len(segs) else query)
            steps.append({"seg": seg, "d": d, "tool": tool, "params": p})
        if not why and (not steps or len(steps) != n):
            why = "steps_mismatch"
        if why:
            stats[f"drop_multi_{why}"] += 1
            if not why.startswith("status_"):
                drops.append({"src": "multi", "query": query, "reason": why})
            continue
        if q in out:
            stats["multi_dup_query"] += 1
            continue
        out[q] = {"kind": "multi", "query": query, "id": "multi", "steps": steps}
    return out


# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
def iter_plan_calls(pattern: str, must_contain: str = "", stats: collections.Counter | None = None):
    """must_contain 非空时，只产出 system 含该串的调用（用于锁定当前版 prompt）。"""
    for f in sorted(glob.glob(pattern)):
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
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
            if must_contain and must_contain not in msgs[0]["content"]:
                if stats is not None:
                    stats["log_prompt_outdated"] += 1
                continue
            yield (nq(HINT.sub("", m.group(1))),
                   [{"role": "system", "content": msgs[0]["content"]},
                    {"role": "user", "content": msgs[-1]["content"]}], out)


# ---------------------------------------------------------------------------
# 判定
# ---------------------------------------------------------------------------
def judge(raw: str, g: dict, virtual: bool = False) -> tuple[str, list]:
    its, err = parse_oneshot(raw or "")
    if not its:
        return ("parse_fail" if err else "empty_plan"), []
    gs = g["steps"]
    if len(its) != len(gs):
        return "step_count", its
    for s, it in zip(gs, its):
        if it.dep:
            return "dep_wrong", its
        if not it.tool:
            return "tool_missing", its
    for s, it in zip(gs, its):
        if (it.d, TOOL_ALIAS.get(it.tool, it.tool)) != (s["d"], s["tool"]):
            return "tool_wrong", its
    for s, it in zip(gs, its):
        mp = it.params
        if virtual and s["d"] == "device":
            try:
                mp = device_schema.compile_tolerant(s["tool"], mp)       # 后处理：虚拟 → 真实
            except ValueError:
                return "param_diff", its
            if not same_params(s["tool"], s["params"], mp):
                return "param_diff", its
            continue
        a = canon_params(_sort_str_lists(s["params"]), s["tool"])
        b = canon_params(_sort_str_lists(mp), s["tool"])
        if a != b and _loose(a) != _loose(b):
            return "param_diff", its
    return "match", its


def _raw_is_clean(raw: str, its, virtual: bool = False) -> bool:
    """match 的原始输出可否原样作 chosen：纯 JSON 数组，且不含系统代填参数。
    虚拟模式下设备步还须是虚拟写法（有 action，没有 operation/object），否则会教模型回到真实写法。"""
    s = (raw or "").strip()
    if not (s.startswith("[") and s.endswith("]")) or "```" in s:
        return False
    if virtual:
        for it in its:
            if (it.d == "device" and it.tool and device_schema.is_virtual_tool(it.tool)
                    and it.tool not in device_schema.PASSTHROUGH
                    and ("action" not in it.params or {"operation", "object"} & set(it.params))):
                return False
    return all(not (set(it.params) & catalog.system_filled(it.tool)) for it in its)


def rebuild(g: dict, its, virtual: bool = False) -> str:
    """按标注重建计划。q 沿用模型改写的条件：步数相同，且（单步，或各步 域:工具 与标注一致）。
    多步且工具链不一致时（如顺序颠倒），模型第 k 句的 q 对不上标注第 k 个工具，改用标注分句。"""
    gs = g["steps"]
    use_q = len(its) == len(gs) and (
        len(gs) == 1 or all((it.d, TOOL_ALIAS.get(it.tool, it.tool)) == (s["d"], s["tool"])
                            for s, it in zip(gs, its)))
    plan = []
    for k, s in enumerate(gs):
        q = (its[k].q if use_q and its[k].q else "") or s["seg"]
        params = s["vparams"] if virtual and "vparams" in s else s["params"]
        plan.append({"q": q, "d": s["d"], "vt": f"{s['d']}:{s['tool']}", "params": params})
    return compact(plan)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold-dir", default=str(DEF_GOLD_DIR))
    ap.add_argument("--gold-csv", default="",
                    help="改用单步金标 CSV（domain,query,tool,tool_param），如 testset/train_manual/train.csv；"
                         "给出时忽略 --gold-dir 与多意图标注")
    ap.add_argument("--log", default=DEF_LOG)
    ap.add_argument("--outdir", default=str(DEF_OUT))
    ap.add_argument("--prefix", default="train_new_20261116")
    ap.add_argument("--include-review", action="store_true", help="多意图标注含『需人工核对』的行也用")
    ap.add_argument("--device-virtual", action="store_true",
                    help="设备域虚拟 schema 模式（对应 HC_DEVICE_VIRTUAL=1 的日志），见模块说明")
    args = ap.parse_args()

    stats: collections.Counter = collections.Counter()
    drops: list[dict] = []
    gold_dir = Path(args.gold_dir)
    if args.gold_csv:
        gold_dir = Path(args.gold_csv)
        gold = load_csv(gold_dir, stats, drops)
    else:
        gold = load_single(gold_dir, stats, drops)
        multi = load_multi(gold_dir / "multi_intent.csv", args.include_review, stats, drops)
        for q in set(gold) & set(multi):
            stats["gold_overlap_single_vs_multi"] += 1
        gold.update({q: g for q, g in multi.items() if q not in gold})

    virtual = args.device_virtual
    if virtual:
        # 标注设备步转虚拟写法；无法表示的样本丢弃
        for q in list(gold):
            ok = True
            for s in gold[q]["steps"]:
                if s["d"] == "device":
                    vp = device_schema.virtualize_params(s["tool"], s["params"])
                    if vp is None:
                        ok = False
                        break
                    s["vparams"] = vp
            if not ok:
                stats["drop_device_not_representable"] += 1
                drops.append({"src": gold[q]["id"], "query": gold[q]["query"],
                              "reason": "device_not_representable"})
                del gold[q]
    marker = device_schema.render_block() if virtual else ""
    calls: dict[str, dict] = {}
    for q, msgs, out in iter_plan_calls(args.log, marker, stats):
        if q not in gold:
            stats["log_no_gold"] += 1
            continue
        calls.setdefault(q, {"msgs": msgs, "outs": []})["outs"].append(out)

    sft, dpo, audit = [], [], []
    by_kind = collections.Counter()
    for q, g in gold.items():
        c = calls.get(q)
        if not c:
            stats["gold_no_log"] += 1
            continue
        vs = [(raw, *judge(raw, g, virtual)) for raw in c["outs"]]
        good = next((v for v in vs if v[1] == "match" and _raw_is_clean(v[0], v[2], virtual)), None)
        chosen = good[0].strip() if good else rebuild(g, vs[0][2], virtual)
        sft.append({"messages": c["msgs"] + [{"role": "assistant", "content": chosen}]})
        by_kind[g["kind"]] += 1
        seen: set[str] = set()
        for raw, vd, its in vs:
            stats[f"{g['kind']}_{vd}"] += 1
            rec = {"kind": g["kind"], "src": g["id"], "query": g["query"], "verdict": vd,
                   "gold": " | ".join(f"{s['d']}:{s['tool']}" for s in g["steps"]),
                   "model": " | ".join((f"{i.d}:{i.tool}" if i.tool else f"{i.d}:dep{i.dep}") for i in its),
                   "chosen": chosen, "model_output": raw, "is_dpo": "0"}
            if vd != "match":
                key = vd + "\x00" + (compact([[i.d, i.tool, i.params, i.dep] for i in its]) if its else raw)
                if key in seen:
                    stats["dpo_dup_rejected"] += 1
                else:
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
    cols = ["kind", "src", "query", "verdict", "is_dpo", "gold", "model", "chosen", "model_output"]
    with open(out / f"{args.prefix}.audit.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(audit)

    verdicts: dict[str, collections.Counter] = {}
    for r in audit:
        verdicts.setdefault(r["kind"], collections.Counter())[r["verdict"]] += 1
    report = {"gold_dir": str(gold_dir), "log": args.log, "gold_usable": len(gold),
              "sft": len(sft), "dpo": len(dpo), "sft_by_kind": dict(by_kind),
              "verdicts": {k: dict(v) for k, v in verdicts.items()},
              "accuracy": {k: round(v["match"] / max(1, sum(v.values())), 4) for k, v in verdicts.items()},
              "counts": dict(sorted(stats.items()))}
    (out / f"{args.prefix}.report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
