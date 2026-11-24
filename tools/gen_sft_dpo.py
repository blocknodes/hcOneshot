#!/usr/bin/env python3
"""由 all_eval.csv（金标）× baseline 一把出日志 生成 SFT / DPO（ms-swift 格式）。

接替 hcSlime/benchmark/gen_sft_dpo_oneshot.py（旧版：旧评测集 + 旧 prompt 的 `tool` 输出格式）。

数据源
  金标  all_eval.csv（domain 为文件前缀 vod/educ/edu/...；tool_param 已去 retext；
        tool_param_alt 为可接受的备选答案；同 query 多行时首行做 chosen，其余也算对）。
  主日志 hcSmith/inception/_raw/*.jsonl：当前 prompt（`vt` 输出格式），提供
        system/user（逐字取自日志，含「当前时间」行与【判定提示】）和 baseline 输出。
  辅日志 hcSmith/hcProxy/multi_intetion/_raw/*.jsonl：旧 prompt（`tool` 格式 + retext）。
        只用其 baseline 输出补充 DPO 的 rejected：先转成当前格式（vt=d:tool、去 retext），
        判错且与主日志 rejected 语义不同才收，prompt 仍用主日志那条。

样本
  SFT  每个能在主日志找到 prompt 的金标 query 一条；assistant = 金标计划
       [{"q", "d", "vt", "params"}]（紧凑 JSON，与模型线上输出同形）。
       q 取模型自己的改写（单条输出时），否则用原 query；
       系统代填参数（retext、audio_search.query、fan.messages）不进 chosen（与当前 prompt 一致）。
  DPO  baseline 判错 → chosen=金标计划，rejected=baseline 原始输出。
       判错口径：解析失败 / 空计划 / 多条 / 域:工具不同 / 参数语义不同。
       参数比对前经 app.parser 同款处理（去 retext、顶层参数折回），检索树按语义归一。

用法：python3 tools/gen_sft_dpo.py [--outdir testset/sftdpo] [--prefix oneshot]
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

from app import catalog  # noqa: E402
from app.parser import parse_json_loose, parse_oneshot  # noqa: E402

logging.getLogger("app.parser").setLevel(logging.WARNING)

WS = ROOT.parent
DEF_GOLD = ROOT / "all_eval.csv"
DEF_MAIN = str(WS / "hcSmith/inception/_raw/*.jsonl")
DEF_AUX = str(WS / "hcSmith/hcProxy/multi_intetion/_raw/*.jsonl")
DEF_OUT = ROOT / "testset" / "sftdpo"

DOMAIN_MAP = {"vod": "vod", "educ": "children", "edu": "education", "music": "music",
              "audio": "audio", "sports": "sports", "tv_control": "device"}
TOOL_ALIAS = {"vod_search": "vod_search_all", "educ_search": "educ_search_all",
              "educ_relate_recommend": "educ_relate_search",
              "sports_match_data_search": "sports_rank_search"}
# 自由文本透传字段：措辞天然各异，不参与比对；chosen 用模型措辞
FREE_TEXT = {("audio_search", "query"), ("fan_knowledge_agent", "messages")}

PLAN_SYS = "你是电视语音助手。读用户请求"
HINT = re.compile(r"\n【判定提示】.*$", re.S)


def nq(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def compact(o) -> str:
    return json.dumps(o, ensure_ascii=False, separators=(",", ":"))


def prune(v):
    """去空值（空串/None/空容器），与 prompt「用户没提的可选字段不要出现」一致。"""
    if isinstance(v, dict):
        out = {k: prune(x) for k, x in v.items()}
        return {k: x for k, x in out.items() if x not in ("", None, [], {})}
    if isinstance(v, list):
        out = [prune(x) for x in v]
        return [x for x in out if x not in ("", None, [], {})]
    return v.strip() if isinstance(v, str) else v


# ---------------------------------------------------------------------------
# 检索树语义归一（同旧版 gen_sft_dpo_oneshot.canon_query）
# ---------------------------------------------------------------------------
def canon_query(v):
    if isinstance(v, list):
        return [canon_query(x) for x in v]
    if not isinstance(v, dict):
        return v
    d = {k: canon_query(x) for k, x in v.items()}
    if "field" in d:
        if "values" in d:
            vals = d["values"] if isinstance(d["values"], list) else [d["values"]]
            if len(vals) == 1:
                d.pop("values", None)
                d.pop("operator", None)
                d["value"] = vals[0]
            else:
                d["values"] = sorted(vals) if all(isinstance(z, str) for z in vals) else vals
                if d.get("operator") not in ("and", "or"):
                    d["operator"] = "or"
        else:
            d.pop("operator", None)
        return d
    for lg in ("and", "or"):
        if lg in d and isinstance(d[lg], list):
            d[lg] = sorted(d[lg], key=lambda x: json.dumps(x, sort_keys=True, ensure_ascii=False))
    keys = set(d) - {"operator"}
    if keys in ({"and"}, {"or"}):
        lg = next(iter(keys))
        if isinstance(d[lg], list) and len(d[lg]) == 1:
            return d[lg][0]
    d.pop("operator", None)
    return d


def canon_params(p: dict, tool: str) -> str:
    o = {k: v for k, v in p.items() if k not in catalog.SYSTEM_FILLED_PARAMS
         and (tool, k) not in FREE_TEXT}
    o = prune(o)
    if "query" in o:
        o["query"] = canon_query(o["query"])
    return json.dumps(o, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------------
# 载入
# ---------------------------------------------------------------------------
def load_gold(path: Path, drops: list, stats: collections.Counter) -> dict[str, dict]:
    """{归一 query: {row, d, tool, params(chosen), accept:set(canon)}}。"""
    out: dict[str, dict] = {}
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        q = nq(r["query"])
        d = DOMAIN_MAP.get(r["domain"].strip())
        tool = TOOL_ALIAS.get(r["tool"].strip(), r["tool"].strip())
        why = ""
        if not q or not d or not tool:
            why = "bad_label"
        elif tool not in catalog.valid_tools_for(d):
            why = "tool_not_in_catalog"
        params = None
        if not why:
            tp = r["tool_param"].strip()
            if tp:
                params = parse_json_loose(tp)
                if not isinstance(params, dict):
                    why = "param_unparseable"
            elif tool == "fan_knowledge_agent":
                params = {}                      # messages 由模型措辞/q 补
            else:
                why = "param_missing"
        if why:
            stats[f"drop_gold_{why}"] += 1
            drops.append({"query": r["query"], "domain": r["domain"], "tool": r["tool"],
                          "tool_param": r["tool_param"], "reason": why})
            continue
        accept = {canon_params(params, tool)}
        alt = parse_json_loose(r.get("tool_param_alt") or "")
        for a in alt if isinstance(alt, list) else []:
            if isinstance(a, dict):
                accept.add(canon_params(a, tool))
        if q in out:                             # 同 query 多行：首行做 chosen，其余算对
            g = out[q]
            if g["d"] == d and g["tool"] == tool:
                g["accept"] |= accept
                stats["gold_dup_merged"] += 1
            else:
                stats["gold_dup_conflict_tool"] += 1
            continue
        out[q] = {"row": r, "d": d, "tool": tool, "params": prune(params), "accept": accept}
    return out


def iter_plan_calls(pattern: str):
    """逐条产出一把出首轮计划调用：(归一 query, messages[system,user], 原始输出, 是否新格式)。"""
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
            if out is None:
                continue
            m = re.search(r"用户请求：(.*)$", msgs[-1]["content"], re.S)
            if not m:
                continue
            q = nq(HINT.sub("", m.group(1)))
            new_fmt = '"vt"' in msgs[0]["content"][:2000]
            yield q, [{"role": "system", "content": msgs[0]["content"]},
                      {"role": "user", "content": msgs[-1]["content"]}], out, new_fmt


# ---------------------------------------------------------------------------
# 判定
# ---------------------------------------------------------------------------
def judge(raw: str, g: dict) -> tuple[str, str, list]:
    """返回 (verdict, 模型参数 canon, intents)；verdict=='match' 为正确。"""
    v = parse_json_loose(raw)
    if not isinstance(v, (list, dict)):
        return "parse_fail", "", []
    intents, err = parse_oneshot(raw)
    if err and not intents:
        return "parse_fail", "", []
    if not intents:
        return "empty_plan", "", []
    if len(intents) != 1:
        return "multi_step", "", intents
    it = intents[0]
    if not it.tool:
        return "tool_missing", "", intents
    if (it.d, it.tool) != (g["d"], g["tool"]):
        return "tool_wrong", "", intents
    mc = canon_params(it.params, it.tool)
    return ("match" if mc in g["accept"] else "param_diff"), mc, intents


def to_new_format(intents) -> str:
    """旧 prompt 的输出（tool/retext）转当前格式，供作 rejected。"""
    out = []
    for it in intents:
        item = {"q": it.q}
        if it.d:
            item["d"] = it.d
        if it.dep:
            item["dep"] = it.dep
        else:
            if it.vt:
                item["vt"] = it.vt
            item["params"] = it.params
        out.append(item)
    return compact(out)


def build_chosen(g: dict, intents, query: str) -> str:
    """金标计划。q 用模型自己的改写（单条输出时），否则用原 query。
    系统代填参数（retext、audio_search.query、fan.messages，见 catalog.system_filled）不进 chosen：
    当前 prompt 明确要求模型不填，chosen 里带上就是在教模型违背 prompt。"""
    q = intents[0].q if len(intents) == 1 and intents[0].q else query
    params = {k: v for k, v in g["params"].items() if k not in catalog.system_filled(g["tool"])}
    return compact([{"q": q, "d": g["d"], "vt": f"{g['d']}:{g['tool']}", "params": params}])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(DEF_GOLD))
    ap.add_argument("--main-log", default=DEF_MAIN)
    ap.add_argument("--aux-log", default=DEF_AUX, help="置空则不用辅日志")
    ap.add_argument("--outdir", default=str(DEF_OUT))
    ap.add_argument("--prefix", default="oneshot")
    ap.add_argument("--no-rebuild", action="store_true",
                    help="日志里没有的金标 query 直接丢弃，不用别条的 system 补 prompt（system 含动态 context 时必须开）")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    stats: collections.Counter = collections.Counter()
    drops: list[dict] = []
    gold = load_gold(Path(args.gold), drops, stats)

    # 主日志：每 query 取首条调用的 prompt；所有调用的输出都参与判定
    main_calls: dict[str, dict] = {}
    for q, msgs, out, new_fmt in iter_plan_calls(args.main_log):
        if not new_fmt:
            stats["main_skip_old_prompt"] += 1
            continue
        if q not in gold:
            stats["main_skip_no_gold"] += 1
            continue
        c = main_calls.setdefault(q, {"msgs": msgs, "outs": []})
        c["outs"].append(out)
    aux_outs: dict[str, list[str]] = collections.defaultdict(list)
    if args.aux_log:
        for q, _msgs, out, _new in iter_plan_calls(args.aux_log):
            if q in gold:
                aux_outs[q].append(out)
            else:
                stats["aux_skip_no_gold"] += 1

    # 主日志没跑到的金标 query（benchmark 时走了 MT 改写等）：prompt 用主日志同款 system +
    # 首条调用的「当前时间」行补齐；这类 query 的 rejected 只能来自辅日志。
    base_sys, now_line = "", ""
    if main_calls:
        first = next(iter(main_calls.values()))["msgs"]
        sys_cnt = collections.Counter(c["msgs"][0]["content"] for c in main_calls.values())
        base_sys = sys_cnt.most_common(1)[0][0]
        now_line = first[1]["content"].split("\n", 1)[0] if first[1]["content"].startswith("当前时间") else ""

    sft, dpo, audit = [], [], []
    by_domain = collections.Counter()
    for q, g in gold.items():
        c = main_calls.get(q)
        query = g["row"]["query"].strip()
        if not c:
            if args.no_rebuild:
                stats["no_main_log_dropped"] += 1
                continue
            if not base_sys:
                stats["no_main_log_dropped"] += 1
                continue
            stats["no_main_log_prompt_rebuilt"] += 1
            user = f"{now_line}\n用户请求：{query}" if now_line else f"用户请求：{query}"
            c = {"msgs": [{"role": "system", "content": base_sys},
                          {"role": "user", "content": user}], "outs": []}
        verdicts = [(raw, *judge(raw, g)) for raw in c["outs"]]
        # chosen 的 q/自由文本：优先取判对那次输出的措辞，其次首条；无主日志输出则用原 query
        ref = next((v for v in verdicts if v[1] == "match"), verdicts[0] if verdicts else None)
        chosen = build_chosen(g, ref[3] if ref else [], query)
        msgs = c["msgs"]
        sft.append({"messages": msgs + [{"role": "assistant", "content": chosen}]})
        by_domain[g["d"]] += 1

        seen_rej: set[str] = set()
        cands = [("main", raw, vd, mc, its) for raw, vd, mc, its in verdicts]
        for raw in aux_outs.get(q, []):
            vd, mc, its = judge(raw, g)
            cands.append(("aux", to_new_format(its) if its else raw, vd, mc, its))
        for src, raw, vd, mc, its in cands:
            rec = {"query": query, "domain": g["d"], "tool": g["tool"], "source": src,
                   "verdict": vd, "chosen": chosen, "model_output": raw,
                   "gold_params_canon": canon_params(g["params"], g["tool"]),
                   "model_params_canon": mc,
                   "model_vt": "|".join(it.vt or f"{it.d}:dep{it.dep}" for it in its)}
            stats[f"{src}_{vd}"] += 1
            if vd != "match":
                key = vd + "\x00" + (mc or compact([[i.q, i.vt, i.params, i.dep] for i in its])
                                     if its else raw)
                if key in seen_rej:
                    stats[f"{src}_dup_rejected"] += 1
                    rec["is_dpo"] = "0(dup)"
                else:
                    seen_rej.add(key)
                    dpo.append({"messages": msgs + [{"role": "assistant", "content": chosen}],
                                "rejected_response": raw})
                    stats[f"dpo_from_{src}"] += 1
                    rec["is_dpo"] = "1"
            audit.append(rec)

    def write_jsonl(name, rows):
        with open(outdir / name, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_jsonl(f"{args.prefix}.sft.jsonl", sft)
    write_jsonl(f"{args.prefix}.dpo.jsonl", dpo)
    write_jsonl(f"{args.prefix}.dropped.jsonl", drops)
    cols = ["query", "domain", "tool", "source", "verdict", "is_dpo", "model_vt",
            "gold_params_canon", "model_params_canon", "chosen", "model_output"]
    for name, rows in ((f"{args.prefix}.audit.csv", audit),
                       (f"{args.prefix}.dpo.csv", [r for r in audit if r.get("is_dpo") == "1"])):
        with open(outdir / name, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    main_v = collections.Counter(r["verdict"] for r in audit if r["source"] == "main")
    report = {"gold": args.gold, "main_log": args.main_log, "aux_log": args.aux_log,
              "gold_queries": len(gold), "sft": len(sft), "dpo": len(dpo),
              "main_verdicts": dict(main_v),
              "main_accuracy": round(main_v["match"] / max(1, sum(main_v.values())), 4),
              "counts": dict(sorted(stats.items())), "sft_by_domain": dict(by_domain)}
    (outdir / f"{args.prefix}.report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
