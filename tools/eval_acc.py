#!/usr/bin/env python3
"""评测结果打分：tool 准确率 / tool+param 准确率（按域、按工具、混淆、错例）。

输入：评测结果 CSV（列：domain, query, tool, tool_param, tool_param_alt, 工具链, 参数 …）
  · 金标：tool / tool_param（tool_param_alt 为「或者」备选，命中任一即对）；
  · 预测：工具链首个工具（`A | B` 中 B 是 turnSlow 兜底，不计）、参数数组第 1 项。

口径
  · tool：金标经 TOOL_ALIAS 归一（vod_search→vod_search_all 等），与预测首工具相等。
  · tool+param：tool 对，且参数语义一致：
      - 去系统代填字段：retext、audio_search.query、fan 的 messages/riskRespStrategy/sceneTypeIntentList；
      - 去空值（""/[]/{}/None）；query 布尔树语义归一（单子节点解包、单值 values→value、
        and/or 子节点顺序无关）—— 复用 hcSlime/benchmark/gen_sft_dpo.canon_params；
      - strict：标量原样比；loose：标量统一转小写字符串（"0" 与 0、"CEC" 与 "cec" 视为同）。
  · 金标参数缺失（非 fan 且 tool_param 为空，漏标）的行：计入 tool，不计入 tool+param 分母。

用法：python3 tools/eval_acc.py all_eval_result_rag.csv [--out 错例.csv]
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "hcSlime" / "benchmark"))
from gen_sft_dpo import canon_params  # noqa: E402

TOOL_ALIAS = {"vod_search": "vod_search_all", "educ_search": "educ_search_all",
              "educ_relate_recommend": "educ_relate_search",
              "sports_match_data_search": "sports_rank_search", "edu_search_all": "edu_search"}
DROP_ALL = {"retext"}
DROP_BY_TOOL = {"audio_search": {"query"},
                "fan_knowledge_agent": {"messages", "riskRespStrategy", "sceneTypeIntentList"}}
DOMAIN_ORDER = ["vod", "educ", "edu", "music", "audio", "sports", "tv_control"]


def loads(s):
    try:
        return json.loads(s) if s and s.strip() else None
    except json.JSONDecodeError:
        return None


def _loose(v):
    if isinstance(v, dict):
        return {k: _loose(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_loose(x) for x in v]
    if v is None:
        return None
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return str(v).strip().lower()


def canon(p, tool, loose=False):
    if not isinstance(p, dict):
        p = {}
    drop = DROP_ALL | DROP_BY_TOOL.get(tool, set())
    p = {k: v for k, v in p.items() if k not in drop}
    if loose:
        p = _loose(p)
    return canon_params(p, query_keys=("query",))


def pred_of(r):
    chain = (r.get("工具链") or "").split("|")[0].strip()
    ps = loads(r.get("参数") or "")
    p0 = ps[0] if isinstance(ps, list) and ps else (ps if isinstance(ps, dict) else {})
    return chain, p0


def pct(a, b):
    return f"{a / b * 100:5.1f}%" if b else "   - "


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.csv, encoding="utf-8-sig")))

    tot = collections.Counter()
    by_dom = collections.defaultdict(collections.Counter)
    by_tool = collections.defaultdict(collections.Counter)
    conf = collections.Counter()
    errs = []
    for r in rows:
        g_tool = TOOL_ALIAS.get(r["tool"], r["tool"])
        p_tool, p_par = pred_of(r)
        gp = loads(r["tool_param"])
        alts = loads(r.get("tool_param_alt") or "") or []
        labeled = gp is not None or g_tool == "fan_knowledge_agent"
        golds = [gp or {}] + [a for a in alts if isinstance(a, dict)]
        t_ok = p_tool == g_tool
        s_ok = t_ok and canon(p_par, g_tool) in {canon(g, g_tool) for g in golds}
        l_ok = t_ok and canon(p_par, g_tool, True) in {canon(g, g_tool, True) for g in golds}
        for c in (tot, by_dom[r["domain"]], by_tool[g_tool]):
            c["n"] += 1
            c["tool"] += t_ok
            if labeled:
                c["pn"] += 1
                c["strict"] += s_ok
                c["loose"] += l_ok
            else:
                c["unlabeled"] += 1
        if not t_ok:
            conf[(g_tool, p_tool or "<空>")] += 1
        if not t_ok or (labeled and not l_ok):
            errs.append({"domain": r["domain"], "query": r["query"], "err_type":
                         "tool" if not t_ok else "param", "gold_tool": g_tool, "pred_tool": p_tool,
                         "gold_param": canon(gp or {}, g_tool, True),
                         "pred_param": canon(p_par, g_tool, True)})

    print(f"文件 {args.csv}  共 {tot['n']} 条（金标参数缺失 {tot['unlabeled']} 条，只计 tool）\n")
    hdr = f"{'domain':>11} {'n':>5} {'tool':>7} {'tool+param(strict)':>19} {'tool+param(loose)':>18}"
    print(hdr)
    print("-" * len(hdr))
    for d in [x for x in DOMAIN_ORDER if x in by_dom] + [x for x in by_dom if x not in DOMAIN_ORDER]:
        c = by_dom[d]
        print(f"{d:>11} {c['n']:>5} {pct(c['tool'], c['n']):>7} {pct(c['strict'], c['pn']):>19} "
              f"{pct(c['loose'], c['pn']):>18}")
    print("-" * len(hdr))
    print(f"{'ALL':>11} {tot['n']:>5} {pct(tot['tool'], tot['n']):>7} {pct(tot['strict'], tot['pn']):>19} "
          f"{pct(tot['loose'], tot['pn']):>18}")

    print("\n按工具（金标工具，n≥20，按 tool 准确率升序）")
    print(f"{'tool':>38} {'n':>5} {'tool':>7} {'t+p(loose)':>11}")
    for t, c in sorted(by_tool.items(), key=lambda kv: kv[1]["tool"] / kv[1]["n"]):
        if c["n"] >= 20:
            print(f"{t:>38} {c['n']:>5} {pct(c['tool'], c['n']):>7} {pct(c['loose'], c['pn']):>11}")

    print("\nTop 工具混淆（金标 -> 预测）")
    for (g, p), n in conf.most_common(15):
        print(f"   {n:4d}  {g} -> {p}")

    out = args.out or str(Path(args.csv).with_name(Path(args.csv).stem + "_errors.csv"))
    with open(out, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["domain", "query", "err_type", "gold_tool", "pred_tool",
                                           "gold_param", "pred_param"])
        w.writeheader()
        w.writerows(errs)
    print(f"\n错例 {len(errs)} 条 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
