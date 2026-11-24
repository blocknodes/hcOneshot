#!/usr/bin/env python3
"""统计评测结果各域准确率：tool 准确率、tool+param 准确率。

输入：all_eval_result.csv（列 domain/query/tool/tool_param/tool_param_alt + 工具链/参数）
  工具链  实际执行的工具，多步用 " | " 分隔（如 turnSlow 兜底 `vod_search_all | vod_fuzzy_search`）
  参数    JSON 数组，与工具链逐位对应

判定口径（与 tools/gen_sft_dpo.py 同一套 canon_params）
  tool        金标工具（经 TOOL_ALIAS 归一）== 工具链【第 1 个】工具（模型首轮的选择）
  tool+param  tool 对，且第 1 步参数与金标（或 tool_param_alt 备选）语义一致：
              不比 retext / audio_search.query / fan.messages（系统代填），空值不计，
              检索树按语义归一（单子节点 and/or、values→value、子节点顺序等写法差异不算错）；
              叶子数值与数字字符串等价（is_fee 0 == "0"）；fan 参数由引擎重建，只比工具
  另给两个参考口径：
    tool(严格)  工具链只有 1 步且等于金标（多步/兜底都算错）
    tool(链含)  工具链任一步等于金标
  金标漏标（非 fan 且 tool_param 为空）的行不计入 tool+param 分母（单列 param_missing）。

金标默认取 all_eval.csv（按 query 覆盖结果文件自带的金标列，便于金标修订后直接重算）。

用法：python3 tools/eval_stats.py [--input all_eval_result.csv] [--gold all_eval.csv] [--detail out.csv]
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.parser import parse_json_loose  # noqa: E402
from tools.gen_sft_dpo import TOOL_ALIAS, canon_params  # noqa: E402

DOMAIN_ORDER = ["vod", "educ", "edu", "music", "audio", "sports", "tv_control"]


def _num_to_str(o):
    """叶子数值转字符串：金标里 is_fee 等偶有写成整数 0 的，schema 要求字符串 "0"，语义相同。"""
    if isinstance(o, dict):
        return {k: _num_to_str(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_num_to_str(x) for x in o]
    if isinstance(o, (int, float)) and not isinstance(o, bool):
        return str(o)
    return o


def canon(p, tool: str) -> str:
    # fan_knowledge_agent 的参数全部由引擎重建（messages/riskRespStrategy/sceneTypeIntentList），
    # 不是模型输出，不参与比对。
    if tool == "fan_knowledge_agent":
        return "{}"
    return canon_params(_num_to_str(p), tool)


def judge(r: dict) -> dict:
    gold_tool = TOOL_ALIAS.get(r["tool"].strip(), r["tool"].strip())
    chain = [TOOL_ALIAS.get(t.strip(), t.strip()) for t in (r.get("工具链") or "").split("|") if t.strip()]
    params = parse_json_loose(r.get("参数") or "")
    params = params if isinstance(params, list) else []
    first = chain[0] if chain else ""
    out = {"gold_tool": gold_tool, "pred_tool": first, "chain": " | ".join(chain),
           "tool_ok": first == gold_tool,
           "tool_strict_ok": chain == [gold_tool],
           "tool_any_ok": gold_tool in chain}

    tp = (r.get("tool_param") or "").strip()
    if tp:
        gp = parse_json_loose(tp)
        gold_ok = isinstance(gp, dict)
    elif gold_tool == "fan_knowledge_agent":
        gp, gold_ok = {}, True                     # fan 只标工具
    else:
        gp, gold_ok = None, False                  # 漏标
    out["param_scored"] = gold_ok
    out["gold_canon"] = canon(gp, gold_tool) if gold_ok else ""
    p0 = params[0] if params and isinstance(params[0], dict) else None
    out["pred_canon"] = canon(p0, gold_tool) if p0 is not None else ""
    ok = False
    if gold_ok and out["tool_ok"] and p0 is not None:
        accept = {out["gold_canon"]}
        alt = parse_json_loose(r.get("tool_param_alt") or "")
        for a in alt if isinstance(alt, list) else []:
            if isinstance(a, dict):
                accept.add(canon(a, gold_tool))
        ok = out["pred_canon"] in accept
    out["tool_param_ok"] = ok
    return out


def pct(a: int, b: int) -> str:
    return f"{a / b * 100:5.1f}%" if b else "   - "


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "all_eval_result.csv"))
    ap.add_argument("--gold", default=str(ROOT / "all_eval.csv"),
                    help="按 query 用该文件的 tool/tool_param/tool_param_alt 覆盖结果文件里的金标列；置空则用结果文件自带的")
    ap.add_argument("--detail", default="", help="逐行判定明细 CSV（默认 <input>.detail.csv）")
    args = ap.parse_args()
    src = Path(args.input)
    rows = list(csv.DictReader(open(src, encoding="utf-8-sig")))
    if args.gold:
        # 按去空白的 query 匹配；金标里同一 query 多行时，首行做主答案，其余同工具的行并入备选
        gold: dict[str, list[dict]] = collections.defaultdict(list)
        for g in csv.DictReader(open(args.gold, encoding="utf-8-sig")):
            gold[re.sub(r"\s+", "", g["query"])].append(g)
        hit = 0
        for r in rows:
            gs = gold.get(re.sub(r"\s+", "", r["query"]))
            if not gs:
                continue
            hit += 1
            g0 = gs[0]
            alts = []
            for g in gs:
                a = parse_json_loose(g.get("tool_param_alt") or "")
                alts += a if isinstance(a, list) else []
                if g is not g0 and g["tool"] == g0["tool"] and g["tool_param"].strip():
                    p = parse_json_loose(g["tool_param"])
                    if isinstance(p, dict):
                        alts.append(p)
            r["tool"], r["tool_param"] = g0["tool"], g0["tool_param"]
            r["tool_param_alt"] = json.dumps(alts, ensure_ascii=False) if alts else ""
        print(f"金标: {args.gold}（按 query 覆盖 {hit}/{len(rows)} 行）\n")

    stat: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    tstat: dict[tuple[str, str], collections.Counter] = collections.defaultdict(collections.Counter)
    wrong: dict[tuple[str, str], collections.Counter] = collections.defaultdict(collections.Counter)
    detail = []
    for r in rows:
        j = judge(r)
        tk = (r["domain"].strip(), j["gold_tool"])
        if not j["tool_ok"]:
            wrong[tk][j["pred_tool"] or "(空)"] += 1
        for s in (stat[r["domain"].strip()], stat["ALL"], tstat[tk]):
            s["n"] += 1
            s["tool"] += j["tool_ok"]
            s["tool_strict"] += j["tool_strict_ok"]
            s["tool_any"] += j["tool_any_ok"]
            s["n_param"] += j["param_scored"]
            s["tool_param"] += j["tool_param_ok"]
            s["param_missing"] += not j["param_scored"]
        detail.append({**{k: r.get(k, "") for k in ("domain", "query", "tool", "tool_param")}, **j})

    doms = [d for d in DOMAIN_ORDER if d in stat] + sorted(set(stat) - set(DOMAIN_ORDER) - {"ALL"}) + ["ALL"]
    head = f"{'域':<11}{'样本':>6}{'tool':>9}{'tool+param':>12}{'tool(严格)':>11}{'tool(链含)':>11}{'漏标':>6}"
    print(head)
    print("-" * len(head.encode("gbk")))
    for d in doms:
        s = stat[d]
        print(f"{d:<11}{s['n']:>6}{pct(s['tool'], s['n']):>9}{pct(s['tool_param'], s['n_param']):>12}"
              f"{pct(s['tool_strict'], s['n']):>11}{pct(s['tool_any'], s['n']):>11}{s['param_missing']:>6}")
    print("\ntool+param 分母 = 样本 - 漏标；tool 取工具链第 1 步。")

    # ---- 按金标工具 ----
    dom_rank = {d: i for i, d in enumerate(DOMAIN_ORDER)}
    tkeys = sorted(tstat, key=lambda k: (dom_rank.get(k[0], 99), -tstat[k]["n"], k[1]))
    trows = []
    for k in tkeys:
        s = tstat[k]
        top = wrong[k].most_common(2)
        trows.append({"domain": k[0], "tool": k[1], "n": s["n"],
                      "tool_acc": round(s["tool"] / s["n"], 4),
                      "tool_param_acc": round(s["tool_param"] / s["n_param"], 4) if s["n_param"] else "",
                      "tool_strict_acc": round(s["tool_strict"] / s["n"], 4),
                      "tool_any_acc": round(s["tool_any"] / s["n"], 4),
                      "param_missing": s["param_missing"],
                      "top_wrong_tool": "; ".join(f"{t}×{c}" for t, c in top)})
    print(f"\n{'域':<11}{'金标工具':<40}{'样本':>5}{'tool':>8}{'tool+param':>11}{'漏标':>5}  错选最多")
    for t in trows:
        tp = f"{t['tool_param_acc'] * 100:5.1f}%" if t["tool_param_acc"] != "" else "   - "
        print(f"{t['domain']:<11}{t['tool']:<40}{t['n']:>5}{t['tool_acc'] * 100:7.1f}%{tp:>11}"
              f"{t['param_missing']:>5}  {t['top_wrong_tool']}")

    out = Path(args.detail) if args.detail else src.with_suffix(".detail.csv")
    cols = ["domain", "query", "tool", "gold_tool", "pred_tool", "chain", "tool_ok", "tool_param_ok",
            "tool_strict_ok", "tool_any_ok", "param_scored", "gold_canon", "pred_canon", "tool_param"]
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(detail)
    summary = {d: {k: stat[d][k] for k in ("n", "tool", "tool_param", "n_param", "tool_strict", "tool_any")}
               | {"tool_acc": round(stat[d]["tool"] / stat[d]["n"], 4),
                  "tool_param_acc": round(stat[d]["tool_param"] / max(1, stat[d]["n_param"]), 4)}
               for d in doms}
    by_tool = out.with_name(out.stem + ".by_tool.csv")
    with open(by_tool, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(trows[0]))
        w.writeheader()
        w.writerows(trows)
    out.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                                 encoding="utf-8")
    print(f"明细: {out}\n按工具: {by_tool}\n汇总: {out.with_suffix('.summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
