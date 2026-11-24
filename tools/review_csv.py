#!/usr/bin/env python3
"""评测结果 → 人工审核 CSV：金标 vs 预测、工具/参数是否正确、可读的参数 diff。

判定口径与 tools/eval_stats.py 完全一致（同一 judge / canon）。
diff 规则（在归一化参数上比）：
  · 检索树 query 拆成叶子条件集合（field=value / field∈[...] / field:from~to，not 下的记为 NOT(...)），
    按集合比：「缺」= 金标有预测无，「多」= 预测有金标无；
  · 其余顶层参数按键比：缺 / 多 / 值不同（金标→预测）。
排序：工具错 → 参数错 → 漏标 → 正确；同类按域、金标工具。Excel 直接打开（utf-8-sig）。

用法：python3 tools/review_csv.py --input all_eval_result_rag_compact4.csv [--output xxx.review.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from eval_stats import DOMAIN_ORDER, judge  # noqa: E402


def _leafs(node, neg: bool = False) -> list[str]:
    """检索树 → 叶子条件字符串列表（忽略 and/or 结构，只看条件集合）。"""
    out: list[str] = []
    if isinstance(node, list):
        for x in node:
            out += _leafs(x, neg)
    elif isinstance(node, dict):
        if "field" in node:
            f = node["field"]
            if "values" in node:
                s = f"{f}∈[{','.join(map(str, node['values']))}]"
                if node.get("operator") == "and":
                    s += "(and)"
            elif "from" in node or "to" in node:
                s = f"{f}:{node.get('from', '')}~{node.get('to', '')}"
            else:
                s = f"{f}={node.get('value', '')}"
            out.append(f"NOT({s})" if neg else s)
        else:
            for k, v in node.items():
                out += _leafs(v, neg ^ (k == "not"))
    elif node not in (None, ""):
        out.append(f"(非检索树:{node})")
    return out


def _fmt(v) -> str:
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, separators=(",", ":"))


def param_diff(gold: dict, pred: dict) -> str:
    parts = []
    if "query" in gold or "query" in pred:
        g, p = _leafs(gold.get("query")), _leafs(pred.get("query"))
        miss = [x for x in g if x not in p]
        extra = [x for x in p if x not in g]
        if miss:
            parts.append("query缺: " + "；".join(miss))
        if extra:
            parts.append("query多: " + "；".join(extra))
        if not miss and not extra and json.dumps(gold.get("query"), sort_keys=True, ensure_ascii=False) \
                != json.dumps(pred.get("query"), sort_keys=True, ensure_ascii=False):
            parts.append("query条件相同但 and/or/not 结构不同")
    for k in sorted(set(gold) | set(pred)):
        if k == "query":
            continue
        if k not in pred:
            parts.append(f"缺 {k}={_fmt(gold[k])}")
        elif k not in gold:
            parts.append(f"多 {k}={_fmt(pred[k])}")
        elif gold[k] != pred[k]:
            parts.append(f"{k}: {_fmt(gold[k])} → {_fmt(pred[k])}")
    return " | ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "all_eval_result_rag_compact4.csv"))
    ap.add_argument("--gold", default=str(ROOT / "all_eval.csv"), help="置空则用结果文件自带金标列")
    ap.add_argument("--output", default="")
    args = ap.parse_args()
    src = Path(args.input)
    rows = list(csv.DictReader(open(src, encoding="utf-8-sig")))
    if args.gold:                                   # 同 eval_stats：按去空白 query 覆盖金标列
        import re
        gold = {}
        for g in csv.DictReader(open(args.gold, encoding="utf-8-sig")):
            gold.setdefault(re.sub(r"\s+", "", g["query"]), g)
        for r in rows:
            g = gold.get(re.sub(r"\s+", "", r["query"]))
            if g:
                for c in ("tool", "tool_param", "tool_param_alt"):
                    r[c] = g.get(c, "")

    out_rows = []
    for r in rows:
        j = judge(r)
        if j["tool_ok"] and j["tool_param_ok"]:
            verdict = "正确"
        elif not j["tool_ok"]:
            verdict = "工具错"
        elif not j["param_scored"]:
            verdict = "漏标(不计参数)"
        else:
            verdict = "参数错"
        diff = ""
        if j["tool_ok"] and j["param_scored"] and not j["tool_param_ok"]:
            g = json.loads(j["gold_canon"] or "{}")
            p = json.loads(j["pred_canon"] or "{}")
            diff = param_diff(g, p) or "（归一后写法差异，见规范化列）"
        elif not j["tool_ok"]:
            diff = f"工具: {j['gold_tool']} → {j['pred_tool'] or '(空)'}"
        out_rows.append({
            "判定": verdict, "域": r["domain"], "query": r["query"],
            "金标工具": j["gold_tool"], "预测工具(首步)": j["pred_tool"], "工具链": j["chain"],
            "工具正确": "是" if j["tool_ok"] else "否",
            "参数正确": ("是" if j["tool_param_ok"] else "否") if j["param_scored"] else "-",
            "diff": diff,
            "金标参数": r.get("tool_param", ""), "金标备选": r.get("tool_param_alt", ""),
            "预测参数": r.get("参数", ""),
            "金标(规范化)": j["gold_canon"], "预测(规范化)": j["pred_canon"],
            "人工结论": "", "备注": "",
        })
    rank = {"工具错": 0, "参数错": 1, "漏标(不计参数)": 2, "正确": 3}
    dom = {d: i for i, d in enumerate(DOMAIN_ORDER)}
    out_rows.sort(key=lambda x: (rank[x["判定"]], dom.get(x["域"], 99), x["金标工具"], x["query"]))
    out = Path(args.output) if args.output else src.with_suffix(".review.csv")
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    from collections import Counter
    c = Counter(x["判定"] for x in out_rows)
    print(f"{len(out_rows)} 行 -> {out}\n  " + "  ".join(f"{k}:{c[k]}" for k in rank))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
