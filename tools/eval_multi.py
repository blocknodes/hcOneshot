#!/usr/bin/env python3
"""多意图（串行/并行）评测统计：bench_serial_*_result.csv。

输入列：tools（金标工具，"+" 连接）、tool_param（金标参数 JSON 数组，null=该步未标）、exec_mode、n_steps、
        source、tools_ok、工具链（预测，" | " 连接）、参数（预测参数 JSON 数组）。
指标（参数比对口径同 tools/eval_stats.canon：去 retext 等代填、检索树语义归一、数值≈数字字符串）：
  步数正确      预测步数 == 金标步数
  工具链(有序)  预测工具序列 == 金标序列
  工具链(无序)  多重集合相等（并行不要求顺序）
  步级工具      按位置对齐的工具准确率（分母=金标步数）
  步级参数      工具对且该步金标参数非 null 的步里，参数一致的比例
  整条全对      工具链(并行用无序/串行用有序) 正确，且所有已标参数的步参数一致
用法：python3 tools/eval_multi.py [--input bench_serial_result.csv] [--detail out.csv]
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from app.parser import parse_json_loose  # noqa: E402
from eval_stats import canon  # noqa: E402
from gen_sft_dpo import TOOL_ALIAS  # noqa: E402
from clean_testset import _strip_k12  # noqa: E402


def _tools(s: str, sep: str) -> list[str]:
    return [TOOL_ALIAS.get(t.strip(), t.strip()) for t in (s or "").split(sep) if t.strip()]


def judge(r: dict) -> dict:
    gt, pt = _tools(r["tools"], "+"), _tools(r.get("工具链", ""), "|")
    gp = parse_json_loose(r.get("tool_param") or "")
    gp = gp if isinstance(gp, list) else [None] * len(gt)
    # 与 all_eval 同口径：去掉 edu 默认补的 k12_category=[同步课,同步拓展课]（query 说了「同步」的保留）
    gp = [_strip_k12(x, r.get("query") or "")[0] if isinstance(x, dict) else x for x in gp]
    pp = parse_json_loose(r.get("参数") or "")
    pp = pp if isinstance(pp, list) else []
    parallel = r.get("exec_mode") == "parallel"
    ordered = pt == gt
    unordered = collections.Counter(pt) == collections.Counter(gt)
    chain_ok = unordered if parallel else ordered

    # 参数按工具对齐：串行按位置；并行按工具名配对（同名按出现顺序）
    pairs: list[tuple[int, int | None]] = []
    if parallel:
        used: set[int] = set()
        for i, t in enumerate(gt):
            j = next((j for j, x in enumerate(pt) if x == t and j not in used), None)
            if j is not None:
                used.add(j)
            pairs.append((i, j))
    else:
        pairs = [(i, i if i < len(pt) and pt[i] == gt[i] else None) for i in range(len(gt))]
    step_tool = sum(1 for _, j in pairs if j is not None)
    scored = ok = 0
    bad = []
    for i, j in pairs:
        g = gp[i] if i < len(gp) else None
        if not isinstance(g, dict) or j is None:
            continue
        scored += 1
        p = pp[j] if j < len(pp) and isinstance(pp[j], dict) else {}
        if canon(g, gt[i]) == canon(p, gt[i]):
            ok += 1
        else:
            bad.append(f"#{i + 1} {gt[i]}: 金标{canon(g, gt[i])} 预测{canon(p, gt[i])}")
    all_ok = chain_ok and ok == scored
    return {"n_gold": len(gt), "n_pred": len(pt), "steps_ok": len(pt) == len(gt),
            "ordered_ok": ordered, "unordered_ok": unordered, "chain_ok": chain_ok,
            "step_tool": step_tool, "scored": scored, "param_ok": ok, "all_ok": all_ok,
            "gold_chain": "+".join(gt), "pred_chain": "+".join(pt), "param_diff": " ; ".join(bad)}


def pct(a, b):
    return f"{a / b * 100:5.1f}%" if b else "   - "


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "bench_serial_result.csv"))
    ap.add_argument("--detail", default="")
    args = ap.parse_args()
    src = Path(args.input)
    rows = list(csv.DictReader(open(src, encoding="utf-8-sig")))
    js = [judge(r) for r in rows]

    def table(title, keyf):
        st: dict = collections.defaultdict(collections.Counter)
        for r, j in zip(rows, js):
            for k in (keyf(r), "ALL"):
                s = st[k]
                s["n"] += 1
                for f in ("steps_ok", "ordered_ok", "unordered_ok", "chain_ok", "all_ok"):
                    s[f] += j[f]
                s["gs"] += j["n_gold"]
                s["st"] += j["step_tool"]
                s["sc"] += j["scored"]
                s["po"] += j["param_ok"]
        print(f"\n== {title}")
        print(f"{'':<16}{'样本':>6}{'步数对':>8}{'链有序':>8}{'链无序':>8}{'步级工具':>9}{'步级参数':>9}{'整条全对':>9}")
        for k in sorted(st, key=lambda x: (x == "ALL", -st[x]["n"], x)):
            s = st[k]
            print(f"{str(k)[:16]:<16}{s['n']:>6}{pct(s['steps_ok'], s['n']):>8}{pct(s['ordered_ok'], s['n']):>8}"
                  f"{pct(s['unordered_ok'], s['n']):>8}{pct(s['st'], s['gs']):>9}{pct(s['po'], s['sc']):>9}"
                  f"{pct(s['all_ok'], s['n']):>9}")

    table("按执行模式", lambda r: r["exec_mode"])
    table("按步数", lambda r: f"{r['n_steps']}步")
    table("按来源", lambda r: r["source"])
    table("按金标工具是否校验通过(tools_ok)", lambda r: f"tools_ok={r['tools_ok']}")
    table("按域组合 Top", lambda r: r["gold"])

    errs = collections.Counter()
    for j in js:
        if not j["steps_ok"]:
            errs[f"步数 {j['n_gold']}→{j['n_pred']}"] += 1
        elif not j["chain_ok"]:
            errs["步数对但工具错"] += 1
        elif not j["all_ok"]:
            errs["工具对但参数错"] += 1
    print("\n== 错误构成", dict(errs.most_common()))
    out = Path(args.detail) if args.detail else src.with_suffix(".detail.csv")
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        cols = ["id", "query", "seg", "exec_mode", "n_steps", "source", "tools_ok"]
        w = csv.writer(f)
        w.writerow(cols + ["gold_chain", "pred_chain", "steps_ok", "chain_ok", "all_ok", "param_diff",
                           "tool_param", "参数"])
        for r, j in zip(rows, js):
            w.writerow([r.get(c, "") for c in cols] + [j["gold_chain"], j["pred_chain"], j["steps_ok"],
                                                       j["chain_ok"], j["all_ok"], j["param_diff"],
                                                       r.get("tool_param", ""), r.get("参数", "")])
    print(f"明细: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
