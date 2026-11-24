#!/usr/bin/env python3
"""runtime_execute.py 产出（期望工具/期望参数/预测工具/预测参数）打分，口径同 tools/eval_stats.judge。

用法：python3 tools/score_rt.py a.csv [b.csv ...]   # 多个文件并排对比各域 tool / tool+param
      --detail 输出逐行判定到 <csv>.judge.csv
"""
from __future__ import annotations

import argparse
import collections
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from eval_stats import DOMAIN_ORDER, judge  # noqa: E402


def score(path: str, detail: bool = False):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    st = collections.defaultdict(collections.Counter)
    out = []
    for r in rows:
        x = {"tool": r.get("期望工具", ""), "tool_param": r.get("期望参数", ""),
             "工具链": r.get("预测工具") or r.get("工具链", ""), "参数": r.get("预测参数") or r.get("参数", ""),
             "domain": r.get("domain", "")}
        j = judge(x)
        d = r["domain"].strip()
        for s in (st[d], st["ALL"]):
            s["n"] += 1
            s["tool"] += j["tool_ok"]
            s["n_param"] += j["param_scored"]
            s["tp"] += j["tool_param_ok"]
        out.append({"domain": d, "query": r["query"], "gold_tool": j["gold_tool"], "pred_tool": j["pred_tool"],
                    "tool_ok": int(j["tool_ok"]), "tool_param_ok": int(j["tool_param_ok"]),
                    "gold_canon": j["gold_canon"], "pred_canon": j["pred_canon"], "err": r.get("err", "")})
    if detail:
        p = Path(path).with_suffix(".judge.csv")
        with open(p, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--detail", action="store_true")
    a = ap.parse_args()
    sts = [score(p, a.detail) for p in a.csv]
    doms = [d for d in DOMAIN_ORDER if d in sts[0]] + ["ALL"]
    print(f"{'域':<11}{'n':>6}" + "".join(f"  {Path(p).stem[:28]:>28}" for p in a.csv))
    for d in doms:
        cells = []
        for s in sts:
            s = s[d]
            cells.append(f"t {s['tool'] / max(1, s['n']) * 100:5.1f} tp {s['tp'] / max(1, s['n_param']) * 100:5.1f}")
        print(f"{d:<11}{sts[0][d]['n']:>6}" + "".join(f"  {c:>28}" for c in cells))


if __name__ == "__main__":
    main()
