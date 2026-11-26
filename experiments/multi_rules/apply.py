#!/usr/bin/env python3
"""把 raw_plans.json（可选经 rules.postprocess）写成评测 csv（金标列取自 3domain_revised.csv，预测列=工具链/参数），
再交给 tools/eval_3domain.py 打分。用法：python3 experiments/multi_rules/apply.py [--no-rules] [--out X.csv]"""
import argparse, csv, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
import rules  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--no-rules", action="store_true")
ap.add_argument("--raw", default=str(Path(__file__).parent / "raw_plans.json"))
ap.add_argument("--gold", default=str(ROOT / "3domain_revised.csv"))
ap.add_argument("--out", default=str(Path(__file__).parent / "result.csv"))
a = ap.parse_args()
raw = {r["query"]: r for r in json.load(open(a.raw, encoding="utf-8"))}
rows = list(csv.DictReader(open(a.gold, encoding="utf-8-sig")))
for r in rows:
    steps = raw[r["query"]]["steps"]
    if not a.no_rules:
        steps = rules.postprocess(r["query"], steps)
    r["工具链"] = " | ".join(s["tool"] for s in steps)
    r["参数"] = json.dumps([s.get("params") or {} for s in steps], ensure_ascii=False)
    r["err"] = ""
with open(a.out, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
