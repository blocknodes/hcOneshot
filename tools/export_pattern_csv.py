#!/usr/bin/env python3
"""把 samples.jsonl 导出成训练 CSV（合并 train.csv + 按域 train_<域>.csv），可在生成中途随时跑。
用法：python3 tools/export_pattern_csv.py [目录，默认 testset/train_pattern_v2] [源文件，默认有 samples.clean.jsonl 就用它]
"""
import collections, csv, json, sys
from pathlib import Path
d = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "testset/train_pattern_v2")
src = sys.argv[2] if len(sys.argv) > 2 else ("samples.clean.jsonl" if (d / "samples.clean.jsonl").exists() else "samples.jsonl")
rows = [json.loads(l) for l in open(d / src, encoding="utf-8") if l.strip()]
hdr = ["domain", "pid", "query", "tool", "tool_param", "seed_query", "replaced"]
def row(s):
    return [s["src_domain"], s["pid"], s["query"], s["tool"],
            json.dumps(s["params"], ensure_ascii=False, separators=(",", ":")), s["seed_query"],
            json.dumps(s.get("replaced") or {}, ensure_ascii=False)]
rows.sort(key=lambda s: (s["src_domain"], s["pid"], s["seed_query"]))
byd = collections.defaultdict(list)
for s in rows: byd[s["src_domain"]].append(s)
with open(d / "train.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f); w.writerow(hdr); w.writerows(row(s) for s in rows)
for dom, l in byd.items():
    with open(d / f"train_{dom}.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f); w.writerow(hdr); w.writerows(row(s) for s in l)
print(len(rows), {k: len(v) for k, v in byd.items()})
