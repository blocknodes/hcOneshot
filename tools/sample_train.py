#!/usr/bin/env python3
"""从 train.csv 按种子分组，每个种子随机抽 N 条样本。

分组键：seed_query + tool（与 gen_train.py 生成时的种子口径一致）。
不足 N 条的种子全取。固定随机种子，结果可复现；输出保持原文件的列与行序。

用法：
  python3 tools/sample_train.py --n 3
  python3 tools/sample_train.py --n 5 --input testset/feishu/train.csv --output out.csv --seed 42
默认输出：<input 同目录>/train_n{N}.csv
"""
from __future__ import annotations

import argparse
import collections
import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOMAIN_ORDER = ["vod", "educ", "edu", "music", "audio", "sports", "tv_control"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True, help="每个种子最多抽几条")
    ap.add_argument("--input", default=str(ROOT / "testset" / "feishu" / "train.csv"))
    ap.add_argument("--output", default="", help="默认 <input 同目录>/train_n{N}.csv")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    args = ap.parse_args()
    if args.n < 1:
        raise SystemExit("--n 须 >= 1")

    src = Path(args.input)
    with open(src, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        rows = list(reader)
    for c in ("seed_query", "tool", "domain"):
        if c not in cols:
            raise SystemExit(f"{src} 缺列 {c}（现有列: {cols}）")

    groups: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    for i, r in enumerate(rows):
        groups[(r["seed_query"], r["tool"])].append(i)

    rng = random.Random(args.seed)
    picked: list[int] = []
    for key in sorted(groups):                     # 排序后再抽，结果与字典顺序无关
        idx = groups[key]
        picked += idx if len(idx) <= args.n else rng.sample(idx, args.n)
    picked.sort()                                  # 保持原文件行序

    out = Path(args.output) if args.output else src.with_name(f"train_n{args.n}.csv")
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows[i] for i in picked)

    short = sum(1 for v in groups.values() if len(v) < args.n)
    print(f"输入 {src}：{len(rows)} 条，{len(groups)} 个种子（不足 {args.n} 条的 {short} 个，全取）")
    print(f"每种子抽 {args.n} 条（seed={args.seed}）→ {len(picked)} 条 → {out}")
    by = collections.Counter(rows[i]["domain"] for i in picked)
    for d in DOMAIN_ORDER + sorted(set(by) - set(DOMAIN_ORDER)):
        if by[d]:
            print(f"  {d:<11}{by[d]:>6}  {by[d] / len(picked):6.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
