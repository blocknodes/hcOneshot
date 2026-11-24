#!/usr/bin/env python3
"""给 para_all.csv 没覆盖到的评测 query（金标修订后新增/改写的 858 条）补同义改写。

做法与 gen_para.py 相同（同一 prompt、同一槽值校验），只是输入换成 all_eval.csv（新列名）里
「seed 不在 para_all」的那部分。产物：lib/exp/para_gap.csv（列 domain,query,tool,tool_param,seed_query）。

用法：python3 experiments/retrieval/gen_para_gap.py [--n 3]
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import csv
import json
import re

from common import HERE, ROOT, chat_many, nq
from gen_para import GEN, _src_values


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--gold", default=str(ROOT / "testset/feishu/20261116/all_eval.csv"))
    ap.add_argument("--lib", default=str(HERE / "lib/para_all.csv"))
    args = ap.parse_args()
    seeds = {nq(r["seed_query"]) for r in csv.DictReader(open(args.lib, encoding="utf-8-sig"))}
    seen, uniq = set(), []
    for r in csv.DictReader(open(args.gold, encoding="utf-8-sig")):
        k = nq(r["query"])
        if k in seeds or k in seen:
            continue
        seen.add(k)
        uniq.append({"domain": r["domain"], "query": r["query"].strip(), "tool": r["期望工具"],
                     "tool_param": r["期望参数"]})
    raws = asyncio.run(chat_many([(GEN.format(n=args.n), r["query"]) for r in uniq], 32))
    drop = collections.Counter()
    out, used = [], {nq(r["query"]) for r in uniq} | seeds
    for r, raw in zip(uniq, raws):
        m = re.search(r"\[.*\]", raw or "", re.S)
        try:
            cands = json.loads(m.group(0)) if m else []
        except json.JSONDecodeError:
            cands, drop["bad_json"] = [], drop["bad_json"] + 1
        try:
            p = json.loads(r["tool_param"]) if r["tool_param"].strip() else {}
        except json.JSONDecodeError:
            p = {}
        need = _src_values(p, r["query"])
        for c in cands if isinstance(cands, list) else []:
            c = str(c).strip()
            if not c or nq(c) in used:
                drop["dup"] += 1
                continue
            if [v for v in need if v not in c]:
                drop["slot_missing"] += 1
                continue
            used.add(nq(c))
            out.append({**r, "query": c, "seed_query": r["query"]})
    path = HERE / "lib/exp/para_gap.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["domain", "query", "tool", "tool_param", "seed_query"])
        w.writeheader()
        w.writerows(out)
    per = collections.Counter(nq(o["seed_query"]) for o in out)
    print(f"seeds {len(uniq)}  paraphrases {len(out)}  drop {dict(drop)}  "
          f"seeds with 0 kept {sum(1 for r in uniq if per[nq(r['query'])] == 0)}  -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
