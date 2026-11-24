#!/usr/bin/env python3
"""为评测 query 构建「同义改写」检索库：只换说法、不换槽值，标注沿用原 query 的金标。

校验：改写必须包含金标里所有「出现在原 query 中的字符串取值」（保证标注对改写仍成立）；
      与原句相同 / 与其他改写重复 → 丢。
产物：lib/para_<domains>.csv（列同 train.csv：domain,query,tool,tool_param,seed_query）

用法：python3 experiments/retrieval/gen_para.py --domains vod,educ [--n 5]
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import csv
import json
import re

from common import HERE, ROOT, chat_many, load_gold, nq

GEN = """你是电视语音助手的数据工程师。给你一条用户对电视说的话，请写出 {n} 种【意思完全相同】的不同口语说法。
要求：
1. 意图、动作、所有具体信息（片名、人名、角色名、年份、数字、集数、语言、免费/付费、年龄等）都不变，原词照抄，不要替换成别的值，也不要增删条件。
2. 只改句式和口头禅，像真实用户会说的话，彼此之间也要有明显区别。
3. 不要加入原话没有的信息（不要加「最火」「最新」「好看的」之类修饰）。
只输出 JSON 字符串数组，不要解释。"""


def _src_values(params, query: str) -> list[str]:
    out = []

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str) and x and x in query:
            out.append(x)
    walk(params)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="vod,educ")
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()
    doms = args.domains.split(",")
    rows = [r for r in csv.DictReader(open(ROOT / "all_eval.csv", encoding="utf-8-sig")) if r["domain"] in doms]
    # 同一 query 多行时只用首行
    seen, uniq = set(), []
    for r in rows:
        if nq(r["query"]) not in seen:
            seen.add(nq(r["query"]))
            uniq.append(r)
    system = GEN.format(n=args.n)
    raws = asyncio.run(chat_many([(system, r["query"]) for r in uniq], 64))

    drop = collections.Counter()
    out, used = [], set(nq(r["query"]) for r in uniq)
    for r, raw in zip(uniq, raws):
        m = re.search(r"\[.*\]", raw or "", re.S)
        try:
            cands = json.loads(m.group(0)) if m else []
        except json.JSONDecodeError:
            cands, drop["bad_json"] = [], drop["bad_json"] + 1
        p = json.loads(r["tool_param"]) if r["tool_param"].strip() else {}
        need = _src_values(p, r["query"])
        for c in cands if isinstance(cands, list) else []:
            c = str(c).strip()
            if not c:
                continue
            if nq(c) in used:
                drop["dup"] += 1
                continue
            miss = [v for v in need if v not in c]
            if miss:
                drop["slot_missing"] += 1
                continue
            used.add(nq(c))
            out.append({"domain": r["domain"], "query": c, "tool": r["tool"],
                        "tool_param": r["tool_param"], "seed_query": r["query"]})
    lib = HERE / "lib"
    lib.mkdir(exist_ok=True)
    path = lib / f"para_{'_'.join(doms)}.csv"
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["domain", "query", "tool", "tool_param", "seed_query"])
        w.writeheader()
        w.writerows(out)
    per = collections.Counter(o["seed_query"] for o in out)
    print(f"seeds {len(uniq)}  paraphrases {len(out)}  drop {dict(drop)}  "
          f"seeds with 0 kept {sum(1 for r in uniq if per[r['query']] == 0)}  -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
