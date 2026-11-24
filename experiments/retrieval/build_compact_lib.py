#!/usr/bin/env python3
"""压缩 few-shot 库：同一「分组键」只留 K 个不同 seed 的各 1 条改写。

层次化检索里名字归实体库（L2）、口径归规则库（L1），few-shot（L3）只负责「这种说法 → 这个工具+参数写法」，
所以同一写法的几十条近义句、只差名字的同构句都可以合并。输入 lib/pattern_all.csv
（列 domain,query,tool,tool_param,seed_query,pattern）。

--by pattern  分组键 = pattern 列（域+工具+参数骨架；设备域的 operation/object 取值会被抹平，太粗）
--by slot     同 label，另把 query 里原样出现的闭集槽值（slot_mask.py 的 slot_vocab）换成 <S>
--by label    分组键 = 域+工具+归一参数，其中原样出现在 query 里的实体库名字换成 <E>
              （「播放庆余年」「播放狂飙」合成一组；设备「打开屏保」「关闭屏保」仍是两组）
选取：组内按 seed 家族大小降序（同大小按 seed 字典序，保证确定性）取前 K 个 seed；
      每个 seed 取第一条与 seed 原句不同的改写（没有就取原句）。

用法：python3 experiments/retrieval/build_compact_lib.py --by label --k 2 [--sqrt 2] [--out lib/compact_label_k2.csv]
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math

from common import HERE, canon, nq


def _label_key(r: dict, names: list[str], slots: list[str] = ()) -> str:
    try:
        c = canon(json.loads(r["tool_param"] or "{}"), r["tool"])
    except json.JSONDecodeError:
        c = r["tool_param"]
    q = nq(r["query"]).lower()
    for n in names:
        if n in c and n in q:
            c = c.replace(n, "<E>")
    for v in slots:                               # --by slot：闭集槽值（学科/年级/类别…）也合并
        if f'"{v}"' in c and v in q:
            c = c.replace(f'"{v}"', '"<S>"')
    return f"{r['domain']}|{r['tool']}|{c}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(HERE / "lib/pattern_all.csv"))
    ap.add_argument("--by", choices=["pattern", "label", "slot"], default="pattern")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--sqrt", type=float, default=0.0,
                    help=">0 时每组取 max(K, ceil(c*sqrt(组内 seed 数))) 个 seed：大组（无参数的 fuzzy/问答类）说法多样，多留几条")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    stem = {"pattern": "compact_k%d", "label": "compact_label_k%d", "slot": "compact_slot_k%d"}[a.by] % a.k + (f"_s{a.sqrt:g}" if a.sqrt else "")
    out = a.out or str(HERE / f"lib/{stem}.csv")
    rows = list(csv.DictReader(open(a.src, encoding="utf-8-sig")))
    names: list[str] = []
    slots: list[str] = []
    if a.by in ("label", "slot"):
        from app import entity
        names = sorted((n for n in entity._lexicon().by if len(n) >= 2), key=len, reverse=True)
    if a.by == "slot":
        import slot_mask
        slots = [v for v, _t in slot_mask._vocab()]
    by: dict[str, dict[str, list[dict]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        key = r["pattern"] if a.by == "pattern" else _label_key(r, names, slots)
        by[key][nq(r["seed_query"] or r["query"])].append(r)
    keep = []
    for key in sorted(by):
        quota = max(a.k, math.ceil(a.sqrt * math.sqrt(len(by[key])))) if a.sqrt else a.k
        fams = sorted(by[key].items(), key=lambda x: (-len(x[1]), x[0]))[:quota]
        for seed, rs in fams:
            keep.append(next((r for r in rs if nq(r["query"]) != seed), rs[0]))
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(keep)
    print(f"{len(rows)} 条 / {len(by)} 组（--by {a.by}）→ {len(keep)} 条 → {out}")


if __name__ == "__main__":
    main()
