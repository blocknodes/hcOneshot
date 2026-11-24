#!/usr/bin/env python3
"""按 pattern 组织 few-shot 库：覆盖评测数据里的每种 pattern，标注统一取当前金标。

pattern = 域 + 工具 + 参数骨架（action/operation 取值、query 树里的字段集合与 and/or/not、sort 键、其余顶层键名），
          见 skeleton()。评测集（testset/feishu/20261116/all_eval.csv）共约 496 种。

候选句来源（都只换说法、不换槽值）：
  · para_all.csv 里该 seed 的同义改写（旧金标时代生成，标注一律改成当前金标）；
  · lib/exp/para_gap.csv：para_all 没覆盖的新 seed 的改写（gen_para_gap.py）。
  评测 query 原句不进库（exclude_self 本来也会把它排掉，进库只占位置）。

选取：
  1. 每个 seed 取 --per-seed 条改写（默认 1；检索时同 seed 本来只出 1 条，多放无益）；
  2. 同一精确标注（域+工具+归一参数）最多 --per-label 条（0=不限），去掉设备域大量同标注的冗余；
  3. 每种 pattern 至少 --floor 条：不足时从该 pattern 其余改写里补。

用法：python3 experiments/retrieval/build_pattern_lib.py --out lib/exp/pat_s1.csv [--per-seed 1 --per-label 0 --floor 3]
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

from common import HERE, ROOT, TOOL_ALIAS, canon, nq

GOLD = ROOT / "testset/feishu/20261116/all_eval.csv"


def _fields(n, out: list[str]) -> None:
    if isinstance(n, dict):
        if "field" in n:
            out.append(n["field"] + ("[]" if "values" in n else "") + ("~" if "from" in n else ""))
            return
        for k, v in n.items():
            if k in ("and", "or", "not"):
                out.append(k)
                _fields(v, out)
    elif isinstance(n, list):
        for x in n:
            _fields(x, out)


def skeleton(domain: str, tool: str, p) -> str:
    if not isinstance(p, dict):
        return f"{domain}|{tool}|?"
    parts = []
    for k in sorted(p):
        v = p[k]
        if k == "retext":
            continue
        if k == "query":
            f: list[str] = []
            _fields(v, f)
            parts.append("query(" + ",".join(sorted(f)) + ")")
        elif k == "sort":
            parts.append("sort(" + ",".join(sorted(v)) + ")" if isinstance(v, dict) else "sort")
        elif k in ("action", "operation", "intent"):
            parts.append(f"{k}={v}")
        elif v not in ("", None, [], {}):
            parts.append(k)
    return f"{domain}|{tool}|" + ";".join(parts)


def load_gold() -> dict[str, dict]:
    gold: dict[str, dict] = {}
    for r in csv.DictReader(open(GOLD, encoding="utf-8-sig")):
        k = nq(r["query"])
        if k in gold:
            continue
        tool = TOOL_ALIAS.get(r["期望工具"].strip(), r["期望工具"].strip())
        tp = r["期望参数"].strip()
        try:
            p = json.loads(tp) if tp else {}
        except json.JSONDecodeError:
            p = None
        gold[k] = {"domain": r["domain"].strip(), "query": r["query"].strip(), "tool": r["期望工具"].strip(),
                   "tool_param": tp, "pattern": skeleton(r["domain"].strip(), tool, p),
                   "label": (r["domain"].strip(), tool, canon(p, tool) if p is not None else tp)}
    return gold


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-seed", type=int, default=1)
    ap.add_argument("--per-label", type=int, default=0)
    ap.add_argument("--floor", type=int, default=3)
    ap.add_argument("--src", nargs="+", default=[str(HERE / "lib/para_all.csv"), str(HERE / "lib/exp/para_gap.csv")])
    args = ap.parse_args()
    gold = load_gold()
    fam: dict[str, list[str]] = collections.defaultdict(list)
    for src in args.src:
        for r in csv.DictReader(open(src, encoding="utf-8-sig")):
            s, q = nq(r["seed_query"]), r["query"].strip()
            if s in gold and nq(q) not in gold and q not in fam[s]:
                fam[s].append(q)

    chosen: list[tuple[str, str]] = []          # (seed, paraphrase)
    used: set[str] = set()
    per_label: collections.Counter = collections.Counter()
    per_pat: collections.Counter = collections.Counter()
    spare: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    for s, g in gold.items():
        for i, q in enumerate(fam.get(s, [])):
            ok = i < args.per_seed and (not args.per_label or g["label"][2] in ("{}", "")
                                        or per_label[g["label"]] < args.per_label)
            if ok:
                chosen.append((s, q))
                used.add(nq(q))
                per_label[g["label"]] += 1
                per_pat[g["pattern"]] += 1
            else:
                spare[g["pattern"]].append((s, q))
    # 没有任何改写的 seed：原句进库（评测本句时被 exclude_self 排掉，只给同 pattern 的其他句当示例）
    for s, g in gold.items():
        if s not in fam:
            spare[g["pattern"]].append((s, g["query"]))
    floor_add = 0
    for pat, cands in spare.items():
        for s, q in cands:
            if per_pat[pat] >= args.floor:
                break
            if nq(q) not in used:
                chosen.append((s, q))
                used.add(nq(q))
                per_pat[pat] += 1
                floor_add += 1

    out = Path(args.out)
    out = out if out.is_absolute() else HERE / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["domain", "query", "tool", "tool_param", "seed_query", "pattern"])
        for s, q in chosen:
            g = gold[s]
            w.writerow([g["domain"], q, g["tool"], g["tool_param"], g["query"], g["pattern"]])
    pats = {g["pattern"] for g in gold.values()}
    covered = {gold[s]["pattern"] for s, _ in chosen}
    seeds_cov = len({s for s, _ in chosen})
    print(f"{out.name}: {len(chosen)} 条  seed 覆盖 {seeds_cov}/{len(gold)}  pattern 覆盖 {len(covered)}/{len(pats)}"
          f"  floor 补 {floor_add}  每 pattern<{args.floor} 的 {sum(1 for p in pats if per_pat[p] < args.floor)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
