#!/usr/bin/env python3
"""生成 dummy DPO 数据（只用于试跑训练流程，不保证 rejected 是真实模型错误）。

system  = 当前线上口径：HC_CATALOG_COMPACT=<--compact>（默认 2）目录 + few-shot 动态 context（与 app/engine 首轮一致）
user    = 当前时间行 + "用户请求：<query>"
chosen  = train.csv 金标计划 [{"q","d","vt","params"}]（去系统代填参数）
rejected= 对金标做一处随机扰动：换同域其他工具 / 删一个参数 / 改一个叶子取值 / 输出空计划
格式同 testset/sftdpo/*.dpo.jsonl（ms-swift：messages + rejected_response）。

用法：python3 tools/gen_dummy_dpo.py [--n 16384] [--seed 0] [--compact 2] [--out ...]
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# --compact 需在导入 app 前生效
_c = sys.argv[sys.argv.index("--compact") + 1] if "--compact" in sys.argv else "2"
os.environ["HC_CATALOG_COMPACT"] = _c
os.environ["HC_FEWSHOT_ENABLED"] = "1"
os.environ["HC_CATALOG_FILE"] = str(ROOT / "catalog.full.json")

from app import catalog, fewshot  # noqa: E402
from app.prompts import build_oneshot_prompt  # noqa: E402

DOMAIN = {"vod": "vod", "educ": "children", "edu": "education", "music": "music",
          "audio": "audio", "sports": "sports", "tv_control": "device"}
ALIAS = {"vod_search": "vod_search_all", "educ_search": "educ_search_all",
         "educ_relate_recommend": "educ_relate_search", "sports_match_data_search": "sports_rank_search"}
NOW = "当前时间：2026-11-03 21:00 星期二（北京时间）"


def dump(o) -> str:
    return json.dumps(o, ensure_ascii=False, separators=(",", ":"))


def leaves(o, path=()):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, path + (k,))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, path + (i,))
    else:
        yield path, o


def set_at(o, path, v):
    for k in path[:-1]:
        o = o[k]
    o[path[-1]] = v


def perturb(plan: list[dict], rng: random.Random) -> list[dict]:
    p = copy.deepcopy(plan)
    it = p[0]
    d, tool = it["vt"].split(":", 1)
    kinds = ["tool"]
    if it["params"]:
        kinds += ["drop", "value", "value"]
    kinds += ["empty"]
    k = rng.choice(kinds)
    if k == "tool":
        others = sorted(catalog.valid_tools_for(d) - {tool})
        if others:
            it["vt"] = f"{d}:{rng.choice(others)}"
            return p
        k = "empty"
    if k == "drop":
        it["params"].pop(rng.choice(sorted(it["params"])))
        return p
    if k == "value":
        ls = [(pa, v) for pa, v in leaves(it["params"]) if isinstance(v, str) and pa[-1] not in ("field", "operator")]
        if ls:
            pa, v = rng.choice(ls)
            set_at(it["params"], pa, v[: max(1, len(v) // 2)] + "x" if len(v) > 1 else v + "x")
            return p
        it["params"] = {}
        return p
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compact", type=int, default=2, help="目录档位 HC_CATALOG_COMPACT（0/1/2/3）")
    ap.add_argument("--train", default=str(ROOT / "testset" / "feishu" / "train.csv"))
    ap.add_argument("--out", default="", help="默认 testset/sftdpo/dummy_c<compact>.dpo.jsonl")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rows = [r for r in csv.DictReader(open(args.train, encoding="utf-8-sig"))
            if r["domain"] in DOMAIN and r["query"].strip()]
    rows = rng.sample(rows, min(args.n, len(rows))) if len(rows) >= args.n else \
        [rng.choice(rows) for _ in range(args.n)]
    static = fewshot.fix_static(build_oneshot_prompt(catalog.render_oneshot_block(), ""))
    out = Path(args.out) if args.out else ROOT / "testset" / "sftdpo" / f"dummy_c{args.compact}.dpo.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            d = DOMAIN[r["domain"]]
            tool = ALIAS.get(r["tool"].strip(), r["tool"].strip())
            try:
                params = json.loads(r["tool_param"]) if r["tool_param"].strip() else {}
            except json.JSONDecodeError:
                params = {}
            params = {k: v for k, v in params.items() if k not in catalog.system_filled(tool)}
            q = r["query"].strip()
            chosen = [{"q": q, "d": d, "vt": f"{d}:{tool}", "params": params}]
            rejected = perturb(chosen, rng)
            if dump(rejected) == dump(chosen):
                rejected = []
            msgs = [{"role": "system", "content": static + fewshot.context_block(q)},
                    {"role": "user", "content": f"{NOW}\n用户请求：{q}"},
                    {"role": "assistant", "content": dump(chosen)}]
            f.write(json.dumps({"messages": msgs, "rejected_response": dump(rejected)}, ensure_ascii=False) + "\n")
            n += 1
    print(f"{n} 条 -> {out}  static={len(static)} chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
