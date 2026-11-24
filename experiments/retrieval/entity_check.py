#!/usr/bin/env python3
"""实体库开/关对比（app/entity.py）：单意图评测集，few-shot 开/关 × 实体库 off/on/lso。

  off  不加实体块（当前线上口径）
  on   加实体块，库原样（库含评测金标抽出的实体 → 偏乐观：等价于「媒资库里有这个名字」）
  lso  加实体块，leave-seed-out：去掉只来自本题（及本题改写）的实体条目，只留别的题里也出现过的
few-shot 三种：on=线上口径（库含本题改写，偏乐观）；off=不加；lso=few-shot 库去掉本题改写家族（模拟新 query）。
system 与 app 首轮一致：compact=4 目录 + 虚拟设备 schema + few-shot（pattern_all.csv）+ 实体块（末尾）。
设备步按 app 后处理（compile_tolerant）还原真实参数后判分，口径同 common.judge。

用法：python3 experiments/retrieval/entity_check.py [--gold testset/feishu/20261116/all_eval.csv] [--fewshot on off lso]
输出：results/entity_<fewshot>.csv（逐题三档对比）+ 终端汇总
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import csv
import os

from common import HERE, ROOT, DOMAIN_ORDER, TOOL_ALIAS, canon, chat_many, load_gold, parse_oneshot, user_msg

os.environ.update(HC_CATALOG_COMPACT="4", HC_DEVICE_VIRTUAL="1", HC_FEWSHOT_EXCLUDE_SELF="1",
                  HC_FEWSHOT_LIB=str(HERE / "lib/pattern_all.csv"), HC_FEWSHOT_MULTI_LIB="")
from app import catalog, config, device_schema, entity, fewshot  # noqa: E402
from app.prompts import build_oneshot_prompt  # noqa: E402

ARMS = ("off", "on", "lso")


def _fewshot_leave_seed_out() -> None:
    """few-shot 库也做 leave-seed-out：检索时去掉本题的改写家族（seed == 本题），模拟「新 query」。"""
    orig = fewshot._Index.search
    if getattr(orig, "_lso", False):
        return

    def search(self, q, k, min_score, exclude_self):
        qn = fewshot._nq(q)
        return [i for i in orig(self, q, k + 1, min_score, exclude_self) if self.docs[i]["seed"] != qn][:k]
    search._lso = True
    fewshot._Index.search = search


def judge(raw: str, g: dict) -> dict:
    intents, _ = parse_oneshot(raw or "")
    first = intents[0] if intents else None
    tool = TOOL_ALIAS.get(first.tool, first.tool) if first else ""
    params = dict(first.params) if first and isinstance(first.params, dict) else {}
    if first and (first.d or catalog.domain_of_tool(tool)) == "device":
        try:
            params = device_schema.compile_tolerant(tool, params)
        except ValueError:
            tool, params = "", {}
    tool_ok = tool == g["tool"]
    ok = False
    if g["params"] is not None and tool_ok:
        accept = {canon(g["params"], g["tool"])} | {canon(a, g["tool"]) for a in g["alts"]}
        ok = canon(params, g["tool"]) in accept
    return {"tool": tool, "d": first.d if first else "", "tool_ok": tool_ok,
            "scored": g["params"] is not None, "tp_ok": ok, "pred": canon(params, g["tool"]) if first else ""}


def _gold(path: str) -> list[dict]:
    """20261116 列名（业务域,query,期望工具,期望参数,domain）转 load_gold 口径。"""
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    if rows and "tool" not in rows[0]:
        tmp = HERE / "cache" / "entity_gold.csv"
        tmp.parent.mkdir(exist_ok=True)
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["domain", "query", "tool", "tool_param", "tool_param_alt"])
            for r in rows:
                w.writerow([r["domain"], r["query"], r["期望工具"], r["期望参数"], ""])
        path = str(tmp)
    return load_gold(__import__("pathlib").Path(path))


def _acc(js: list[dict], key: str) -> str:
    n = sum(j["scored"] for j in js) if key == "tp_ok" else len(js)
    return f"{sum(j[key] for j in js) / max(1, n) * 100:5.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(ROOT / "testset/feishu/20261116/all_eval.csv"))
    ap.add_argument("--fewshot", nargs="+", default=["on", "off"], choices=["on", "off", "lso"],
                    help="on=线上口径；off=不加 few-shot；lso=few-shot 库去掉本题改写家族（模拟新 query）")
    ap.add_argument("--concurrency", type=int, default=64)
    args = ap.parse_args()
    golds = _gold(args.gold)
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    for fs in args.fewshot:
        os.environ["HC_FEWSHOT_ENABLED"] = "0" if fs == "off" else "1"
        if fs == "lso":
            _fewshot_leave_seed_out()
        config.get_settings.cache_clear()
        base = build_oneshot_prompt(catalog.render_oneshot_block(), "")
        sys0 = [fewshot.system_prompt(base, g["query"]) for g in golds]
        blocks = {"off": [""] * len(golds),
                  "on": [entity.render(g["query"]) for g in golds],
                  "lso": [entity.render(g["query"], exclude_seed=g["query"]) for g in golds]}
        res = {}
        for arm in ARMS:
            raws = asyncio.run(chat_many([(s + b, user_msg(g["query"]))
                                          for s, b, g in zip(sys0, blocks[arm], golds)], args.concurrency))
            res[arm] = [judge(r, g) for r, g in zip(raws, golds)]
        print(f"\n===== few-shot {fs}（{len(golds)} 题）=====")
        print(f"{'子集':<16}{'arm':<5}{'n':>6}{'tool':>8}{'tool+param':>12}{'修好':>6}{'变坏':>6}")
        for arm_sub in ("on", "lso"):
            hit = [i for i, b in enumerate(blocks[arm_sub]) if b]
            for name, idx in ((f"全量", range(len(golds))), (f"命中实体({arm_sub})", hit)):
                for arm in ("off", arm_sub):
                    js = [res[arm][i] for i in idx]
                    fix = sum(1 for i in idx if res[arm][i]["tp_ok"] and not res["off"][i]["tp_ok"])
                    brk = sum(1 for i in idx if not res[arm][i]["tp_ok"] and res["off"][i]["tp_ok"])
                    fb = f"{fix:>6}{brk:>6}" if arm != "off" else ""
                    print(f"{name:<16}{arm:<5}{len(js):>6}{_acc(js, 'tool_ok'):>8}{_acc(js, 'tp_ok'):>12}{fb}")
        print("按域（tool+param，off / on / lso；括号内为命中实体题数 on/lso）")
        for d in DOMAIN_ORDER:
            idx = [i for i, g in enumerate(golds) if g["domain"] == d]
            if not idx:
                continue
            cells = " / ".join(_acc([res[a][i] for i in idx], "tp_ok") for a in ARMS)
            print(f"  {d:<11}{len(idx):>5}  {cells}  ({sum(bool(blocks['on'][i]) for i in idx)}"
                  f"/{sum(bool(blocks['lso'][i]) for i in idx)})")
        with open(out_dir / f"entity_fewshot_{fs}.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["domain", "query", "gold_tool", "gold_canon", "entity_on", "entity_lso"]
                       + [f"{a}_{k}" for a in ARMS for k in ("tool", "tool_ok", "tp_ok", "pred")])
            for i, g in enumerate(golds):
                w.writerow([g["domain"], g["query"], g["tool"],
                            canon(g["params"], g["tool"]) if g["params"] is not None else "",
                            blocks["on"][i].strip(), blocks["lso"][i].strip()]
                           + [res[a][i][k] for a in ARMS for k in ("tool", "tool_ok", "tp_ok", "pred")])
        cnt = collections.Counter(bool(b) for b in blocks["on"])
        print(f"  实体块非空：on {cnt[True]} 题，lso {sum(bool(b) for b in blocks['lso'])} 题；"
              f"逐题结果 -> {out_dir / f'entity_fewshot_{fs}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
