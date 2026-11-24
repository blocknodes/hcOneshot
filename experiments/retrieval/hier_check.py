#!/usr/bin/env python3
"""层次化检索（L1 规则 + L2 实体 + L3 few-shot）离线对比：单意图评测集 all_eval（4679 题），直接调 hcProxy。

不改 app/ 任何文件：context 在本脚本里按 app.fewshot.context_block 的拼法组装（rules → examples → 实体块），
system = fix_static(compact=4 目录 prompt) + context，与 app 首轮一致（虚拟设备 schema）。

每个 arm = few-shot 库 × topk × 规则库 × 实体块 × 占位符检索：
  规则库   builtin = fewshot.py 内置（只有少儿一条，当前线上）；snap = lib/exp/rules_snapshot_20261121.csv
  实体块   off / on（库原样，≈媒资库覆盖到）/ lso（去掉只来自本题的实体条目）
  占位符   名字用 app.entity.mask 换成「E影视片名」等（"all" 时再用 slot_mask 把学科/年级/类别等闭集槽值换成
           「E学科」「E年级」…），再从同一 few-shot 库检一次，与原句检索交替合并
few-shot 口径：lso = 检索去掉本题改写家族（seed == 本题），模拟新 query（泛化性看这个）；on = 线上口径。

用法：python3 experiments/retrieval/hier_check.py --arms base hier_k3 ... [--fewshot lso on]
输出：results/hier_<arm>_<fs>.csv（逐题）+ results/hier_summary.csv（汇总追加）+ 终端表
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import os
import time

from common import HERE, ROOT, DOMAIN_ORDER, canon, chat_many, nq, user_msg

LIB = HERE / "lib"
RULES_SNAP = str(LIB / "exp/rules_snapshot_20261121.csv")
os.environ.update(HC_CATALOG_COMPACT="4", HC_DEVICE_VIRTUAL="1", HC_FEWSHOT_EXCLUDE_SELF="1",
                  HC_FEWSHOT_MULTI_LIB="", HC_ENTITY_ENABLED="0", HC_FEWSHOT_ENABLED="1",
                  HC_FEWSHOT_ENTITY_HINT="0",
                  # 规则库的「实体类型触发」用 fewshot_entity_dict；只用于触发规则，不注入 hint、不做该词典的占位符检索
                  HC_FEWSHOT_ENTITY_DICT=str(LIB / "entity_dict.csv"))
from app import catalog, config, entity, fewshot  # noqa: E402
from app.prompts import build_oneshot_prompt  # noqa: E402
from entity_check import _gold, judge  # noqa: E402
import slot_mask  # noqa: E402

# name: (few-shot 库, topk, 规则库, 实体块, 占位符检索：False / True=名字 / "all"=名字+闭集槽值)
ARMS = {
    "base":          ("pattern_all.csv", 4, "builtin", "off", False),   # 当前线上
    "base_rule":     ("pattern_all.csv", 4, "snap", "off", False),
    "base_hier":     ("pattern_all.csv", 4, "snap", "on", True),
    "k1":            ("compact_k1.csv", 4, "builtin", "off", False),
    "k3":            ("compact_k3.csv", 4, "builtin", "off", False),
    "k3_rule":       ("compact_k3.csv", 4, "snap", "off", False),
    "k3_ent":        ("compact_k3.csv", 4, "builtin", "on", False),
    "k3_mask":       ("compact_k3.csv", 4, "builtin", "off", True),
    "hier_k1":       ("compact_k1.csv", 4, "snap", "on", True),
    "hier_k2":       ("compact_k2.csv", 4, "snap", "on", True),
    "hier_k3":       ("compact_k3.csv", 4, "snap", "on", True),
    "hier_k5":       ("compact_k5.csv", 4, "snap", "on", True),
    "hier_k3_t2":    ("compact_k3.csv", 2, "snap", "on", True),
    "hier_k3_elso":  ("compact_k3.csv", 4, "snap", "lso", True),
    "hier_k5_elso":  ("compact_k5.csv", 4, "snap", "lso", True),
    "L1":            ("compact_label_k1.csv", 4, "builtin", "off", False),
    "L2":            ("compact_label_k2.csv", 4, "builtin", "off", False),
    "L3":            ("compact_label_k3.csv", 4, "builtin", "off", False),
    "hier_L1":       ("compact_label_k1.csv", 4, "snap", "on", True),
    "hier_L2":       ("compact_label_k2.csv", 4, "snap", "on", True),
    "hier_L3":       ("compact_label_k3.csv", 4, "snap", "on", True),
    "hier_L2_elso":  ("compact_label_k2.csv", 4, "snap", "lso", True),
    "hier_L3_elso":  ("compact_label_k3.csv", 4, "snap", "lso", True),
    "L2s2":          ("compact_label_k2_s2.csv", 4, "builtin", "off", False),
    "hier_L2s2":     ("compact_label_k2_s2.csv", 4, "snap", "on", True),
    "hier_L2s3":     ("compact_label_k2_s3.csv", 4, "snap", "on", True),
    "hier_L2s2_elso": ("compact_label_k2_s2.csv", 4, "snap", "lso", True),
    "hier_L2s3_elso": ("compact_label_k2_s3.csv", 4, "snap", "lso", True),
    "S2s2":          ("compact_slot_k2_s2.csv", 4, "builtin", "off", False),
    "hier_S2s2":     ("compact_slot_k2_s2.csv", 4, "snap", "on", "all"),
    "hier_S2s3":     ("compact_slot_k2_s3.csv", 4, "snap", "on", "all"),
    "hier_L2s3_all": ("compact_label_k2_s3.csv", 4, "snap", "on", "all"),
    "hier_S2s3_elso": ("compact_slot_k2_s3.csv", 4, "snap", "lso", "all"),
    "base_hier_all": ("pattern_all.csv", 4, "snap", "on", "all"),
    "hier_L2s3_all_t3": ("compact_label_k2_s3.csv", 3, "snap", "on", "all"),
    "hier_L2s3_all_t2": ("compact_label_k2_s3.csv", 2, "snap", "on", "all"),
    "hier_L2s3_all_elso": ("compact_label_k2_s3.csv", 4, "snap", "lso", "all"),
    "base_t2":       ("pattern_all.csv", 2, "builtin", "off", False),
    "base_hier_all_t2": ("pattern_all.csv", 2, "snap", "on", "all"),
    "base_hier_elso": ("pattern_all.csv", 4, "snap", "lso", True),
}

_IDX: dict[str, tuple] = {}


class _Masker:
    """给 fewshot._MaskedIndex 用：库句用完整实体库（mode=all 时再加闭集槽值）换占位符。"""

    def __init__(self, mode: str) -> None:
        self.mode = mode

    def mask(self, q: str) -> str:
        return slot_mask.mask_all(q) if self.mode == "all" else entity.mask(q, exclude_seed="")


def _qmask(q: str, mode: str, ent_ex: str) -> str:
    return slot_mask.mask_all(q, exclude_seed=ent_ex) if mode == "all" else entity.mask(q, exclude_seed=ent_ex)


def _indexes(lib: str, mode: str = "ent"):
    if lib not in _IDX:
        _IDX[lib] = (fewshot._Index(LIB / lib), {})
    base, m = _IDX[lib]
    if mode not in m:
        m[mode] = fewshot._MaskedIndex(base, _Masker(mode))
    return base, m[mode]


def _search(idx, q: str, k: int, ex_seed: str) -> list[int]:
    s = config.get_settings()
    hits = idx.search(q, k + 8, s.fewshot_min_score, True)
    return [i for i in hits if not ex_seed or idx.docs[i]["seed"] != ex_seed][:k]


def _examples(lib: str, q: str, k: int, mask: bool, ent_ex: str | None, ex_seed: str) -> list[int]:
    """同 fewshot._search_merged：占位符检索与原句检索交替合并、按 seed 去重。"""
    if not mask:
        return _search(_indexes(lib)[0], q, k, ex_seed)
    mode = "all" if mask == "all" else "ent"
    base, midx = _indexes(lib, mode)
    hits = _search(base, q, k, ex_seed)
    mq = _qmask(q, mode, ent_ex)
    if mq == nq(q):
        return hits
    mh = _search(midx, mq, k, ex_seed)
    out, seeds = [], set()
    for pair in zip(mh + [None] * k, hits + [None] * k):
        for i in pair:
            if i is None or i in out or base.docs[i]["seed"] in seeds:
                continue
            seeds.add(base.docs[i]["seed"])
            out.append(i)
    return out[:k]


def context(name: str, q: str, fs: str) -> tuple[str, int, bool, bool]:
    lib, topk, rules, ent, mask = ARMS[name]
    ex_seed = nq(q) if fs == "lso" else ""
    ent_ex = q if ent == "lso" else ""
    parts = []
    rb = fewshot.rules_block(q)
    if rb:
        parts.append(rb)
    hits = _examples(lib, q, topk, mask, ent_ex, ex_seed)
    if hits:
        base, _ = _indexes(lib)
        parts.append("【相似请求的标注示例（按关键词检索，供参考工具选择与参数写法；以本轮请求为准，不要照抄取值）】\n"
                     + "\n\n".join(base.docs[i]["text"] for i in hits))
    ctx = ("\n" + "\n".join(parts)) if parts else ""
    eb = "" if ent == "off" else entity.render(q, exclude_seed=ent_ex)
    return ctx + eb, len(hits), bool(rb), bool(eb)


def run_arm(name: str, fs: str, golds: list[dict], base_sys: str, concurrency: int):
    lib, topk, rules, ent, mask = ARMS[name]
    os.environ["HC_FEWSHOT_RULES_LIB"] = RULES_SNAP if rules == "snap" else ""
    config.get_settings.cache_clear()
    fewshot._RLIB = None
    ctxs = [context(name, g["query"], fs) for g in golds]
    raws = asyncio.run(chat_many([(base_sys + c[0], user_msg(g["query"])) for c, g in zip(ctxs, golds)],
                                 concurrency))
    js = [judge(r, g) for r, g in zip(raws, golds)]
    for j, c in zip(js, ctxs):
        j["ctx"], j["shots"], j["rule"], j["ent"] = c
    n = len(js)
    nlib = len(_indexes(lib)[0].docs)
    summ = {"arm": name, "fewshot": fs, "lib": lib, "lib_size": nlib, "topk": topk, "rules": rules,
            "entity": ent, "mask": str(mask), "n": n,
            "tool": sum(j["tool_ok"] for j in js) / n,
            "tool_param": sum(j["tp_ok"] for j in js) / max(1, sum(j["scored"] for j in js)),
            "avg_shots": sum(j["shots"] for j in js) / n,
            "rule_hit": sum(j["rule"] for j in js) / n, "ent_hit": sum(j["ent"] for j in js) / n,
            "avg_ctx_chars": sum(len(j["ctx"]) for j in js) / n}
    for d in DOMAIN_ORDER:
        idx = [i for i, g in enumerate(golds) if g["domain"] == d]
        summ[f"tp_{d}"] = sum(js[i]["tp_ok"] for i in idx) / max(1, sum(js[i]["scored"] for i in idx))
    return js, summ


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(ROOT / "testset/feishu/20261116/all_eval.csv"))
    ap.add_argument("--arms", nargs="+", default=["base", "hier_k3"], choices=list(ARMS))
    ap.add_argument("--fewshot", nargs="+", default=["lso"], choices=["lso", "on"])
    ap.add_argument("--concurrency", type=int, default=64)
    args = ap.parse_args()
    golds = _gold(args.gold)
    config.get_settings.cache_clear()
    base_sys = fewshot.fix_static(build_oneshot_prompt(catalog.render_oneshot_block(), ""))
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    rows = []
    for fs in args.fewshot:
        for name in args.arms:
            t = time.time()
            js, summ = run_arm(name, fs, golds, base_sys, args.concurrency)
            rows.append(summ)
            print(f"[{name}/{fs}] lib={summ['lib_size']} t+p={summ['tool_param'] * 100:.1f}% "
                  f"tool={summ['tool'] * 100:.1f}% shots={summ['avg_shots']:.2f} rule={summ['rule_hit']:.2f} "
                  f"ent={summ['ent_hit']:.2f} ({time.time() - t:.0f}s)", flush=True)
            with open(out / f"hier_{name}_{fs}.csv", "w", encoding="utf-8-sig", newline="") as f:
                w = csv.writer(f)
                w.writerow(["domain", "query", "gold_tool", "gold_canon", "tool", "tool_ok", "tp_ok", "pred",
                            "shots", "rule", "ent", "ctx"])
                for g, j in zip(golds, js):
                    w.writerow([g["domain"], g["query"], g["tool"],
                                canon(g["params"], g["tool"]) if g["params"] is not None else "",
                                j["tool"], j["tool_ok"], j["tp_ok"], j["pred"], j["shots"], j["rule"], j["ent"],
                                j["ctx"]])
    print(f"\n{'arm':<14}{'fs':<5}{'lib':>7}{'topk':>5}{'shots':>7}{'tool':>8}{'t+p':>8}  "
          + " ".join(f"{d[:6]:>6}" for d in DOMAIN_ORDER))
    for s in rows:
        print(f"{s['arm']:<14}{s['fewshot']:<5}{s['lib_size']:>7}{s['topk']:>5}{s['avg_shots']:>7.2f}"
              f"{s['tool'] * 100:>7.1f}%{s['tool_param'] * 100:>7.1f}%  "
              + " ".join(f"{s[f'tp_{d}'] * 100:>6.1f}" for d in DOMAIN_ORDER))
    summ_path = out / "hier_summary.csv"
    new = not summ_path.exists()
    with open(summ_path, "a", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
