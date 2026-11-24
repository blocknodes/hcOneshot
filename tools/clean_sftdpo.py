#!/usr/bin/env python3
"""清洗 SFT/DPO：剔除「金标有问题、模型其实对」的样本。

对每条 DPO（chosen=金标，rejected=模型输出）按以下规则判定（命中即处理，原因记入报告）：
  A. 等价写法（DPO 剔除，SFT 保留）：
     · eval_stats 口径判为一致（数值与数字字符串等价，如 is_fee 0 == "0"）
     · 仅大小写不同（金标把英文写成小写：sdr转hdr / dj版）
  B. 金标与 prompt 当前时间不一致（DPO 与 SFT 都剔除）：
     query 含相对时间词，且差异落在时间/日期字段（金标按旧日期写死，prompt 里是别的「当前时间」）
     · 实体值只差空格/标点（AC 米兰 vs AC米兰），其余一致
  C. 疑似金标取值错（DPO 与 SFT 都剔除）：
     开放型实体字段（人名/片名/歌名/球队等）上，金标取值与模型取值字面相似度 < 0.5、金标值不在 query 里、
     模型值在 query 里（如 query「王宝强」金标 person_name=安妮海瑟薇）。字面相近的归一化差异不算。
  D. 人工结论（可选 --review）：review.csv 里「人工结论」含「金标错」→ 两边都剔除；含「模型对」→ 只剔 DPO
SFT 里命中 B/C/D(金标错) 的同 prompt 样本一并剔除。

用法：python3 tools/clean_sftdpo.py --prefix testset/sftdpo/train2_c4 [--review all_eval_result_rag_compact4.review.csv]
产物：<prefix>.clean.sft.jsonl / <prefix>.clean.dpo.jsonl / <prefix>.clean.report.csv
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from app.parser import parse_oneshot  # noqa: E402
from eval_stats import canon  # noqa: E402
from gen_train import REL_TIME  # noqa: E402

# keywords 不参与：qqmusic 无槽位时 schema 规定填默认「热门」（金标对）
ENTITY_FIELDS = {"person_name", "title", "actor", "director", "role", "song", "singer", "album",
                 "sport_team", "sport_team_home", "sport_team_guest", "sport_star", "ip"}
_PUNCT = re.compile(r"[\s·・.,，。!！?？'\"《》“”:：\-—_]")


def _norm_v(v: str) -> str:
    return _PUNCT.sub("", v).lower()
TIME_KEYS = re.compile(r"time|date|release_time|sport_time")


def _q(msgs) -> str:
    m = re.search(r"用户请求：(.*)$", msgs[1]["content"], re.S)
    return re.sub(r"\n【判定提示】.*$", "", m.group(1), flags=re.S).strip() if m else ""


def _first(raw: str):
    its, _ = parse_oneshot(raw or "")
    return its[0] if its else None


def _entity_values(o, out: dict[str, set]) -> None:
    if isinstance(o, dict):
        if "field" in o:
            f = o["field"]
            for v in ([o.get("value")] + list(o.get("values") or [])):
                if isinstance(v, str) and v:
                    out.setdefault(f, set()).add(v)
        for k, v in o.items():
            if k in ENTITY_FIELDS:
                for x in (v if isinstance(v, list) else [v]):
                    if isinstance(x, str) and x:
                        out.setdefault(k, set()).add(x)
            _entity_values(v, out)
    elif isinstance(o, list):
        for x in o:
            _entity_values(x, out)


def classify(query: str, chosen_raw: str, rejected_raw: str) -> tuple[str, str]:
    """返回 (动作, 原因)；动作：keep / drop_dpo / drop_both。"""
    g, p = _first(chosen_raw), _first(rejected_raw)
    if not g or not p or g.tool != p.tool or g.d != p.d:
        return "keep", ""
    gc, pc = canon(g.params, g.tool), canon(p.params, p.tool)
    if gc == pc:
        return "drop_dpo", "A:等价写法(数值/字符串)"
    if gc.lower() == pc.lower():
        return "drop_dpo", "A:仅大小写不同"
    gj, pj = json.loads(gc), json.loads(pc)
    diff_keys = {k for k in set(gj) | set(pj) if gj.get(k) != pj.get(k)}
    if REL_TIME.search(query):
        flat_g, flat_p = json.dumps(gj, ensure_ascii=False), json.dumps(pj, ensure_ascii=False)
        dates_g = set(re.findall(r"20\d{6}|20\d\d-\d\d-\d\d", flat_g))
        dates_p = set(re.findall(r"20\d{6}|20\d\d-\d\d-\d\d", flat_p))
        if (any(TIME_KEYS.search(k) for k in diff_keys) or dates_g != dates_p) and dates_g:
            # 去掉日期后其余一致 → 只差在基准日
            strip = lambda s: re.sub(r"20\d{6}( \d\d:\d\d:\d\d)?|20\d\d-\d\d-\d\d( \d\d:\d\d:\d\d)?", "D", s)
            if strip(flat_g) == strip(flat_p):
                return "drop_both", "B:相对时间基准日不一致"
    ge, pe = {}, {}
    _entity_values(gj, ge)
    _entity_values(pj, pe)
    import difflib
    only_punct = bool(diff_keys)
    for f in set(ge) & set(pe) & ENTITY_FIELDS:
        gonly, ponly = ge[f] - pe[f], pe[f] - ge[f]
        if not (gonly and ponly):
            continue
        if {_norm_v(v) for v in gonly} == {_norm_v(v) for v in ponly}:
            continue                                       # 只差空格/标点/大小写 → 见下方等价判定
        only_punct = False
        sim = max(difflib.SequenceMatcher(None, a, b).ratio() for a in gonly for b in ponly)
        if sim < 0.5 and all(v not in query for v in gonly) and all(v in query for v in ponly):
            return "drop_both", f"C:金标{f}与query无关（{'/'.join(sorted(gonly))} vs {'/'.join(sorted(ponly))}）"
    # 实体值只差空格/标点（AC 米兰 vs AC米兰、泰勒·弗里茨 vs 泰勒弗里茨），其余参数一致 → 等价
    if only_punct and _norm_v(json.dumps(gj, ensure_ascii=False, sort_keys=True)) == \
            _norm_v(json.dumps(pj, ensure_ascii=False, sort_keys=True)):
        return "drop_dpo", "A:实体值仅空格/标点不同"
    return "keep", ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True, help="如 testset/sftdpo/train2_c4")
    ap.add_argument("--review", default="", help="可选：带「人工结论」列的 review.csv")
    args = ap.parse_args()
    pre = Path(args.prefix)
    sft = [json.loads(l) for l in open(f"{pre}.sft.jsonl", encoding="utf-8")]
    dpo = [json.loads(l) for l in open(f"{pre}.dpo.jsonl", encoding="utf-8")]

    manual: dict[str, str] = {}
    if args.review:
        for r in csv.DictReader(open(args.review, encoding="utf-8-sig")):
            c = (r.get("人工结论") or "").strip()
            if c:
                manual[re.sub(r"\s+", "", r["query"])] = c

    drop_sft_prompt: set[str] = set()
    kept, report, cnt = [], [], collections.Counter()
    for d in dpo:
        q = _q(d["messages"])
        act, why = classify(q, d["messages"][2]["content"], d["rejected_response"])
        mc = manual.get(re.sub(r"\s+", "", q), "")
        if "金标错" in mc:
            act, why = "drop_both", f"D:人工-{mc}"
        elif "模型对" in mc and act == "keep":
            act, why = "drop_dpo", f"D:人工-{mc}"
        cnt[why.split(":")[0] if why else "保留"] += 1
        if act == "drop_both":
            drop_sft_prompt.add(d["messages"][0]["content"] + "\x00" + d["messages"][1]["content"])
        if act == "keep":
            kept.append(d)
        else:
            report.append({"动作": act, "原因": why, "query": q, "chosen": d["messages"][2]["content"],
                           "rejected": d["rejected_response"]})
    sft_kept = [s for s in sft
                if s["messages"][0]["content"] + "\x00" + s["messages"][1]["content"] not in drop_sft_prompt]

    def w(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    w(f"{pre}.clean.sft.jsonl", sft_kept)
    w(f"{pre}.clean.dpo.jsonl", kept)
    with open(f"{pre}.clean.report.csv", "w", encoding="utf-8-sig", newline="") as f:
        cw = csv.DictWriter(f, fieldnames=["动作", "原因", "query", "chosen", "rejected"])
        cw.writeheader()
        cw.writerows(sorted(report, key=lambda r: r["原因"]))
    reasons = collections.Counter(r["原因"].split("（")[0] for r in report)
    print(f"DPO {len(dpo)} → {len(kept)}（剔除 {len(report)}）；SFT {len(sft)} → {len(sft_kept)}")
    for k, v in reasons.most_common():
        print(f"  {v:4d}  {k}")
    print(f"产物：{pre}.clean.sft.jsonl / {pre}.clean.dpo.jsonl / {pre}.clean.report.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
