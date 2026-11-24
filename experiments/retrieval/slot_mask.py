#!/usr/bin/env python3
"""层次化检索 L2 的离线原型：名字（实体库）+ 闭集槽值（学科/年级/版本/类别/标签/赛事…）统一换占位符。

  · 名字：app.entity.mask（实体库 entity_lib.csv，支持 leave-seed-out）；
  · 槽值：lib/slot_vocab.csv（value,domain,field,count）。由本文件 build 子命令从 few-shot 库的金标里抽：
    只收闭集字段（CLOSED），取值须原样出现在 query 里，出现 ≥ MIN_COUNT 次；另加年级/册次正则。
    设备域不做（operation/object 本身就是要学的写法）。
占位符形如「E学科」「E年级」「E影视类别」：BM25 时同类槽值互相匹配，few-shot 只需覆盖「句式 → 写法」。

用法：python3 experiments/retrieval/slot_mask.py build   # 生成 lib/slot_vocab.csv
"""
from __future__ import annotations

import collections
import csv
import json
import re
import sys

from common import HERE, nq

VOCAB = HERE / "lib/slot_vocab.csv"
MIN_COUNT = 3
# (评测域, 字段) → 占位符
CLOSED = {
    ("edu", "first_subject"): "E学科", ("edu", "grade"): "E年级", ("edu", "material_version"): "E教材版本",
    ("edu", "figures"): "E学段", ("edu", "semester"): "E学期",
    ("vod", "category"): "E影视类别", ("vod", "tag"): "E影视标签", ("vod", "area"): "E地区",
    ("vod", "language"): "E语言",
    ("educ", "content_type"): "E少儿形式", ("educ", "children_second_genre"): "E少儿类别",
    ("educ", "children_third_genre"): "E少儿类别", ("educ", "language"): "E语言", ("educ", "country"): "E地区",
    ("sports", "sport_game"): "E赛事", ("sports", "sport_name"): "E运动项目",
    ("sports", "sport_rank_type"): "E榜单", ("sports", "sport_competition_phase"): "E赛段",
    ("music", "tag"): "E音乐标签",
}
_REGEX = [
    (re.compile(r"(?:[一二三四五六七八九]|[1-9])年级"), "E年级"),
    (re.compile(r"(?:初|高)[一二三]"), "E年级"),
    (re.compile(r"(?:上|下)(?:册|学期)"), "E学期"),
]


def _leaves(n, out):
    if isinstance(n, dict):
        if "field" in n:
            for v in (n.get("values") or [n.get("value")]):
                if isinstance(v, str):
                    out.append((n["field"], v))
            return
        for k, v in n.items():
            if k in ("and", "or", "not"):
                _leaves(v, out)
    elif isinstance(n, list):
        for x in n:
            _leaves(x, out)


def build(src=HERE / "lib/pattern_all.csv") -> None:
    cnt: collections.Counter = collections.Counter()
    for r in csv.DictReader(open(src, encoding="utf-8-sig")):
        try:
            p = json.loads(r["tool_param"] or "{}")
        except json.JSONDecodeError:
            continue
        if not isinstance(p, dict):
            continue
        o: list = []
        _leaves(p.get("query"), o)
        for k, v in p.items():
            if isinstance(v, str) and k != "query":
                o.append((k, v))
        q = nq(r["query"])
        for f, v in o:
            if (r["domain"], f) in CLOSED and len(v) >= 2 and v in q:
                cnt[(v, r["domain"], f)] += 1
    with open(VOCAB, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["value", "domain", "field", "count"])
        for (v, d, f), c in sorted(cnt.items(), key=lambda x: (-x[1], x[0])):
            if c >= MIN_COUNT:
                w.writerow([v, d, f, c])
    print(f"{sum(c >= MIN_COUNT for c in cnt.values())} 个槽值 → {VOCAB}")


_VOC: list[tuple[str, str]] | None = None


def _vocab() -> list[tuple[str, str]]:
    """[(value, 占位符)]，按长度降序；同一 value 取出现最多的字段。"""
    global _VOC
    if _VOC is None:
        best: dict[str, tuple[int, str]] = {}
        for r in csv.DictReader(open(VOCAB, encoding="utf-8-sig")):
            tag, c = CLOSED[(r["domain"], r["field"])], int(r["count"])
            if r["value"] not in best or c > best[r["value"]][0]:
                best[r["value"]] = (c, tag)
        _VOC = sorted(((v, t) for v, (c, t) in best.items()), key=lambda x: -len(x[0]))
    return _VOC


def _tags() -> list[str]:
    """所有占位符（槽值 + 实体库），长度降序。"""
    from app import entity
    ts = set(CLOSED.values()) | {t for _r, t in _REGEX}
    ts |= {"E" + entity._DOMAIN_CN.get(d, d) + f.split("（")[0] for (d, _f), f in entity._FIELD_CN.items()}
    return sorted(ts, key=len, reverse=True)


def mask_slots(q: str) -> str:
    """闭集槽值 → 占位符（最长优先、不重叠；已是占位符的片段不再替换）。"""
    for rx, tag in _REGEX:
        q = rx.sub(tag, q)
    voc = _vocab()
    tags = _tags()
    out, i = [], 0
    while i < len(q):
        t = next((t for t in tags if q.startswith(t, i)), None)     # 已有占位符原样保留
        if t:
            out.append(t)
            i += len(t)
            continue
        v = next((v for v, _t in voc if q.startswith(v, i)), None)
        if v:
            out.append(dict(voc)[v])
            i += len(v)
        else:
            out.append(q[i])
            i += 1
    return "".join(out)


def mask_all(q: str, exclude_seed: str = "") -> str:
    """名字（实体库，可 leave-seed-out）→ 槽值，两层占位符。"""
    from app import entity
    return mask_slots(entity.mask(q, exclude_seed=exclude_seed))


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        build()
    else:
        for s in sys.argv[1:]:
            print(s, "->", mask_all(s))
