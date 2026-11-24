#!/usr/bin/env python3
"""从已标注数据的参数里抽实体名，编译成「名字 → 类型」词典（原型；正式版应换成媒资库/曲库全量导出）。

类型：歌曲 / 歌手 / 影视 / 少儿 / 人名
来源：music_*_search 的 song、singer；vod_search_all 的 title、actor、director；
      educ_search_all 的 title、role。名字必须原样出现在 query 里（避免把归一后的值当名字）。
过滤：长度 < 2、纯数字、类别词/常用词黑名单。

用法：python3 tools/build_entity_dict.py [--out experiments/retrieval/lib/entity_dict.csv]
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = [ROOT / "testset/train/train.csv", ROOT / "experiments/retrieval/lib/pattern_all.csv"]
STOP = set("""儿歌 歌曲 音乐 电影 电视剧 综艺 纪录片 动画 动画片 动漫 卡通 绘本 故事 早教 歌 曲子 新歌 老歌 热歌
流行 经典 抖音 网络 华语 英文 中文 粤语 国语 少儿 儿童 宝宝 小朋友 免费 高清 全集 第一集 最新 热门 好看
视频 节目 片子 影片 大片 喜剧 动作 爱情 悬疑 古装 都市 科幻 恐怖 战争 武侠 英语 数学 语文 拼音 识字 颜色
小鸡 小猫 小狗 小兔 恐龙 汽车 火车 飞机 公主 王子 妈妈 爸爸 宝贝 朋友 生日 新年 春节 圣诞""".split())
FIELDS = {
    "music": {"song": "歌曲", "singer": "歌手"},
    "vod_search_all": {"title": "影视", "actor": "人名", "director": "人名"},
    "educ_search_all": {"title": "少儿", "role": "少儿"},
}


def _leaves(n, out):
    if isinstance(n, dict):
        if "field" in n:
            vs = n.get("values") if "values" in n else [n.get("value")]
            for v in vs or []:
                if isinstance(v, str):
                    out.append((n["field"], v))
            return
        for k, v in n.items():
            if k in ("and", "or", "not"):
                _leaves(v, out)
    elif isinstance(n, list):
        for x in n:
            _leaves(x, out)


def extract(tool: str, p: dict) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if tool.startswith("music_") and tool.endswith("search"):
        for k, t in FIELDS["music"].items():
            out += [(t, v) for v in p.get(k) or [] if isinstance(v, str)]
    elif tool in FIELDS:
        lv: list = []
        _leaves(p.get("query"), lv)
        out += [(FIELDS[tool][f], v) for f, v in lv if f in FIELDS[tool]]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "experiments/retrieval/lib/entity_dict.csv"))
    a = ap.parse_args()
    csv.field_size_limit(10**9)
    cnt: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for src in SRC:
        for r in csv.DictReader(open(src, encoding="utf-8-sig")):
            try:
                p = json.loads(r.get("tool_param") or "{}")
            except json.JSONDecodeError:
                continue
            if not isinstance(p, dict):
                continue
            q = r.get("query") or ""
            for t, v in extract((r.get("tool") or "").strip(), p):
                v = v.strip()
                if len(v) < 2 or v.isdigit() or v in STOP or v not in q:
                    continue
                cnt[v][t] += 1
    with open(a.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "types", "freq"])
        for v, c in sorted(cnt.items(), key=lambda x: (-sum(x[1].values()), x[0])):
            w.writerow([v, "|".join(t for t, _ in c.most_common()), sum(c.values())])
    tc = collections.Counter(c.most_common(1)[0][0] for c in cnt.values())
    print(f"{len(cnt)} 个名字 {dict(tc)}，跨类型 {sum(len(c) > 1 for c in cnt.values())} → {a.out}")


if __name__ == "__main__":
    main()
