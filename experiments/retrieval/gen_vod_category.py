#!/usr/bin/env python3
"""vod_search_all 的 category 增强：按金标口径用模板造句，标签由规则直接生成（不经模型）。

口径（统计自 testset/feishu/20261116/all_eval.csv）：
  电影   ← 电影 / 影片 / 片子 / 大片 / <题材>片 / 新片
  电视剧 ← 电视剧 / 剧 / 连续剧 / <题材>剧 / 新剧
  综艺   ← 综艺 / 综艺节目 / 真人秀
  纪录片 ← 纪录片
  <题材>片 / <题材>剧 → tag=<题材> + category；热门/爆款 → sort.hot，高分/口碑 → sort.rate，最新/刚上线 → sort.new
  年份 → release_time yyyy0101~yyyy1231；N 分以上 → rate from N to *；免费 → is_fee "0"

两套互不重叠的实体与句式：
  --kind probe  评测用（不进库）  → testset/probe/vod_category_probe.csv
  --kind aug    进库的增强示例   → experiments/retrieval/lib/exp/vod_category_aug.csv
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CAT_WORDS = {
    "电影": ["电影", "影片", "片子", "大片"],
    "电视剧": ["电视剧", "剧", "连续剧"],
    "综艺": ["综艺", "综艺节目", "真人秀"],
    "纪录片": ["纪录片"],
}
TAG_SUFFIX = {"电影": "片", "电视剧": "剧"}          # <题材>片 / <题材>剧

ENT = {
    "probe": {
        "area": ["香港", "韩国", "日本", "美国", "泰国"],
        "tag_movie": ["悬疑", "武侠", "科幻", "战争", "恐怖"],
        "tag_tv": ["古装", "都市", "悬疑", "谍战"],
        "tag_show": ["竞技", "脱口秀", "旅行"],
        "actor": ["周星驰", "赵丽颖", "黄渤", "孙俪", "梁朝伟"],
        "director": ["王家卫", "冯小刚"],
        "year": [2019, 2021, 2024],
        "rate": ["8", "8.5"],
    },
    "aug": {
        "area": ["台湾", "英国", "法国", "印度", "内地", "国产"],
        "tag_movie": ["动作", "爱情", "犯罪", "灾难", "奇幻", "喜剧"],
        "tag_tv": ["家庭", "青春", "历史", "刑侦", "仙侠"],
        "tag_show": ["音乐", "恋爱", "美食", "喜剧"],
        "actor": ["吴京", "杨幂", "胡歌", "张译", "周冬雨", "成龙", "雷佳音", "倪妮"],
        "director": ["陈凯歌", "徐克", "宁浩", "贾樟柯"],
        "year": [2015, 2017, 2018, 2020, 2022, 2023, 2025],
        "rate": ["7", "7.5", "9"],
    },
}

# 句式模板：{c}=类别词（含 <题材>片 形式时由 {tc} 给出）；两套不重叠
TPL = {
    "probe": {
        "area": ["{area}的{c}有哪些", "想看{area}{c}"],
        "tag": ["有没有{tc}推荐", "想看点{tc}"],
        "actor": ["找几部{actor}主演的{c}", "{actor}有什么好看的{c}"],
        "director": ["{director}拍过哪些{c}"],
        "year": ["{year}年的{tc}推荐一下", "{year}年上映的{c}"],
        "rate": ["评分{rate}分以上的{area}{c}"],
        "free": ["有没有免费的{tc}"],
        "hot": ["最近很火的{area}{c}"],
        "rate_sort": ["口碑最好的{tc}"],
        "new": ["刚上线的{area}{c}"],
    },
    "aug": {
        "area": ["给我搜一下{area}{c}", "{area}拍的{c}都有啥", "来几部{area}{c}", "想瞅瞅{area}{c}", "放几部{area}{c}"],
        "tag": ["帮我找{tc}", "推荐点{tc}看看", "我要看{tc}", "有{tc}吗", "{tc}来几部"],
        "actor": ["{actor}演过的{c}", "搜{actor}的{c}", "{actor}参演的{c}有哪些", "{actor}的{c}都给我找出来"],
        "director": ["{director}导演的{c}", "{director}执导过的{c}有哪些", "{director}都导过什么{c}"],
        "year": ["{year}年出的{c}", "{year}年拍的{tc}", "{year}年{area}{c}", "{year}年的{area}{c}"],
        "rate": ["{rate}分以上的{c}", "豆瓣{rate}分往上的{tc}", "{area}{rate}分以上的{c}"],
        "free": ["不要钱的{c}", "免费能看的{tc}", "{area}免费{c}"],
        "hot": ["热门{c}", "现在大家都在看的{tc}", "爆火的{area}{c}", "{area}最近大火的{c}"],
        "rate_sort": ["高分{c}推荐", "评价好的{tc}", "好评如潮的{tc}", "评分最高的{area}{c}"],
        "new": ["最新上的{c}", "新出的{tc}", "新上架的{area}{c}", "最近更新的{area}{c}"],
    },
}


def leaf(f, v):
    return {"field": f, "value": v}


def build(kind: str, n_per: int, seed: int) -> list[dict]:
    rnd = random.Random(seed)
    E, T = ENT[kind], TPL[kind]
    out, seen = [], set()
    for cat, words in CAT_WORDS.items():
        tags = {"电影": E["tag_movie"], "电视剧": E["tag_tv"], "综艺": E["tag_show"], "纪录片": ["历史", "自然"]}[cat]
        for slot, tpls in T.items():
            for _ in range(n_per):
                tpl = rnd.choice(tpls)
                w = rnd.choice(words)
                tag = rnd.choice(tags)
                # <题材>片 / <题材>剧：电影、电视剧用后缀写法，其余用「<题材><类别词>」
                if "{tc}" in tpl:
                    tc = f"{tag}{TAG_SUFFIX[cat]}" if cat in TAG_SUFFIX and rnd.random() < 0.6 else f"{tag}{w}"
                    use_tag = True
                else:
                    tc, use_tag = "", False
                vals = {"c": w, "tc": tc, "area": rnd.choice(E["area"]), "actor": rnd.choice(E["actor"]),
                        "director": rnd.choice(E["director"]), "year": rnd.choice(E["year"]), "rate": rnd.choice(E["rate"])}
                q = tpl.format(**vals)
                if q in seen:
                    continue
                seen.add(q)
                leaves = [leaf("category", cat)]
                if use_tag:
                    leaves.append(leaf("tag", tag))
                if "{area}" in tpl:
                    leaves.append(leaf("area", vals["area"]))
                if "{actor}" in tpl:
                    leaves.append(leaf("actor", vals["actor"]))
                if "{director}" in tpl:
                    leaves.append(leaf("director", vals["director"]))
                if "{year}" in tpl:
                    y = vals["year"]
                    leaves.append({"field": "release_time", "from": f"{y}0101", "to": f"{y}1231"})
                if "{rate}" in tpl:
                    r = vals["rate"] if "." in vals["rate"] else vals["rate"] + ".0"
                    leaves.append({"field": "rate", "from": r, "to": "*"})
                if slot == "free":
                    leaves.append(leaf("is_fee", "0"))
                p = {"action": "search", "query": leaves[0] if len(leaves) == 1 else {"and": leaves}}
                sort = {"hot": "hot", "rate_sort": "rate", "new": "new"}.get(slot)
                if sort:
                    p["sort"] = {sort: {"order": "desc"}}
                out.append({"domain": "vod", "query": q, "tool": "vod_search_all",
                            "tool_param": json.dumps(p, ensure_ascii=False), "seed_query": q})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["probe", "aug"], required=True)
    ap.add_argument("--n-per", type=int, default=0)
    a = ap.parse_args()
    n = a.n_per or (2 if a.kind == "probe" else 6)
    rows = build(a.kind, n, seed=7 if a.kind == "probe" else 11)
    if a.kind == "probe":
        path = ROOT / "testset/probe/vod_category_probe.csv"
        cols = ["业务域", "query", "期望工具", "期望参数", "domain"]
        recs = [{"业务域": "影视", "query": r["query"], "期望工具": r["tool"], "期望参数": r["tool_param"], "domain": "vod"} for r in rows]
    else:
        path = ROOT / "experiments/retrieval/lib/exp/vod_category_aug.csv"
        cols = ["domain", "query", "tool", "tool_param", "seed_query"]
        recs = rows
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(recs)
    print(f"{a.kind}: {len(recs)} 条 → {path}")


if __name__ == "__main__":
    main()
