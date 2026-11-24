#!/usr/bin/env python3
"""从标注数据抽实体库：名字 → 所属域 + 槽位（+ 影视片名常见 category）。

只收「名字类」槽位（片名/人名/歌名/球队等），属性类槽位（category/tag/area/年级…）的取值反过来
当停用表：同一个词若在属性槽位出现过（如「动画」「音乐」），不作为实体收录。
取值必须在该行 query 里原样出现（去空白、不分大小写），防止收进改写/归一后的值。

输出 CSV 列：name,domain,field,count,hint,seeds
  · domain 用 app 域名（vod/children/music/sports）；field 为参数里的槽位名；
  · hint：vod 片名在同行里最常搭配的 category（占比 ≥50% 才填），如「电影」；
  · seeds：该条实体来源行的 seed_query（JSON 数组，去空白），评测时做 leave-seed-out 用。

用法：python3 tools/build_entity_lib.py [--src a.csv b.csv ...] [--out experiments/retrieval/lib/entity_lib.csv]
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_DOMAIN = {"vod": "vod", "educ": "children", "music": "music", "sports": "sports"}
# (评测域, 槽位)：只收名字类槽位
_ENTITY = {
    ("vod", "title"), ("vod", "actor"), ("vod", "director"), ("vod", "role"), ("vod", "person_name"),
    ("educ", "title"), ("educ", "role"),
    ("music", "song"), ("music", "singer"), ("music", "ip"), ("music", "album"),
    ("sports", "sport_team"), ("sports", "sport_star"), ("sports", "sport_game"),
}
# 默认来源：训练集 + 单意图评测金标（评测时按 seeds 做 leave-seed-out，见 experiments/retrieval/entity_check.py）
_SRCS = ("testset/feishu/train.csv", "testset/feishu/20261116/all_eval.csv")
# 国家/地区名常被标成球队（「意大利」），但在「意大利电影」里是地区，不收
_PLACES = set("""中国 美国 英国 法国 德国 意大利 西班牙 葡萄牙 荷兰 比利时 瑞士 瑞典 挪威 丹麦 芬兰 冰岛 波兰 捷克
俄罗斯 乌克兰 希腊 土耳其 日本 韩国 朝鲜 印度 泰国 越南 新加坡 马来西亚 印尼 菲律宾 澳大利亚 新西兰 加拿大 墨西哥
巴西 阿根廷 智利 秘鲁 哥伦比亚 乌拉圭 埃及 南非 伊朗 沙特 以色列 克罗地亚 塞尔维亚 奥地利 匈牙利 爱尔兰 苏格兰
香港 台湾 澳门 内地 大陆 欧美 欧洲 亚洲 非洲 美洲""".split())
_COMMON_FREQ = 300
_SKIP_KEYS = {"retext", "action", "messages", "query", "sort", "intent"}


def _nq(s: str) -> str:
    return re.sub(r"\s+", "", s or "").lower()


def _vals(v) -> list[str]:
    if isinstance(v, list):
        return [str(x).strip() for x in v if isinstance(x, (str, int, float))]
    if isinstance(v, (str, int, float)) and not isinstance(v, bool):
        return [str(v).strip()]
    return []


def _slots(params: dict) -> list[tuple[str, list[str], dict | None]]:
    """(槽位, 取值列表, 所在 and 节点)：query 布尔树里的 field 节点 + 顶层参数。"""
    out: list[tuple[str, list[str], dict | None]] = []

    def walk(n, parent):
        if isinstance(n, dict):
            if "field" in n:
                out.append((str(n["field"]), _vals(n.get("values", n.get("value"))), parent))
                return
            for k in ("and", "or", "not"):
                if k in n:
                    walk(n[k], n)
        elif isinstance(n, list):
            for x in n:
                walk(x, parent)

    q = params.get("query")
    if isinstance(q, (dict, list)):
        walk(q, q if isinstance(q, dict) else None)
    for k, v in params.items():
        if k not in _SKIP_KEYS:
            out.append((k, _vals(v), None))
    return out


def _category(node: dict | None) -> list[str]:
    cats: list[str] = []
    if isinstance(node, dict):
        for c in node.get("and") or []:
            if isinstance(c, dict) and c.get("field") == "category":
                cats += _vals(c.get("values", c.get("value")))
    return cats


def _rows(src: Path) -> list[dict]:
    """兼容两种列名：domain,query,tool,tool_param[,seed_query] / 业务域,query,期望工具,期望参数,domain。"""
    out = []
    for r in csv.DictReader(open(src, encoding="utf-8-sig")):
        out.append({"domain": r.get("domain") or "", "query": r.get("query") or "",
                    "tool_param": r.get("tool_param") or r.get("期望参数") or "",
                    "seed_query": r.get("seed_query") or ""})
    return out


def build(srcs: list[Path]) -> list[dict]:
    rows = [r for src in srcs for r in _rows(src)]
    attr_vals: set[str] = set()
    ents: dict[tuple[str, str, str], dict] = {}
    for r in rows:
        dom = (r.get("domain") or "").strip()
        try:
            params = json.loads(r.get("tool_param") or "{}")
        except json.JSONDecodeError:
            continue
        if not isinstance(params, dict):
            continue
        qn = _nq(r.get("query") or "")
        seed = re.sub(r"\s+", "", r.get("seed_query") or r.get("query") or "")
        for field, vals, node in _slots(params):
            if (dom, field) not in _ENTITY:
                attr_vals.update(_nq(v) for v in vals)
                continue
            for v in vals:
                if not (2 <= len(v) <= 20) or re.fullmatch(r"[\d\W_]+", v) or _nq(v) not in qn:
                    continue
                e = ents.setdefault((v, _DOMAIN[dom], field),
                                    {"count": 0, "cats": collections.Counter(), "seeds": set()})
                e["count"] += 1
                e["seeds"].add(seed)
                if dom == "vod" and field == "title":
                    e["cats"].update(_category(node))
    import jieba
    jieba.setLogLevel(60)
    jieba.initialize()
    out = []
    for (name, d, f), e in ents.items():
        # 短常用词（颜色/启蒙/国际/童话）在别的语境里多半不是名字：≤3 字且 jieba 词频 ≥300 的不收
        common = len(name) <= 3 and (jieba.dt.FREQ.get(name) or 0) >= _COMMON_FREQ
        if _nq(name) in attr_vals or name in _PLACES or common:
            continue
        hint = ""
        if e["cats"]:
            cat, n = e["cats"].most_common(1)[0]
            if n * 2 >= e["count"]:
                hint = cat
        out.append({"name": name, "domain": d, "field": f, "count": e["count"], "hint": hint,
                    "seeds": json.dumps(sorted(e["seeds"]), ensure_ascii=False)})
    out.sort(key=lambda x: (x["name"], -x["count"], x["domain"], x["field"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", default=[str(ROOT / p) for p in _SRCS])
    ap.add_argument("--out", default=str(ROOT / "experiments/retrieval/lib/entity_lib.csv"))
    args = ap.parse_args()
    out = build([Path(p) for p in args.src])
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "domain", "field", "count", "hint", "seeds"])
        w.writeheader()
        w.writerows(out)
    by = collections.Counter((x["domain"], x["field"]) for x in out)
    print(f"{len(out)} 条实体（{len({x['name'] for x in out})} 个名字）-> {args.out}")
    for k, n in sorted(by.items()):
        print(f"  {k[0]}.{k[1]}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
