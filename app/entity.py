"""实体库（entity lexicon）：按本轮 query 词典匹配名字，告诉模型「这个名字在库里是什么」。

解决「用户只说名字」时的落域/槽位歧义：如「播放三生三世十里桃花」里的名字在库里是影视片名（电视剧），
不是歌曲。匹配是纯字符串最长匹配（无 LLM、无网络），不改变「单轮 1 次 LLM」。

开关 HC_ENTITY_ENABLED（默认关）。开启后首轮 system prompt 最末尾追加「实体库参考」块
（在 few-shot / 槽值块之后，前缀 KV cache 不受影响）。

库 CSV 列：name,domain,field,count[,hint,seeds]（由 tools/build_entity_lib.py 生成）。
  · 同一名字可有多条（多种类型），按 count 降序展示；
  · seeds 仅供评测做 leave-seed-out（exclude_seed），运行期不用。
确定性：同一 query + 同一库 → 同一文本。库缺失/坏行一律降级为空块，不影响主干。
"""
from __future__ import annotations

import csv
import json
import logging
import re
import threading
from pathlib import Path

from .config import _BASE, get_settings

logger = logging.getLogger("hcOneshot.entity")

_DOMAIN_CN = {"vod": "影视", "children": "少儿", "music": "音乐", "sports": "体育",
              "education": "教育", "audio": "有声"}
_FIELD_CN = {
    ("vod", "title"): "片名", ("vod", "actor"): "演员", ("vod", "director"): "导演",
    ("vod", "role"): "角色", ("vod", "person_name"): "人物",
    ("children", "title"): "节目名", ("children", "role"): "角色",
    ("music", "song"): "歌曲名", ("music", "singer"): "歌手", ("music", "ip"): "IP（影视/动画作品的相关歌曲）",
    ("music", "album"): "专辑",
    ("sports", "sport_team"): "球队", ("sports", "sport_star"): "运动员", ("sports", "sport_game"): "赛事",
}
_HEADER = ("【实体库参考】（系统按本轮请求里的名字查实体库得到，只说明名字在库里属于哪个域的哪类槽位）\n"
           "用法：只用来判断名字指代什么、落哪个域、填哪个槽位；用户原话里有明确线索"
           "（如「歌/听/唱」「电影/电视剧」「动画片」「比赛」）时以原话为准；槽值仍按用户原话填写。\n")
_MAX_TYPES = 3


def _nq(s: str) -> str:
    return re.sub(r"\s+", "", s or "").lower()


class _Lexicon:
    def __init__(self, path: Path) -> None:
        self.by: dict[str, list[dict]] = {}
        for r in csv.DictReader(open(path, encoding="utf-8-sig")):
            name, d, f = (r.get("name") or "").strip(), (r.get("domain") or "").strip(), (r.get("field") or "").strip()
            if len(_nq(name)) < 2 or not d or not f:
                continue
            try:
                count = int(r.get("count") or 1)
                seeds = frozenset(json.loads(r.get("seeds") or "[]"))
            except (ValueError, TypeError):
                continue
            ents = self.by.setdefault(_nq(name), [])
            same = next((e for e in ents if e["domain"] == d and e["field"] == f), None)
            if same:                              # 只差大小写/空白的同类条目合并（NBA / nba）
                same["count"] += count
                same["seeds"] |= seeds
                continue
            ents.append({"name": name, "domain": d, "field": f, "count": count,
                         "hint": (r.get("hint") or "").strip(), "seeds": seeds})
        for v in self.by.values():
            v.sort(key=lambda e: (-e["count"], e["domain"], e["field"]))
        self.max_len = max((len(k) for k in self.by), default=0)

    def match(self, query: str, exclude_seed: str = "") -> list[tuple[str, list[dict]]]:
        """从左到右最长匹配，命中不重叠。exclude_seed：去掉只来自该 seed 的条目（评测防泄漏）。"""
        q = _nq(query)
        ex = re.sub(r"\s+", "", exclude_seed or "")
        out, i = [], 0
        while i < len(q):
            for n in range(min(self.max_len, len(q) - i), 1, -1):
                ents = self.by.get(q[i:i + n])
                if ents and ex:
                    ents = [e for e in ents if not e["seeds"] or e["seeds"] - {ex}]
                if ents:
                    out.append((q[i:i + n], ents))
                    i += n
                    break
            else:
                i += 1
        return out


_LEX: _Lexicon | None = None
_LEX_PATH = ""
_LOCK = threading.Lock()


def _lexicon() -> _Lexicon | None:
    global _LEX, _LEX_PATH
    p = Path(get_settings().entity_lib)
    path = p if p.is_absolute() else _BASE / p
    with _LOCK:
        if _LEX is None or _LEX_PATH != str(path):
            try:
                _LEX = _Lexicon(path)
                _LEX_PATH = str(path)
                logger.info("实体库载入：%s（%d 个名字）", path, len(_LEX.by))
            except (OSError, csv.Error) as exc:
                logger.warning("实体库载入失败（降级为无实体块）：%s", exc)
                return None
    return _LEX


def _label(e: dict) -> str:
    d, f = e["domain"], e["field"]
    s = f"{_DOMAIN_CN.get(d, d)}·{_FIELD_CN.get((d, f), f)}（{d}.{f}"
    return s + (f"，常见 category={e['hint']}）" if e["hint"] else "）")


def render(query: str, exclude_seed: str = "") -> str:
    """匹配 + 渲染；无命中返回空串。不看开关（评测可直接调用）。"""
    lex = _lexicon()
    if lex is None:
        return ""
    hits = lex.match(query, exclude_seed)[:get_settings().entity_max_hits]
    if not hits:
        return ""
    lines = [f"- {ents[0]['name']}：" + "；".join(_label(e) for e in ents[:_MAX_TYPES]) for _, ents in hits]
    return "\n" + _HEADER + "\n".join(lines) + "\n"


# 评测钩子：非空时 mask() 的 query 侧匹配按该 seed 做 leave-seed-out（运行期恒为空）
_EXCL_SEED = ""


def mask(query: str, exclude_seed: str | None = None) -> str:
    """名字换成类型占位符（如「播放庆余年」→「播放E影视片名」），供 few-shot 按句式检索；无库/无命中原样返回。"""
    lex = _lexicon()
    if lex is None:
        return query
    hits = lex.match(query, _EXCL_SEED if exclude_seed is None else exclude_seed)
    if not hits:
        return query
    q, pos = _nq(query), 0
    for n, ents in hits:                          # match 结果按从左到右、互不重叠
        e = ents[0]
        tag = "E" + _DOMAIN_CN.get(e["domain"], e["domain"]) \
            + _FIELD_CN.get((e["domain"], e["field"]), e["field"]).split("（")[0]
        i = q.find(n, pos)
        q, pos = q[:i] + tag + q[i + len(n):], i + len(tag)
    return q


def context_block(query: str) -> str:
    """引擎调用口：关闭或失败返回空串。"""
    if not get_settings().entity_enabled:
        return ""
    try:
        return render(query)
    except Exception as exc:  # noqa: BLE001
        logger.warning("实体块生成失败（降级）：%s", exc)
        return ""
