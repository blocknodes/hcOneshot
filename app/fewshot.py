"""动态 few-shot context（实验移植自 experiments/retrieval：L1 关键词规则 + L3 BM25 相似示例）。

开关 HC_FEWSHOT_ENABLED（默认关）。开启后首轮 system prompt 末尾追加：
  1. 规则提示：query 命中模式（目前只有少儿检索口径）→ 注入标注口径；
  2. 相似示例：按关键词 BM25（jieba 词 + 字二元组）从标注库检索 top-k，渲染成 few-shot。
同时修正静态 prompt 里与金标冲突的两处（设备锚例、「拿不准选通用检索」）。

标注库 CSV 列：domain,query,tool,tool_param[,seed_query]（domain 为评测域名 vod/educ/edu/...）。
确定性：同一 query + 同一库 → 同一文本。失败（库缺失/坏行）一律降级为空块，不影响主干。
"""
from __future__ import annotations

import collections
import csv
import json
import logging
import math
import re
import threading
from pathlib import Path

from .config import _BASE, get_settings

logger = logging.getLogger("hcOneshot.fewshot")

_DOMAIN = {"vod": "vod", "educ": "children", "edu": "education", "music": "music",
           "audio": "audio", "sports": "sports", "tv_control": "device"}
_DOMAIN |= {v: v for v in _DOMAIN.values()}
_TOOL_ALIAS = {"vod_search": "vod_search_all", "educ_search": "educ_search_all",
               "educ_relate_recommend": "educ_relate_search",
               "sports_match_data_search": "sports_rank_search"}

# ---------------------------------------------------------------------------
# 静态 prompt 修正
# ---------------------------------------------------------------------------
_STATIC_FIXES = [
    ('"params":{"object":"volume","value":"30"}',
     '"params":{"operation":"设置","object":"音量","value":"30"}'),
    ("  拿不准就选该域最通用的检索工具。vt 的域须与 \"d\" 一致。",
     "  vt 的域须与 \"d\" 一致。检索类工具在 search 与 fuzzy 之间按各工具描述的适用场景区分。"),
]


def fix_static(prompt: str) -> str:
    for old, new in _STATIC_FIXES:
        prompt = prompt.replace(old, new)
    return prompt


# ---------------------------------------------------------------------------
# L1 规则
# ---------------------------------------------------------------------------
_KID = (r"少儿|儿童|孩子|宝宝|宝贝|幼儿|小朋友|早教|启蒙|儿歌|绘本|动画|动漫|卡通|童话|"
        r"汪汪队|佩奇|熊出没|喜羊羊|超级飞侠|奥特曼|贝瓦|宝宝巴士|巧虎|海底小纵队|猪猪侠|小马宝莉|"
        r"迪迦|葫芦娃|托马斯|布鲁伊|巴巴爸爸|小猪|艾莎|冰雪奇缘|芭比|变形警车")

_RULES = [
    {"id": "children_search", "match": [_KID], "exclude": [],
     "text": """【口径提示：少儿检索（children:educ_search_all / educ_relate_search）】
- 参数只有 query（及可选 sort）：不要加 action；不要加 sort（即使用户说最火/最新/热门/继续播放）。
- content_type 只在用户明确说了「动画/动画片/卡通」时才加，且只填一个值（动画/动画片→"动画"，卡通→"卡通"），
  不要用 values/or 展开；只说片名时不要补 content_type。
- 用户说「动漫」→ {"field":"children_second_genre","value":"动漫"}，不用 content_type。
- 「XX年以后/之后/至今」→ release_time 的 to 填【当前时间】当天日期（yyyyMMdd），不填 "*"。
- language 按用户说法：说「XX版」→ 国语版/英文版/日语版（中文版→国语版，英语版→英文版，日文版→日语版）；
  只说「英语/日语」→ 英语/日语。排除语言（「不要英语的」）用 {"not":{"field":"language","value":"英语"}}。
- title 只填作品名本身，去掉「的视频/的故事/全屏播放/开始/所有」等动作或修饰词；不要擅自补全或改写片名。
- 只报一个角色名（光头强、小丸子、jojo 等可能是角色也可能是片名）→ {"or":[{"field":"role",...},{"field":"title",...}]}。
- 不要使用目录里没有的字段（如 rate、channel、is_over）；用户的评分/频道条件无对应字段就不传。
例：2023年以后的动画片 → {"query":{"and":[{"field":"content_type","value":"动画"},{"field":"release_time","from":"20230101","to":"<当前日期>"}]}}
例：播放彼得兔 → {"query":{"field":"title","value":"彼得兔"}}"""},
]


def rules_block(query: str) -> str:
    return "\n".join(r["text"] for r in _RULES
                     if any(re.search(p, query) for p in r["match"])
                     and not any(re.search(p, query) for p in r["exclude"]))


# ---------------------------------------------------------------------------
# L3 BM25 相似示例
# ---------------------------------------------------------------------------
_K1, _B = 1.2, 0.75
_STOP = set("的了吗呢吧啊呀我你他她它们给帮请要想看听放播放一下个些点再把被和与或及就都也还在是有")


def _nq(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def _tok(s: str) -> list[str]:
    import jieba
    s = re.sub(r"[\s，。！？、,.!?《》“”\"'；;：:（）()]", "", s or "")
    return [w for w in jieba.lcut(s) if w not in _STOP] + [s[i:i + 2] for i in range(len(s) - 1)]


class _Index:
    def __init__(self, path: Path) -> None:
        import jieba
        jieba.setLogLevel(60)
        self.docs: list[dict] = []
        for r in csv.DictReader(open(path, encoding="utf-8-sig")):
            d = _DOMAIN.get((r.get("domain") or "").strip())
            tool = _TOOL_ALIAS.get((r.get("tool") or "").strip(), (r.get("tool") or "").strip())
            q = (r.get("query") or "").strip()
            if not (d and tool and q):
                continue
            tp = (r.get("tool_param") or "").strip()
            try:
                params = json.loads(tp) if tp else {}
            except json.JSONDecodeError:
                continue
            plan = [{"q": q, "d": d, "vt": f"{d}:{tool}", "params": params}]
            self.docs.append({"q": q, "qn": _nq(q),
                              "text": f"用户：{q}\n{json.dumps(plan, ensure_ascii=False, separators=(',', ':'))}"})
        self.tf = [collections.Counter(_tok(d["q"])) for d in self.docs]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = sum(self.len) / max(1, len(self.len))
        df: collections.Counter = collections.Counter()
        self.post: dict[str, list[int]] = collections.defaultdict(list)
        for i, t in enumerate(self.tf):
            for w in t:
                df[w] += 1
                self.post[w].append(i)
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def search(self, q: str, k: int, min_score: float, exclude_self: bool) -> list[int]:
        qn = _nq(q)
        sc: dict[int, float] = collections.defaultdict(float)
        for w in set(_tok(q)):
            idf = self.idf.get(w)
            if not idf:
                continue
            for i in self.post[w]:
                f = self.tf[i][w]
                sc[i] += idf * f * (_K1 + 1) / (f + _K1 * (1 - _B + _B * self.len[i] / self.avg))
        out = []
        for i, s in sorted(sc.items(), key=lambda x: (-x[1], x[0])):
            if exclude_self and self.docs[i]["qn"] == qn:
                continue
            if s < min_score:
                break
            out.append(i)
            if len(out) >= k:
                break
        return out


_IDX: _Index | None = None
_IDX_PATH = ""
_LOCK = threading.Lock()


def _lib_path() -> Path:
    p = Path(get_settings().fewshot_lib)
    return p if p.is_absolute() else _BASE / p


def _index() -> _Index | None:
    global _IDX, _IDX_PATH
    path = _lib_path()
    with _LOCK:
        if _IDX is None or _IDX_PATH != str(path):
            try:
                _IDX = _Index(path)
                _IDX_PATH = str(path)
                logger.info("few-shot 库载入：%s（%d 条）", path, len(_IDX.docs))
            except (OSError, csv.Error) as exc:
                logger.warning("few-shot 库载入失败（降级为无示例）：%s", exc)
                return None
    return _IDX


def examples_block(query: str) -> str:
    s = get_settings()
    idx = _index()
    if idx is None or not idx.docs:
        return ""
    hits = idx.search(query, s.fewshot_topk, s.fewshot_min_score, s.fewshot_exclude_self)
    if not hits:
        return ""
    return ("【相似请求的标注示例（按关键词检索，供参考工具选择与参数写法；以本轮请求为准，不要照抄取值）】\n"
            + "\n\n".join(idx.docs[i]["text"] for i in hits))


def warmup() -> None:
    """开启时在启动期预建索引，避免首个请求等待（约数秒）。"""
    if get_settings().fewshot_enabled:
        _index()


def context_block(query: str) -> str:
    """规则 + 示例；关闭或失败返回空串。"""
    if not get_settings().fewshot_enabled:
        return ""
    try:
        parts = [b for b in (rules_block(query), examples_block(query)) if b]
    except Exception as exc:  # noqa: BLE001
        logger.warning("few-shot context 生成失败（降级）：%s", exc)
        return ""
    return ("\n" + "\n".join(parts)) if parts else ""


def system_prompt(base_prompt: str, query: str) -> str:
    """首轮 system：开启时修正静态冲突 + 末尾追加动态 context；关闭时原样返回。"""
    if not get_settings().fewshot_enabled:
        return base_prompt
    return fix_static(base_prompt) + context_block(query)
