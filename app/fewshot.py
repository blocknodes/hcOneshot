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
import contextvars
import csv
import datetime
import json
import logging
import math
import re
import threading
from pathlib import Path

from . import device_schema
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


_RLIB: list[dict] | None = None
_RLIB_PATH = ""


def _rule_lib() -> list[dict] | None:
    """规则库（CSV 列 id,domain,priority,match,require,exclude,entity,text）；未配置或失败返回 None。"""
    global _RLIB, _RLIB_PATH
    lib = (get_settings().fewshot_rules_lib or "").strip()
    if not lib:
        return None
    path = Path(lib) if Path(lib).is_absolute() else _BASE / lib
    with _LOCK:
        if _RLIB is None or _RLIB_PATH != str(path):
            try:
                rows = []
                for r in csv.DictReader(open(path, encoding="utf-8-sig")):
                    rows.append({"id": r["id"], "priority": int(r.get("priority") or 9),
                                 "match": re.compile(r["match"]) if r.get("match") else None,
                                 "require": re.compile(r["require"]) if r.get("require") else None,
                                 "exclude": re.compile(r["exclude"]) if r.get("exclude") else None,
                                 "entity": {t for t in (r.get("entity") or "").split("|") if t},
                                 "text": r["text"].strip()})
                _RLIB, _RLIB_PATH = sorted(rows, key=lambda x: x["priority"]), str(path)
                logger.info("规则库载入：%s（%d 条）", path, len(rows))
            except (OSError, csv.Error, re.error, KeyError) as exc:
                logger.warning("规则库载入失败（退回内置规则）：%s", exc)
                return None
    return _RLIB


def matched_rules(query: str) -> list[str]:
    """命中的规则 id（按优先级，最多 fewshot_rules_max 条）。触发：正则 match 命中，或实体词典命中 entity 类型；
    require 须同时命中；exclude 命中则不触发。"""
    lib = _rule_lib()
    if lib is None:
        return []
    ent, _ = _entity()
    etypes = {t for _s, _e, n in ent.match(query) for t in ent.types[n]} if ent is not None else set()
    out = []
    for r in lib:
        hit = bool(r["match"] and r["match"].search(query)) or bool(r["entity"] & etypes)
        if not hit or (r["require"] and not r["require"].search(query)) or (r["exclude"] and r["exclude"].search(query)):
            continue
        out.append(r)
    return [r["id"] for r in out[: get_settings().fewshot_rules_max]]


def rules_block(query: str) -> str:
    lib = _rule_lib()
    if lib is None:
        return "\n".join(r["text"] for r in _RULES
                         if any(re.search(p, query) for p in r["match"])
                         and not any(re.search(p, query) for p in r["exclude"]))
    ids = set(matched_rules(query))
    return "\n".join(r["text"] for r in lib if r["id"] in ids)


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
        virtual = get_settings().device_virtual
        skipped = 0
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
            if d == "device" and virtual:
                params = device_schema.virtualize_params(tool, params)
                if params is None:                # 虚拟 schema 无法表示的示例不进库
                    skipped += 1
                    continue
            plan = [{"q": q, "d": d, "vt": f"{d}:{tool}", "params": params}]
            # seed：同一种子的改写家族共用，检索时每个家族只取一条（缺列时退化为 query 本身）
            self.docs.append({"q": q, "qn": _nq(q), "seed": _nq(r.get("seed_query") or q),
                              "text": f"用户：{q}\n{json.dumps(plan, ensure_ascii=False, separators=(',', ':'))}"})
        if skipped:
            logger.info("设备虚拟 schema：%d 条设备示例无法表示，已跳过", skipped)
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
        seeds: set[str] = set()
        for i, s in sorted(sc.items(), key=lambda x: (-x[1], x[0])):
            if exclude_self and self.docs[i]["qn"] == qn:
                continue
            if s < min_score:
                break
            seed = self.docs[i].get("seed") or self.docs[i]["qn"]
            if seed in seeds:                     # 同一种子的改写只留分数最高的一条，避免 top-k 被同义句占满
                continue
            seeds.add(seed)
            out.append(i)
            if len(out) >= k:
                break
        return out


    def top_score(self, q: str, exclude_self: bool = True) -> float:
        """本轮 query 在库里的最高 BM25 分（排除与 query 完全相同的条目）；无命中为 0。"""
        qn = _nq(q)
        sc: dict[int, float] = collections.defaultdict(float)
        for w in set(_tok(q)):
            idf = self.idf.get(w)
            if not idf:
                continue
            for i in self.post[w]:
                f = self.tf[i][w]
                sc[i] += idf * f * (_K1 + 1) / (f + _K1 * (1 - _B + _B * self.len[i] / self.avg))
        return max((v for i, v in sc.items() if not (exclude_self and self.docs[i]["qn"] == qn)), default=0.0)


_IDX: _Index | None = None
_IDX_PATH = ""
_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# 实体链接：词典最长匹配 → 类型占位符检索 + 实体参考块（不调模型）
# ---------------------------------------------------------------------------
_ENT_LABEL = {"歌曲": "歌曲（音乐域点歌）", "歌手": "歌手（音乐域）", "影视": "影视作品（vod）",
              "少儿": "少儿内容（children）", "人名": "人名（演员/导演）", "设备": "设备功能（device）"}


class _EntityDict:
    def __init__(self, path: Path) -> None:
        self.types: dict[str, list[str]] = {}
        self.labels: dict[str, str] = {}          # 可选 label 列：该名字的完整说明（如设备对象的标准写法），优先于类型标签
        for r in csv.DictReader(open(path, encoding="utf-8-sig")):
            n = (r.get("name") or "").strip()
            ts = [t for t in (r.get("types") or "").split("|") if t in _ENT_LABEL]
            if len(n) >= 2 and ts:
                self.types[n] = ts
                if (r.get("label") or "").strip():
                    self.labels[n] = r["label"].strip()
        self.by_first: dict[str, list[str]] = collections.defaultdict(list)
        for n in sorted(self.types, key=len, reverse=True):
            self.by_first[n[0]].append(n)

    def match(self, q: str) -> list[tuple[int, int, str]]:
        """从左到右最长匹配，返回 [(start, end, name)]，互不重叠。"""
        out, i = [], 0
        while i < len(q):
            hit = next((n for n in self.by_first.get(q[i], ()) if q.startswith(n, i)), None)
            if hit:
                out.append((i, i + len(hit), hit))
                i += len(hit)
            else:
                i += 1
        return out

    def mask(self, q: str) -> str:
        # 设备对象不换占位符：「<设备>设置」会把无线网设置、壁画设置这类不同工具的句子混到一起
        spans = [x for x in self.match(q) if self.types[x[2]][0] != "设备"]
        for s, e, n in reversed(spans):
            q = q[:s] + f"<{self.types[n][0]}>" + q[e:]
        return q


class _MaskedIndex(_Index):
    """与基础库共用 docs（渲染原句示例），只把检索文本换成占位符版。"""

    def __init__(self, base: _Index, ent: _EntityDict) -> None:  # noqa: D401
        self.docs = base.docs
        self.tf = [collections.Counter(_tok(ent.mask(d["q"]))) for d in self.docs]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = sum(self.len) / max(1, len(self.len))
        df: collections.Counter = collections.Counter()
        self.post = collections.defaultdict(list)
        for i, t in enumerate(self.tf):
            for w in t:
                df[w] += 1
                self.post[w].append(i)
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}


_ENT: _EntityDict | None = None
_MIDX_ENT: _MaskedIndex | None = None
_ENT_KEY = ""


def _entity() -> tuple[_EntityDict | None, _MaskedIndex | None]:
    """实体词典与占位符索引（随基础库/词典路径变化重建）；未配置或失败返回 (None, None)。"""
    global _ENT, _MIDX_ENT, _ENT_KEY
    lib = (get_settings().fewshot_entity_dict or "").strip()
    idx = _index()
    if not lib or idx is None:
        return None, None
    path = Path(lib) if Path(lib).is_absolute() else _BASE / lib
    key = f"{path}|{_IDX_PATH}"
    with _LOCK:
        if _ENT is None or _ENT_KEY != key:
            try:
                ent = _EntityDict(path)
                _MIDX_ENT, _ENT, _ENT_KEY = _MaskedIndex(idx, ent), ent, key
                logger.info("实体词典载入：%s（%d 个名字）", path, len(ent.types))
            except (OSError, csv.Error) as exc:
                logger.warning("实体词典载入失败（降级为不做实体链接）：%s", exc)
                return None, None
    return _ENT, _MIDX_ENT


def entity_block(query: str) -> str:
    ent, _ = _entity()
    if ent is None or not get_settings().fewshot_entity_hint:
        return ""
    seen, lines = set(), []
    for _s, _e, n in ent.match(query):
        if n not in seen:
            seen.add(n)
            lines.append(f"- {n}：" + (ent.labels.get(n) or " / ".join(_ENT_LABEL[t] for t in ent.types[n])))
    if not lines:
        return ""
    return ("【实体参考】（系统按名称匹配媒资词典得到，仅用于判断名字是什么；用户明确说了类别词或平台时以用户为准）\n"
            + "\n".join(lines))


def _search_merged(idx: _Index, q: str, k: int, min_score: float, exclude_self: bool) -> list[int]:
    """原句检索 + 占位符检索（有实体命中时）交替合并、去重，取前 k。"""
    hits = idx.search(q, k, min_score, exclude_self)
    ent, midx = _entity()
    if ent is None or midx is None or idx is not _IDX:
        return hits
    mq = ent.mask(q)
    if mq == q:
        return hits
    mh = midx.search(mq, k, min_score, exclude_self)
    out, seen, seeds = [], set(), set()
    for pair in zip(mh + [None] * k, hits + [None] * k):
        for i in pair:
            if i is None or i in seen:
                continue
            seed = idx.docs[i].get("seed") or idx.docs[i]["qn"]
            if seed in seeds:
                continue
            seen.add(i)
            seeds.add(seed)
            out.append(i)
    return out[:k]



# ---------------------------------------------------------------------------
# 示例日期换算：标注库里的相对时间（明天/周六/上周…）已按标注时的基准日写成绝对日期，
# 原样渲染会让模型照抄旧日期。渲染时按示例 query 里的相对词，把旧日期换成「当前时间」下的日期。
# 绝对日期的示例（「9月14日」「12月8日」）不含相对词，保持原样。
# ---------------------------------------------------------------------------
_NOW: contextvars.ContextVar = contextvars.ContextVar("fewshot_now", default=None)
_LIB_BASES = (datetime.date(2026, 9, 11), datetime.date(2026, 8, 24))   # 标注库使用过的基准日
_WD = "一二三四五六日"
_REL_DAY = {"前天": -2, "昨天": -1, "昨日": -1, "今天": 0, "今日": 0, "今晚": 0, "明天": 1, "明日": 1, "后天": 2}
_RE_WEEKDAY = re.compile(r"((?<!一)下|上|本|这)?(?:周|星期|礼拜)([一二三四五六日天])")
_RE_WEEK = re.compile(r"((?<!一)下|上|本|这)(?:周|星期|礼拜)(?![一二三四五六日天末])")
_RE_DATE8 = re.compile(r"(?<!\d)(20\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01]))(?!\d)")


def _rel_dates(query: str, today: datetime.date) -> list[datetime.date] | None:
    """示例 query 里的相对时间在 today 下对应的日期（单日 [d]，整周 [周一, 周日]）；无相对词返回 None。"""
    for w, off in _REL_DAY.items():
        if w in query:
            return [today + datetime.timedelta(days=off)]
    mon = today - datetime.timedelta(days=today.weekday())
    m = _RE_WEEKDAY.search(query)
    if m:
        wk = {"上": -7, "下": 7}.get(m.group(1) or "", 0)
        i = _WD.index("日" if m.group(2) == "天" else m.group(2))
        return [mon + datetime.timedelta(days=wk + i)]
    m = _RE_WEEK.search(query)
    if m:
        a = mon + datetime.timedelta(days={"上": -7, "下": 7}.get(m.group(1), 0))
        return [a, a + datetime.timedelta(days=6)]
    return None


def _rebase_dates(text: str, now: datetime.date | None) -> str:
    """把示例（「用户：q\n计划 JSON」）里按标注基准日写的相对日期换成 now 下的日期。"""
    if now is None or "\n" not in text:
        return text
    head, _, plan = text.partition("\n")
    q = head[3:] if head.startswith("用户：") else head
    new = _rel_dates(q, now)
    if new is None:
        return text
    found = set(_RE_DATE8.findall(plan))
    if not found:
        return text
    for base in _LIB_BASES:
        old = _rel_dates(q, base)
        if old is None or len(old) != len(new):
            continue
        mp = {o.strftime("%Y%m%d"): n.strftime("%Y%m%d") for o, n in zip(old, new)}
        if any(k in found for k in mp):
            return head + "\n" + _RE_DATE8.sub(lambda m: mp.get(m.group(1), m.group(1)), plan)
    return text


def _render(text: str) -> str:
    return _rebase_dates(text, _NOW.get())


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
    hits = _search_merged(idx, query, s.fewshot_topk, s.fewshot_min_score, s.fewshot_exclude_self)
    if not hits:
        return ""
    return ("【相似请求的标注示例（按关键词检索，供参考工具选择与参数写法；以本轮请求为准，不要照抄取值）】\n"
            + "\n\n".join(_render(idx.docs[i]["text"]) for i in hits))


# ---------------------------------------------------------------------------
# 多意图（串/并行）：实验见 experiments/retrieval/multi_context.py（arm multi_clause）
# ---------------------------------------------------------------------------
# 连接词切分（只在明确的串/并连接词处切；「2020年之后」「一起看」「还有吗」「再见」等不切）
_CONNECT = re.compile(r"(?:并且|同时|一起(?!看|观看|主持|唱|在|玩|学)|顺便|顺带|之后就|(?<![\d年月日号点分秒天周])之后|然后|随后|"
                      r"(?<!意甲|英超|中超|西甲|德甲|法甲)接下来(?!的)|接着|再帮我|再把|再给我|(?<![不一音量声音])再(?!见|次|来一|看一|听一|放一|推荐|选|大|小|高|低))")
# 无连接词时的纯标点切分：仅当至少两段都含动作/设备词
_ACT = re.compile(r"播放|放|打开|关闭|关掉|关|调|设置|切换|换|看|听|搜索|找|推荐|开|启动|进入|来|声音|音量|亮度|"
                  r"全屏|小屏|静音|暂停|快进|快退|定时|屏保|壁纸|模式")
_MRULE = """【多意图拆单提示】本句含多个独立动作：每个动作各输出一条普通条目（各自 d/vt/params），
条目顺序即执行顺序，与用户说出的顺序一致；「然后/接着/随后/之后/再」等只表示先后执行，不是指代依赖，不要写成依赖条目。
设备操作（音量、亮度、全屏、关机、换一个/下一首、定时等）单独成条，不要并进内容检索条目。"""


def split_clauses(q: str) -> list[str]:
    """按连接词切子句；无连接词时仅在「≥2 段都含动作/设备词」时按标点切。返回长度 ≥2 的子句列表。"""
    strip = " ，,。！!～~；;"
    parts = [p.strip(strip) for p in _CONNECT.split(q or "")]
    parts = [re.sub(r"^(帮我|请|给我|麻烦)", "", p) for p in parts if len(p) >= 2]
    if len(parts) >= 2:
        return parts
    ps = [p.strip() for p in re.split(r"[，,；;。！!～~]+", q or "") if len(p.strip()) >= 2]
    if len(ps) >= 2 and sum(bool(_ACT.search(p)) for p in ps) >= 2:
        return ps
    return [q] if q else []


def virtualize_plan(plan: str) -> str | None:
    """多步计划 JSON 里的设备步参数改成虚拟写法；任一设备步无法表示或 JSON 非法返回 None。"""
    try:
        steps = json.loads(plan)
    except json.JSONDecodeError:
        return None
    if not isinstance(steps, list):
        return None
    for s in steps:
        vt = str(s.get("vt") or "")
        if vt.startswith("device:") and isinstance(s.get("params"), dict):
            p = device_schema.virtualize_params(vt.partition(":")[2], s["params"])
            if p is None:
                return None
            s["params"] = p
    return json.dumps(steps, ensure_ascii=False, separators=(",", ":"))


class _MultiIndex(_Index):
    """多意图库：CSV 列 query, plan（多步计划 JSON），渲染为「用户：… / 计划」示例。"""

    def __init__(self, path: Path) -> None:  # noqa: D401  (复用 _Index 的检索，只换载入)
        import jieba
        jieba.setLogLevel(60)
        self.docs = []
        virtual = get_settings().device_virtual
        for r in csv.DictReader(open(path, encoding="utf-8-sig")):
            q, plan = (r.get("query") or "").strip(), (r.get("plan") or "").strip()
            if q and plan and virtual:
                plan = virtualize_plan(plan)      # 含无法表示的设备步 → None，整条跳过
            if q and plan:
                self.docs.append({"q": q, "qn": _nq(q), "text": f"用户：{q}\n{plan}"})
        self.tf = [collections.Counter(_tok(d["q"])) for d in self.docs]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = sum(self.len) / max(1, len(self.len))
        df: collections.Counter = collections.Counter()
        self.post = collections.defaultdict(list)
        for i, t in enumerate(self.tf):
            for w in t:
                df[w] += 1
                self.post[w].append(i)
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}


_MIDX: _MultiIndex | None = None
_MIDX_PATH = ""


def _multi_index() -> _MultiIndex | None:
    global _MIDX, _MIDX_PATH
    lib = (get_settings().fewshot_multi_lib or "").strip()
    if not lib:
        return None
    path = Path(lib) if Path(lib).is_absolute() else _BASE / lib
    with _LOCK:
        if _MIDX is None or _MIDX_PATH != str(path):
            try:
                _MIDX = _MultiIndex(path)
                _MIDX_PATH = str(path)
                logger.info("多意图 few-shot 库载入：%s（%d 条）", path, len(_MIDX.docs))
            except (OSError, csv.Error) as exc:
                logger.warning("多意图 few-shot 库载入失败（降级为单意图口径）：%s", exc)
                return None
    return _MIDX


def multi_context_block(query: str) -> str:
    """≥2 子句且配置了多意图库时：拆单提示 + 多意图示例 + 各子句单意图示例/规则；否则空串。"""
    clauses = split_clauses(query)
    midx = _multi_index()
    if midx is None:
        return ""
    s = get_settings()
    if len(clauses) < 2:
        # 规则没切出子句时，按检索分数判断：多意图库最高分 / 单意图库最高分 ≥ 阈值 → 仍注入多意图示例
        if s.fewshot_multi_trigger != "score":
            return ""
        idx = _index()
        sm, ss = midx.top_score(query), (idx.top_score(query) if idx is not None else 0.0)
        if sm <= 0 or sm < s.fewshot_multi_ratio * max(ss, 1e-6):
            return ""
        clauses = [query]
    parts = [_MRULE]
    eb = entity_block(query)
    if eb:
        parts.append(eb)
    for c in [query] + clauses:                        # 整句 + 子句级规则提示（去重）
        r = rules_block(c)
        if r and r not in parts:
            parts.append(r)
    hits = midx.search(query, s.fewshot_multi_topk, s.fewshot_min_score, s.fewshot_exclude_self)
    if hits:
        parts.append("【相似多意图请求的标注示例（按关键词检索；参考拆分方式、顺序、工具与参数写法，以本轮请求为准）】\n"
                     + "\n\n".join(_render(midx.docs[i]["text"]) for i in hits))
    idx = _index()
    if idx is not None:
        seen, ex = set(), []
        for c in clauses:
            for i in _search_merged(idx, c, s.fewshot_clause_topk, 3.0, True):
                if i not in seen:
                    seen.add(i)
                    ex.append(_render(idx.docs[i]["text"]))
        if ex:
            parts.append("【各子句相似请求的单意图标注示例（供参考每个子句的工具与参数写法）】\n" + "\n\n".join(ex))
    return "\n" + "\n".join(parts)


def warmup() -> None:
    """开启时在启动期预建索引，避免首个请求等待（约数秒）。"""
    if get_settings().fewshot_enabled:
        _index()
        _multi_index()
        _entity()
        _rule_lib()


def context_block(query: str) -> str:
    """规则 + 示例；关闭或失败返回空串。"""
    if not get_settings().fewshot_enabled:
        return ""
    try:
        multi = multi_context_block(query)
        if multi:
            return multi
        parts = [b for b in (rules_block(query), entity_block(query), examples_block(query)) if b]
    except Exception as exc:  # noqa: BLE001
        logger.warning("few-shot context 生成失败（降级）：%s", exc)
        return ""
    return ("\n" + "\n".join(parts)) if parts else ""


def system_prompt(base_prompt: str, query: str, now: datetime.datetime | None = None) -> str:
    """首轮 system：开启时修正静态冲突 + 末尾追加动态 context；关闭时原样返回。
    now：当前时间（北京时间），用于把示例里的相对日期换算到当前日期；缺省不换算。"""
    if not get_settings().fewshot_enabled:
        return base_prompt
    tok = _NOW.set(now.date() if now else None)
    try:
        return fix_static(base_prompt) + context_block(query)
    finally:
        _NOW.reset(tok)
