"""单意图后处理规则（HC_SINGLE_RULES=1）：LLM 已给出 d/tool/params 之后，按稳定的口径约定做确定性修补。

边界：只修补模型已产出的单条计划（改 tool/params/domain），不替代 LLM 落域与选工具；
多意图计划由 multi_rules 负责，这里不动。任何规则只在命中明确文字特征时才改，未命中原样返回。
patch(query, domain, tool, params, now) → (domain, tool, params)
"""
from __future__ import annotations

import datetime
import re
from typing import Any

_DOM_OF_PREFIX = (("music_", "music"), ("vod_", "vod"), ("educ_", "educ"), ("edu_", "edu"),
                  ("audio_", "audio"), ("sports_", "sports"), ("tvchannel_", "music"))


def _dom(tool: str) -> str:
    for p, d in _DOM_OF_PREFIX:
        if tool.startswith(p):
            return d
    return ""


# ---------------- vod ----------------
_PLAY_VERB = re.compile(r"播放|播一|放一|放个|我要看|我想看|想看|要看|开始播|来播|打开|点播|回放|继续看|看一下|看下")
_HOT_WORDS = re.compile(r"热度|最热|很火|热门|火的|人气")
_PERSONAL = re.compile(r"我的|我看过|喜好|推荐|个性|猜你|按我|根据我")


def _vod_params_has_title(params: dict) -> bool:
    def walk(o: Any) -> bool:
        if isinstance(o, dict):
            if o.get("field") == "title":
                return True
            return any(walk(v) for v in o.values())
        if isinstance(o, list):
            return any(walk(x) for x in o)
        return False
    return walk(params.get("query"))


def _vod(query: str, tool: str, params: dict) -> tuple[str, dict]:
    if tool == "vod_search_all" and params.get("action") == "play":
        # 点播须同时满足：有播放类动词，且（指名了片名 或 句子以「播放」起头）；否则只是检索
        has_verb = bool(_PLAY_VERB.search(query))
        has_title = _vod_params_has_title(params)
        if not has_verb or not (has_title or query.startswith("播放")):
            params = {**params, "action": "search"}
    if tool == "vod_search_all" and "影片" in query and isinstance(params.get("query"), dict) \
            and "category" not in str(params["query"]):
        q = params["query"]
        params = {**params, "query": {"and": [*(q["and"] if isinstance(q.get("and"), list) else [q]),
                                              {"field": "category", "value": "电影"}]}}
    if tool == "vod_personalized_search" and _HOT_WORDS.search(query) and not _PERSONAL.search(query):
        cat = params.get("category")
        if cat:
            tool = "vod_search_all"
            params = {"action": "search", "query": {"field": "category", "value": cat},
                      "sort": {"hot": {"order": "desc"}}}
    return tool, params


# ---------------- educ ----------------
_EDUC_HIST = re.compile(r"(刚刚|刚才|上次|上一次|之前|以前|前几天|最近几天|最近|历史)(看过?|播放|播过|放过?|观看)|历史记录|播放历史|看过的"
                        r"|(前天|昨天)(看|播|放)过?的|刚刚的")
_EDUC_SIM = re.compile(r"(跟|和|同|像).{1,16}(相似|类似|同类|一样|同一类型|雷同)")
_CN_NUM = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _day_range(now: datetime.datetime, days_ago: int) -> str:
    d = (now - datetime.timedelta(days=days_ago)).date()
    return f"{d} 00:00:00 TO {d} 23:59:59"


def _educ(query: str, tool: str, params: dict, now: datetime.datetime) -> tuple[str, dict]:
    if tool in ("educ_search_all", "educ_fuzzy_search") and _EDUC_HIST.search(query):
        p: dict = {}
        if "前天" in query:
            p["time"] = _day_range(now, 2)
        elif "昨天" in query:
            p["time"] = _day_range(now, 1)
        return "educ_history", p
    if tool == "educ_search_all" and _EDUC_SIM.search(query):
        q = params.get("query")
        if isinstance(q, dict) and "title" in str(q):
            tool = "educ_relate_search"
    if tool == "educ_search_all":
        params = _educ_age(query, params)
    if tool == "educ_relate_search" and re.search(r"相似|类似|雷同|一样|同类", query):
        q = params.get("query")
        for node in ((q.get("and") or [q]) if isinstance(q, dict) else []):
            v = node.get("value") if isinstance(node, dict) else None
            if node.get("field") == "title" and isinstance(v, str) and len(v) > 2 \
                    and v[0] in "同跟和" and query.startswith(v):
                node["value"] = v[1:]
    return tool, params


def _educ_age(query: str, params: dict) -> dict:
    """单个明确年龄「N岁」→ age_range from=to=N；只说「宝宝」没有年龄数字 → 不凭空加 age_range。"""
    q = params.get("query")
    if not isinstance(q, dict):
        return params
    nodes = q.get("and") if isinstance(q.get("and"), list) else [q]
    m = re.search(r"(\d+|[一二两三四五六七八九十])\s*岁", query)
    rng = bool(re.search(r"到|至|以下|以上|以内|-|~", query))
    for n in list(nodes):
        if not (isinstance(n, dict) and n.get("field") == "age_range"):
            continue
        if m and not rng:
            v = m.group(1)
            v = str(_CN_NUM.get(v, v))
            n["from"], n["to"] = v, v
        elif not m and "宝宝" in query and "岁" not in query:
            if nodes is q.get("and"):
                nodes.remove(n)
                if len(nodes) == 1:
                    params = {**params, "query": nodes[0]}
    return params


# ---------------- music ----------------
_TOPLIST = re.compile(r"([\u4e00-\u9fa5A-Za-z0-9]{1,14}榜)")
_SING = re.compile(r"^(?:我想|我要|想要|帮我|给我)?(?:唱|演唱|点唱|K歌)")


def _music(query: str, tool: str, params: dict, domain: str = "") -> tuple[str, dict]:
    if tool == "vod_history" and (domain == "music" or re.search(r"听|歌|曲|音乐", query)):
        tool = "music_song_history"
    if tool == "music_song_history":
        return tool, {k: v for k, v in params.items() if k == "retext"}
    if tool == "music_song_search" and "收藏" in query and not re.search(r"收藏的?(歌手|专辑)", query):
        return "music_song_favorite_search", {k: v for k, v in params.items() if k == "retext"}
    if tool == "music_song_search":
        if _SING.search(query):
            tool = "music_ksong_search"
        else:
            m = _TOPLIST.search(query)
            if m and "tag" in params and not params.get("toplist") and "榜单" not in query:
                p = {k: v for k, v in params.items() if k != "tag"}
                p["toplist"] = [m.group(1) if not m.group(1).startswith(("播放", "放", "听")) else
                                re.sub(r"^(?:播放|放|听|来一首|我想听)", "", m.group(1))]
                params = p
        if "全歌版" in query and "version" not in params:
            params = {**params, "version": ["全歌版"]}
        tag = params.get("tag")
        if isinstance(tag, list) and tag == ["最新"]:
            params = {**params, "tag": ["新歌"]}
        if re.search(r"最新(的)?(歌|音乐)", query) and not [k for k in params if k not in ("retext", "sort")] \
                and "new" in str(params.get("sort")):
            params = {k: v for k, v in params.items() if k != "sort"}
            params["tag"] = ["新歌"]
        if not [k for k in params if k != "retext"]:
            tool = "music_song_recommend"
    return tool, params


# ---------------- sports ----------------
def _sports(query: str, tool: str, params: dict) -> tuple[str, dict]:
    if tool == "sports_match_search":
        q = params.get("query")
        if isinstance(q, dict) and q.get("field") == "sport_team" and isinstance(q.get("value"), str) \
                and re.sub(r"[\s的]", "", query) == q["value"].replace(" ", ""):
            return "sports_team_search", {"sport_team": q["value"]}
        s = str(q)
        if "亚运会" in s and re.search(r"直播|赛场|节目|决赛|现场|视频", query) and "预约" not in query:
            if re.search(r"赛场|节目|比赛直播|播放.*决赛", query):
                return "sports_vod_search", params
    return tool, params


# ---------------- edu / audio 问答 ----------------
_QA_EDU = re.compile(r"怎么办|怎么才能|怎么帮|该如何|如何|应该|几科|翻译成|哪些类型|从哪|为什么|吗[？?]?$|^总结|^翻译|翻译一下|区别|是什么意思")
_NOT_SEARCH = re.compile(r"播放|我要看|我想看|我想学|我要学|找|推荐|搜索|打开|有没有|帮我|给我|学习|课程|教程|视频|网课")
_QA_AUDIO = re.compile(r"排行榜|哪些主播|哪个主播|主播有哪些|是哪个.*主讲|有没有讲|讲没讲|最犀利|最近更新了哪|是谁|是什么类型|讲了什么|讲的是什么|核心观点|主讲|吗[？?]?$")
_AUDIO_PLAY = re.compile(r"播放|我要听|想听|来一|放一|给我放|打开")


def _qa(query: str, domain: str, tool: str) -> str:
    if domain == "edu" or tool.startswith("edu_"):
        if tool in ("edu_fuzzy_search", "educ_fuzzy_search", "edu_search") and _QA_EDU.search(query) \
                and not re.search(r"^(?:请|帮我|麻烦)?(?:播放|我要|我想|推荐|搜索|打开|给我|学习|找|有没有|看|学)", query):
            return "fan_knowledge_agent"
    if tool in ("audio_search", "vod_fuzzy_search", "edu_fuzzy_search") and _QA_AUDIO.search(query) \
            and not _AUDIO_PLAY.search(query) and (domain in ("audio", "") or tool == "audio_search"):
        return "fan_knowledge_agent"
    return tool


def patch(query: str, domain: str, tool: str, params: dict[str, Any] | None,
          now: datetime.datetime | None = None) -> tuple[str, str, dict[str, Any]]:
    """返回 (domain, tool, params)；未命中规则原样返回。"""
    params = dict(params or {})
    now = now or datetime.datetime.now()
    d = domain or _dom(tool)
    if not tool or tool == "fan_knowledge_agent":
        return domain, tool, params
    if tool == "audio_history" and "上周" in query and isinstance(params.get("time"), str):
        mon = (now - datetime.timedelta(days=now.weekday() + 7)).date()
        params["time"] = f"{mon} 00:00:00 TO {mon + datetime.timedelta(days=6)} 23:59:59"
    new_tool = _qa(query, d, tool)
    if new_tool != tool:
        return domain, new_tool, {}
    if d == "vod" or tool.startswith("vod_"):
        tool, params = _vod(query, tool, params)
    if d == "educ" or tool.startswith("educ_"):
        tool, params = _educ(query, tool, params, now)
    if d == "music" or tool.startswith("music_"):
        tool, params = _music(query, tool, params, d)
    if d == "sports" or tool.startswith("sports_"):
        tool, params = _sports(query, tool, params)
    return domain, tool, params
