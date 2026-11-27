"""多意图拆分规则后处理（不训练模型）：对引擎首轮产出的计划 [{q,d,tool,params}] 做确定性修正。

与 app/ 完全解耦：只吃/吐 steps 列表。口径来自 3domain 评测集的修订版标注：
  1 音乐/有声 + 设备（循环/随机/音量/屏幕/定时/开关机）两步组合；最多保留 2 步
  2 关机（休眠）类步骤在组合里不保留；返回主页/打开 X 页面这类导航步不保留
  3 改口（换成/不对/算了还是…）只保留最后一次
  4 儿歌/评书/故事等落有声 audio_search；泛指「音乐/歌曲」落 music_song_recommend
  5 「第N首」类按序号选择落 playback_control，不再重复搜索
"""
from __future__ import annotations

import re

CN_NUM = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
MEDIA = {"music_song_search", "music_song_recommend", "music_ksong_search", "educ_search_all", "audio_search",
         "music_song_qqmusic_search", "music_song_mv_search", "music_song_favorite_search", "music_song_history",
         "audio_history", "vod_search_all", "fan_knowledge_agent"}
SEARCHY = {"music_song_search", "music_ksong_search", "educ_search_all", "music_song_recommend", "vod_search_all"}
KIDS = re.compile(r"儿歌|童谣|小鸡|猪猪侠|鲨鱼一家|jojo|JOJO|评书|相声|故事|有声|喜马拉雅")
GENERIC_SONG = {"音乐", "歌曲", "歌", "轻音乐", "好听的音乐", "好听的歌", "一首音乐", "一首歌", "其它音乐", "其他音乐", "歌声", "曲", "好听的"}
MODE_RE = re.compile(r"单曲循环|循环播放|随机播放|顺序播放|重新播放|列表循环")
VOL_RE = re.compile(r"声音|音量|大点声|小点声|大声|小声")
SCREEN_OFF = re.compile(r"熄屏|息屏|关屏|黑屏|闭屏|关闭屏幕|关掉画面|关闭画面|关画面")
ORD_RE = re.compile(r"第([一二三四五六七八九十\d]+)(首|个|部|条|集|曲)|最后一(首|个)")
CORRECT_RE = re.compile(r"换成|改成|不对|算了|还是|不要了|不是|改为")
WAKE_RE = re.compile(r"^(你好哈利|海信小聚|你好小聚|小聚小聚|哈利哈利|你好|哈利)[，, ]*$")
NAV_RE = re.compile(r"^(把)?(返回|回到|回|进入|打开|切换到|去)(.*)(主页|首页|专区|历史|记录|收藏|频道)")

_DEV = {"date_time": "", "device": "", "location": ""}


def _num(s: str):
    s = s.strip()
    if s.isdigit():
        return int(s)
    if s and all(c in CN_NUM for c in s):
        if s == "十":
            return 10
        if len(s) == 1:
            return CN_NUM[s]
        if s[0] == "十":
            return 10 + CN_NUM[s[1]]
        if s[-1] == "十":
            return CN_NUM[s[0]] * 10
        if len(s) == 3 and s[1] == "十":
            return CN_NUM[s[0]] * 10 + CN_NUM[s[2]]
    return None


def _st(tool, params, q="", d=""):
    return {"q": q, "d": d, "tool": tool, "params": params}


def _screen_off(q=""):
    return _st("screensaver_control", {"operation": "关闭", "object": "屏幕"}, q, "device")


def _fullscreen(q=""):
    return _st("screen_layout", {"operation": "打开", "object": "全屏"}, q, "device")


def _power_on(q=""):
    return _st("power_control", {"operation": "打开", "object": "开机"}, q, "device")


def _recommend(q=""):
    return _st("music_song_recommend", {}, q, "music")


def _audio(q=""):
    return _st("audio_search", {"action": "play", "query": q}, q, "audio")


def _playback(query):
    m = MODE_RE.search(query)
    if not m:
        return None
    w = m.group(0)
    if w == "重新播放":
        p = {"operation": "重播", "object": "播放控制"}
    elif w == "列表循环":
        p = {"operation": "循环播放", "object": "播放列表"}
    else:
        p = {"operation": w, "object": "播放列表"}
    return _st("playback_control", p, w, "device")


def _volume(query):
    """按 query 里的音量说法生成 numeric_adjust 步。"""
    seg = query
    m = re.search(r"(调到|开到|放到|关到|到|加到|加)(\d+|[零一二两三四五六七八九十]+)(?![点些下])", query)
    val = _num(m.group(2)) if m else None
    if re.search(r"加\s*(\d+|[零一二两三四五六七八九十]+)", seg) and val is not None:
        op = "提高"
    elif re.search(r"关到", seg) and val is not None:
        op = "降低"
    elif val is not None:
        op = "设置"
    elif re.search(r"大|高|升|加|响", seg.split("声音")[-1] if "声音" in seg else seg):
        op = "提高"
    elif re.search(r"小|低|降|减|轻", seg.split("声音")[-1] if "声音" in seg else seg):
        op = "降低"
    else:
        op = "提高"
    p = {"operation": op, "object": "音量"}
    if val is not None:
        p["value"] = str(val)
    return _st("numeric_adjust", p, "音量", "device")


def _is_power_off(s):
    p = s.get("params") or {}
    txt = f"{p.get('operation', '')}{p.get('object', '')}{s.get('q', '')}"
    return s["tool"] == "power_control" and ("关" in txt) and "重启" not in txt and "开机" not in p.get("object", "开机x")


def _generic_search(s):
    """搜索类步骤是否没有具体实体（只有泛指音乐词）。"""
    p = s.get("params") or {}
    vals = []
    for k, v in p.items():
        if k in ("retext", "action", "query", "sort"):
            continue
        vals += v if isinstance(v, list) else [v]
    vals = [str(v) for v in vals if v not in ("", None)]
    return all(v in GENERIC_SONG or v == "好听" or re.fullmatch(r"(声音|音量)?到?\d*", v)
               or re.fullmatch(r"(我要|我想|想)?听?(一?首)?(好听的)?(音乐|歌曲|歌)", v) for v in vals)


def _hallucinated_generic(s):
    """song 值没出现在本步 q 里，而 q 以泛指词结尾（「息屏播放音乐」被填成 song=海信小聚）。"""
    p = s.get("params") or {}
    vals = [str(v) for k in ("song", "singer", "tag") for v in (p.get(k) or []) if v]
    return bool(vals) and not any(v in s.get("q", "") for v in vals) and bool(re.search(r"(音乐|歌曲|歌)$", s.get("q", "")))


_VERSION_FIX = {"dj": "dj版", "DJ": "dj版", "dj版": "dj版", "DJ版": "dj版", "Dj版": "dj版", "remix": "remix版", "Remix版": "remix版",
                "live": "live版", "Live版": "live版", "LIVE版": "live版", "免费": "免费版"}


def normalize_params(tool: str, params):
    """枚举/重复值归一（不改结构）：music_* 的 version 大小写与 tag→version（dj/免费）、数组去重。"""
    if not isinstance(params, dict) or not tool.startswith("music_"):
        return params
    p = dict(params)
    for k, v in list(p.items()):
        if isinstance(v, list):
            seen = []
            for x in v:
                if x not in seen:
                    seen.append(x)
            p[k] = seen
    if tool == "music_song_search":
        ver = [_VERSION_FIX.get(x, x) for x in (p.get("version") or [])]
        tags = []
        for t in (p.get("tag") or []):
            if _VERSION_FIX.get(t) in ("dj版", "免费版"):
                ver.append(_VERSION_FIX[t])
            else:
                tags.append(t)
        if ver:
            p["version"] = list(dict.fromkeys(ver))
        if "tag" in p:
            if tags:
                p["tag"] = tags
            else:
                p.pop("tag")
    return p


def _educ_title(params) -> str:
    """educ_search_all 的 query 树里取 title 叶子的值（educ→music 归一时当歌名）。"""
    def walk(n):
        if isinstance(n, dict):
            if n.get("field") == "title" and n.get("value"):
                return str(n["value"])
            for v in n.values():
                r = walk(v)
                if r:
                    return r
        elif isinstance(n, list):
            for v in n:
                r = walk(v)
                if r:
                    return r
        return ""
    return walk(params or {})


def _trim(query: str, steps: list[dict]) -> list[dict]:
    """模型已拆出 ≥3 步时，剔除不属于「内容 + 设备」口径的多余步：关机步、页面跳转步、被改口替换的前一步。
    只删不增、不改写、不重排；不足 3 步原样返回。"""
    if len(steps) < 3:
        return steps
    steps = [s for s in steps if not (_is_power_off(s) and len(steps) > 1)]
    steps = [s for s in steps if not (s["tool"] == "fan_knowledge_agent" and WAKE_RE.search(s.get("q", "")) and len(steps) > 2)]
    if len(steps) > 2:
        steps = [s for s in steps if s["tool"] != "common_control"]
    if len(steps) > 2 and any(s["tool"] in ("music_song_search", "music_song_recommend", "audio_search") for s in steps):
        steps = [s for s in steps if s["tool"] != "vod_search_all"]
    while len(steps) > 2 and steps[0]["tool"] in MEDIA and NAV_RE.search(steps[0].get("q", "")):
        steps = steps[1:]
    if len(steps) > 2 and steps[0]["tool"] in ("music_song_history", "audio_history"):
        steps = steps[1:]
    if CORRECT_RE.search(query):
        def grp(t):
            return "media" if t in SEARCHY or t == "audio_search" else t
        out: list[dict] = []
        for s in steps:
            if out and grp(out[-1]["tool"]) == grp(s["tool"]) and grp(s["tool"]) in ("media", "numeric_adjust"):
                out[-1] = s
            else:
                out.append(s)
        steps = out
    return steps


def patch(query: str, steps: list[dict]) -> list[dict]:
    """服务端用（HC_MULTI_RULES）：只修补模型已给出的计划。
    · 每步的 d/tool/params 逐步修补；
    · 模型已拆出 ≥3 步时，只【删除】多余步（关机/页面跳转/被改口替换的步）；
    不拆、不增、不重排步骤，也不改各步 q —— 拆句与步骤结构由模型决定。"""
    # 前提：模型已拆出 ≥2 个意图。单意图计划一律原样返回，不做任何修改。
    if len(steps) < 2:
        return [dict(s) for s in steps]
    out = postprocess(query, steps, patch_only=True)
    if len(out) != len(steps):
        return [dict(s) for s in steps]
    res = []
    for a, b in zip(steps, out):
        if not b.get("tool") and a.get("tool"):
            b = dict(a)
        res.append({**a, "d": b.get("d", a.get("d")), "tool": b.get("tool", a.get("tool")),
                    "params": b.get("params", a.get("params"))})
    return _trim(query, res)


def postprocess(query: str, steps: list[dict], patch_only: bool = False) -> list[dict]:
    """离线实验用的完整规则（含增删/重排步骤）；patch_only=True 时只做第 1 段逐步修补。"""
    steps = [dict(s) for s in steps]
    q = query
    kids = bool(KIDS.search(q))
    has_ord = ORD_RE.search(q)

    # ---- 1 工具归一 ----
    for s in steps:
        t = s["tool"]
        if t in ("educ_search_all", "music_song_search", "music_song_recommend", "music_ksong_search") and kids \
                and not re.search(r"QQ|qq|酷狗|酷我", q):
            s.update(tool="audio_search", d="audio", params={"action": "play", "query": s.get("q") or q})
        elif t == "audio_history" and not kids:
            s.update(tool="music_song_history", d="music", params={"retext": s.get("q") or q})
        elif t == "educ_search_all":
            title = _educ_title(s.get("params"))
            s.update(tool="music_song_search", d="music", params={"song": [title]} if title else {})
        elif t == "music_ksong_search" and not re.search(r"K歌|k歌|卡拉|点歌|伴奏|麦克风", q):
            s.update(tool="music_song_search", d="music")
        if s["tool"] == "music_song_search" and (_generic_search(s) or _hallucinated_generic(s)):
            s.update(tool="music_song_recommend", params={})
        if s["tool"] == "mode_control" and SCREEN_OFF.search(q):
            s.update(_screen_off(s.get("q", "")))
        if s["tool"] == "audio_control" and SCREEN_OFF.search(q):
            s.update(_screen_off(s.get("q", "")))

    for s in steps:
        t, p = s["tool"], dict(s.get("params") or {})
        if t == "source_switch" and re.search(r"QQ|qq", s.get("q", "") + q):
            s.update(tool="music_song_qqmusic_search", d="music", params={"retext": s.get("q") or q})
        elif t == "music_song_qqmusic_search" and re.search(r"酷狗|酷我", q) and not re.search(r"QQ|qq", q):
            kw = p.get("keywords") or []
            s.update(tool="music_song_search", d="music", params={"song": kw} if kw else {})
        elif t == "vod_search_all" and MODE_RE.search(q) and not re.search(r"电影|电视剧|综艺|动漫|纪录片", q):
            m = re.search(r"《?(.+?)》?$", s.get("q", "").replace("播放", ""))
            s.update(tool="music_song_search", d="music", params={"song": [m.group(1)]} if m else {})
        elif t == "fan_knowledge_agent" and re.search(r"播放|听|放", s.get("q", "") or q) and re.search(r"音乐|歌", q):
            s.update(_audio(q) if kids else _recommend(q))
        elif t == "music_song_history" and re.search(r"播放.+", q) and not has_ord:
            m = re.search(r"播放(.+?)(?:[，,]|声音|音量|$)", q)
            if m and not patch_only:
                c = m.group(1)
                mm = re.match(r"(.+?)的(.+)", c)
                params = {"singer": [mm.group(1)], "song": [mm.group(2)]} if mm else {"song": [c]}
                steps.insert(steps.index(s) + 1, _st("music_song_search", params, "播放" + c, "music"))
        elif t == "playback_control" and "循环播放" in q and "单曲循环" not in q:
            p["operation"] = "循环播放"; s["params"] = p
        elif t == "power_control" and "重启" in q:
            s["params"] = {"operation": "关闭", "object": "重启"}
        elif t == "screen_layout" and "屏幕" in q:
            p["object"] = "屏幕"; s["params"] = p
        elif t == "timer_control" and re.search(r"还有多久|多长时间|剩余", q):
            s["params"] = {"operation": "查询", "object": "剩余关机时长"}
        elif t == "numeric_adjust" and p.get("operation") == "设置" and re.search(r"[声音量][^，,]*加到\d|加\s*\d|加到[零一二两三四五六七八九十]", s.get("q", "") or q):
            p["operation"] = "提高"; s["params"] = p
        elif t == "numeric_adjust" and p.get("operation") == "设置" and re.search(r"关到", q):
            p["operation"] = "降低"; s["params"] = p
        elif t == "playback_control" and p.get("operation") == "下" and ORD_RE.search(s.get("q", "") + q):
            s["params"] = {"operation": "下", "object": "播放列表"}
        elif t == "numeric_adjust" and p.get("object") == "声音":
            p["object"] = "音量"; s["params"] = p
        elif t == "numeric_adjust" and p.get("operation") == "设置" and not p.get("value") and p.get("object") == "音量":
            p["operation"] = "提高"; s["params"] = p
        elif t == "screensaver_control" and re.search(r"定时|倒计时|分钟后|小时后", q):
            s["tool"] = "timer_control"; s["params"] = {"operation": "打开", "object": "熄屏"}
        elif t == "source_switch" and re.search(r"U盘|u盘|USB|usb", s.get("q", "") + q):
            p["value"] = "usb"
            if p.get("operation") == "设置":
                p["operation"] = "打开"
            s["params"] = p
        if t in ("music_song_search",) and isinstance(p.get("tag"), list):
            tg = [x for x in p["tag"] if not MODE_RE.search(str(x))]
            if tg != p["tag"]:
                p = dict(p); p.pop("tag") if not tg else p.update(tag=tg); s["params"] = p
    if patch_only:
        return steps
    if any(s["tool"] == "screensaver_control" for s in steps) and "全屏" not in q and "放大" not in q:
        steps = [s for s in steps if s["tool"] != "screen_layout"]

    # ---- 2 剔除：关机 / 导航 / 无关兜底 ----
    steps = [s for s in steps if not (_is_power_off(s) and len(steps) > 1)]
    steps = [s for s in steps if not (s["tool"] == "fan_knowledge_agent" and WAKE_RE.search(s.get("q", "")) and len(steps) > 2)]
    if len(steps) > 2:
        steps = [s for s in steps if s["tool"] != "common_control"]
    if len(steps) > 2 and any(s["tool"] in ("music_song_search", "music_song_recommend", "audio_search") for s in steps):
        steps = [s for s in steps if s["tool"] != "vod_search_all"]
    while len(steps) > 2 and steps[0]["tool"] in MEDIA and NAV_RE.search(steps[0].get("q", "")):
        steps = steps[1:]
    if len(steps) > 2 and steps[0]["tool"] in ("music_song_history", "audio_history"):
        steps = steps[1:]

    # ---- 3 改口：保留最后一次 ----
    if CORRECT_RE.search(q):
        def grp(t):
            return "media" if t in SEARCHY or t == "audio_search" else t
        out: list[dict] = []
        for s in steps:
            if out and grp(out[-1]["tool"]) == grp(s["tool"]) and grp(s["tool"]) in ("media", "numeric_adjust"):
                out[-1] = s
            else:
                out.append(s)
        steps = out

    # ---- 4 序号选择（第N首）----
    if has_ord:
        n = _num(has_ord.group(1)) if has_ord.group(1) else None
        if n == 1:
            pp = {"operation": "播放", "object": "播放列表", "value": "1"}
        elif re.search(r"从第|专辑", q):
            pp = {"operation": "跳转", "object": "播放控制", **({"value": str(n)} if n else {})}
        else:
            pp = {"operation": "下", "object": "播放列表"}
        pb = _st("playback_control", pp, has_ord.group(0), "device")
        keep = [s for s in steps if s["tool"] != "playback_control"]
        # 序号步（q 含序号的搜索步）剔除，由 playback_control 取代
        keep = [s for s in keep if not (s["tool"] in MEDIA and ORD_RE.search(s.get("q", "")) and len(keep) > 1)]
        if len(keep) > 1:
            keep = keep[:1]
        for s in keep:
            if s["tool"] == "fan_knowledge_agent":
                m = re.match(r"(.+?)专辑", q)
                s.update(tool="music_song_search", d="music", params={"album": [m.group(1)]} if m else {})
        if not keep:
            keep = [_recommend(q)]
        steps = keep + [pb]

    # ---- 5 缺失意图补全 ----
    tools = [s["tool"] for s in steps]
    has_media = any(t in MEDIA for t in tools)
    if MODE_RE.search(q) and "playback_control" not in tools:
        pb = _playback(q)
        # 「X单曲循环唱」之类：ksong 归一已处理
        steps.append(pb)
    if VOL_RE.search(q) and "numeric_adjust" not in tools and not re.search(r"亮度", q):
        steps.append(_volume(q))
    if SCREEN_OFF.search(q) and "screensaver_control" not in tools and "timer_control" not in tools:
        steps.append(_screen_off(q))
    if "全屏" in q and "screen_layout" not in tools:
        steps.append(_fullscreen(q))
    if re.search(r"开机|重启", q) and "power_control" not in tools and not re.search(r"关机|定时|还有", q):
        p = _power_on(q)
        if "重启" in q:
            p["params"] = {"operation": "关闭", "object": "重启"}
        steps.insert(0, p)
    tools = [s["tool"] for s in steps]
    has_media = any(t in MEDIA for t in tools)
    if not has_media and len(steps) <= 2:
        if kids:
            m = KIDS.search(q)
            steps.append(_audio(m.group(0) if m else q))
        elif re.search(r"音乐|歌|曲", q):
            steps.append(_recommend(q))
        else:
            # 「熄屏遇君」「小美满单曲循环」：去掉设备提示词后剩下的就是内容名
            rest = re.sub(SCREEN_OFF.pattern + "|" + MODE_RE.pattern + r"|声音.*|音量.*|全屏|播放|开机|重启", "", q).strip("，。, ")
            if rest and len(rest) <= 10:
                steps.append(_st("music_song_search", {"song": [rest]}, rest, "music"))

    # ---- 6 步数上限 + 次序 ----
    if len(steps) > 2:
        # 去重同类，再按优先级保留 媒资 + 设备各一
        seen, out = set(), []
        for s in steps:
            if s["tool"] in seen:
                continue
            seen.add(s["tool"]); out.append(s)
        steps = out
    steps = _order(q, steps)
    return steps


_CUES = {
    "power_control": r"开机|重启|启动",
    "screensaver_control": SCREEN_OFF.pattern,
    "screen_layout": r"全屏",
    "numeric_adjust": r"声音|音量|大点声|小点声|大声|小声",
    "playback_control": MODE_RE.pattern + "|" + ORD_RE.pattern,
    "timer_control": r"定时|还有",
}


def _order(q, steps):
    """按用户说话顺序排：设备步取其提示词位置，媒资步取第一个媒资词位置（开机/重启恒在最前）。"""
    if len(steps) < 2:
        return steps
    if steps[0]["tool"] == "power_control":
        return steps
    pos = []
    for s in steps:
        t = s["tool"]
        if t == "playback_control":
            pos.append(10 ** 6)
        elif t in _CUES:
            m = re.search(_CUES[t], q)
            pos.append(m.start() if m else None)
        else:
            m = re.search(r"播放|放|听|唱|音乐|歌|儿歌|相声|评书|故事", q)
            pos.append(m.start() if m else None)
    if any(p is None for p in pos):
        return steps
    # 媒资步的提示词会与「全屏播放」中的「播放」同位，用稳定排序，设备步优先于同位媒资步
    idx = sorted(range(len(steps)), key=lambda i: (pos[i], steps[i]["tool"] in MEDIA))
    return [steps[i] for i in idx]
