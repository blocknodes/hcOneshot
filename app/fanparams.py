"""fan_knowledge_agent（泛知识问答智能体）请求参数构造 —— **逐字节移植自 hcTools/app/fanparams.py**。

hcOneshot 为何需要它：一把出里模型直接产 params，而目录块为省 token 把 fan 的
`messages` 压成 `string`、丢掉了 riskRespStrategy/sceneTypeIntentList —— 模型照目录填
就不合真实契约。故 step 组装时统一经 rebuild_fan_params 规范化。
本文件与 hcTools 保持同口径；改词表请两处同步（hcTools 是权威）。

—— 以下为 hcTools 原文（verbatim）——

契约来源：飞书《泛知识智能体 0915》
https://hisense.feishu.cn/wiki/OXZNwA9BDidXU7kQIJxcUkJJngi
接口：POST /knowledge/qa/v3/agent（V2 为 /knowledge/qa/v2/agent）。

必填只有 messages；riskRespStrategy 产品侧要求非空（默认 all_replace）。
场景通过 sceneTypeIntentList: [{sceneType, intent}] 下发：sceneType 区分业务
（video 影视 / child 少儿 / audio_qa 有声 / sport_qa 体育 / agent_music 音乐 /
education_qa 教育 / device_control_qa 设备控制；开放域知识为 knowledge_qa），
intent 只对音乐、体育两个业务定义，产品据 intent 决定是否下发媒资以及匹配哪套
系统提示词。intent 缺省合法（契约的 paramError 示例就只给了 sceneType），因此
非音乐/体育场景只下发 sceneType。

【已废弃 2026-09-16】旧实现下发顶层 `sceneType` 与 `need_media`：前者不在契约内
（契约是 sceneTypeIntentList 列表），后者契约里根本没有该字段——媒资是否下发由
intent 决定，不再由客户端算 bool。
"""

from __future__ import annotations

import re

# --------------------------------------------------------------------------
# sceneType：按内容域词表映射，首个命中即返回，未命中为开放域知识 knowledge_qa。
# 顺序即优先级：**体育先于音乐**——「勇士和爵士的比赛」「快船队和爵士队」里的
# 爵士是犹他爵士队而非音乐体裁，体育词表命中即判 sport_qa 可消除该歧义。
# --------------------------------------------------------------------------
_SCENE_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"少儿|动画|卡通|佩奇|汪汪队|幼儿|绘本|儿歌"), "child"),
    (re.compile(r"教育|学习|课程|知识(?:点|课)|课文|老师(?:讲|教)|学校|学科|教师|教研|"
                r"数学|物理|化学|"
                r"公式|古诗|单词"), "education_qa"),
    # 体育：赛事名（世俱杯/欧冠…）、项目名、知名运动员与球队（契约示例
    # 「梅西的身高是多少？」「预测2026年世俱杯的冠军是哪支队伍」均无“体育/比赛”二字）。
    # 撞车加环视：詹姆斯·邦德是影视人物(vod_search)；《赛车总动员》是动画电影。
    (re.compile(
        r"体育|比赛|赛事|比分|球队|球员|运动员|联赛|赛季|"
        r"世界杯|世俱杯|欧洲杯|亚洲杯|美洲杯|欧冠|亚冠|奥运|全运|"
        r"NBA|CBA|英超|西甲|意甲|德甲|法甲|中超|中甲|"
        r"足球|篮球|排球|网球|乒乓球|羽毛球|棒球|冰球|斯诺克|台球|"
        r"田径|马拉松|游泳(?!.{0,4}(?:知识|安全|常识))|跳水|体操|举重|格斗|拳击|"
        r"自由搏击|UFC|F1|赛车(?!总动员)|滑冰|滑雪|"
        r"季后赛|总决赛|半决赛|决赛|夺冠|金牌|积分榜|"
        r"梅西|C罗|c罗|姆巴佩|内马尔|哈兰德|贝林厄姆|维尼修斯|萨拉赫|凯恩|"
        r"詹姆斯(?!·?邦德)|库里|杜兰特|乔丹|科比|东契奇|约基奇|姚明|"
        r"樊振东|马龙|孙颖莎|陈梦|王曼昱|全红婵|覃海洋|苏炳添|刘翔|李娜|郑钦文|"
        r"德约科维奇|纳达尔|费德勒|阿尔卡拉斯|"
        r"皇马|巴萨|拜仁|曼联|曼城|利物浦|切尔西|阿森纳|尤文|国米|湖人|勇士|"
        r"凯尔特人|雄鹿|快船|掘金|独行侠"), "sport_qa"),
    # 音乐：体裁词也算（契约示例「介绍下摇滚乐的发展历程」无“音乐/歌曲”二字）。
    # 以下词与其他域撞车，一律加否定环视（括号内为撞车用例，golden 均非音乐）：
    #   爵士+队 → 犹他爵士队(sports_team_search)；蓝调+音效 → 设备音效模式(mode_control)；
    #   演唱 → 「播放BILIBILI跨年演唱会」/「歌剧小二黑结婚」实为影视(vod_fuzzy_search)；
    #   乐器名 → 「古筝名曲教学入门」「打开免费的二胡」实为教育课程(edu_fuzzy_search)。
    (re.compile(r"音乐|歌曲|歌星|歌手|专辑|曲子|MV|旋律|音符|"
                r"摇滚(?!乐?队)|民谣|说唱|嘻哈|交响|"
                r"爵士(?!队)|蓝调(?!音效)|"
                r"曲风|乐曲|乐章|乐理|乐评|乐队"), "agent_music"),
    # audio_qa 须排在 video 之前：「广播剧天涯客的配音演员」里的「剧/演员」
    # 是广播剧这一有声内容的固有组成部分，不应被影视词表吞掉。
    # 「小说」加负向环视：「小说改编的电视剧/电影」是影视内容，不应吞成 audio_qa。
    (re.compile(r"广播剧|有声|听书|电台|播客|评书|"
                r"小说(?!\s*.{0,8}(?:改编|影视|电视剧|电影|网剧|剧集|短剧|翻拍))"),
     "audio_qa"),
    (re.compile(r"影视|电影|电视剧|剧集|综艺|纪录片|影片|剧(?:里|中)?|演员|导演|演(?:的|过|着)?"),
     "video"),
    (re.compile(r"设备|遥控|音量|亮度|画质|机顶盒|投屏|关机|开机"), "device_control_qa"),
]

# --------------------------------------------------------------------------
# agent_music 的 intent（契约只定义两类）
#   music_chat_qa 不绑定特定歌曲的通用音乐领域知识问答 → 纯问答链路，不下媒资
#   music_info_qa 指向具体歌曲/歌手/乐队/专辑的客观属性事实查询 → 需执行歌曲检索
# 契约明确：问句一旦绑定特定歌曲，即便问的是音乐类型也归 music_info_qa
# （如「《七里香》是什么曲风」）。
# --------------------------------------------------------------------------
_MUSIC_ENTITY = re.compile(
    r"《[^》]+》"                                    # 书名号 = 绑定具体作品
    r"|歌手|歌星|乐队|主唱|专辑|单曲|这首歌|那首歌|歌曲集"
    r"|歌名|歌词|作曲|作词|写词|填词|原唱|翻唱|合唱团"
    r"|唱(?:的|过|了)|演(?:唱|绎)过|出(?:自|道)哪"
    r"|榜|排行"                                      # 榜单需执行歌曲检索 → info 链路
    # 「歌曲XX」直接跟歌名（契约示例「歌曲泡沫的相关信息」），排除「歌曲的相关信息」。
    r"|歌曲[一-龥A-Za-z0-9·]{2,6}"
)
# 「X的歌/专辑」也属绑定，但需排除「好听的歌」这类纯修饰词（左边界要紧贴「的」）。
_MUSIC_OWNED = re.compile(r"[一-龥A-Za-z·]{1,6}的(?:歌|歌曲|专辑|单曲|音乐|作品)")
_MUSIC_OWNED_STOP = re.compile(
    r"好听|经典|免费|最新|热门|流行|伤感|开心|欢快|安静|动感|舒缓|轻快|"
    r"怀旧|伤感|好听|网络|抖音|适合|这种|那种|什么|所有|全部|一首|几首"
)

# --------------------------------------------------------------------------
# sport_qa 的 intent（契约定义十类；sports_chat_qa 为体育兜底）
# 判定顺序＝由具体到宽泛，兜底最后。
# --------------------------------------------------------------------------
_SP_LINEUP = re.compile(r"阵容|首发|大名单|上场球员|排兵布阵|谁上(?:场|阵)")
_SP_EVENT = re.compile(
    r"黄牌|红牌|点球|越位|乌龙|犯规|换人|角球|"
    r"(?:几|多少)(?:张|个|次|粒|球).{0,4}(?:牌|球|进球)|比赛事件"
)
_SP_RANK = re.compile(r"榜|排名|排行|积分")
_SP_SUMMARY = re.compile(r"总结|回顾|复盘|战报|汇总|盘点|赛况|战绩|情况")
# 单场/多场判据：_sports_intent 单靠 _SP_VS（对阵双方）区分——命中即单场
# （match_summary/match_forecast），否则即多场/周期级（summary/forecast）。
# 不另设「本周/多场」词表：多场本就是 else 分支，词表无从参与判定。
# 单场识别：A 和 B 之间用「比赛/球赛/对决/大战」连接即视为单场。
# 只写“比赛”会漏「快船队和爵士队的球赛预测下谁能赢啊」这类球赛说法，
# 漏判会把它降级成赛事级 sports_forecast。
_SP_VS = re.compile(r"对阵|对战|对垒|迎战|VS|vs|Vs|"
                    r"和.{1,8}(?:的)?(?:比赛|球赛|对决|大战|决赛)")
_SP_FORECAST = re.compile(
    r"预测|谁能赢|谁会赢|谁赢|哪(?:个|支|队)赢|能不能赢|会不会赢|胜负|输赢|夺冠"
)
_SP_TITLE = re.compile(r"冠军|捧杯|捧起|晋级|出线|最终排名|登顶|卫冕|前三|名次")
_SP_INFO = re.compile(r"在哪|哪里|几点|什么时候|时间|场地|场馆|举办|赛程|开赛|对阵")
_SP_PERSON = re.compile(
    r"身高|体重|年龄|多大|出生|生日|国籍|简介|资料|履历|生涯|成就|荣誉|"
    r"效力|转会|身价|教练|运动员|球员"
)


def fan_scene_type(query: str, hint: str = "") -> str:
    """内容域词 → sceneType；未命中为开放域知识 knowledge_qa。

    hint 是调用方（某个具体域）给出的场景，优先于词表：词表可能漏（如
    「来点香港商台榜的歌」只出现「歌」不出现「歌曲」，词表判不出音乐），
    而 music 域调到这里时场景本就是 agent_music，调用方比词表更可信。
    """
    q = query or ""
    if hint:
        return hint
    for pattern, scene in _SCENE_RULES:
        if pattern.search(q):
            return scene
    return "knowledge_qa"


def _music_intent(q: str) -> str:
    if _MUSIC_ENTITY.search(q):
        return "music_info_qa"
    m = _MUSIC_OWNED.search(q)
    if m and not _MUSIC_OWNED_STOP.search(m.group(0)):
        return "music_info_qa"
    return "music_chat_qa"


def _sports_intent(q: str) -> str:
    if _SP_LINEUP.search(q):
        return "sports_match_lineup_search"
    if _SP_EVENT.search(q):
        return "sports_match_event_qa_search"
    if _SP_RANK.search(q):
        return "sports_rank_llm_search"
    if _SP_SUMMARY.search(q):
        # 单场（有对阵双方）→ match_summary；否则多场/周期级 → summary
        return "sports_match_summary" if _SP_VS.search(q) else "sports_summary"
    if _SP_FORECAST.search(q):
        # 单场对阵 → match_forecast；赛事级（冠军/晋级/最终排名）→ forecast
        return "sports_match_forecast" if _SP_VS.search(q) else "sports_forecast"
    if _SP_TITLE.search(q):
        return "sports_forecast"
    if _SP_INFO.search(q):
        return "sports_match_info_search"
    if _SP_PERSON.search(q):
        return "sports_person_search"
    return "sports_chat_qa"


def fan_intent(query: str, scene: str = "") -> str:
    """场景子意图；只有音乐(agent_music)与体育(sport_qa)定义，其余返回 ""。"""
    q = query or ""
    scene = scene or fan_scene_type(q)
    if scene == "agent_music":
        return _music_intent(q)
    if scene == "sport_qa":
        return _sports_intent(q)
    return ""


def fan_scene_intents(query: str, hint: str = "") -> list[dict]:
    """sceneTypeIntentList；intent 为空时只下发 sceneType（契约允许缺省）。"""
    q = query or ""
    scene = fan_scene_type(q, hint)
    item: dict = {"sceneType": scene}
    intent = fan_intent(q, scene)
    if intent:
        item["intent"] = intent
    return [item]


def fan_params(query: str, scene_hint: str = "") -> dict:
    """fan_knowledge_agent 完整请求参数：messages 必填，riskRespStrategy 非空。

    scene_hint：域内已知场景时由调用方传入（音乐域传 "agent_music"，sports 域
    传 "sport_qa"），避免词表漏判把本域场景降级成 knowledge_qa。
    """
    q = query or ""
    return {
        "messages": [{"role": "user", "content": q}],
        "riskRespStrategy": "all_replace",
        "sceneTypeIntentList": fan_scene_intents(q, scene_hint),
    }


# --------------------------------------------------------------------------
# 各域本域知识问答（fan_knowledge_agent）的 sceneType 与兜底路由
#
# 每个具体域（vod/music/...）在自身 pipeline 里把 fan_knowledge_agent 作为
# 「第二层兜底」（LLM 两层都失败后），参数由 fan_params(query, scene_hint) 重建，
# 携带本域 sceneType（video/agent_music/...），让下游走该场景的知识问答链路，
# 而不是掉进无 sceneType 的通用 qa。第一层兜底仍是域的模糊检索工具。
# --------------------------------------------------------------------------
QA_SCENE_BY_DOMAIN = {
    "vod": "video",
    "children": "child",
    "education": "education_qa",
    "music": "agent_music",
    "audio": "audio_qa",
    "sports": "sport_qa",
    "device": "device_control_qa",
}


def domain_fan_params(domain_key: str, query: str) -> dict:
    """域内兜底：本域知识问答参数（带 sceneType）。只对 7 个具体域有 hint。"""
    return fan_params(query, scene_hint=QA_SCENE_BY_DOMAIN.get(domain_key, ""))


# --------------------------------------------------------------------------
# LLM 全部失败时的两层兜底判据：query 像「知识问答」→ 本域 fan；否则 → 本域检索工具。
# 只做问法形态的轻量词形判断（为什么/是什么/怎么/吗/谁/介绍一下/…），不落规则文件。
# --------------------------------------------------------------------------
_QA_RE = re.compile(
    r"为什么|是什么|是什么意思|什么是|实际是|怎么样|怎么回事|怎么回事|"
    r"怎么(?:办|写|做|玩|调|用|回|弄|改|关|开|装|登录)|如何|咋|"
    r"谁(?:是|的|来|在|找|演|唱|有)|哪些|哪些的|有什么|有没有什么|"
    r"[吗嘛吧呢][？?]?$|介绍一下|介绍下|有哪些|讲讲|说说|谈谈|"
    r"多少(?:岁|时|钱)|在哪(?:里)?|哪里|哪个|几岁|何时|何种|何谓|"
    r"代表|含义|来着|啥意思|有什么区别|区别是|一样吗|可不可以|能不能"
)


def looks_like_qa(query: str) -> bool:
    """query 是否呈问句/知识问答形态（两层兜底里的第二层判据）。"""
    return bool(_QA_RE.search(query or ""))


def rebuild_fan_params(params: Any, scene_hint: str = "", fallback_query: str = "") -> dict:
    """LLM 裸输出的 fan 参数 → 重建为 fan 契约（messages 数组 + sceneTypeIntentList + riskRespStrategy）。

    LLM 可能给 messages 为字符串 / 含 content 的 dict / 数组三种形态，按 qa 域既有契约统一；
    取不到时回退 fallback_query（调用方用用户原话兜底），保证 fan 必收到用户原话。
    """
    q = ""
    if isinstance(params, dict):
        msgs = params.get("messages")
        if isinstance(msgs, str):
            q = msgs
        elif isinstance(msgs, list) and msgs:
            first = msgs[0]
            q = first.get("content", "") if isinstance(first, dict) else str(first)
        elif isinstance(msgs, dict):
            q = msgs.get("content", "")
        elif isinstance(params.get("content"), str):
            q = params["content"]
        elif not msgs:
            # 无 messages 也回退空串，fan_params 保证结构合法
            q = ""
    if not (q or "").strip():
        q = fallback_query or ""
    return fan_params(q, scene_hint)


__all__ = ["fan_scene_type", "fan_intent", "fan_scene_intents", "fan_params",
           "QA_SCENE_BY_DOMAIN", "domain_fan_params", "looks_like_qa", "rebuild_fan_params"]
