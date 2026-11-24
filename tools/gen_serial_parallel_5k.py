#!/usr/bin/env python3
"""生成串行/并行多意图数据（5k）：并行=无 dep，串行=每步带 dep。

素材：all_eval.csv 的真实金标行（domain/query/tool/tool_param）。
拼装：取 2–3 条互不相同、域不冲突的金标子句，用连接词模板拼成一句多意图 query；
      gold 直接由子句金标合成（params 天然合法，与线上同口径）。
      · 并行：连接词取「同时/顺便/一起/另外/还有」，dep_steps 为空。
      · 串行：连接词取「然后/接着/随后/之后/再/接下来」，第 2..n 步带 dep 指向第 1 步，
              仍照常带 vt/params（一把出，不做指代消解）。

列同 bench_serial_train_10k.csv，另加 dep_steps 标记哪些步带 dep。
用法：python3 tools/gen_serial_parallel_5k.py [--n 5000] [--seed 20261110]
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from device_phrases import (DEVICE_PHRASES, SURFACE_W, NULL_P,  # noqa: E402
                            clean_device_params, may_lead, may_follow,
                            POS_EVIDENCE, NONLEAD_TOOLS, _POS_BY_QT, pos_weight,
                            content_may_lead, content_may_follow, CONTENT_FOLLOW_REG,
                            CONTENT_POS_EVIDENCE)
from clean_testset import _strip_k12  # noqa: E402  与评测 judge() 同一套 edu 字段口径

ROOT = Path(__file__).resolve().parent.parent
SEED_CSV = ROOT / "all_eval.csv"
CATALOG = ROOT / "catalog.full.json"
OUT = ROOT / "testset" / "feishu" / "bench_dep_5k.csv"

# 旧工具名 → 当前目录工具名（同 tools/gen_train.py 的 TOOL_ALIAS）
TOOL_ALIAS = {"vod_search": "vod_search_all", "educ_search": "educ_search_all",
              "educ_relate_recommend": "educ_relate_search",
              "sports_match_data_search": "sports_rank_search"}

CN = {"device": "设备", "vod": "影视", "music": "音乐", "education": "教育",
      "children": "少儿", "sports": "体育", "audio": "有声"}

# 设备工具集（与 ref_device_vocab.DEVICE_TOOLS 同口径）。用于判断一个子句是不是
# 设备侧——注意【不能看 gold 列】：CDX 跨标语域里 gold 是内容域（影视/音乐…）
# 但 tool 是设备工具（playback_control/source_switch…）。参考集里带接缝标点/
# 语气词的非末子句全部由【工具域】决定（0/186 的 CDX 子句带标点），故按 tool 判。
_DEVICE_TOOL_SET = {
    "numeric_adjust", "mode_control", "screen_layout", "playback_control",
    "screensaver_control", "common_control", "power_control", "display_control",
    "source_switch", "timer_control", "network_control", "smart_camera",
    "audio_control", "demo_control", "screen_lift_rotation",
    "solve_picture_sound_problem_control",
}

# 域 → 线上金标的 domain 列名（all_eval 用 tv_control/educ/edu，bench 用中文）
DOM_ALIAS = {"tv_control": "device", "educ": "children", "edu": "education"}

# 连接词池：并行（不表先后/依赖）与串行（表先后）。
# 只用真人口语里出现的顺承/并列词，且同一条 query 内不重复（见 make_query）。
#
# 权重：真人挑连接词并不是均匀的——「然后/还有/再/顺便」张口就来，
# 「完事再/紧接着/顺带」是偶发。等概率抽取会让 16 个词各占 6%，读起来像从表里
# 轮着挑，这正是「生硬」的来源之一。故按口语频率重复入池（重复次数=权重）。
# 权重直接对齐 bench_serial_train_10k.csv 的实测频率（那是用户指定的语感基准）：
#   并行：同时 49% / 一起 26% / 顺便 25%（参考集里「还有/另外」几乎不用于并行）
#   串行：接着·再·然后·随后·接下来·之后 六项各约 15-18%，基本均匀
# 之前我按「想当然的口语频率」加权，与参考集差得很远——尤其并行档把「还有/另外」
# 放到了 25% 以上，而参考集里它们占比近乎为零。
# 把两种模式的连接词逐一对齐 bench_serial_train_10k.csv 的实测清单（按 exec_mode 分开统计）：
#   并行（5000 行）：只有四个，且几乎等概率——一起 25.6 / 顺便 25.1 / ，同时 25.1 / 同时 24.2。
#     没有裸逗号并列（0%），更没有「另外/还有/顺便/顺带/另外呀」这类我上一版凭空加的词。
#   串行（4991 行）：八种——，13.7 / 接着 13.0 / 随后 12.8 / 然后 12.8 / 空 12.6 /
#     再 12.1 / 接下来 11.8 / 之后就 11.2。「空」=两个子句直接连写、不加任何分隔符；
#     参考集里的「之后」100% 是「之后就」，从无孤立的「之后」；串行也从不给连接词加前导逗号
#     （「，接着」0 次），这点和并行正好相反。
# 我上一版把两者混用一个池、还额外补了 6 个自造词，是这批数据「生硬」的主要来源。
_SERI_W = [("，", 14), ("接着", 13), ("随后", 13), ("然后", 13), ("", 13),
           ("再", 12), ("接下来", 12), ("之后就", 11)]
_PARA_W = [("同时", 25), ("一起", 25), ("顺便", 25), ("，同时", 25)]
PARA = [c for c, w in _PARA_W for _ in range(w)]
SERI = [c for c, w in _SERI_W for _ in range(w)]

# 真人口语并非总是「子句+连词+子句」。多数时候只是把两件事顺着说出来，
# 中间用逗号甚至不接词。纯逗号并列要占相当比例，否则通篇一个句式，读起来像模板。
BARE = ["，", "，", "，", "，", "，"]
# 句末语气词：真人常带，去掉后仍是完整子句，不影响金标。
# 子句内语气词：参考集行尾实测只有 吧9.2% 啊8.8% 呢8.2% 呗0.2%，无 哦/哈/呀/哪/啦。
TAIL_PART = ["", "", "", "", "", "", "吧", "啊", "呢"]
# 首句「呼名 + 动词引导」可叠加（"小维帮我…""小聚给我…"）。
# 但参考集 bench_serial_train_10k.csv 里呼名只占 0.2%——真人对自己家的电视说话
# 很少先喊名字，直接说事即可。我原来给了 32%，是整批数据最显眼的「不像人话」特征。
# 呼名保留但降为低频，改用「哎/那个/诶」这类无称呼的语气起头撑起开头多样性。
# 唤醒词：参考集只有「小聚」（25 行 / 0.25%，含 13 行"海信小聚"），无「小维/哎/诶/那个」。
WAKE = ["小聚", "海信小聚", "小聚"]
LEAD_VERB2 = ["帮我", "给我", "麻烦", "帮我", "给我"]
# 整句礼貌引导：四个词在参考集里近乎均分（给我 382 / 请 356 / 帮我 345 / 麻烦 333）。
# 加在整句最前面，属于【拼接时生成】而非首子句自带（见 _lead_head 的说明）。
_LEAD4 = ["请", "帮我", "给我", "麻烦"]
_LEAD4_W = [356, 345, 382, 333]
# P(加整句引导 | 首子句工具)，实测自 bench_serial_train_10k.csv 的【裸起首】行
# （分母剔除首子句自带请求语的行）。n<60 的工具并入默认值。
_ADD_LEAD = {
    "numeric_adjust": 0.229, "screen_layout": 0.173, "playback_control": 0.222,
    "power_control": 0.226, "vod_search": 0.167, "vod_search_all": 0.063,
    "audio_search": 0.049, "educ_search": 0.105, "educ_search_all": 0.071,
    "music_song_search": 0.118, "vod_fuzzy_search": 0.171, "edu_fuzzy_search": 0.149,
    "edu_search": 0.229, "fan_knowledge_agent": 0.175, "common_control": 0.209,
    "music_song_recommend": 0.066, "sports_team_search": 0.191, "sports_vod_search": 0.093,
    "sports_match_search": 0.170, "mode_control": 0.115, "music_song_history": 0.053,
    "sports_rank_search": 0.260, "source_switch": 0.000, "tvchannel_music_search": 0.133,
    "music_ksong_search": 0.227, "educ_fuzzy_search": 0.243,
    "music_song_favorite_search": 0.095,
}
_ADD_LEAD_DEFAULT = 0.110
# 整句引导【贴不贴、贴哪个】取决于首子句的【动词类型】（2026-11-11 第十一轮）。
# 上面那张 _ADD_LEAD 只给了"某工具贴引导的总概率"，但四个引导词的**选择**原来是无条件
# 均分（_LEAD4_W = 356/345/382/333），于是把「麻烦」也贴到了播放句上——参考集里几乎不这么干：
#   · 贴引导的概率：**播放动词起首**的行 5.3%（152/2866），**非播放动词起首** 22.9%（1636/7134），
#     差 4.3 倍。真人不会说「麻烦播放X」「帮我播放X」——想看点播内容是直接说
#     「我想看/我要看/播放X」，只有对**设置类**动作才用征询口吻（"麻烦关掉电视""帮我调音量"）。
#   · 引导词分布也随首句动词类型走：
#       播放句上：请 0.487 / 给我 0.336 / 帮我 0.151 / 麻烦 0.026
#       非播放句：请 0.258 / 帮我 0.285 / 给我 0.254 / 麻烦 0.203
#     即播放句几乎只用「请/给我」，从不「麻烦」。
# 我原来在这两处都按均分 + 统一概率贴，实测（mine vs REF）：
#   播放句贴引导 0.105 vs 0.053（2×）、非播放句 0.193 vs 0.229（欠贴）；「麻烦播放」15 行 vs 1 行。
# 下面两张表直接从参考集收割（口径：整句以「请/帮我/给我/麻烦」起头、且该词【不属于】子句本身——
# 即剥掉引导词后剩下的头是裸句；与 _lead_head 的判定一致）。_ATTACH_LEAD 覆盖上面的总概率。
_ATTACH_LEAD = {"play": 0.053, "nonplay": 0.229}
# 逐 (工具, 首句动词类型) 的贴引导概率，直接从参考集收割（口径：首子句【本身不带】
# 请求语，整句却以 请/帮我/给我/麻烦 起头）。播放句上大量工具实测 p=0.000
# （educ_search 0/204、vod_search_all 0/314、playback_control 0/72、source_switch 0/84）
# —— 真人想看点什么就直接说「播放/我想看X」，绝不说「麻烦播放X」。
_ATTACH_BY_TOOL_NONPLAY = {
    "audio_search": 0.161, "common_control": 0.261,
    "edu_fuzzy_search": 0.184, "edu_search": 0.263, "educ_fuzzy_search": 0.258,
    "educ_search": 0.236, "educ_search_all": 0.203,
    "fan_knowledge_agent": 0.179, "mode_control": 0.171,
    "music_ksong_search": 0.227, "music_song_history": 0.161,
    "music_song_qqmusic_search": 0.077, "music_song_recommend": 0.250,
    "music_song_search": 0.264, "numeric_adjust": 0.229,
    "playback_control": 0.273, "power_control": 0.233, "screen_layout": 0.229,
    "sports_match_reservation": 0.250, "sports_match_search": 0.233,
    "sports_rank_search": 0.260, "sports_team_search": 0.191,
    "sports_vod_search": 0.174, "tvchannel_music_search": 0.164,
    "vod_fuzzy_search": 0.201, "vod_person_search": 0.167,
    "vod_personalized_search": 0.269, "vod_relate_search": 0.160,
    "vod_search": 0.191, "vod_search_all": 0.193,
}
_ATTACH_TOOL_FALLBACK = {"play": 0.002, "nonplay": 0.223}
_LEAD4_BY_TYPE = {
    "play":    ["请", "帮我", "给我", "麻烦"],
    "nonplay": ["请", "帮我", "给我", "麻烦"],
}
# 权重口径必须与"生成器自己贴的那一次"一致：统计时只取【首子句本身不带请求语】的行，
# 否则会把种子自带的「我想看X」也算成贴引导，把分布带偏（我第一版就是混进了自带行，
# 得到 play 档 请.487/麻烦.026，与真实的 .233/.140 差一倍）。
_LEAD4_W_BY_TYPE = {
    "play":    [233, 256, 372, 140],
    "nonplay": [254, 243, 266, 237],
}
# 首句动词 → 贴引导概率（参考集口径：首子句本身不带请求语，整句却冠了 请/帮我/给我/麻烦）。
# 【关键：不是"播放句一律不贴"，而是按【具体动词】三分】——实测：
#   播放 0/1537、打开 0/373、放 0/159、找 0/121、搜 0/103、看 0/70、查 0/48、听 0/41、
#   来一首 0/20  —— 这些动词【从不】被冠引导（真人说"播放X"，不说"麻烦播放X"）；
#   唱 15/59=0.254、推荐 14/49=0.286、学习 8/31=0.258 —— 这三个【可以】冠（"帮我推荐首歌"
#   "给我唱个儿歌""请学习这套课"），因为它们天然带"替我做/教我"的请求语义；
#   其余（含设备句）0.222。
# 我第一版只按"是否播放类动词"二分，把 唱/推荐/学习 也划进不贴档，结果 play 档整体 0.037
# 而 REF 0.019 看着还行、但里面 麻烦 占比 0.405（REF 0.140）——因为漏掉的正是那三类，
# 剩下的全落在设备/其他动词上。三分后才逐类吻合。
_LEAD_TOOLS_VERB = {
    "played":  ("播放", "放", "听", "看", "搜", "查", "找", "来一首", "来点", "打开",
                "放一", "放个", "看一", "听一", "播"),
    "askable": ("唱", "推荐", "学习", "学"),      # 天然带请求语义，可冠引导
}
def _head_type(head: str) -> str:
    for v in _LEAD_TOOLS_VERB["askable"]:
        if head.startswith(v):
            return "nonplay"                        # 可冠档：并入 nonplay 的贴率
    for v in _LEAD_TOOLS_VERB["played"]:
        if head.startswith(v):
            return "play"
    return "nonplay"
# 话题开启词：真人常先起个头再说事（"对了""哦对""顺便问下"），这类词不改语序也不影响金标。
# 只在首子句前加，且与呼名互斥（"小维对了…"不通），用概率控制。
# 参考集 0 次「对了/还有啊/顺便问下」——话题开启词是我自造的，保留变量以免引用报错，但不再使用。
PREFACE = []
# 末尾征询：真人下完指令常补一句（"行吗""可以吧"），纯语气，不影响金标。
# 句尾礼貌语：种子常自带（"请播黄河在咆哮谢谢"），避免再叠一层轻松/致谢
_POLITE_TAIL = re.compile(r"(谢谢|多谢|麻烦了|行吗|行不行|可以吧|可以吗|成吗|好吧|好不好)$")
# 句末语气词后缀：带「吧」的征询（可以吧/好吧）叠在「…吧」后面会重复同一语素
# （"音量设成30吧，可以吧"），听起来絮叨。有它时换用另一类征询。
_BA_TAIL = re.compile(r"(吧|呗)$")
ENQUIRE = ["", "", "", "", "", "行吗", "可以吧", "好吧", "谢谢",
           "行不行", "好不好", "可以吗", "成吗"]
# 「吧」系征询：与句末「吧」语素重合，需避开
_ENQUIRE_BA = {"可以吧", "好吧"}
_ENQUIRE_NONBA = [e for e in ENQUIRE if e and e not in _ENQUIRE_BA]
# 请求软化：前缀在动词核之前（"能不能放X"）。参考集里只占子句的 0.47%
# （能不能 43 + 可不可以 4 / 20243 个子句）；「要不」参考集 0 次。
# 我原来按 10% 的概率加，实测整卷 4.06%，是参考的 8 倍——礼貌层级不是越多越像人，
# 真人对自己家电视绝大多数时候是直接下命令。
SOFTEN = ["能不能", "可不可以"]
SOFT_FLAT = tuple(SOFTEN)
# 时间/情境铺垫：真人常先交代当下的处境（"我现在想听…"）。纯前缀，不影响金标。
# 参考集 0 次「我现在/我这会儿…」，不再使用。
TIME_PRE = []
# 感叹/情绪起头：真人常有（"哎呀这音量"），当前语料完全缺失。纯语气词，独立成档。
# 参考集 0 次「哎呀/哎哟」（感叹起头是我自造的），不再使用。
EXCLAIM = []
# 「描述当前毛病」类子句（"人声听不清""太刺眼"）陈述的是一种状态，不是序列里的一步，
# 前面接「完事再/接着/随后」很别扭；只允许裸并列或「另外/顺便」这类非顺承词。
STATE_TOOLS = {"solve_picture_sound_problem_control"}
# 「描述毛病」子句的连接词【按模式分开】——参考集实测：
#   并行：同时 33 / 顺便 14 / 一起 12（共 59 条症状句），【无一例外】都用并行的连接词，
#         裸逗号 0 条。
#   串行：接着 9 / 随后 7 / 接下来 6 / 之后 4 / 然后 3（+ 直接连写的 12 条），
#         也就是串行的常规顺承词，没有特殊待遇。
# 故症状句并不需要专属词表——并行用 PAR、串行用常规串行池即可（我此前给了一张
# 不分模式的 ["，","顺便"]，于是在并行里造出「唱一个落花吧，屏幕不够锐利帮我修改一下！」
# 这类参考集 0 例的裸逗号行）。
#   ⚠️ 2026-11-11 修正：这里原来写死 ["，", "顺便"]，串行症状句照用，于是 13 行串行
#   用「顺便」接症状句（"播放史记的有声内容顺便画面忽明忽暗的帮我调节一下"）。参考集
#   串行含 solve_picture_sound_problem_control 的 63 行里「顺便」**0 次**——上面注释早就
#   写对了（"串行的常规顺承词，没有特殊待遇"），是代码没跟上。串行改走 SERI 池。
STATE_CONN_PAR = ["，同时", "一起", "顺便"]

# 顺承副词：子句【自身】以这些词开头时，前面不再挂连接词，直接连写。
# 依据 bench_serial_train_10k.csv 实测（见 make_query 里的说明）：seg 以 接着/再/随后/
# 然后/接下来/之后就 起头的，前面 100% 是直接连写、从无连接词。
_SEQ_ADV = ("接着", "再", "随后", "然后", "接下来", "之后就")
SEQUENTIAL = {"，然后", "，接着", "，随后", "，之后", "，接下来", "，紧接着", "，完事再", "，再"}
# 句中迟疑/换气（"播放X，那个，音量调到30"）：真人边想边说时极常见，当前语料 0%。
# 它不含任何先后语义，纯停顿，故并行/串行都能用；金标不变。
# 参考集 0 次「，那个，/，嗯，/，我看看，」，不再使用。
HESITATE = []
# 句首迟疑（"嗯，帮我放个X"）：与人名呼语、话题开启词互斥，独立一档
# 参考集 0 次句首迟疑词，不再使用。
HEAD_HESITATE = []
# 话题式「就」用：子句以名词话题起头时，「就」应插在话题与动词之间
# （"接着音效模式就调成流行音效"）。交替项必须「长的在前」，否则
# 「声音模式调成游戏模式」会先匹配到「声音」，生成「声音就模式调成…」这种断词。
_TOPIC_RE = re.compile(
    r"^(?:把)?(?P<topic>图像模式|声音模式|音效模式|信号源|清晰度|对比度|"
    r"音量|声音|亮度|色度|屏幕|画面|全屏|分屏|屏保|壁纸|壁画)(?P<rest>.+)$")

# 结尾清理：去掉尾部语气词/标点，便于拼接
TAIL = re.compile(r"[~～。！!？?，,、；;\s]+$")
HEAD = re.compile(r"^(小聚|你好|小维|海信小聚)[，,：:\s]*")

# 纯chat/无法作为子句的行关键词（这些行 tool_param 多为空或不适合并入多意图）
BAD_Q = re.compile(r"^(你好|在吗|你是谁|谢谢|再见|哈喽|hello)[~～。！!？?]*$")
# 疑问尾/疑问词（"…那个片段吗""哪个台有…""有哪些"）：并入多意图后语义崩，剔除
ASK_Q = re.compile(r"[吗么呢吧]\s*[？?]?$|[？?]|(哪个台|哪一部|哪些|是什么|多少|几点|谁|"
                   r"怎么|为什么|如何|能不能帮我看看)")
# 前置序数/指代（"第3个""这一部""它"）：并入多意图后失去所指，剔除
REF_Q = re.compile(r"第[0-9一二三四五六七八九十]+个|这[一二三四五六七八九十]?[部首个本集]|"
                   r"那一[部首个]|^(它|他|她|这个|那个)")

# 谓语/动作检验：真人说的话必有动词或明确动作。all_eval 里 26.6% 是「能力项标签」
# （"分屏功能""HDMI接口选择""亮度数值50"），并进多意图后语法断裂，必须剔除。
# 要求【句首或紧随主语后】就是动作词，避免把"英雄诀播放""张震讲故事之鬼鲸泥鳅"当完整句收进来。
PRED_HEAD = re.compile(
    r"^(我?想|我要|给我|帮我|请|麻烦|能不能|可以|来|放|播|听|看|搜|找|查|推荐|打开|关闭|关掉|"
    r"调|设置|换成|切换|切到|跳到|快进|快退|倒退|收藏|预约|学|唱|买|启动|停止|退出|返回|"
    r"下载|安装|介绍|讲|念|显示|提高|降低|恢复|继续|进入|开始|用|介绍|打开)")
# 句首是内容名词时，名词后须紧跟动作（"小谢尔顿第五季播放"这类倒装也自然，故允许句尾动词）
PRED_TAIL = re.compile(r"(播放|播一下|放一下|看一下|听一下|听听|看看|搜一下|找一下|来一首|来一段|来点)$")
# 明显是纯内容碎片（无任何动作）：名字/名词堆叠
NOUN_JUNK = re.compile(r"^(的|了|吧|呢|啊)|[之]+.*$")


def is_full_sentence(q: str) -> bool:
    """判断是否为真人会说的完整句子（含动作），而非能力项标签/名词碎片。"""
    if PRED_HEAD.match(q) or PRED_TAIL.search(q):
        return True
    # 允许 "把X调到N"/"X帮我调一下" 这类把字句、求助句
    if re.match(r"^(把|将)", q) and re.search(r"调|设|换|切|关|开|放|播", q):
        return True
    if re.search(r"(帮我|给我|麻烦)", q) and re.search(r"调|设|换|切|关|开|放|播|找|搜", q):
        return True
    return False


def clean(q: str) -> str:
    q = HEAD.sub("", q.strip())
    # 去句首/句中口语填充词（"也嗯我想听"→"我想听"；"呃/啊/哎/哦"）
    q = re.sub(r"^[嗯啊呃哎哦唉呀嘛欸诶呐]+", "", q)
    q = re.sub(r"[，,]?[嗯呃哎唉欸诶]+", "", q)
    q = TAIL.sub("", q)
    return q.strip()


def _flat(o) -> list[str]:
    """取参数里所有字符串叶子值（用于判断并列条件是否被拆开，这里只用于去重/健康检查）。"""
    out = []

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str) and x:
            out.append(x)

    walk(o)
    return out


# 工具 → 金标域【覆盖表】（2026-11-11）。catalog 的 domain 是【工具权威域】，不是
# bench_serial_train_10k.csv 的【金标域】——两者在 fan_knowledge_agent 上不一致：
#   · catalog 里 fan_knowledge_agent 有 7 条（audio/children/device/education/music/
#     sports/vod 各一），load_catalog_tools 用 dict 推导，【最后一条 vod 胜出】，
#     于是我的问句子句一律打上 gold=影视；
#   · 参考集里 fan_knowledge_agent 出现 273 次，标【教育】255（93%）、【音乐】18。
# 这是「标注口径 ≠ 目录口径」，评测 judge() 只比工具与参数、不看我写的域，但域会
# 决定 gold 列与搭档兼容性，写错就整批错。故按【参考集实测主导金标】覆盖。
# 口径：该工具在参考集里的众数金标（n≥5 才覆盖，小样本不猜）。实测只有这一个工具
# 需要覆盖——educ_search/vod_search/educ_relate_recommend/sports_match_data_search
# 的 catalog 域（经 TOOL_ALIAS 归一后）与参考集主导金标 100% 一致。
_GOLD_OVERRIDE = {"fan_knowledge_agent": "education"}


def load_catalog_tools() -> dict[str, str]:
    """tool 名 → 权威域（catalog.full.json）。用于把旧工具名归一 + 校验工具存在。"""
    tools = json.load(open(CATALOG, encoding="utf-8"))["tools"]
    dom = {t["tool"]: t["domain"] for t in tools}
    dom.update(_GOLD_OVERRIDE)          # 覆盖「目录域 ≠ 参考集主导金标」的工具
    return dom


# ---------------------------------------------------------------------------
# 内容子句的「句式换框」：all_eval 里的内容句平均被复用 4.4 次，且句式高度一致
# （"播放X""我想听X""帮我找X"）。这里在不改动金标参数的前提下替换句首框架词，
# 使同一内容以不同说法出现。
#
# 关键约束：params.action 决定语义（play=起播 / search=查找），只能在同族内替换，
# 不能把 "搜索X"(search) 改写成 "播放X"(play)，否则 query 与金标不一致。
# ---------------------------------------------------------------------------
# 句首框架 = 可选「请求语」 + 可选「动词」。必须整体吃掉，否则 「给我播放X」 只吃掉
# 「给我」，剩余部分仍以「播放」开头，换框后会变成「给我放播放X」这种双动词残句。
_REQUEST = r"(?:请你|请|帮我|给我|麻烦|我要|我想|我很想)"
# 注意：交替项按「长的在前」排列。若短式排在前面（如 `搜` 在 `搜一下` 前），
# 「搜一下X」会被切成「搜」+「一下X」，换框后得到「搜一下一下X」。
_VERB = (r"(?:搜一下|放一下|听一下|看一下|查一下|唱一下|播一下|学一下|找一下|挑一下|看下|"
         r"来一首|来点|搜索|查找|播放|推荐|学习|放个|"
         r"播|放|听|看|唱|搜|找|查|来|打开|进入|学|用|挑)")
_FRAME = re.compile(rf"^(?:{_REQUEST})?(?:{_VERB})?(?P<rest>.*)$")
_HAS_FRAME = re.compile(rf"^(?:{_REQUEST})?(?:{_VERB})")

# 「看」类域（视频/影视/少儿/体育/教育）与「听」类域（有声/音乐）动词不能混：
# "我想听鲨鱼的电影" 是错的。故框架词按域分流。
_WATCH_DOMAINS = {"vod", "children", "sports"}
_LISTEN_DOMAINS = {"audio", "music"}
_PLAY_WATCH = ["播放", "我要看", "给我放", "放个", "我想看", "给我播放", "看"]
_PLAY_LISTEN = ["播放", "我想听", "给我放", "放个", "我要听", "给我听", "听"]
# 教育域：「我想学X」换成「我要看X」会改变意图（学→看），故只用中性起播词，不引入看/听。
_PLAY_EDU = ["播放", "给我放", "放个", "打开", "进入"]

# 【换框词的参考集句首频次】——第二十三轮。frame_variants 每轮把每个换框词都当
# 【等权】变体产出，可参考集里句首词根本不是等权的：实测（n=10000，最长匹配）
# 播放 931.5 / 打开 837.5 / 进入 241 / 我要看 149 / 我想看 120 / 我想听 117 /
# 我要听 58 / 找 50 / 搜索 44 / 看 36.5 / 听 40.5 / 帮我找 22 / 给我放 12.5 /
# 查看一下 11.5 / 找一下 10.5 / 帮我搜 9 / 查找 9 / 看下 5 / 给我听 5 /
# 给我播放 4 / 搜一下 4 / 放个 3.5 / 搜 3.5 / 查看 3 / 查 1.5（每 5k 减半）。
# 等权产出把低频档（给我放/给我听/看下/找一下）抬到与主力档同频：实测 MINE 5k
# 给我放 26 vs 参考 12.5、给我听 20 vs 5、看 53 vs 36.5、听 55 vs 40.5、给我播放 9 vs 4。
# 这里给每个换框词一个【相对权重】= 参考频次 / 该表均值，用于 _extra 的变体配权。
# 逐表归一（除以该表均值）→ 各表变体总质量近似不变（∑rel ≈ 表长），只是把质量
# 按参考频次重新分配。参考集里查无此词的（不该出现）给一个很小的兜底值。
_FRAME_REF_FREQ = {
    "播放": 931.5, "打开": 837.5, "进入": 241.0, "我要看": 149.0, "我想看": 120.0,
    "我想听": 117.0, "我要听": 58.0, "找": 50.0, "搜索": 44.0, "听": 40.5, "看": 36.5,
    "帮我找": 22.0, "给我放": 12.5, "查看一下": 11.5, "找一下": 10.5, "帮我搜": 9.0,
    "查找": 9.0, "看下": 5.0, "给我听": 5.0, "给我播放": 4.0, "搜一下": 4.0,
    "放个": 3.5, "搜": 3.5, "查看": 3.0, "查": 1.5,
}


def _swap_weights(swaps: list[str]) -> list[float]:
    """换框词 → 相对权重（参考频次 / 本表均值），使 ∑rel ≈ len(swaps)（总质量守恒）。"""
    fs = [_FRAME_REF_FREQ.get(w, 1.0) for w in swaps]
    mean = sum(fs) / len(fs)
    return [f / mean for f in fs]
# 检索框换词表。REF 实测口径（n=10000）：找100 搜索88 帮我找44 查看一下23 找一下21
# 帮我搜18 查找18 看下10 搜一下8 搜7。原表混进了 `查一下`——REF 全卷 **0 次**
# （真人是 `查看一下`/`查找`/`查`），是自造词；实测 MINE 5k 产 `查一下` 9 次、REF 0。
# 改用参考集里真实存在的 `查看一下`。注意本表【长度即变体质量】——frame_variants 对
# 每条目产一个等权变体（权重 lw*VAR_W），故条目数必须保持不变（原 8 条），
# 只换词不扩表，否则变体总量随之放大。两个 `搜`系（搜一下/搜）与 `查` 是参考集的低频档，
# 保留但不再与主力档同权（本表无序、等权，故低频词混入即被抬频——这里靠少列入来控制）。
_SEARCH_SWAP = ["找", "搜索", "帮我找", "查看一下", "找一下", "帮我搜", "查找", "看下"]
# 种子句尾的体态后缀（"…来听""…免费听""…听听"）：换框后与新句首动词重复，需剥掉。
# 只收「来/免费/收听/听听」这类无歧义后缀，「听」单字可能与标题同名，不碰。
_TAIL_MOD = re.compile(r"(?:来听|来看|免费听|免费看|收听|听听|看看)$")
# 「续播/历史」类句子的框架词不能换（"继续播放我昨晚听的" 换成 "搜一下…" 会改变语义）
_NO_SWAP = re.compile(r"^(继续|接着|打开刚才|查看|看下|进入|换成)")
# 平台名后还挂着一个动词的种子（"打开爱奇艺播放逐玉""用QQ音乐播放折子戏"）：
# 剥掉句首框架后 rest = "爱奇艺播放逐玉"，rest 的【首字】不是动词，于是换框会拼出
# "播放爱奇艺播放逐玉" 这种三动词残句。参考集 10000 行里此式 0 例（平台名一律写成
# 「动词+平台+内容」或「在平台里面+动词」）。故 rest 里【平台名后紧跟动词】时拒绝换框。
_PLATFORM = (r"(?:爱奇艺|腾讯视频|腾讯|优酷视频|优酷|芒果TV|芒果|哔哩哔哩|B站|西瓜视频|"
             r"抖音|咪咕视频|咪咕|网易云音乐|网易云|QQ音乐|QQ|酷狗|酷我|喜马拉雅)")
# 平台名之后（可隔「里面/里/上」等方位词）紧跟起播动词 = 句中已有动词，不能再换框。
_PLAT_MID_VERB = re.compile(rf"{_PLATFORM}(?:里面|里|上|中)?(?:{_VERB})")

# 句首框架的动词表与请求语表（_REQUEST / _VERB）保留：换框变体（frame_variants）
# 与问句族切分仍要用。早先还有一张 _FRAME_W「动词→参考集占比」表用于句内配权，
# 现已删除——内容子句的权重统一为【该子句自己的参考集频次】，框架占比由对 lw
# 求和自然给出（见 pick_content 的说明），那张表成了重复计算。


def frame_variants(q: str, params: dict, dom: str) -> list[tuple[str, float]]:
    """同 action 族内替换句首框架词，返回 [(变体, 相对权重)]（不含原句）。参数不变，金标不变。

    权重见 _swap_weights：按参考集【句首词实测频次】配权，而非等权。等权会把
    给我放/给我听/看下 这类低频档抬到与播放/找 同频（实测 MINE 给我听 20 vs 参考 5）。
    """
    if not isinstance(params, dict):
        return []
    if _NO_SWAP.match(q) or not _HAS_FRAME.match(q):
        return []
    act = params.get("action")
    if act == "play":
        if dom in _LISTEN_DOMAINS:
            swaps = _PLAY_LISTEN
        elif dom == "education":
            swaps = _PLAY_EDU
        else:
            swaps = _PLAY_WATCH
    elif act == "search":
        swaps = _SEARCH_SWAP
    else:
        return []          # 无 action 语义，换框架易失真，不动
    rest = _FRAME.match(q).group("rest")
    # rest 必须以内容起头（若仍以动词起头，说明原句是「我想 再听…」这类嵌套，换框会出双动词）
    if not rest or re.match(_VERB, rest):
        return []
    # rest 里「平台名后紧跟动词」时句中已有动词（"打开爱奇艺播放逐玉" → rest="爱奇艺播放逐玉"），
    # 换框会拼出 "播放爱奇艺播放逐玉" 三动词残句（参考集 0 例）。拒绝换框，原句照用。
    if _PLAT_MID_VERB.search(rest):
        return []
    # 种子常把结尾礼貌语黏在标题上（"请播黄河在咆哮谢谢" → rest="黄河在咆哮谢谢"），
    # 换框后「来一个黄河在咆哮谢谢」会让人以为"谢谢"是标题一部分。补逗号断开。
    rest = re.sub(r"^(.*?)(谢谢|多谢|麻烦了)$", r"\1，\2", rest)
    # 有些种子在句尾也挂了体态词（"我想找紫金陈的文学作品来听"）。换框后头部已有动词，
    # 尾部再留「来听/听听」就成了 "听一下…来听"。尾部体态词与头部重复时删掉。
    rest = _TAIL_MOD.sub("", rest)
    if not rest:
        return []
    # 变体不能以「…一下一下」「放个一个」等叠词收尾，且换框词与 rest 首词不能同类叠加
    _ws = _swap_weights(swaps)
    out: list[tuple[str, float]] = []
    for s, _w in zip(swaps, _ws):
        v = s + rest
        if v == q or re.search(r"(一下|一个|个|首|段|点)\1", v):
            continue
        # rest 已自带量词开头（"一下儿童故事…"）时，换框词不能再带量词，否则「放个一下…」
        if re.match(r"^(一下|个|首|段|点|下)", rest) and re.search(
                r"(一下|个|首|段|点)$", s):
            continue
        # 换框词后直接接光杆「个」（"播放个格林童话""我要听个X"）不顺口；
        # 但「个人/个别…」是词首，不在此列。
        if re.match(rf"^(?:{_REQUEST})?(?:播放|播|放|听|看|唱|搜|找|查)个(?!人|别)", v):
            continue
        out.append((v, _w))
    return out


def load_seeds(min_params: bool, dom_of: dict[str, str]) -> dict[str, list[dict]]:
    """按域收集可用金标子句。min_params=False 允许空参工具（fuzzy/recommend 等）。

    域以 catalog 为准（all_eval 的 domain 列只是标注习惯，工具才是真身）；
    工具名按 TOOL_ALIAS 归一到目录现名，不在目录里的丢弃。
    """
    by = collections.defaultdict(list)
    seen = set()
    with open(SEED_CSV, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            q = clean(r["query"])
            # 个别种子把礼貌语黏在内容上（"请播黄河在咆哮谢谢"，金标 title 仍是"黄河在咆哮"）。
            # 若剥掉它后 query 的其余部分仍与金标一致，就断开成独立小句，免得参数看着像含"谢谢"。
            m = re.match(r"^(.*?)[，,]?(谢谢|多谢|麻烦了)$", q)
            if m and m.group(1).strip():
                q = m.group(1).strip().rstrip("，,")
            # 个别种子句首动词叠写（"请播放打开QQ音乐大碗宽面"），读着像病句。保留后一个。
            q = re.sub(r"^(请|帮我|给我)?(?:播放|播)(打开|进入|搜索|查找)", r"\1\2", q)
            # 有 9 条内容搜索种子误用了设备动词起头（"调到恐怖惊悚类有声书音频" 的金标是
            # audio_search，不是 numeric_adjust）。这是标注时的笔误，照抄进多意图会与前半句
            # 语义打架。按域改回内容动词。
            tool_pre = TOOL_ALIAS.get((r["tool"] or "").strip(), (r["tool"] or "").strip())
            if dom_of.get(tool_pre) not in (None, "device") and re.match(
                    r"^(调到|调成|切到|换到)", q):
                rest_ = re.sub(r"^(调到|调成|切到|换到)", "", q)
                q = ("我找" + rest_) if rest_ else q
            # 「听一会孙子兵法」：种子把「听一会」写成了「听一下」的变体，但「一会」在此
            # 是误用（应为「一会儿」或直接「听」）。归一成「听一下」。
            q = re.sub(r"^(听|看|放)一会", r"\1一下", q)
            if not q or BAD_Q.match(q) or ASK_Q.search(q) or REF_Q.search(q):
                continue
            if not is_full_sentence(q):     # 残句/能力项标签/名词碎片，剔除
                continue
            if len(q) < 5 or len(q) > 26:
                continue
            tool = TOOL_ALIAS.get((r["tool"] or "").strip(), (r["tool"] or "").strip())
            if not tool:
                continue
            # 域以 catalog 为准；工具不在目录 → 丢
            dom = dom_of.get(tool)
            if dom is None:
                continue
            if dom == "children":
                dom = "children"
            if dom not in CN:
                continue
            try:
                pm = json.loads(r["tool_param"]) if (r["tool_param"] or "").strip() else {}
            except json.JSONDecodeError:
                continue
            if not isinstance(pm, dict):
                continue
            if min_params and not pm:
                continue
            # 相对时间词会与「当前时间」行冲突，跳过（同 gen_train 口径）
            if re.search(r"今天|明天|昨天|最近|这周|下周|今年|上午|下午|晚上|凌晨", q):
                continue
            key = (dom, tool, q)
            if key in seen:
                continue
            seen.add(key)
            # 参数去掉系统代填字段（retext/device/location/date_time 保留原样但清空默认）
            pm = {k: v for k, v in pm.items() if k not in ("retext",)}
            by[dom].append({"q": q, "d": dom, "tool": tool, "params": pm})
    return by


# 播控类工具：需要「已有内容在播」才有意义（上一集/快进/跳转/倍速/暂停…），
# 不能与纯检索、预约、关机类搭配，否则语义矛盾（"预约球赛，完了快进5分钟"）。
PLAYBACK_TOOLS = {"playback_control"}
# 允许在同一条 query 内重复出现的工具（其余一律不许，见 compatible()）
_REPEATABLE = {"numeric_adjust", "playback_control", "educ_search_all"}
# 明确会起播的内容工具（action=play 或本身即播放语义）
PLAY_TOOLS = {"music_song_search", "music_song_mv_search", "music_song_qqmusic_search",
              "music_ksong_search", "music_song_recommend", "music_song_semantic_search",
              "music_song_favorite_search", "music_song_history", "tvchannel_music_search",
              "audio_history", "sports_match_reservation"}


def provides_playback(s: dict) -> bool:
    """该子句是否会让电视进入「播放中」状态（播控类子句的前置条件）。

    检索类（search）不一定起播，不算；但「播放/我想听/放个」等明确起播语则不排除。
    """
    if s["d"] == "device":
        return False
    p = s.get("params") or {}
    if isinstance(p, dict):
        if p.get("action") == "play":
            return True
        if p.get("action") == "search":
            return False
    # 句子本身含明确起播动词 → 视为会起播。
    # 注意：裸露的「放/播/听/看」也计（"放卡通电影""放我收藏的金曲"都是起播），否则会漏判。
    if re.search(r"[播放听看]|来一首|来一段|进入|打开|给我唱|我想唱|我要唱", s["q"]):
        return True
    # 模糊检索类工具（无参数）通常是「找/看」内容，保守视为可能起播
    return s["tool"] in PLAY_TOOLS


# 并行中会互相打架的动作（暂停/退出/下一首）：并行句里不加，串行里要有起播前置
# 并行中语义空转 / 易歧义的动作：暂停、退出、上下切换（"播放A，顺便换一首"自相矛盾）
DULL_LIKE = re.compile(r"^(暂停|不听了|退出播放|别放了|下一首|上一首|换一首|下一集|上一集|"
                       r"下一部|换一部|换一个|重播|再听一遍|继续播放|接着放)$")


# 全局动作类设备子句：关机/开机/重启/定时关机——只能作为整句的最后一步，
# 且不与其他动作「并行」（"帮我打开物理书，另外关机"这种并列很怪）。
GLOBAL_LIKE = {"power_control", "timer_control"}
# 「开关机/重启」类：与 timer_control 不同，它们【可以】出现在起播步之前
# （"关掉电视随后播放第1集"，参考集 443 行首位）。见 order_ok 的说明。
POWER_TOOLS = {"power_control"}


def is_global_like(s: dict) -> bool:
    return s["d"] == "device" and s["tool"] in GLOBAL_LIKE


# 光杆变量子句："亮度""音量""声音"（无动词，只有名词）。参考集 69 次里 44 次（64%）
# 紧跟在【同样是调节动作】的子句之后（"音量调高顺便亮度""声音调小再亮度"）——是"顺带把
# 另一个维度也捎上"的省略说法。而我把它当独立子句随机摆位，于是在【内容子句】后面也挂
# 一个（"播放免费的赛罗奥特曼顺便亮度""放歌，亮度呢"），成了答非所问的残句。
# 参考集里这种「前面是内容句、后面跟光杆变量」的组合【一次都没有】（0/69）。
_BARE_VAR = re.compile(r"^(把|将)?(亮度|音量|声音)$")
# 「停播」与「换轨」互斥：同一句里既说"别再放了/不听了/停止播放"又说"下一首/上一集"，
# 语义打架（参考集 0 例）。也无「停播+重新起播新内容」的组合（参考集 0 例）。
# 【2026-11-11 扩表】把「暂停」与「跳转/快进」也纳进来：参考集 10000 行里
# 暂停类子句与换轨/寻址类子句同现 0 次（"暂停播放一起快进5分钟""暂停顺便放下一首"
# 都是我先前的产出）。暂停是"停住不动"，快进/下一首是"往前走"，同一句里拧着。
_STOP_PB = re.compile(r"(停止播放|别放了|别再放|不听了|不看了|别唱了|停下来|退出播放|"
                      r"关掉声音|关闭声音|关声音|不放了|暂停|停一下)")
_SWITCH_PB = re.compile(r"(下一首|下一集|下一部|下一个|下一曲|下一期|"
                        r"上一首|上一集|上一部|上一个|上一曲|上一期|"
                        r"换一首|换下一|换一个|回到上一|切到下一|切到上一|倒回上一|"
                        r"快进|快退|跳到|切下一|切上一|往后切|往前切|下一[步个])")
# 「打开/进入静音」与「调音量」互斥：静音是要把声音关掉的，同一句里再调音量自相矛盾
# （"把声音调到最大接着打开静音""音量开到十一起打开静音"）。参考集 10000 行 0 例——
# 参考集里静音与音量句同现的 17 行【全部】是「取消静音 + 音量调大」（"取消静音顺便
# 大点声音"），一次反例都没有。我原先把静音当普通 numeric_adjust 说法随机配对，
# 产出 5 行矛盾句。判据只看「打开静音」一侧，不碰「取消静音」。
_MUTE_ON = re.compile(r"(打开静音|开启静音|进入静音|开静音|调到静音|静音模式|静音开启|"
                      r"静音功能|声音静音|静音设置|静音管理|静音控制|静音开关|"
                      r"关闭声音|关掉声音|关声音|把声音关掉|声音关掉|关上声音|别出声)")
_MUTE_OFF = re.compile(r"(取消静音|退出静音|关闭静音|关掉静音|关静音|解除静音|恢复声音|打开声音)")
# 音量/响度「调节」子句（不含静音自身）。只在【设备子句】上判，避免内容句
# "搜索…有声音频"（"声音"是"有声"的一部分）被误判成音量子句。
_VOL_ADJ = re.compile(r"(音量|声音|响度|大点|小点|大声|小声|大些|小些|开大|调大|调小|"
                      r"调高|调低|加大|减小|增大|调低|提高|降低)")
# 「音量调大」方向（用于"取消静音 + 调大"这条参考集 17/17 的合法搭配）。
_VOL_UP = re.compile(r"(大|高|开|加|提高|增|升|最大|最高)")
# 「调节动作」判据：被光杆变量省略掉的那个动词。参考集 69 条光杆变量里 69 条都紧挨着
# 这样一条子句（"音量关小一起亮度""声音调高顺便亮度"），无一例外。
# 【2026-11-10 扩表】：原来只收「调开大提高降低」这类动词，把我的短语库里一整档
# 【裸加减/数值】说法漏判了——"声音减1""亮度减50""对比度数值30""开大些声音"这些
# 与参考集 "声音减小""音量降低" 同义，却因正则不认而被判成"光杆变量没搭档"，
# 整行作废。实测 numeric_adjust 池里 30 条这样被误杀，正是光杆「亮度」参考 69→我 9、
# 「小点」18→3 的直接原因（抽到后 80% 被拒，渴死在拒绝循环里）。
_ADJ_VERB = re.compile(r"(调|开大|开小|开高|开低|大点|小点|大些|小些|小点声|大点声|大声点|小声点|"
                       r"高一点|低一点|提高|降低|"
                       r"增大|减小|加大|减少|减到|加到|"
                       r"[加减][0-9一二三四五六七八九十]+|数值|"
                       r"关小|关大|调高|调低|调大|调小|"
                       r"放到|设到|设为|设置|加|减|升|降)")


def compatible(a: dict, b: dict, mode: str) -> bool:
    if a["tool"] == b["tool"] and a["q"] == b["q"]:
        return False
    # 全局动作类（关机/定时/重启）：不能两条同时出现（"关机，再定时30分钟关机" 自相矛盾），
    # 但跟别的动作并列完全正常——参考集里 power_control+numeric_adjust 有 181 行、
    # 关机搭 common_control 有 116 行。原来一律要求"串行且放最后"，把 power_control
    # 压到 3.3%（参考 7.0%）。顺序问题改由 order_ok 统一把全局动作挪到最后。
    if is_global_like(a) and is_global_like(b):
        return False
    if a["d"] == b["d"] == "device":
        # 同一设备工具在一句里重复出现，只有参考集里真实存在的三种才允许：
        #   numeric_adjust 12.5%（"屏幕亮度提高一起声音大些"）、playback_control 2.8%、
        #   educ_search_all 4.3%。其余设备工具（screen_layout / screensaver_control /
        #   common_control / mode_control / power_control …）在参考集里重复率一律 0%
        #   ——"打开分屏一起打开全屏""调出壁纸顺便开启壁画"这类自相矛盾的两步是我的生成器
        #   独有的（上一版 screen_layout 重复 24 行、screensaver 9 行、common 10 行）。
        if a["tool"] == b["tool"] and a["tool"] not in _REPEATABLE:
            return False
        return True
    if a["d"] == b["d"]:
        return False
    # 播控类（上一集/快进/暂停）【不再】要求内容搭档本身带起播动词。
    # 参考集里 playback 挂在内容句上的 CD/DC 行共 1021 条，其中 701 条（69%）的内容句
    # 本身【不带】起播动词（"火的综艺啊 别播了""六年级下册数学北师大版 切歌"
    # "男篮世界杯助攻榜 回到上一首"）——真人是在指着「我正在看/听的那个东西」说话，
    # 语境里其实已经在播，不必再显式说"播放"。我原先一律拒掉这类，是单槽 playback
    # 只有参考一半（9.4% vs 16.4%）的主因。故放宽：播控与任何内容句都可搭配
    # （设备+设备本就允许，见上面注释里的 773 行"下一首一起声音减"）。
    return True


# 文本层「起播意图」：与 params.action 无关，只看用户怎么说（"我要看X"即使金标 search 也算起播）。
# 锚定到子句开头（剥掉礼貌词后），避免"播放次数多的综艺"（播放=热度）这类名词性误判。
_LEAD_STRIP = re.compile(r"^(那个|哎|小聚|小维|帮我|给我|请|麻烦|能不能|可以|我是?|我想要?)+")
_PLAY_HEAD = re.compile(r"^(播放|播|放|听|看|唱|来一首|来一段|来点|我要看|我想听|我要听|"
                        r"开始播放|继续播放|给我放|给我播)")


def looks_like_play(q: str) -> bool:
    s = _LEAD_STRIP.sub("", q)
    s = re.sub(r"^(要|想|让我|要听|要看)", "", s)
    return bool(_PLAY_HEAD.match(s))


# power 定序专用：与 looks_like_play 同源但**多含 `打开`**。
# 为什么不直接改 looks_like_play 加 `打开`：它被 `order_ok` 的"一行最多一个起播动作"
# 规则消费，动它会连带改变那条规则的边界（"打开纪录片"会被算作第二个起播动作而整行作废）。
# 这里只需要判断"内容句是不是让电视演点什么"，`打开音乐/打开电影长安的荔枝` 显然属于
# 播放侧（参考集里 power 在它前面起头的比例 0.54，与 `播放X` 同档）。
_IS_PLAY_HEAD = re.compile(r"^(播放|播|放|听|看|唱|打开|来一首|来一段|来点|我要看|我想听|"
                           r"我要听|开始播放|继续播放|给我放|给我播)")


def _is_play(q: str) -> bool:
    s = _LEAD_STRIP.sub("", q)
    s = re.sub(r"^(要|想|让我|要听|要看)", "", s)
    return bool(_IS_PLAY_HEAD.match(s))




# 设备工具 → (先于起播内容句的次数, 作为首位设备句的总次数)，懒加载自
# ref_device_vocab.lead_dev_before_play（模块级，因为 order_ok 是模块级函数）。
# 见 order_ok 里那段说明：语序要按参考集实测，不能自造"非 power 一律禁止"。
_LEAD_DEV_PLAY: dict[str, tuple[int, int]] = {}


def _load_lead_dev_play(pb) -> dict[str, tuple[int, int]]:
    global _LEAD_DEV_PLAY
    if not _LEAD_DEV_PLAY:
        try:
            from ref_device_vocab import lead_dev_before_play as _ldbp
            _LEAD_DEV_PLAY = _ldbp(provides_playback=pb)
        except Exception as _e:
            print(f"[lead-dev-before-play] 不可用（{_e}）")
    return _LEAD_DEV_PLAY


def order_ok(segs: list[dict], mode: str, from_reg: bool = False) -> bool:
    """整句层面的顺序/搭配约束。

    from_reg=True 表示本行是【逐字照搬 REF 的联合语域】（DC 或 CD 跨标），语序与工具
    都是 REF 原文，位置类启发式不得再对它二次裁决——否则会像 2026-11-11 那样，把
    DC 语域里 20 条 `timer_control` 起头的行（REF 原文 `关闭电视接着播放睡衣小电视`）
    整族拒掉：`播放睡衣小电视` 在 REF 有 20 行、全由 timer 起头的对承载，而我产出 0。
    """
    # 定时类（timer_control）必须位于最后一步：参考集 281 行无一例外（"45分钟后关机"
    # 只能是最后说的事）。关机/开机/重启（power_control）【不】受此限——参考集里它
    # 首位 443 行、末位 497 行，几乎是抛硬币（"关掉电视随后播放第1集""关机再放一首歌听"
    # 这类"先关掉再换一集"的串行说法很常见）。原先把两者一并钉死在末位，导致
    # power_control 作为首子句的占比从参考的 47% 掉到 0%，详见 main() 里的换序逻辑。
    # 【例外】语域行不查：REF 原文里确有 timer 起头的 DC 行（见上方 docstring）。
    if not from_reg:
        for i, s in enumerate(segs):
            if is_global_like(s) and s["tool"] == "timer_control" and i != len(segs) - 1:
                return False
    # 关机/定时只能出现在【串行】里。参考集 5000 行并行中 power_control/timer_control
    # 出现 0 次（1221 条全部落在串行），因为并行语义是"两件事同时发生"——
    # "关电视+同时调音量"本身自相矛盾：电视都关了，调音量给谁看。
    # 我原来按工具占比抽设备子句时不区分模式，产出了 116 行并行关机/定时。
    if mode == "parallel" and any(is_global_like(s) for s in segs):
        return False
    # 「先X」只能在首位（"先返回首页然后关闭电视"）；参考集 26 个含「先」的子句里
    # 只有 4 个不在首位，而我的库是从参考集收割来的「先返回首页」等短语，被随机摆进
    # 第二位后读起来像"四十然后先返回首页"这种时间颠倒。
    for i, s in enumerate(segs):
        if i > 0 and s["q"].startswith("先"):
            return False
    # 顺承副词（接着/再/随后/然后/接下来/之后就）起头的子句【不能进并行】。
    # 参考集 5000 行并行里以顺承副词起头的子句 0 条（4 条全在串行："关闭电视再播放视频"），
    # 因为「接着播放」这类词本身已声明先后，与并行的"同时发生"语义冲突。
    # 我的短语库里有「接着播放」「接着播放视频」，被随机摆进并行后拼出
    # "全屏播放同时接着播放"这类自相矛盾的句子。这里直接拒掉整个并行组合。
    if mode == "parallel" and any(s["q"].startswith(_SEQ_ADV) for s in segs):
        return False
    # 「先X」子句同样不能进并行：参考集 5000 行并行里以「先」起头的子句 0 条
    # （26 条全在串行）。"先返回主页一起快进三分钟"——并行里说"先"本身就矛盾。
    if mode == "parallel" and any(s["q"].startswith("先") for s in segs):
        return False
    # 「别放了/停止播放/不听了」与「下一首/上一集/换一首」互斥：一句里既叫停又切轨，
    # 语义打架（"别再放了，下一集""切到下一首顺便不听了关掉"）。参考集 10000 行 0 例。
    _whole = "#".join(s["q"] for s in segs)
    if _STOP_PB.search(_whole) and _SWITCH_PB.search(_whole):
        return False
    # 「打开/进入静音」与「调音量」互斥（参考集 0 例，见 _MUTE_ON 的说明）。
    # 只拦"开静音"一侧；"取消静音 + 音量调大"是参考集 17/17 的合法搭配，不能误伤——
    # 故先把「取消静音」类子句从音量判据里排除。
    _muteon = [s for s in segs if _MUTE_ON.search(s["q"]) and not _MUTE_OFF.search(s["q"])]
    if _muteon:
        _vols = [s for s in segs
                 if s["d"] == "device" and not _MUTE_ON.search(s["q"]) and not _MUTE_OFF.search(s["q"])]
        if any(_VOL_ADJ.search(v["q"]) for v in _vols):
            return False
    # 「退出静音/取消静音」也是同一个道理的另一半：参考集里静音句与音量句同现的 17 行
    # 【全部】是「取消静音 + 音量调大」。说"取消静音"却同时把音量调小（"取消静音一起
    # 音量减到五""请声音调小再退出静音"）在参考集 0 例——取消静音之后又调小，语义拧着。
    _unmute = [s for s in segs if _MUTE_OFF.search(s["q"])]
    if _unmute:
        _vols = [s for s in segs
                 if s["d"] == "device" and not _MUTE_ON.search(s["q"]) and not _MUTE_OFF.search(s["q"])
                 and _VOL_ADJ.search(s["q"])]
        if any(not _VOL_UP.search(v["q"]) for v in _vols):
            return False
    # 光杆变量子句（"亮度""音量""声音"，无动词）必须紧挨着一条「调节动作」子句——
    # 参考集 69/69 如此（"音量关小一起亮度""声音调高顺便亮度"）。否则就是"播放免费的
    # 赛罗奥特曼顺便亮度"这类答非所问的残句（参考集 0 例）。相邻即可，前后都算。
    for i, s in enumerate(segs):
        if _BARE_VAR.match(s["q"].strip()):
            nb = [x for j, x in enumerate(segs) if j != i]
            # 判据用【搭档工具】而非文字正则：参考集 69 条光杆变量，搭档工具 69/69 全是
            # numeric_adjust。文字正则无论如何补都会漏（"声音减1""对比度数值30""大声点"
            # 与参考 "声音减小" 同义），一漏就整行作废。改判工具后不再依赖措辞。
            if not any(x["tool"] == "numeric_adjust" for x in nb):
                return False
    # 光杆「小点/大点/小点声/大点声」（省略了目标）必须紧挨着一条【带音量/声音对象】的子句——
    # 参考集 18 条裸「小点」全部如此（"小点，同时声音打开全屏播放""小点一起音量大屏播放"），
    # 0 条例外。否则就是"帮我调亮度，小点"这类指代不明的残句（该句讲的是亮度、不是音量）。
    for i, s in enumerate(segs):
        if s["q"].strip() in ("小点", "大点"):
            others = [x["q"] for j, x in enumerate(segs) if j != i]
            if not any(("音量" in o or "声音" in o) for o in others):
                return False
    # 「降厄」是参考集里一个只会成对出现的说法：25 条【全部】紧挨着「亮度降到10」
    # （"声音降厄顺便亮度降到10""音量降厄一起亮度降到十"），单独出现 0 条。
    # "音量降厄接下来关机"读起来是残句。沿用参考集的搭配约束：必须配一条亮度下调子句。
    for i, s in enumerate(segs):
        if "降厄" in s["q"]:
            others = [x["q"] for j, x in enumerate(segs) if j != i]
            if not any("亮度" in o for o in others):
                return False
    # 串行：设备控制步【多数】要排在内容起播步之后（"帮我调音量，随后播放…"倒置不自然），
    # 但"哪些设备工具能当起头句、先于一条起播内容句"是【按工具的位置条件性】，不是
    # "非 power 一律禁止"。旧规则（非 power 设备句不得先于起播句）是我自造的，
    # 对参考集自己的 5000 行串行实测【误拒 146 行，全部是 DC 形态】——REF 真的写
    # "关闭屏幕随后播放音乐""退出广告看西游记""大屏播放第1集吧""大点声然后最强大脑"。
    # 正确口径（ref_device_vocab.lead_dev_before_play，n=参考集实测）：
    #   power 0.316、mode 0.356、common 0.047、playback 0.028、screen_layout 0.017、
    #   screen_lift_rotation 1.000（仅 6 例）、numeric 0.000、source_switch 0.000…
    # 即：只有【实测从不】先于起播句的设备工具（numeric/source_switch/…）才拒绝；
    # 有先于起播句实例的工具一律放行。见 [[lw-is-frequency-no-frame-layer]] 同源的教训：
    # 语序事实要从参考集量，不能从直觉编。
    if mode == "serial":
        first_play = next((i for i, s in enumerate(segs) if provides_playback(s)), None)
        if first_play is not None:
            for s in segs[:first_play]:
                if s["d"] != "device":
                    continue
                _ev = _LEAD_DEV_PLAY.get(s["tool"])
                if _ev is None:
                    continue               # 该工具无首位设备句样本 → 不拒绝
                if _ev[0] == 0:
                    return False           # 实测从不先于起播句 → 拒
    # 播控类子句必须前面已经出现过「会起播」的内容子句，或者干脆是设备内部的搭配
    # （"下一首一起声音减""暂停播放顺便关闭静音"——参考集里这类 D+D 有 773 行，
    #  播控挂在旁边的音量/静音/屏保上是自洽的，不预设另有内容子句）。
    # 之前只查 segs[:i]，导致播控位于首位时被一律拒（"下一首一起声音减""快进到三分钟同时音量大些"
    # 都是参考集里真实存在的行，播控在首位、搭档是另一条设备子句）。改为查除自己之外的所有子句，
    # 与 compatible() 同口径：只要还搭着一条设备子句，或另有一条会起播的内容子句，就成立。
    # （原「播控必须有起播前置」的规则已删：参考集里 69% 的 playback+内容行，内容句
    #   本身不带起播动词（"火的综艺啊 别播了"）。这与 compatible() 的口径一致化。）
    # （原「并行设备子句最多 1 条」的规则已删：参考集并行里 31.2% 是两条设备子句
    #   ——"打开护眼顺便利马亮度调到十二""快进到三分钟同时音量大些""全屏一起声音大1"
    #   共 1539 行。这条规则把并行 D+D 全灭了（我上一版并行 2 设备 = 0 行），
    #   也连带压低了 playback_control——参考集 683 条 playback 是挂在设备子句上的。）
    # 并行：播控类与内容子句并列，只有在该内容子句【本身就会起播】时才成立
    # （"播放百科故事顺便节奏太慢换一首"——前半句把内容放起来，后半句才是换歌的动作）。
    # 参考集这样的并行行有 633 条（影视112/教育112/音乐107/体育81/有声80/少儿66），
    # 而我原来一刀切地禁掉所有「并行播控+内容」，加上 compatible() 里同一条规则，
    # 把 playback_control 压到 6.2%（参考 12.7%）。这里改成与 compatible() 同口径：
    # 只禁止「播控 + 不会起播的纯检索内容句」。
    # 一条 query 里最多一个「内容起播」动作，否则像列清单（"播放A，还有播放B"）。
    # 这里按【句子文本】判断，不看 params.action——金标常标 action=search 但用户说的是"我要看"。
    # 【语域行豁免】：内容+内容语域（REF 42 行、"播放音乐>>我要看汪汪队"）两条都是起播，
    # 是 REF 原文，不得按此规则拒（否则 CC 语域产出恒为 0）。
    if not from_reg:
        play_text = [s for s in segs if s["d"] != "device" and looks_like_play(s["q"])]
        if len(play_text) > 1:
            return False
    # 同一设备对象不重复设置（"图像模式调成游戏模式，再图像模式切换到鲜艳模式"自相矛盾）
    dev_objs = [s["params"].get("object") for s in segs
                if s["d"] == "device" and isinstance(s.get("params"), dict)]
    dev_objs = [o for o in dev_objs if o]
    if len(dev_objs) != len(set(dev_objs)):
        return False
    # 同类幅值调整（提高后又降低）语义打架："声音小一点，完事再声音大一点"
    dev_ops = [(s["params"].get("object"), s["params"].get("operation"))
               for s in segs if s["d"] == "device" and isinstance(s.get("params"), dict)]
    ups = {o for o, op in dev_ops if op == "提高"}
    downs = {o for o, op in dev_ops if op == "降低"}
    if ups & downs:
        return False
    return True


def join_domains(segs: list[dict]) -> str:
    return "+".join(CN[s["d"]] for s in segs)


# 口语化引导：给首个子句加自然前缀（不改动子句本身的槽值，因而不影响金标）。
# 注意：子句若已以「我/帮/请/给」开头，不再叠加，避免「麻烦我要听…」这类重复。
LEAD = ["", "", "", "", "", "", "那个，", "哎，", "小聚，", "小维，"]
LEAD_VERB = ["帮我", "麻烦", "给我"]
# 连接词后的语气缓冲：只用不改变语序/语义的单字语气，且不与连接词本身语义重叠
# （"再"+缓冲会变成"完事再就放"，故缓冲只留"就"，且只用在与"再"不冲突的连接词后）
FILLER = ["", "", "", "", "", "", "", "", "就"]


# 动词引导只能加在以动词/意图词起头的子句前（否则"帮我彼得兔全屏播放"）
_VERB_START = re.compile(
    r"^(播放|播|听|看|放|搜|找|查|推荐|来|打开|关|调|设|换|切|跳|快进|快退|"
    r"收藏|预约|学|唱|启动|停止|继续|进入|恢复|取消|静音|开|要|想|把|将|"
    # 下面这些是剥掉「给我/帮我」后可能露出的动作词（"给我讲一下""给我暂停"
    # "给我再放一遍""给我用2倍速放""给我单曲循环""给我顺序播"）。
    r"讲|说|暂停|再|用|单曲|顺序|随机|帮忙|根据|写|点)")


def _lead_head(head: str, rng: random.Random, tool: str = "", dom: str = "") -> str:
    """给首子句加口语引导。真人会说「小维帮我…」「哎，麻烦…」，呼名与动词引导是可以叠加的。"""
    if re.match(r"^(我|帮我|请|给我|麻烦|能不能|可以)", head):
        # 子句自带人称/请求语，最多贴个语气起头，不再叠动词引导（同参考集口径：多数裸说）
        if rng.random() < 0.0025:          # 参考集唤醒词 25/10000 = 0.25%
            return rng.choice(WAKE) + head
        return head

    can_verb = bool(_VERB_START.match(head))
    # 先剥离已有的「我想/我要/帮我…」请求语，取出纯动词短语——很多附加成分要贴在动词上
    m_req = re.match(r"^(我想|我要|我很想|请|请你|帮我|给我|麻烦)+", head)
    req_, core = (m_req.group(0), head[m_req.end():]) if m_req else ("", head)
    # 请求软化（"能不能放X"）：只加在动词核前，不叠在已有请求语前。
    # 软化后必须标记，否则后面再叠 LEAD_VERB2 会得到「麻烦能不能来一个…」这类双重请求语。
    # 【2026-11-11 限定设备位】参考集 47 个 softening 子句【全部】是设备金标
    # （"能不能关闭屏保""能不能打开壁纸"），内容金标 0 —— 真人不会对"电视演什么"用
    # "能不能播放X"这种征询口吻（那是在自己家点播，不是在求别人）。我原来不限定域，
    # 产出「能不能播放仙侠电视剧」7 行、"能不能放猫和老鼠"。只在设备子句上软化。
    softened = False
    if (can_verb and dom == "device" and rng.random() < 0.006
            and not req_ and not core.startswith(SOFT_FLAT)):
        core = rng.choices(SOFTEN, weights=[43, 4])[0] + core     # 参考集 能不能43 : 可不可以4
        softened = True
    head = req_ + core
    r = rng.random()
    # 起头项逐一对齐 bench_serial_train_10k.csv（而不是凭"像人话"的直觉往里加）。
    # 参考集首字分布全是动词/请求语本身：播 1544、我 906、请 496、帮 492、给 469、关 387、
    # 打 380、麻 336、音 316、声 303……而下面这些我原来自造的起头，参考集里一概为 0：
    #   我现在/我这会儿/我待会/我刚想起来 0；哎呀/哎哟 0；对了/哦对了/哎对了/还有啊/
    #   顺便问下/对了顺便 0；，那个，/，就是，/，嗯，/，我看看， 0；哎/诶 0。
    # 唯一真实存在的是唤醒词「小聚」（25 行 / 0.25%，含"海信小聚"）。
    # 于是把所有自造档全部删掉，只留：裸句 + 0.25% 唤醒词 + 整句礼貌引导（见下）。
    #
    # ---------------------------------------------------------------------------
    # 整句礼貌引导（2026-11-10 补）——参考集 14.2% 的行在整句最前面加一个请求语
    # （请/帮我/给我/麻烦），而该请求语【不属于首子句本身】。证据：
    #   · 「麻烦」作为整句起头出现 333 次，可它出现在子句内部只有 3 次；
    #     「请」整句起头 356 次 vs 子句内 134 次；四个词合计「拼接时加的」1416
    #     vs「子句自带的」366（79% 是拼句时加的）。
    #   · 它只加在【裸起首子句】上（"声音""全屏""音量""关闭""返回""大屏""亮度"
    #     "国王杯射手榜""迈克尔杰克逊演唱会经典曲目"），首子句自带请求语时从不叠加。
    #   · 四个词近乎均分：给我 382 / 请 356 / 帮我 345 / 麻烦 333。
    #   · 加与不加强烈依赖【首子句的工具】：设备类 17-23%（numeric_adjust 0.229、
    #     power_control 0.226、playback_control 0.222、common_control 0.209），
    #     edu_search 0.229、sports_rank_search 0.260，而 vod_search_all 只有 0.063、
    #     audio_search 0.049、source_switch 0.000。整体 16.0%（1395/8723 裸起首行）。
    #
    # 我原来只在【动词起头】时以 4% 概率贴 帮我/给我/麻烦，既漏掉了裸名词起首、
    # 又漏掉了「请」，总量只有参考的 1/10——整卷因此通篇以裸祈使句开头，正是
    # 用户说的"生硬"。这里改为按首子句工具查表，起手词按参考集配比抽。
    # ---------------------------------------------------------------------------
    # 整句引导只贴在【裸祈使】起首。首句若已是【状态描述句】（"运动画面不清晰帮我调节一下"
    # "声音有杂音帮我调一下"——描述毛病、句内自带「帮我…」），再冠一个「给我/请」就成了
    # "给我运动画面不清晰帮我调节一下"，一句里叠两层请求语且语义错位。
    # 参考集这类「状态句作首句」10000 行里 0 例，故句内已含「帮我/给我」时一律不加整句引导。
    if not req_ and tool and "帮我" not in head and "给我" not in head:
        # 【2026-11-11 第十一轮】贴引导的概率与选词都【取决于首句动词类型】。
        # 原来只看工具、选词无条件均分，于是把「麻烦」也贴到播放句上：
        # 实测我「麻烦播放」15 行 vs REF 1 行，播放句贴引导 0.105 vs REF 0.053。
        # 首句动词三分（见 _LEAD_TOOLS_VERB）：播放/放/听/看/搜/查/找/打开 一律不贴；
        # 唱/推荐/学习 与设备句同档（0.21 左右）。
        _ht = _head_type(head)
        if _ht == "play":
            _p = _ATTACH_TOOL_FALLBACK["play"]            # 播放句几乎不贴（REF 6/2494）
        else:
            _p = _ATTACH_BY_TOOL_NONPLAY.get(tool, _ATTACH_TOOL_FALLBACK["nonplay"])
        if rng.random() < _p:
            return rng.choices(_LEAD4_BY_TYPE[_ht],
                               weights=_LEAD4_W_BY_TYPE[_ht])[0] + head
    if r < 0.9975:
        return head                                              # 裸句，绝对多数
    return rng.choice(WAKE) + head                               # 「小聚/海信小聚」贴头


def make_query(segs: list[dict], mode: str, rng: random.Random) -> str:
    """把子句拼成一句自然 query。

    真人说法并非「子句+连词+子句」的固定模板，因此这里按概率在三种形态间变化：
      1) 裸并列（只用一个逗号，不加连词）——占比不低，是最像随口说的形态；
      2) 连词承接（同一条内不重复，避免「还有就是把…还有就是把…」）；
      3) 连词 + 语气缓冲（"然后"+"就"）。
    子句尾部还可能带句末语气词（"吧/啊/呗/哦"），去掉也不影响金标。
    """
    pool = SERI if mode == "serial" else PARA

    def _stem(x: str) -> str:
        # 同义连接词归一到同一语素：顺便/顺带/对了顺便/顺便问下 视为同一个
        if "顺便" in x or "顺带" in x:
            return "顺便"
        if "还有" in x or "还有就是" in x:
            return "还有"
        for k in ("另外", "一起", "同时", "并且", "接下来"):
            if k in x:
                return k
        return x

    head = _lead_head(segs[0]["q"], rng, segs[0].get("tool", ""), segs[0].get("d", ""))
    # 句内若已重复出现「帮我」（如"帮我画面颜色太鲜艳了帮我调一下"），去掉句首引导
    if head.count("帮我") > 1 and head.startswith("帮我"):
        head = head[2:]
    # 「先…」框架：真人安排两步时常说「先看X，再调Y」。只加在首子句前，不改语序。
    # 「先」只能贴在动词前（"先播放X"）；若首句是「小维，播放…」则贴在动词上。
    # 「先…再…」框架参考集只占 0.16%（16/10000）；我给 10% 实测 3.1%，是参考的 20 倍。
    if mode == "serial" and rng.random() < 0.004 and "先" not in head:
        m_lead = re.match(r"^((?:小维|小聚|那个|哎|对了|哦对了|哎对了|还有啊|顺便问下|对了顺便|"
                          r"哎呀|哎哟|天|哟)[，,]?)?(.*)$", head)
        pre_, rest_ = m_lead.group(1) or "", m_lead.group(2)
        if _VERB_START.match(rest_):
            head = f"{pre_}先{rest_}"
    q = head
    used: set[str] = set()
    # 句首的话题开启词与连接词共用同一套语素，避免整句复读（"顺便问下…，对了顺便…"）。
    # 直接头部调用一次 _stem，无需枚举 PREFACE。
    used.add(_stem(head))
    avail = list(pool)
    rng.shuffle(avail)
    prev_conn = ""
    last = len(segs) - 1
    # 请求语省并：真人连续交代两件事时，第二次通常不再重复「帮我/给我」
    # （"帮我找A，帮我把B调成20" → "帮我找A，把B调成20"）。原样保留会让整句像两份
    # 独立指令粘在一起。句首用过的请求语记在 head_frame，后续子句同词则剥掉。
    m_hf = re.match(r"^(?:小维|小聚|那个|哎|哎呀|哎哟|天|哟|诶)?[，,]?"
                    r"(?:嗯，|对了，|哦对了，|哎对了，|还有啊，|顺便问下，|对了顺便，)?"
                    r"(帮我|给我|麻烦|请你|请|我想|我要|能不能|可不可以)", head)
    head_frame = m_hf.group(1) if m_hf else ""
    for i, s in enumerate(segs[1:], start=1):
        body = s["q"]
        # 「描述毛病」的子句不是序列的一步，前面不能接顺承词（"完事再人声听不清"很怪），
        # 只能裸并列或非顺承连接词。
        state_clause = s.get("tool") in STATE_TOOLS
        # 形态选择：连接词池已按参考集逐一对齐（并行只有 4 个词、串行 8 种含裸逗号与直接连写），
        # 不再另设「裸逗号」档——参考集并行 0% 用裸逗号，串行 13.7%，这两种已被池覆盖。
        if state_clause:
            # 并行症状句用并行池，串行症状句用【串行常规池】（参考集串行 63 行实测：
            # 接着11/接下来11/再9/随后8/裸逗号8/直接连写7/然后5/之后就4，「顺便」0）。
            c = rng.choice(STATE_CONN_PAR) if mode == "parallel" else \
                next((x for x in rng.sample(SERI, len(SERI)) if x not in used), "，")
            f = ""
            # 句首已含「顺便」（"对了顺便，"）时，状态子句的连接词不能再带「顺便」
            if any(_stem(x) == _stem(c) for x in used):
                c = "，同时" if mode == "parallel" else "，"
        else:
            c = next((x for x in avail if x not in used), rng.choice(pool))
            # 「顺便/顺带/对了顺便」语义重叠，同句出现两次像复读（"顺便…，对了顺便…"），
            # 故按【关键语素】去重，而不是按整串去重。
            if any(_stem(y) == _stem(c) for y in used):
                # 并行【绝不能】退化成裸逗号：参考集 5000 行并行 100% 都带并行连接词
                # （裸逗号 0 条）。换一个没用过的并行词；实在没有了才用逗号。
                if mode == "parallel":
                    c = next((x for x in PARA if not any(_stem(y) == _stem(x) for y in used)),
                             "，同时")
                else:
                    c = "，"
            used.add(c)
            # 独立的「就」缓冲取消了：参考集 10000 行里「就」只出现在「之后就」这一个
            # 连接词内部（601 行），从没有「同时就打开全屏」这类裸连接词+就的写法，
            # 也没有「音量就调到」这类话题+就的写法。凭空加「就」是最直白的非母语感来源。
            f = ""
        # 子句【自身以顺承副词开头】时，前面不再挂连接词，直接连写。
        # 参考集实测：seg 以 接着/再/随后/然后/接下来/之后就 起头的，前面【从不】有连接词
        # （4/4 全是直接连写："关闭电视再播放视频"）。对照 seg 以「继续」起头的 64 例，
        # 前面可以是任何连接词（同时继续18/接着继续9/随后继续5/…），所以只拦「顺承副词」
        # 这一档，不碰「继续」。我上一版只比首尾字，漏掉了「接下来」+「接着播放」
        # （末字「来」≠首字「接」），于是拼出「接下来接着播放」——参考集 10000 行为 0。
        if any(body.startswith(w) for w in _SEQ_ADV):
            c, f = "", ""
        # 连接词与子句首字重复时换一个（"接下来"+"来一个"→"接下来来一个"、"完事再"+"再听一遍"
        # →"完事再再听一遍"）：既比对整词，也比对末字与首字。无可换则退化为裸逗号。
        # 连接词现在有无逗号两种形态（"同时"/"，同时"），比对前先去掉可能的前导逗号。
        c_core = c[1:] if c.startswith("，") else c
        if c not in ("，", "") and c_core and (body.startswith(c_core) or body[0] == c_core[-1]):
            def _ok(x):
                xc = x[1:] if x.startswith("，") else x
                return (not body.startswith(xc)) and body[0] != xc[-1]
            alt = next((x for x in pool if x not in used and x != c and _ok(x)), None)
            c = alt if alt else "，"
        # 省并：本子句与句首用了同一个请求语时，剥掉（"给我调成70，给我放X" → "…，放X"）。
        # 真人连续交代两件事时第二次不再重复「帮我/给我」；原样保留像两份指令粘在一起。
        # 只剥与句首完全同形的，且保证剥完仍以动词起头（"给我放X" → "放X"）。
        # 必须放在「也」之前：剥掉请求语后 _VERB_START 才成立。
        # 参考集两处请求语的行只有 0.18%（18/10000），我这里 0.44%。除了完全同形的省并，
        # 也要处理「换了一个请求语」的情况（"帮我听X…给我把音量调到50"）：句首已交代过
        # 一件事，第二句再冠一个不同的请求语（帮我→给我→麻烦）同样像两份指令粘连。
        _merged = False
        if head_frame and body.startswith(head_frame):
            stripped = body[len(head_frame):]
            if stripped and _VERB_START.match(stripped):
                body = stripped
                _merged = True
        if not _merged and i > 0:
            # 参考集非首子句的请求语只有「我要(137)/我想(23)」两种——
            # 「请给我/给我/请/帮我」在非首位出现 0 次（那是我自造的粘连：
            # "同时请给我放疯狂动物城""一起我想收听评书"）。故扩展剥离范围。
            # 注意：参考集里 "请让我听X"(9) "给我随机播"(3) "给我讲一下"(7) "请帮我点"(2)
            # 这些组合本就是**整体短语**，剥掉前缀后剩「让我听/随机播/讲一下/帮我点」并不以
            # 标准动词起头，_VERB_START 拦不住也剥不掉，整句就成了"亮度低点一起请让我听X"。
            # 故对非首子句，凡以前述请求语开头的，一律连"让/随机/讲/帮忙/根据"等后续词一起剥。
            m_req2 = re.match(
                # 只匹配请求语本身，不吞后面的形容词（"帮我播放相声"的"相声"是内容不是动词）。
                # 不剥「我要/我想」：参考集非首位的请求语恰恰只有这两种（我要137/我想23），
                # 剥掉反而会把"我要画劳动节手抄报"变成裸动词句，再被误补动词。
                r"^(?:请让我|请帮忙|请帮我|请给我|请根据我的喜好|帮我|给我|"
                r"麻烦(?:你)?|请你|请)", body)
            if m_req2:
                stripped = body[m_req2.end():]
                stripped = re.sub(r"^[，,]?(?:来)?", "", stripped)
                # 剥完仍是完整动作句才用（"给我随机播X"→"播X"、"请让我听X"→"听X"）。
                if stripped and _VERB_START.match(stripped):
                    body = stripped                 # "给我随机播X"→"播X"
                elif stripped:
                    # 剥完若不以动词起头，就在前面补一个该域的中性动词，而不是退回带请求语
                    # 的原文。域分流：看类用「看」、听类用「听」、教育域用「学」
                    # （教育域说"听画劳动节手抄报"是错的——那是"我要画劳动节手抄报"被剥掉
                    #  「我要」后误补动词所致；教育域的动作是学/看/做）。
                    if s["d"] == "education":
                        v = "学"
                    elif s["d"] in _WATCH_DOMAINS:
                        v = "看"
                    else:
                        v = "听"
                    body = v + stripped
        # 「也」追加彻底删掉。我原以为「播放X，也调个音」是口语，实际参考集 10000 行里
        # 含「也」的只有 9 行（0.09%），且全都是内容标题里的「也」（"就算赢不了，我也要试一试"），
        # 没有一例是我这种并列连词用法。我给了 8%，等于凭空塞进 400 行的外语感句式。
        # 非末子句可带句末语气词（"聊着聊着就…"的那种停顿感）
        tailp = ""
        if (rng.random() < (0.03 if i != last else 0.03)
                and not body.endswith(("吧", "啊", "呗", "呢", "哦", "了"))
                # 种子自身可能以礼貌语收尾（"请播黄河在咆哮谢谢"），再缀语气词很怪
                and not _POLITE_TAIL.search(body)):
            # 「吗」结尾的子句（问句框「屏保画册打开好吗」）后面接「吧」＝两种句末语气
            # 打架：吗是疑问、吧是征询，不能叠（"打开好吗吧"）。参考集 10000 行 0 例
            # 「吗吧」，却有「吗啊」2/「好吗呢」1——所以只避「吧」，不避呢/啊。
            _pool = (TAIL_PART if i == last else ["", "吧", "啊"])
            if body.endswith(("吗", "么")):
                _pool = [t for t in _pool if t != "吧"]
            tailp = rng.choice(_pool)
        # 【2026-11-11 接缝标点/语气词的归属修正】——本轮"生硬"反馈的根因。
        #
        # 参考集的接缝只有 12 种（一起/顺便/，同时/同时/，/接着/然后/随后/直接连写/
        # 再/接下来/之后就），【零】例以标点或语气词起头的接缝。我产出 77 行非标接缝
        # （`。同时`/`！接着`/`~一起`/`吧顺便`/裸`。`），占 1.54%，参考集一行都没有。
        #
        # 原因：`tailp`（非末子句的语气词）与 `segp`（接缝标点）只拼进了 query，
        # 没有写回 segs[i]["q"]。于是 seg 列在接缝前就结束了，标点/语气词暴露在接缝上。
        # 参考集的口径正相反——它的 155 个"非末子句带语气词"、344 个"接缝前带标点"
        # 全部把标记写在【子句内部】（如 seg=`来一首歌呢~`，query=`来一首歌呢~，换一个看看`；
        # seg=`开始播放浙江电视台的光荣之路！`，query=`…光荣之路！，同时色度设置为30%`），
        # 接缝本身仍是那 12 种标准连接词。
        #
        # 注意【末子句】的语气词不写回：参考集 1777 个句尾语气词里 1719 个在 seg 之外
        # （seg=`图像模式调为标准模式`，query=`…调为标准模式呢`），写成 seg 内只有 58 例。
        # 这个内外不对称正是参考集的真实口径，不能一刀切。
        #
        # 【2026-11-11 二次修正：只写在内容域子句】。参考集的接缝标点/语气词是
        # 内容域专有的：非末子句里带行尾标点的 344 例【全部】是内容域（教育 185、
        # 影视 53、有声 40、音乐 28、体育 21、少儿 17），带语气词的 160 例同理；
        # 而设备域的 3755 个非末子句【一个都没有】（0/3755）。设备子句是简短指令
        # （"关闭电视""音量减"），真人不会在它后面拖一个「~」或「吧」再接下一件事；
        # 内容子句常是完整的陈述/问句，说完了自然落地。我第一版不分子域，
        # 于是产出 `进入音箱模式~同时…`、`暂停！接着…`、`关电视。之后就…`——参考集零例。
        #
        # 写回必须在【body 定稿之后】（下一段的 `if body != s["q"]` 会重置 segs[i]），
        # 否则本段的追加会被那一步抹掉。
        # 句中迟疑（"播放X，那个，音量调到30"）已删：参考集 0 次「，那个，/，嗯，/，我看看，」。
        # 真人对自己家电视是一次说完的，不存在边想边说的换气停顿。
        # 话题式「就」（"接着音效模式就调成流行音效"）同样取消：参考集里 0 次出现
        # （话题+就 共 0 行）。我原以为它读着顺，实际是自己发明的书面腔。
        # 子句本体已定稿（含省并/也/话题就），写回 segs 副本，使 seg 列与 query 严格一致。
        # 注意 segs[i] 是共享池里的对象，必须复制后再改，否则会污染后续所有行。
        if body != s["q"]:
            segs[i] = dict(s, q=body)
        _seg_is_content = s.get("tool") not in _DEVICE_TOOL_SET
        # 非末子句的语气词：内容域才保留（写回 seg，query 里也出现）；设备域整条丢弃。
        if tailp and i != last:
            if _seg_is_content:
                segs[i] = dict(segs[i], q=segs[i]["q"] + tailp)
            else:
                tailp = ""
        # 非末子句的行尾收束：参考集 344 个「接缝」（3.44%）在前一子句末尾带
        # ！/~ /。 然后【继续接连接词】——"开始播放浙江电视台的光荣之路！，同时色度设置为30%"
        # "来一首歌呢~，换一个看看"。全部出现在第 0 位（前 344/344），即"第一件事说完了，
        # 语气先收一下，再接着说第二件"。这是真人说话最典型的断句痕迹：一口气说到一半
        # 先落地，而不是用逗号一路连写。
        # 语气词（呢/吧/啊）已在上一句尾处理，这里只加纯标点，且不再叠。
        segp = ""
        if (i == 1 and rng.random() < _SEGP_P and tailp == ""
                and not body.endswith(("！", "。", "~", "？", "，"))):
            # 归属修正：segp 是【上一子句】的行尾标点，必须写回 segs[i-1]，
            # 否则它会暴露在接缝上（"关电视。之后就开大点声音"）。参考集口径是
            # seg=`开始播放浙江电视台的光荣之路！`、query=`…光荣之路！，同时色度设置为30%`。
            # 出发点 i==1，故上一子句恒为 segs[0]。
            # 且同 tailp——参考集的接缝标点只出现在【内容域】子句上（344/344），
            # 设备域的 3755 个非末子句一个都没有。上一子句是设备时整条丢弃。
            if segs[i - 1].get("tool") not in _DEVICE_TOOL_SET:
                segp = rng.choices(["！", "~", "。"], weights=[50, 50, 43])[0]
                segs[i - 1] = dict(segs[i - 1], q=segs[i - 1]["q"] + segp)
        # 连接词有「，同时」和「同时」两种形态，后者直接拼接、不加逗号——这不是笔误，
        # 参考集就是这么写的（"声音调到20随后大屏播放""关掉电视随后播放强军战歌~"），
        # 占参考集连接词的 80%。加逗号反而偏离基准。
        # 但标点收束处【保留逗号】是少数派里的多数：参考集 307 个「标点+连接词」接缝里，
        # 带逗号的有 60 个（19.5%，如"！，同时" 10、"？，一起" 35、"~，同时" 7），
        # 不带逗号的 247 个。我原来一律剥掉前导逗号（自以为是"叠标点"），实测这类 0 行
        # vs 参考 60 行，是标点接缝里唯一系统性缺失的形态。按 19.5% 保留。
        if segp and c.startswith("，") and rng.random() >= 0.195:
            c = c[1:]
        # 裸连写（c==""）的复读护栏（2026-11-12，第二十五轮）。护栏在 line ~1068 只比连接词与
        # 子句首字，【显式排除】了 c=="" 的裸连写形态，于是前句末尾与后句开头撞车时无人拦，
        # 拼出「大屏播放播放音量再大点！」「全屏播放播放音量再大点吧」「小点音量音量播放到十」
        # 3 行——参考集 10000 行为 0。实测参考集的容忍度：643 个裸连写接缝里，前句末 2 字
        # 与后句首 2 字相同的【0 个】；1 字相同的恰好 1 例（「小点声音」+「音量播放到10」→
        # 「小点声音音量播放到10吧~」，是参考集原行）。故只拦【2 字重合】，1 字重合必须放过，
        # 否则会误杀参考集自己的合法形态。segp 是标点（！~。），跨标点不可能复读，故只在
        # segp=="" 时判。命中则退化为裸逗号——参考集串行 13.7% 用裸逗号，是合法形态。
        if c == "" and segp == "" and q and len(body) >= 2 and q[-2:] == body[:2]:
            c = "，"
        q += segp + c + f + body + tailp
        prev_conn = c
    # 末尾征询/致谢（"行吗""谢谢"）：真人下完指令常补一句，纯语气不影响金标。
    # 但若句尾已是礼貌语（种子常写"…谢谢"），再补一个就成"谢谢，行不行"式的双重客气。
    enquired = False
    # 末尾征询（"，行吗""，可以吧"）：参考集里 0 次出现——真人下完指令不反问。
    # 我给了 10%，是这批数据最像"客服话术"的部分，直接取消。
    # （参考集句尾的"呢/吧/啊"是语气词，不是征询句，见下面的断句块。）
    if False and rng.random() < 0.10 and not _POLITE_TAIL.search(q):
        e = rng.choice(_ENQUIRE_NONBA if _BA_TAIL.search(q) else ENQUIRE)
        if e:
            q = q.rstrip("，,。") + "，" + e
            enquired = True
    # 句末断句：原语料 100% 是「逗号并列」的连写长句，零句号/问号/波浪号，
    # 看上去每一行都是同一个模板。真人说话常在第二件事前断一次句。
    # 句末标点比例对齐 bench_serial_train_10k.csv：感叹号 9.2%、句号 8.6%、波浪 9.0%。
    # 参考集里「！」是主力（"播放中超联赛！同时音量数值调到80"），我原来一个都没有——
    # 通篇只用逗号连写，正是「不像人在说话」的另一半原因。
    if not enquired:
        # 【结构修正】中段断句与行尾收束原先是同一次抽签的两个互斥档（if/elif/elif），
        # 于是「中段断句」的行必定【没有】行尾收束。但参考集里两者是独立的：
        # 实测断句行 48.8% 有收束（81/166），非断句行 50.7% 有收束（4986/9834）——只差 1.9pp。
        # 这解释了上一轮的取舍困局：调大 _END_T1 补齐 mid_break，就必然从收束档抽走概率，
        # 尾部 χ² 反而从 23 涨到 74（最差格 +5.28σ）。两者本不该争同一份质量。
        #
        # 【二次修正】更根本的是，我那一段断句逻辑本身也搞错了位置：它在 q 的【最后一个】
        # 逗号上换标点（q.rfind("，")），而参考集 10000 行里 143 个中段断句【全在第 0 号接缝】
        # （clause0 之后），末接缝断句 0 例。真人说话是"第一件说完先落地"，不是"最后一件
        # 之前才停顿"。真正对的机制是本函数上半部分的 segp（i==1，写回 segs[0]），
        # 它一向就在正确的位置上。故这里直接删掉整段 rfind 断句，只留行尾收束一层；
        # mid_break 率改由 _SEGP_P 调（见 segp 处），与收束彻底解耦。
        r = rng.random()
        if r < _END_T2:
            # 整句收尾。参考集实测（n=10000）行尾分布：！8.6% ~8.5% 。8.2% 吧6.1% 啊6.0%
            # 呢5.6%，合计约 43% 的行有明确的句末收束（其中 17.8% 是语气词）。
            # 我原来只覆盖 26% 且以标点为主，实测 ！2.9% ~4.7% 。2.7% 啊3.3% 呢3.7%，
            # 整批读起来像一条条没有语调的指令。语气词必须贴在子句末尾（落在子句中会让
            # seg 不再是 query 子串，故下面统一做子串校验）。
            # 末子句本身可能已带语气词（子句内 TAIL_PART 那一档加的），再缀一个就成了
            # 「打开分屏吧吧」「音量调成30吧呢」——参考集里叠语气词只有 5 例（0.05%）。
            # 已有语气词时只允许追加标点。
            # 句末语气词逐一核对 bench_serial_train_10k.csv 的实测行尾（n=10000）：
            #   标点 ！8.6% ~8.5% 。8.2%；语气词 吧9.2% 啊8.8% 呢8.2% 呗0.2%。
            # 参考集行尾【没有】哦/哈/呀/哪/啦/呗(仅0.2%)——我上一版把「哦哈呀」也放进池子，
            # 实测噢1.1%+哈0.9%+呀0.7%，全是自造。池子收成参考集真实存在的四类。
            _P = "吧啊呢"
            cur = q.rstrip()
            already = cur[-1] in _P if cur else False
            # 「吗/么」结尾（问句框「…屏保画册打开好吗」）后不能再缀「吧」：吗是疑问、
            # 吧是征询，两种句末语气打架（"打开好吗吧"），参考集 0 例。呢/啊/标点仍允许
            # （参考集有「吗啊」2/「好吗呢」1）。
            _qtail = cur[-1] in "吗么" if cur else False
            # 权重取参考集实测：！8.6 ~8.5 。8.2 吧9.2 啊8.8 呢8.2（呗 0.2% 并入吧档的低频）
            # 【语气词+标点】的叠缀：参考集 n=10000 行尾实测 吧~0.9% 吧！0.8% 啊~0.8%
            # 啊！0.7% 呢~0.7% 吧。0.7% 啊，0.7% 吧，0.7%（合计约 6.4%）。真人说完一件事，
            # 语气词之后常再带一个收尾标点（"…吧~""…啊！""…呢。"），我原来把语气词与标点
            # 当成一次抽签的互斥选项，这类叠缀实测只有 0.3%（参考 6.4%），是行尾调式唯一
            # 系统性缺失的形态——整批因此少了「说完还带点尾音"的收束感，读着更硬。
            # 权重按参考集行尾【末字符边际分布】反推（n=10000，去掉无收束的 49.7%）：
            #   末字符 ~18.9% ！17.6% 。15.4% 吧13.6% 啊13.1% 呢12.5%（，由下面裸逗号档补）。
            # 叠缀（语气词后再带标点）。
            # 2026-11-11 修正：参考集里【语气词×标点】的 12 种组合全都有，且几乎是
            # 均等的（n=10000 实测：吧~94 吧！76 吧。68 吧，71 啊~76 啊！74 啊。60 啊，70
            # 呢~67 呢！66 呢。66 呢，66，最多与最少只差 1.6 倍）。我上一版池子里只列了
            # 10 种——漏了「啊。」和「呢，」——其中「啊。」在参考集 60 例、我 0 例，是整卷
            # 字符 n-gram 里【唯一的硬零】：REF 有「播放X啊。」这种落地收束，我一句都没有，
            # 读起来就少了一种"说完一件、尾音落平"的调式。故补齐 12 种，权重按 REF 实测
            # 计数（≈1:1:1…），不再人为压低。
            pool = (["", "！", "~", "。", "，"] if already else
                    ["！", "~", "。", "吧", "啊", "呢",
                     "吧~", "吧！", "吧。", "吧，", "啊~", "啊！", "啊。", "啊，",
                     "呢~", "呢！", "呢。", "呢，"])
            wt = ([10, 2, 2, 2, 2] if already else
                  [14, 13, 13, 14, 13, 12,
                   2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
            if _END_WT and not already:
                wt = _END_WT
            if _qtail:
                # 把池子里所有「吧」起头的选项权重清零（保留呢/啊/纯标点）
                wt = [0 if t.startswith("吧") else w for t, w in zip(pool, wt)] \
                    if not already else wt
            cand = q.rstrip("，,。！~") + rng.choices(pool, weights=wt)[0]
            if all(s["q"] in cand for s in segs):
                q = cand
            # 好/吧 之后不再追加第二档：参考集行尾语气词合计 26.4%，两档叠加会超。
            # 原第二档（追加 ~/呢/吧）概率 5%，实测把 ~ 推到 10.7%（参考 8.5%）——删掉。
        elif r < _END_T3:
            # 行尾裸逗号：参考集 7.6% 的行以「，」结尾（parallel 364 / serial 370，两模式等价），
            # 且这 756 行里 744 行句中【没有】任何 ！~。 —— 即整行连写、只在最后留一个逗号，
            # 是语音输入把停顿一并转写出来的痕迹，与行尾标点/语气词互斥。
            # 我原来 0% 这种行，全批要么收尾要么连写，少了这一档自然感。
            # 只在整行确实不含断句标记时追加，避免出现「播放X。音量调到30，」这类叠标点。
            if not any(p in q for p in "！~。"):
                cand = q + "，"
                if all(s["q"] in cand for s in segs):
                    q = cand
    return q


_PAR_CD_B = float(os.environ.get("PAR_CD_B", "0.6842"))
_PAR_DD_B = float(os.environ.get("PAR_DD_B", "0.9820"))
_PAR3_F  = float(os.environ.get("PAR3_F",  "0.0280"))
_SER3_F  = float(os.environ.get("SER3_F",  "0.0210"))
_PAR_CC  = float(os.environ.get("PAR_CC",  "0.0155"))
_PAR_CDX = float(os.environ.get("PAR_CDX", "0.029"))
# CC feed 说明（2026-11-11，第二十一轮）：上面两个值（并行 0.0320→0.0155、
# 串行 0.0080→0.0038）是在 CC 路径【已修复但尚未重标定】时留下的。修复前
# `少儿+少儿` 被下游 compatible 复核整族拒掉、只有 `音乐+少儿` 能活，于是实测
# 产出只有目标的一半；按 realized 反推的 feed 因此约 2 倍。修复豁免后两对都活了，
# 若沿用旧 feed 则产出翻倍（实测 5 seed 均值 少儿+少儿 22 / 音乐+少儿 17，
# 目标 10.5/10）。在 7 seed 上重扫：feed 0.0155/0.0038 → 均值 少儿+少儿 12.9 /
# 音乐+少儿 9.0（目标 10.5/10），落在 CC 仅 21 行的 Poisson 噪声内。见
# [[survivorship-calibration]]：这类 feed 只能按 realized 实测迭代，不能解析推。
_SER_CC  = float(os.environ.get("SER_CC",  "0.0038"))
_SER_CDX = float(os.environ.get("SER_CDX", "0.032"))
_SER_CD_B = float(os.environ.get("SER_CD_B", "0.5840"))
_SER_DD_B = float(os.environ.get("SER_DD_B", "0.9358"))

# 行尾结构（2026-11-11 重构：中段断句与行尾收束原本共享一次抽签，现彻底解耦）。
# _SEGP_P  = 第 0 号接缝（clause0 之后）带断句标点的概率。REF 实测：143/10000=1.43%
#            的非末子句以 。/！/~ 结尾，且 143/143 全在第 0 位、全是内容域子句。
#            另有 23 行（0.23%）标点落在子句【内部】（引号里的台词自带「！」），
#            那部分由语料本身带来，不由这里控制。
#            抽签概率≠产出率：还须过 tailp=="" 且 body 不以标点结尾两道门，
#            实测放行率≈0.65，故概率取 0.022 才落在 1.43%（四个种子实测 1.42–1.64%）。
# _END_T2  = 行尾收束概率；_END_T3 = 收束 + 裸逗号（互斥，故仍用 elif）。
#            【2026-11-11 实测反推】：名义值（收束 45.24 + 裸逗号 5.45）并不等于产出，
#            因为收束掷中后还有「末子句已有语气词 → 只许追加标点」「吗/么 尾 → 清掉吧」
#            两道门，实际收束率低于掷中率、语气词率也偏高。以 4 格行尾（语气/标点/逗号/无）
#            逐格 χ² 对 REF 拟合（种子 14，n=5000，网格 5×5）得最优 0.430/0.486：
#            tone 18.36（REF 17.85）、punct 24.62（25.26）、comma 7.52（7.56）、
#            none 49.50（49.33），四格 χ²=2.0（名义值 0.4524/0.5065 时 χ²=29）。
_SEGP_P = float(os.environ.get("SEGP_P", "0.022"))
_END_T2 = float(os.environ.get("END_T2", "0.430"))
_END_T3 = float(os.environ.get("END_T3", "0.486"))
# 行尾收尾池权重覆盖（逗号分隔，顺序同 pool）。空＝用代码内默认。
_END_WT = [float(x) for x in os.environ["END_WT"].split(",")] if os.environ.get("END_WT") else None
# 池内默认权重（顺序＝pool：！~。吧啊呢 + 12 种「语气词+标点」叠缀）。
# 前六项按 REF 行尾【末字符边际】实测计数（！648 ~619 。626 吧564 啊603 呢564），
# 叠缀按 REF 实测≈每个 60（REF 里 12 种各 60-94，几乎均等）。我上一版给前六项
# 用「14,13,13,14,13,12」的近均匀权重，等于把「吗/呢/啊」抬得比「！」还高，背离 REF。
_END_WT_DEFAULT = [648, 619, 626, 564, 603, 564] + [60] * 12
if _END_WT is None:
    _END_WT = _END_WT_DEFAULT

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    dom_of = load_catalog_tools()
    by = load_seeds(min_params=False, dom_of=dom_of)
    print("评测集可复用完整句（按域）：", {k: len(v) for k, v in sorted(by.items())})

    # 设备子句：用【手写自然口语短语库】（评测集里的设备句多为"能力项标签"，拼接后语法断裂）。
    #
    # 抽样不是均匀的——均匀抽样是这批数据「不像人话」的两个结构性来源之一：
    #   ① 工具占比失衡：我的库里 numeric_adjust 与 mode_control 各 460 条（1:1），
    #      但参考集 bench_serial_train_10k.csv 的设备子句里 numeric 占 34.3%、mode 只占 4.9%，
    #      差了 7 倍。均匀抽出来 mode_control 占 19%（参考 4.9%），一句话里塞"图像模式切成体育模式"
    #      这种 11 字枚举模板，语感立刻变成"念菜单"。
    #   ② 长度失衡：参考集设备子句 59% 在 5 字以内（"全屏播放""关电视""大点声"），
    #      平均 5.0 字；均匀抽样把我库里大量 8 字以上的礼貌长句（"麻烦把音量调到20"）拉平到
    #      平均 7.6 字、只有 19% 在 5 字以内。真人对自己家电视说话是短的。
    # 修法：先按参考集的工具占比抽工具，再在该工具内部按长度加权（短句权重高）。
    # 权重口径：参考集【全体设备子句】的占比（n=13336），不是"带参数子句"的占比。
    # 我中途曾改用带参数占比（因为 numeric_adjust 48.5% 的子句 tool_param 为空），
    # 理由是"空参的我生成不了"——这个理由【是错的】：空参的那 251 种说法恰恰是
    # "大点声"(90) "声音调到二十"(48) "小点音量"(34) "亮度调到零"(28)，
    # 正是我手写库覆盖得最好的短口语度档。空参 ≠ 不可生成，只是参考集没标参数。
    # 用带参数占比把 numeric_adjust 从 33.0% 压到 22.6%，产出 24.8% vs 参考 33.0%，
    # 数值调节这一大块（也是真人最高频的一类操作）整体欠配，占比被让给了
    # screen_layout（17.7% vs 12.9%）。改回全体占比。
    _DEV_TOOL_W = {
        "numeric_adjust": 330, "screen_layout": 129, "playback_control": 127,
        "screensaver_control": 78, "common_control": 72, "power_control": 63,
        "mode_control": 47, "display_control": 27, "source_switch": 23,
        "timer_control": 20, "network_control": 16, "smart_camera": 15,
        "audio_control": 12, "solve_picture_sound_problem_control": 10,
        "demo_control": 5, "screen_lift_rotation": 3,
    }

    def _len_w(s: str) -> float:
        # 短句优先：≤5 字满权重，6-7 字降到三分之一，8 字以上只给八分之一。
        n = len(s)
        return 1.0 if n <= 5 else (0.34 if n <= 7 else 0.13)

    # 逐字数长度校正 _DEV_LEN_CAL 已删除。
    # 它当初是为【手写库】发明的（手写库长度分布偏长，要靠它把 4 字句抬上来），
    # 但 SURFACE_W 现在已经是「参考集实测频次 / _len_w」——参考频次本身就含真实的
    # 长度分布，再乘一遍长度校正等于三重修正。实测它把 5 字句从参考 15% 压到 8.5%、
    # 8-9 字从 9% 抬到 12.7%，且在 degnum 档上造成「声音降低 x5.0、「音量调小 x4.6」
    # 这类尾部说法被系统性夸大。删掉后长度分布直接由参考频次决定，无需再校正。

    _dev_by_tool: dict[str, list[tuple[dict, float]]] = {}
    for q, t, p in DEVICE_PHRASES:
        d = {"q": q, "d": "device", "tool": t,
             "params": clean_device_params(p) if isinstance(p, dict) else None}
        # 长度权重 × 表面形式权重。SURFACE_W 存在的原因：DEVICE_PHRASES 按表面去重，
        # 重复杂一形式的入池次数无法加权，故参考集占比改由 device_phrases.SURFACE_W 给出。
        _dev_by_tool.setdefault(t, []).append(
            (d, _len_w(q) * SURFACE_W.get(q, 1.0)))
    _dev_tool_pool = [t for t, w in _DEV_TOOL_W.items() if t in _dev_by_tool for _ in range(w)]
    # 关机/定时在【串行】里的占比要单独抬高：参考集里它们只出现在串行，且占串行设备
    # 子句的 16.3%（power 13.8% + timer 4.1%，n=6788）——因为"关机"天然是序列的终点
    # （"关掉电视随后播放强军战歌"），并行无从表达。我把它们从并行里禁掉之后，
    # 它们只能落到串行，但池子权重仍是全体口径（power 6.3% + timer 2.0% = 8.3%），
    # 实测串行里只占 9.4%，欠配近一倍。这里给串行单独一个池子。
    _GLOBAL_POOL_W = {"power_control": 138, "timer_control": 41}
    _dev_tool_pool_serial = [t for t, w in {**_DEV_TOOL_W, **_GLOBAL_POOL_W}.items()
                             if t in _dev_by_tool for _ in range(w)]

    # 【单设备位】的专用池（内容+设备 / 设备+内容 行里那个设备槽）。
    # 设备位【单独出现】时的工具分布与它【成对出现】时【完全不同】，不能共用 _DEV_TOOL_W：
    #   · 成对出现（设备+设备）：numeric_adjust 占 33.7%（"音量调到20，同时全屏"这类
    #     一人常说两件设置，数值调节是最容易凑对的一类）。
    #   · 单独出现（内容+设备）：numeric_adjust 只占 17.9%，playback 15.6%、screensaver
    #     15.0%——几乎摊平，且 common/display/network 这些"冷门"工具在这里的占比是
    #     成对时的 2-4 倍（display 5.5% vs 2.7%、network 3.3% vs 1.7%）。
    # 共用 _DEV_TOOL_W 的后果：我的单设备位 numeric_adjust 冲到 35.2%（参考 17.9%），
    # 而 playback/screensaver 只有参考的一半——整卷读起来"十句有八句在调音量"，
    # 这正是「生硬」。故单设备位按参考集【单槽实测占比】单独配权（n=6552）。
    # 【CD 槽专属口径】（2026-11-11 重导）。这两张表原来是从「内容+设备 行里那个设备槽」
    # 的**总体**分布收割的，而那个总体 = CD + DC。DC 现在已有自己的 `_DC_SLOT_W` 并单独
    # 抽取，于是这张表只该承担 CD，可我仍用着含 DC 的旧权重——而 DC 的设备槽 74% 是
    # power（"关电视#播放第1集"），把 power 抬到 12.0%（REF 的 CD 槽只有 5.8%）。
    # 实测后果：串行 C#D 行的设备槽 power 占比 **13% vs REF 6%**（2.2×），
    # 整卷 power 设备share 6.3%→9.0%。这是本轮「生硬」的主因——满屏"播放X，同时关电视"。
    # 下面两张表按 REF 的【CD-only】设备槽重导（ser n=2900 / par n=3332，均已 _pair_tools
    # 重排对齐），CD 槽里 parallel 的 power/timer 恒 0%（并行关闭序列终点类动作）。
    _SINGLE_SLOT_W = {
        "numeric_adjust": 194, "screensaver_control": 173, "playback_control": 168,
        "common_control": 91, "mode_control": 66, "screen_layout": 64,
        "display_control": 64, "source_switch": 44, "network_control": 36,
        "smart_camera": 33, "audio_control": 31, "solve_picture_sound_problem_control": 23,
        "demo_control": 12, "screen_lift_rotation": 3,
    }
    # 串行 CD 槽：power 5.8% / timer 6.4%（timer 在串行 CD 里是真实终点动作
    # "…随后45分钟后关机"，parallel 0%）。numeric/screensaver/playback 三档各约 15%。
    _SINGLE_SLOT_W_SERIAL = {
        "numeric_adjust": 160, "playback_control": 160, "screensaver_control": 140,
        "common_control": 90, "timer_control": 64, "mode_control": 58,
        "power_control": 58, "screen_layout": 56, "display_control": 50,
        "source_switch": 41, "network_control": 33, "smart_camera": 30,
        "solve_picture_sound_problem_control": 22, "audio_control": 21,
        "demo_control": 11, "screen_lift_rotation": 6,
    }
    _dev_tool_pool_single = [t for t, w in _SINGLE_SLOT_W.items()
                             if t in _dev_by_tool for _ in range(w)]
    _dev_tool_pool_single_serial = [t for t, w in _SINGLE_SLOT_W_SERIAL.items()
                                    if t in _dev_by_tool for _ in range(w)]

    # 【设备先行·内容后随】(DC) 行的专用设备池（2026-11-11）。
    # DC 行的设备槽分布与"设备位单独出现"的【总体】分布完全不同——不能共用
    # _SINGLE_SLOT_W_SERIAL：那是 CD+DC 合起来的单槽口径，而 DC 是其中【特定的一类】。
    # 参考集 ser 的 296 条 DC 里，设备槽 220 条是 power_control（74%），
    # 全是"先关掉电视/关机，再起播下一件"（"关电视#播放第1集""关机#放一首歌听"）；
    # 而 _SINGLE_SLOT_W_SERIAL 里 power 只占 12%。
    #
    # 更关键的是【存活偏差】：order_ok 有一条"串行里设备步一般要在起播步之后，
    # 但 power 例外"的规则——非 power 的设备子句出现在起播内容句之前会被整行拒掉。
    # 于是从通用池抽 DC 行时，抽到非 power 的（88%）大多在 order_ok 被毙，
    # DC 产出只有 0.028（参考 0.059）——这一半是"抽错工具"造成的。
    # 按参考集【DC 行自己的】工具分布抽，抽中即通过，产出直接对齐。
    _DC_SLOT_W = {
        "parallel": {"common_control": 22, "screen_layout": 19, "screen_lift_rotation": 12,
                     "mode_control": 11, "numeric_adjust": 10, "playback_control": 7},
        "serial": {"power_control": 220, "timer_control": 20, "common_control": 17,
                   "mode_control": 12, "numeric_adjust": 9, "screen_lift_rotation": 6,
                   "screen_layout": 6, "playback_control": 6},
    }
    _dc_tool_pool = {m: [t for t, w in d.items() if t in _dev_by_tool for _ in range(w)]
                     for m, d in _DC_SLOT_W.items()}

    def pick_device_single(serial: bool = False, pos: str = "", dc: bool = False) -> dict:
        """内容+设备 行里的那个设备槽：按参考集【单槽】占比选工具（分模式）。

        `dc=True` 用于【设备先行·内容后随】的行：改用 DC 专属工具池（见 _DC_SLOT_W），
        否则会因 order_ok 的"非 power 设备不得先于起播句"规则大面积被拒。

        搭档是内容子句，永远不是 numeric_adjust/screen_layout，故把「省略了动词的
        光杆变量」和 "小点/小点声/降厄" 一类【必须有调节动作搭档】的说法整类滤掉——
        参考集里它们从不与内容句同桌（0 例），"播放赛罗奥特曼顺便亮度" 这类残句
        正是这么来的。

        pos="lead"/"follow" 决定说法在该位置是否被参考集见证过（见 may_lead/may_follow）。
        """
        if dc:
            _pool = _dc_tool_pool["serial" if serial else "parallel"]
        else:
            _pool = _dev_tool_pool_single_serial if serial else _dev_tool_pool_single
        # pick_device 在"该工具在目标位置没有合规说法"时返回 None（见那里的说明）。
        # DC 池比通用池窄得多，撞上这种工具的概率更高，故重试几次；仍不行时退回通用池，
        # 不把整行作废（调用方脚下没有 None 检查）。
        for _ in range(6):
            s = pick_device(_pool=_pool, partner_tool="内容", pos=pos)
            if s is not None:
                return s
        return pick_device(_pool=_dev_tool_pool_single_serial if serial
                           else _dev_tool_pool_single, partner_tool="内容", pos=pos)

    # ---------------------------------------------------------------------------
    # 设备×设备 的【联合】工具对（2026-11-10）
    # ---------------------------------------------------------------------------
    # 我原来两个设备位【各自独立】抽工具，于是产出的是两者的近似乘积分布：
    # 实测我的头号组合 (numeric_adjust,numeric_adjust) 只占 11.8%，各组合摊得很平。
    # 但参考集里设备对是【强相关】的联合：numeric_adjust×screen_layout 一对就占 35%
    # （"音量调到20，同时全屏"——先把屏幕铺开再顺手调个音量，是真人最常说的组合），
    # 其后 playback/关机/主页各有固定搭配；30 个有序对里前 4 个吃掉 51%。
    # 独立抽样把这种"人们真的会一起说的两件事"的搭配关系抹平了，读起来就是"随机凑的
    # 两个设置"——正是用户说的"生硬"。故这里直接以参考集的【有序工具对】频次抽样，
    # 定下两个工具后再各自在工具内选说法（说法仍走 lw 权重）。
    # 数据取自 bench_serial_train_10k.csv 里 2 步且两位都是设备工具的行，分模式统计。
    _DEV_PAIR = {
        "parallel": [
            ("numeric_adjust", "screen_layout", 316), ("screen_layout", "numeric_adjust", 290),
            ("numeric_adjust", "numeric_adjust", 178), ("playback_control", "numeric_adjust", 119),
            ("numeric_adjust", "playback_control", 86), ("source_switch", "numeric_adjust", 55),
            ("playback_control", "screen_layout", 52), ("mode_control", "numeric_adjust", 51),
            ("numeric_adjust", "mode_control", 48), ("screen_layout", "playback_control", 39),
            ("common_control", "numeric_adjust", 35), ("numeric_adjust", "common_control", 33),
            ("playback_control", "playback_control", 26), ("playback_control", "common_control", 19),
            ("numeric_adjust", "screensaver_control", 16), ("common_control", "screen_layout", 15),
            ("common_control", "screensaver_control", 13), ("common_control", "playback_control", 13),
            ("screen_lift_rotation", "numeric_adjust", 11), ("screen_layout", "mode_control", 4),
        ],
        "serial": [
            ("screen_layout", "numeric_adjust", 263), ("numeric_adjust", "screen_layout", 225),
            ("power_control", "numeric_adjust", 181), ("playback_control", "numeric_adjust", 138),
            ("common_control", "power_control", 116), ("numeric_adjust", "numeric_adjust", 107),
            ("numeric_adjust", "power_control", 89), ("power_control", "common_control", 68),
            ("numeric_adjust", "playback_control", 57), ("source_switch", "numeric_adjust", 55),
            ("power_control", "playback_control", 45), ("screen_layout", "timer_control", 42),
            ("numeric_adjust", "mode_control", 38), ("playback_control", "screen_layout", 34),
            ("playback_control", "power_control", 34), ("numeric_adjust", "timer_control", 32),
            ("mode_control", "numeric_adjust", 29), ("playback_control", "playback_control", 24),
            ("common_control", "numeric_adjust", 21), ("playback_control", "common_control", 19),
            ("screen_layout", "playback_control", 19), ("screen_layout", "power_control", 18),
            ("numeric_adjust", "common_control", 18), ("numeric_adjust", "screensaver_control", 17),
            ("power_control", "screen_layout", 13), ("screen_lift_rotation", "numeric_adjust", 10),
            ("common_control", "playback_control", 9), ("common_control", "screensaver_control", 8),
            ("screen_layout", "mode_control", 8), ("common_control", "screen_layout", 7),
        ],
    }
    # 只保留两端工具都在当前短语库里的对（个别工具可能一个短语都没有）。
    _dev_pair_pool = {
        m: [x for x in lst if x[0] in _dev_by_tool and x[1] in _dev_by_tool]
        for m, lst in _DEV_PAIR.items()}

    # 「音量」与「声音」在参考集金标里是两个不同的 object，口语里却指同一个量
    # （"声音小些…声音调到7"）。判重目标前先归一，否则 object=音量、文本认成声音，
    # 会被当成两个目标而漏过。
    _TGT_ALIAS = {"声音": "音量", "声": "音量", "音量": "音量"}

    def _dev_target(s: dict) -> str:
        # 设备说法调节的【目标】（音量/亮度/屏幕…），取自金标 object 字段；
        # params 为 null（参考集 24.5% 的说法系统性留白）时退回从文字里认。
        p = s.get("params")
        if isinstance(p, dict):
            t = (p.get("object") or p.get("device") or "").strip()
            if t:
                return _TGT_ALIAS.get(t, t)
        for t in ("音量", "声音", "亮度", "屏幕", "对比度", "色度", "饱和度",
                  "分辨率", "图像模式", "音效模式", "声音模式", "静音"):
            if t in s["q"]:
                return _TGT_ALIAS.get(t, t)
        # 无目标名词、params 又留白的音量说法（参考集原生如「免费大点声」「小点」）：
        # 语意上仍然是在调音量，归到默认目标，否则同目标复核会漏判。
        if s["tool"] == "numeric_adjust" and any(
                v in s["q"] for v in ("大点声", "小点声", "大声", "小声",
                                      "声大", "声小", "大一点声", "小一点声")):
            return "音量"
        if s["tool"] == "numeric_adjust" and s["q"].strip() in ("大点", "小点"):
            return "音量"
        return ""

    # 光杆变量（"亮度""音量""声音"）在参考集里【只】出现在 numeric_adjust×numeric_adjust
    # 这一步对里（69/69），因为它是「省略了动词的调音量/调亮度」——搭档必须也是一条调节动作，
    # 才能把省略的动词补回来。我的抽样器原先不知道这层上下文，把"亮度"抽到 (播放控件, numeric)
    # 这类对里，整行被 order_ok 拒掉；抽到 42 次只活下来 11 次，参考 69 被我产成 11。
    # 同理"小点/小点声"只在 screen_layout 搭档下出现（18/18、24/24，"小点一起大屏播放"），
    # "*降厄"只在含「亮度」的搭档下（25/25），"保持音量"只在 screensaver 下（14/14）——
    # 这几类 order_ok 里已有【文字层】的搭档约束（小点需音量/声音、降厄需亮度），
    # 这里不再按工具重复加码：重复加码会把 (numeric_adjust, screen_layout) 这种
    # 罕见工具对整类抽空，反而把小点/保持音量从 3、6 压到 0。
    # 只有「光杆变量」这一类 order_ok 是按【工具】判的，前置过滤与它同口径、不冲突：
    # 与其抽出来再整行作废（拒绝采样会把同行的其它子句一起重抽，改变联合分布），
    # 不如在抽说法时就避开。
    _BARE_SURFACE = re.compile(r"^(把|将)?(亮度|音量|声音)$")

    def pick_device(serial: bool = False, tool: str = "", _pool=None,
                    partner_tool: str = "", pos: str = "") -> dict:
        _base_pool = _pool if _pool is not None else (
            _dev_tool_pool_serial if serial else _dev_tool_pool)
        # 位置闸门（见 device_phrases.may_lead / may_follow）：参考集的设备说法是
        # 【强位置条件】的——205 个频次≥10 的说法里 83 个从不句首、50 个永远句首，
        # 且 8/16 个设备工具从不句首。我原先按位置无关的权重抽，产出里设备首位子句
        # 39.4% 用了参考集从不放首位的说法（"启动u盘…""音量22…"）。
        #
        # 两段式（2026-11-11 定型）：
        #   · 硬闸门（工具级）：NONLEAD_TOOLS 在首位整类排除——工具从不句首是最强的
        #     语序事实（display 358 次、screensaver 1018、timer 258… 各自总量都很大，
        #     n0=0 是真实语序而非样本不足）。
        #   · 硬闸门（说法级）：在【该位置】有确凿反证的（may_lead/may_follow=False）排除。
        #     证据一律带 tool——同一说法在不同工具下位置条件可【相反】：`关闭屏幕`
        #     作为 screensaver 是 (0,64) 从不句首、作为 screen_lift_rotation 是 (18,0)
        #     【只】句首。聚合证据 (41,64) 会放行"screensaver 起头"（参考 0.0%）。
        #   · 位置专属频次作权重：抽中的说法里，按【该位置的参考集频次】加权
        #     （首位用 n0、跟随位用 n1），与参考集「位置×说法」联合分布同口径。
        #     不能乘 pos_weight 做"软化"：lw 本身已是参考集总频次、已含该说法的
        #     起头倾向，再乘一次位置概率等于双重计数，头部被二次放大
        #     （实测 3-gram |Δ| 0.060→0.075）。
        if pos == "lead":
            _bp = [t for t in _base_pool if t not in NONLEAD_TOOLS]
            if _bp:
                _base_pool = _bp
        tool = tool or rng.choice(_base_pool)
        pool = _dev_by_tool[tool]
        if tool == "numeric_adjust" and partner_tool != "numeric_adjust":
            _f = [(d, w) for d, w in pool if not _BARE_SURFACE.match(d["q"].strip())]
            if _f:
                pool = _f
        if pos:
            _ok = may_lead if pos == "lead" else may_follow
            _idx = 0 if pos == "lead" else 1
            _pf = [(d, w) for d, w in pool if _ok(d["q"].strip(), tool)]
            if not _pf and tool:
                # 点名工具在该位置无合规说法（唯一实例 `screen_lift_rotation` 在首位，
                # 见下），从位置池另抽一个工具——不硬塞违例说法、也不作废整行。
                _bp = [t for t in _base_pool if t not in NONLEAD_TOOLS or pos != "lead"]
                if _bp:
                    tool = rng.choice(_bp)
                    pool = _dev_by_tool[tool]
                    if tool == "numeric_adjust" and partner_tool != "numeric_adjust":
                        _f = [(d, w) for d, w in pool
                              if not _BARE_SURFACE.match(d["q"].strip())]
                        if _f:
                            pool = _f
                    _pf = [(d, w) for d, w in pool if _ok(d["q"].strip(), tool)]
            if not _pf:
                return None
            _pool2 = []
            for d, w in _pf:
                _e = _POS_BY_QT.get((d["q"].strip(), tool)) or \
                    POS_EVIDENCE.get(d["q"].strip())
                _pw = (_e[_idx] if _e and _e[_idx] > 0 else None)
                _pool2.append((d, _pw if _pw else w))
            pool = _pool2
        s = rng.choices([d for d, _ in pool], weights=[w for _, w in pool])[0]
        # 参数未标注（null）对齐：参考集里 24.5% 的设备位 tool_param 为 null，且这是
        # 【按说法系统性】的（"大点声"永远 null、"全屏播放"从不 null）。NULL_P 是参考集里
        # 该说法的 null 频率，按它抽查一次决定该步是否不标参数。eval_multi.judge() 对
        # null 步不计参数分，所以这与参考集口径一致。注意必须 copy——池里的 dict 是共享的，
        # 原地改 params 会把"本次不标"永久写回池子。
        if NULL_P.get(s["q"]) and rng.random() < NULL_P[s["q"]]:
            s = {**s, "params": None}
        return s

    devices = [d for lst in _dev_by_tool.values() for d, _ in lst]
    print(f"设备自然短语库：{len(devices)} 条（加权抽样）")
    # 内容子句：只保留真人会说的完整句（严格过滤后的评测集句子）。
    contents = [s for d, lst in by.items() if d != "device" for s in lst]
    print(f"内容域完整句：{len(contents)} 条")

    # 参考集内容子句收割：all_eval 去重后 1182 条，而参考集里【带参数】的内容子句就有
    # 1214 条不同说法，其中 688 条不在我的池子里（"选择魔道祖师广播剧""NBA 盖帽榜"
    # "有没有环保公益主题的音乐"）。内容子句的表面形式是整卷 4-gram 熵的主要来源，
    # 池子只有参考一半大时就出现 distinct 1061 vs 2681 的差距。金标参数直接取自参考集。
    # 过滤口径与种子池一致（长度/疑问尾/指代/完整句），确保并进多意图不会崩。
    try:
        from ref_device_vocab import harvest_content_counts as _harvest_content
        _content_ref = _harvest_content()
    except Exception as _e:
        print(f"[content harvest] 不可用（{_e}）")
        _content_ref = {}
    _seen_cq = {s["q"] for s in contents}
    _added = 0
    for _q, (_t, _pm, _n) in _content_ref.items():
        # 必须先按 TOOL_ALIAS 归一再查域：参考集用的是旧工具名，dom_of 用的是目录现名。
        # 漏了这一步会把 educ_search(232 种)/vod_search(224 种) 整批丢掉——它们是参考集里
        # 内容子句的第三、第四大来源（合计 456 种），实测内容池因此少了近四分之一的类型。
        # 种子池那条路径（load_seeds 第 355 行）是归过的，收割这条路径原来漏了。
        _t = TOOL_ALIAS.get(_t, _t)
        _dom = dom_of.get(_t)
        if _dom in (None, "device") or _dom not in CN:
            continue
        if _q in _seen_cq:
            continue
        if _pm is None:                 # 参考集刻意留白，照搬 None（judge 对 null 步不计参数分）
            _pq = None
        else:
            try:
                _pq = json.loads(_pm) if isinstance(_pm, str) else _pm
            except json.JSONDecodeError:
                continue
            if not isinstance(_pq, dict):
                continue
        _q2 = clean(_q)
        # 下限与 ref_device_vocab.harvest_content_counts 同步放到 2（见那里的说明）：
        # 参考集内容子句 len<=4 的一档占实例 7.8%，且是最热门的一批（"播放音乐" 80 次）。
        # 这里校验的是 clean() 之后的长度（"快进到三分钟！" 剥标点后仍 6 字，不受影响）。
        if not (2 <= len(_q2) <= 26):
            continue
        # ASK_Q / REF_Q 是给【种子池】写的——那批来自 all_eval 单意图评测集，一个问句
        # 就是一轮 chat（"哪个台有…"），并进多意图后语义崩。但【参考集自己的多意图行】里
        # 问句是合法的一步：实测命中 ASK_Q/REF_Q 的内容子句有 475 个实例、264 种类型，
        # 且 100% 出现在 n_steps=2 里（"转眼的近义词是什么接着打开主题"
        # "播放第三个视频随后打开护眼"）。照搬这些整句，正是用户要的多样性。
        # 故收割路径只保留 BAD_Q（纯寒暄），不再套 ASK_Q/REF_Q。
        # 争议点在【切分】：参考集里问句/指代句都以「，」或「接着/随后」与后句相连，
        # 我原样照搬整句即可，不重新切分，故不会把「那个男主…的古装剧叫什么」切错。
        if BAD_Q.match(_q2):
            continue
        # 【不能套时间词过滤】（2026-11-11）。上面 load_seeds 那条（第 465 行）是为
        # all_eval 种子池写的——那个池子里一整行 query 可以带显式的「当前时间」上下文，
        # 与相对时间词冲突。但参考集自己的内容子句里【时间词是合法且常见的】：
        # 「最近播放量高的新剧」4 次、「播放我最近播放过的歌单」、「我想听我最近听过的
        # 有声书」、「今年播出的国产动画」、「晚上睡不着适合一个人看电影」……共 56 种说法
        # / 105 个实例，全是真人真说的检索条件。我一律照搬整句，不重新生成，故无冲突问题。
        # 实测这条过滤把它们全删了：我的内容子句里含「最近」的 0 个（参考 52）、
        # 含「今天/明天/昨天」0 个（参考 28）——正是 3-gram 差分里最大的欠产簇之一。
        # k12_category 是 edu 工具的系统代填字段，judge() 会在比对前从金标里剥掉
        # （query 含「同步」的除外）。收割时必须同样剥掉，否则存下来的金标比评测口径
        # 多一个字段，183 行会被判 param_ok=False。这里直接复用评测那边的同一函数，
        # 保证「生成的口径」与「评测的口径」是同一个。
        if isinstance(_pq, dict):
            _pq = _strip_k12(_pq, _q2)[0]
        _seen_cq.add(_q)
        # _lw=1.0：收割项的长度分布已是参考集真实分布，不再乘 _c_len_w
        # （那是为修正【种子库】偏长发明的修正，对参考集项是二次修正，会整体压短）。
        # lw【必须带参考集频次】而不是恒 1.0。理由：内容池是【类型均匀】入池的，
        # 而参考集里 null 参数档是"类型少但频次高"（"播放音乐" 80 次、"播放成何体统" 47 次）。
        # 恒 1.0 时我的内容位按【类型占比】分布（null 档 10.9%），而参考集是【频次占比】
        # （null 档 26.8%），于是 content null 只能做到 10.5% vs 参考 26.5%。
        # 带频次后，pick_content 直接以 lw（=参考集频次）作句内权重，故 (工具, 请求语, 动词)
        # 的分布自然落在参考集形状上——频次高的说法就是参考集里说得多的说法。
        contents.append({"q": _q2, "d": _dom, "tool": _t, "lw": float(max(1, _n)),
                         "params": (None if _pq is None else
                                    {k: v for k, v in _pq.items() if k != "retext"})})
        _added += 1
    print(f"参考集内容子句增收：{_added} 条 -> 合计 {len(contents)} 条")

    # 问句子句池（独立成路，见 ref_device_vocab.harvest_question_clauses 的说明）。
    # 这类子句带句尾「？」，不能进普通内容池（clean() 会剥掉标点、改写成语义不同的陈述句），
    # 也不该被 ASK_Q 过滤——那是为 all_eval 种子池写的规则，在参考集里问句是有工具的。
    try:
        from ref_device_vocab import harvest_question_clauses as _harvest_q
        _q_ref = _harvest_q()
    except Exception as _e:
        print(f"[question harvest] 不可用（{_e}）")
        _q_ref = {}
    # fan_knowledge_agent 的权威域在 catalog 里是 vod，但参考集 273 行里 255 行标【教育】。
    # 这是 catalog 与实际标注不一致；问句族按参考集标注走（judge 只比对工具与参数，
    # 域只影响结构统计），故这里不覆盖域、只把工具留在池里。
    _qpool: list[dict] = []
    for _q, (_t, _pm, _n) in _q_ref.items():
        if dom_of.get(_t) in (None, "device") or dom_of.get(_t) not in CN:
            continue
        _pq = _strip_k12(_pm, _q)[0] if isinstance(_pm, dict) else {}
        _qpool.append({"q": _q, "d": dom_of[_t], "tool": _t, "lw": 1.0,
                       "params": {k: v for k, v in _pq.items() if k != "retext"},
                       "w": max(1, _n), "question": True})
    print(f"问句子句池：{len(_qpool)} 条（工具 {len({s['tool'] for s in _qpool})} 种）")
    def _c_len_w(s: str) -> float:
        n = len(s)
        return 1.0 if n <= 8 else (0.42 if n <= 12 else 0.18)

    # 【首位专属权重】（2026-11-11）。见 pick_content("lead") 处的说明——
    # 首位该按 n0（参考集首位频次）配权，而非总频次 lw。用环境变量可离线 A/B。
    _POS_LEAD_W = os.environ.get("POS_LEAD_W", "1") == "1"

    def _c_lead_w(s: dict) -> float:
        """首位抽取用的权重：该说法的【参考集首位频次 n0】，无证据时退回总频次 lw。

        与跟随位对称：跟随位一直按 n1 抽（见 _C_FOLLOW_TABLE），首位却一直按 lw
        （总频次）抽。当某说法的质量大量落在跟随位时（"播放音乐" 总 97 = 首位 62 +
        跟随 18），用总频次配权就把跟随位那部分质量【也算进了首位】，首位份额被抬高。
        实测 music_song_recommend 首位的"播放音乐"份额：REF .373、按 lw 抽 .433、
        按 n0 抽 .373——逐位吻合。整卷 effect：pos0 的"播放音乐" 53 vs 期望 31（1.70×）。
        """
        # 注意：s["lw"] 已含 reg_ratio（见 _REG_DEPOOL 说明），此处不再二次乘。
        _rr = s.get("reg_ratio", 1.0)            # 语域残差比例（n0 证据上也用）
        if not _POS_LEAD_W:
            return s.get("lw", _c_len_w(s["q"]))
        e = CONTENT_POS_EVIDENCE.get(s["q"])
        if e is not None and e[0] > 0:
            return float(e[0]) * _rr             # 首位 n0 也必须按残差缩减，否则双计
        return s.get("lw", _c_len_w(s["q"]))     # 变体/参考集里没有的：退回原权重

    # 换框变体相对父句的权重份额（见下方 "内容句换框" 处的说明）。
    # 扫描口径：对 0/0.1/0.25 各跑一卷，量（a）内容子句 total-variation vs 参考，
    # （b）distinct 类型数 vs 参考同口径期望值，（c）audio_search 的「播放」/「裸标题」
    # 两档相对参考的比值，（d）整卷 4-gram 熵。结果：
    #   VAR_W=0    tv 37.2%  distinct 1630（期望 1612）  播放 1.03 裸 1.11  熵 13.898
    #   VAR_W=0.1  tv 39.6%  distinct 1707（期望 1605）  播放 0.93 裸 0.80  熵 13.910
    #   VAR_W=0.25 tv 40.5%  distinct 1820（期望 1610） 播放 0.75 裸 0.51  熵 13.938
    # VAR_W=0 与参考最贴（帧占比几乎不偏），但熵略低；文档要求「表达多样」，取 0.1
    # 这一档：变体只作小幅补充，主多样性来自收割进池的真实参考子句。
    VAR_W = float(os.environ.get("VAR_W", "0.1"))

    # ---------------------------------------------------------------------------
    # 内容子句权重统一到【参考集频次】（2026-11-10）
    # ---------------------------------------------------------------------------
    # 原先两个来源的权重【差两个数量级】：收割项 lw=参考频次（最高 80），种子项
    # lw=_c_len_w（≤1.0）、换框变体同样 ≤1.0。于是产出极度偏向少数高频收割项——
    # 实测 top-50 类型吃掉 26% 质量（参考 17.7%）、top-1000 吃掉 87%（参考 68.3%）。
    # 语感上就是"翻来覆去那几句"，正是用户说的"不够真实"。
    #
    # 参考集频次本身就是真人怎么说这话的直接度量，且已含真实长度分布（种子项 80%
    # 就在参考集内，mean 2.15 次）。统一到它之后，两源可比，长尾按真实频率浮现。
    # 换框变体不在参考集里，取其父句频次——变体与父句是同一句式的两种说法，
    # 共享权重才不至于把"表达多样性"这条腿饿死。
    try:
        from ref_device_vocab import content_clause_freq as _ccf
        _CONTENT_FREQ = _ccf()
    except Exception as _e:
        print(f"[content freq] 参考集频次不可用（{_e}）")
        _CONTENT_FREQ = {}
    # 【扣除语域覆盖行（2026-11-12，第二十四轮）】——见 content_clause_freq_nonreg。
    # 主池的权重是"该说法的参考频次"，可落在语域表里的说法已由【语域路径】逐字照搬
    # 产出了一份（覆盖率 39.4% vs REF 38.5%，逐位吻合）。不扣就是计两次——实测
    # 语域∩池组的 realized/expected 中位 1.63×（池 0.58 + 语域 1.13），超产的正是
    # `播放免费的小猪佩奇`(2.5×)/`播放第一集`(1.6×)/`播放第12集`(2.0×) 这一族。
    # 主池只配【残差】质量（= 全额 − 语域覆盖行）。残差为 0 的说法主池不再产出，
    # 完全交给语域路径；但其【换框变体】是新增表面、不属任何语域表，故仍按【全额】
    # 频次挂权重（见下方 _fw）。用环境变量 A/B（REG_DEPOOL=0 关回旧行为）。
    _REG_DEPOOL = os.environ.get("REG_DEPOOL", "1") == "1"
    try:
        from ref_device_vocab import content_clause_freq_nonreg as _ccf2
        _CONTENT_FREQ_POOL = _ccf2() if _REG_DEPOOL else _CONTENT_FREQ
    except Exception as _e:
        print(f"[content freq nonreg] 不可用（{_e}）")
        _CONTENT_FREQ_POOL = _CONTENT_FREQ
    for _s in contents:
        _q = _s["q"]
        _full = float(_CONTENT_FREQ.get(_q) or _c_len_w(_q))     # 全额参考频次
        _res = _CONTENT_FREQ_POOL.get(_q, _full)                 # 残差（扣掉语域覆盖行）
        # reg_ratio = 非语域行占比：主池的每一层配权（首位 n0 / 句内 lw）都乘它，
        # 使该说法在【主池】里的总份额按其"非语域残差"比例缩减——语域路径补上其余。
        _s["lw_full"] = _full
        _s["reg_ratio"] = (_res / _full) if _full > 0 else 1.0
        _s["lw"] = _full * _s["reg_ratio"] if _REG_DEPOOL else _full

    # 句式换框：同 action 族内替换句首框架词，扩大表面形式（金标参数不变）
    _seen_c = {s["q"] for s in contents}
    _extra: list[dict] = []
    for s in contents:
        _vs = [(v, w) for v, w in frame_variants(s["q"], s["params"], s["d"])
               if v not in _seen_c]
        if not _vs:
            continue
        # 【变体权重】= 父句频次的 VAR_W 倍，父句自己【不动】。
        #
        # 曾经这里是「家族守恒」：父句与 k 个变体平分父句频次（各 1/(k+1)）。动机是
        # 怕一个 freq=80 的父句带 2 个变体共占 240 质量、把高频句夸大。但守恒有个
        # 系统性副作用：只有【动词起头】的句子才生得出变体（"播放小说"→"听小说""放小说"），
        # 裸标题（"海底小纵队""辽宁本钢"）天然无变体候选。于是守恒把动词档整体除以
        # (1+k)≈2，裸标题档原封不动——实测 audio_search 的「播放」档掉到参考的 0.53 倍、
        # 「裸标题」档 1.04 倍，两个本来并列的最大档被硬生生拉开一倍。
        #
        # 改法：变体是【新增】的表面形式，不该从父句身上扣。给变体一个小份额 VAR_W
        # （父句保持全额 cf），既保留"同一内容多种说法"的表达多样性，又不改变
        # 父句所在的框架档在参考集里的占比——因为该档的总质量仍以父句为主。
        # VAR_W 由离线扫描定（见下方注释），不是拍脑袋。
        #
        # 【第二十三轮：变体再乘换框词相对权重 w】。等权时每个换框词产出的变体质量
        # 相同，于是低频档被抬频（实测 给我听 20 vs 参考 5、给我放 26 vs 12.5）。
        # w = 参考集句首频次 / 本表均值，逐表归一使 ∑w ≈ 表长，总质量近似不变。
        # 变体用【全额】频次（lw_full），不用扣过语域的 lw：变体是新增表面，不在任何
        # 语域表里，若沿用父句已被扣到 0 的残差就会把"同一内容的多种说法"饿死。
        _fw = s.get("lw_full", s.get("lw", _c_len_w(s["q"]))) * VAR_W
        for v, _w in _vs:
            _seen_c.add(v)
            _extra.append({"q": v, "d": s["d"], "tool": s["tool"], "params": s["params"],
                           "lw": _fw * _w, "lw_full": _fw * _w, "reg_ratio": 1.0})
    contents += _extra
    print(f"内容句换框后：{len(contents)} 条")

    # 内容子句同样要偏短。参考集内容子句平均 9.7 字、43% 在 8 字以内（"播放音乐""播放第1集"
    # "我想听刘德华的歌"）；我的种子库里 9-12 字占 50%、13 字以上占 28%，均匀抽出来平均 11.0 字、
    # 只有 21% 在 8 字以内。允许的种子最长 26 字（"查找圆柱与圆锥体积公式推导及应用的精讲课"），
    # 拼进多意图后整句就奔 30 字去，这是 query 长度 20.8 vs 参考 16.7 的主因。
    # （_c_len_w 现在只在【参考集里没有的】少数种子项上兜底，见上面的统一赋权块。）

    # 工具维度再归一化一次：种子库里各内容工具的条数不反映参考集
    # 的真实占比——audio_search 有 297 条种子、educ_search_all 171 条、
    # sports_match_reservation 只有 36 条但在产出里占到 7.8×（因为我互斥规则放宽了）。
    # 目标质量 = 参考集该工具子句占比（全体口径，n=5893），由该工具的候选数均分。
    # 权重直接取参考集【全体内容子句】的实测占比（n=6907），不再凭"我库里有多少条"折算。
    # 上一版就是这么折的，但两级抽样里的第二级（长度×框架）会再一次改变实际占比：
    # 某一工具的子句整体偏短、整体以「播放」起头，就会在第一级达标后于第二级被放大。
    # 实测 vod_search_all 15.2% vs 参考 10.8%、educ_search_all 10.2% vs 6.0%。
    # 这里改为按实测占比配目标质量，并减去各工具在第二级的平均放大系数做一次自洽校正
    # （放大系数由 _cw_mean 计算），使第一级×第二级的合成占比落在参考值上。
    _CONTENT_TOOL_W = {
        "vod_search_all": 173, "educ_search_all": 131, "audio_search": 101,
        "music_song_search": 76, "edu_fuzzy_search": 65, "vod_fuzzy_search": 56,
        "edu_search": 49, "fan_knowledge_agent": 40, "music_song_recommend": 35,
        "sports_vod_search": 29, "sports_match_search": 27, "sports_team_search": 25,
        "music_song_history": 17, "tvchannel_music_search": 15, "sports_rank_search": 15,
        "sports_match_reservation": 14, "music_ksong_search": 13,
        "music_song_favorite_search": 13, "music_song_mv_search": 11,
        "educ_fuzzy_search": 11, "vod_personalized_search": 7,
        "music_song_qqmusic_search": 5, "vod_person_search": 5, "vod_relate_search": 4,
        "audio_history": 3, "vod_history": 2, "educ_history": 1, "educ_relate_search": 1,
        "educ_relate_recommend": 1, "sports_match_data_search": 1,
    }
    # 两级抽样：先按目标占比抽工具，再在该工具内按「长度 × 框架词」抽句子。
    # 不能把工具权重与框架权重【相乘】在一次抽样里——两者会互相作用：
    # 某个工具的子句碰巧都短、又都以「播放」起头，就会同时吃到两个方向的放大，
    # 实测这样做的 music_song_recommend 冲到 6.1%（参考 1.4%）、audio_search 掉到 1.2%。
    # 两级抽样让两件事正交：工具占比只由 _CONTENT_TOOL_W 决定，句内分布只由长度/框架决定。
    _ctool: dict[str, list[dict]] = collections.defaultdict(list)
    for s in contents:
        _ctool[s["tool"]].append(s)

    # ---------------------------------------------------------------------------
    # 跟随位内容注册表（2026-11-11）——见 pick_content("follow") 的说明。
    # ---------------------------------------------------------------------------
    # _C_FOLLOW_TABLE: {说法: n1}（参考集里该说法在【内容域的跟随位】出现的次数）
    # _C_FOLLOW_TOOLS / _C_FOLLOW_TOOLW: 跟随位按工具的质量（educ 180 / vod 123 /
    #   recommend 76 / music_search 37 / playback 22），供"先定工具再定说法"用。
    # 两者都直接来自参考集实测，和总频次表是【两套】分布——跟随位是封闭语域，
    # 用总频次配权会把参考集从不跟随的说法（n1=0）拉进来。
    try:
        from ref_device_vocab import attested_content_leadfreq as _aclf
        _aclf_tab = _aclf()
    except Exception as _e:
        print(f"[content follow table] 不可用（{_e}）")
        _aclf_tab = {}
    _C_FOLLOW_TABLE: dict[str, int] = {}
    for (_q, _g), (_n0, _n1) in _aclf_tab.items():
        if _n1 > 0:
            _C_FOLLOW_TABLE[_q] = _C_FOLLOW_TABLE.get(_q, 0) + _n1
    _CONTENT_ITEM: dict[str, dict] = {}
    for _s in contents:
        _CONTENT_ITEM.setdefault(_s["q"], _s)     # 首见为准（同域同工具，无需再筛）
    _fq_present = [q for q in _C_FOLLOW_TABLE if q in _CONTENT_ITEM]
    _C_FOLLOW_TOOLW: dict[str, int] = collections.Counter()
    for _q in _fq_present:
        _C_FOLLOW_TOOLW[_CONTENT_ITEM[_q]["tool"]] += _C_FOLLOW_TABLE[_q]
    _C_FOLLOW_TOOLS = list(_C_FOLLOW_TOOLW)
    print(f"跟随位注册表：{len(_C_FOLLOW_TABLE)} 种说法（池内 {len(_fq_present)}），"
          f"工具 {dict(_C_FOLLOW_TOOLW)}")

    # ---------------------------------------------------------------------------
    # 【设备先行·内容后随】(DC) 联合语域（2026-11-11）
    # ---------------------------------------------------------------------------
    # 到本轮为止，DC 分支是"设备从 DC 池抽、内容从跟随位注册表抽"【两边独立】。
    # 两个后果叠加，把 DC 整个压歪：
    #   (a) 跟随位内容注册表以【播放句】为主（"播放第1集"…），几乎每条都
    #       provides_playback。于是 DC 行里内容句总在起播，而设备句在它前面；
    #   (b) order_ok 里那条【我自造】的"串行中非 power 设备句不得先于起播句"规则，
    #       把首位的非 power 设备句整类拒掉。实测：DC 池投入 power 0.743，产出 0.994。
    #   而 (b) 这条规则本身是错的——它对参考集自己的 5000 行串行【误拒 146 行，全部
    #   是 DC 形态】：REF 真的写"关闭屏幕随后播放音乐""退出广告看西游记""大屏播放
    #   第1集吧"。正确口径不是"非 power 一律禁"，而是按工具的位置条件性：
    #   power 能先于起播句 0.316、mode 0.356、screen_layout 0.017、numeric 0.000…
    #   （见 ref_device_vocab.lead_dev_before_play）。
    #
    # 修法就是内容跟随位那一轮的同一招：DC 是【封闭联合语域】——REF 串行 DC 共 196 行、
    # 62 个 (设备tool,设备说法,内容tool,内容说法) 对，是"先说把当前画面/声音弄成什么样、
    # 再说下一件看什么"的固定搭配表（关闭屏幕>>只听音乐 12、大点声>>最强大脑 9…）。
    # 直接按对频次联合抽：抽中即合规，机器再不会自造组合、也不会被那条错规则误杀。
    try:
        from ref_device_vocab import dc_pair_register as _dcr
        _DC_REG = _dcr("serial")
        # 并行侧的 DC 也封闭（REF 96 行 / 17 对），且工具构成与串行不同
        # （并行内容位 playback_control 32/96，串行以 vod 搜索为主），故分 mode 各存一张表。
        _DC_REG_PAR = _dcr("parallel")
    except Exception as _e:
        print(f"[DC register] 不可用（{_e}）")
        _DC_REG = {}
        _DC_REG_PAR = {}
    # 【内容金标+设备工具】(CD 方向跨标语域，2026-11-11)。见 ref_device_vocab.
    # cd_cross_register 的说明——这是 dc_pair_register 的镜像：首位子句的 gold 是
    # 【内容域】、工具却是设备工具（"播放云视听小电视"/source_switch、"把音乐关闭"/
    # playback_control、"看中央五台"/source_switch、"播放短视频"/screen_lift_rotation）。
    # 这些是真人说法（云视听小电视 REF 42 行、把音乐关闭 41 行），内容池里永远抽不到
    # （_ctool_pool 只装内容工具），必须联合抽。
    try:
        from ref_device_vocab import cd_cross_register as _ccr
        _CDX_REG = _ccr("serial")
        _CDX_REG_PAR = _ccr("parallel")
    except Exception as _e:
        print(f"[CD-cross register] 不可用（{_e}）")
        _CDX_REG = {}
        _CDX_REG_PAR = {}
    # 【内容+内容】(CC) 联合语域（2026-11-11）。REF 两步句里两个位置都是内容域的
    # 极少（串行 13 行 / 并行 29 行），是封闭的两种搭配（"播放音乐>>我要看汪汪队"、
    # "播放免费的小猪佩奇>>播放免费的宝宝巴士"）。生成器此前一律禁掉（CC≈0），
    # 但 REF 真有这 0.42%——按语域照搬即可，不会自造清单句。
    try:
        from ref_device_vocab import cc_pair_register as _ccpr
        _CC_REG = _ccpr("serial")
        _CC_REG_PAR = _ccpr("parallel")
    except Exception as _e:
        print(f"[CC register] 不可用（{_e}）")
        _CC_REG = {}
        _CC_REG_PAR = {}
    # 【设备+设备】(DD) 联合语域（2026-11-11，第十六轮）。DD 行占整卷约 30%，是最后一块
    # 仍在【独立抽两边】的槽位。原来先用 _DEV_PAIR 抽一对工具、再在各自工具内按 lw 抽说法，
    # 工具对联合、说法层却独立——实测 REF 的 DD 说法对互信息超自身零模型 +0.956 bit，
    # 我只有 +0.0003（几乎完全独立）。耦合实例：`打开下一集>>关掉电视`(1.00)、
    # `小屏播放>>大点声`(1.00)、`亮度15>>打开护眼`(1.00)、`提高亮度>>关闭护眼模式`(1.00)、
    # `下一曲>>音量减`(1.00)。这些是"两件事本就是一件"的真人搭配，独立抽永远产不出——
    # 读起来就是两句不相干的话被逗号连起来。见 ref_device_vocab.dd_pair_register。
    try:
        from ref_device_vocab import dd_pair_register as _ddr
        _DD_REG = _ddr("serial")
        _DD_REG_PAR = _ddr("parallel")
    except Exception as _e:
        print(f"[DD register] 不可用（{_e}）")
        _DD_REG = {}
        _DD_REG_PAR = {}
    # 【内容→设备条件分布】(CD 行，2026-11-11，第十七轮)。CD（内容先行、设备后随）
    # 行占整卷约 62%，是最后一块【两边独立抽】的槽位。实测 REF 的内容说法×设备说法
    # 互信息超自身独立零模型 +0.570 bit，我只有 +0.029。耦合实例：
    #   `播放抖音歌曲`>>`大点声`(1.00)  `只听音乐`>>`关闭屏幕`(1.00)
    #   `换一首歌`>>`音量小点`(1.00)   `播放吞噬星空`>>`大屏播放`(0.92)
    # CD【不是】封闭四元组表（6218 行里 5386 个不同二元组、83% 单例），故不能用
    # 逐字联合语域，改用【条件分布】+ 说法→工具兜底。held-out 泛化检验（REF 对半切）
    # 实测 excess 0.42~0.46 vs REF 0.37~0.43，逐位吻合（是真人规律、非小样本噪声）。
    # 见 ref_device_vocab.cd_cond_register。
    # 【min_count 5→3（2026-11-11，第二十轮）】桶粒度是"表达多样性"的直接旋钮。
    # 实测（末卷文件 vs REF 同 n 的联合多项抽样）：
    #   MC=5：CD 设备位 distinct 偏 ~11%（并行 769 vs REF 856±13，z=-6.9；串行 685 vs 763±12，z=-6.5），
    #         且 MI excess 0.268 < REF 子采样 0.3165±0.0206（**耦合不足**，读起来就是"内容与
    #         设备两句关系松散"）；MC=3：MI 0.308（与 REF 几乎重合）、shape 更好；
    #         MC=2：MI 0.379（**过拟合**，超出 REF 上界）；MC=1：MI 0.500（把单例桶照搬噪声）。
    # MC=5 把 84% 的内容说法推给工具兜底桶，粒度太粗、丢了说法级的耦合与尾部覆盖；MC=3
    # 覆盖 34–36% 而 MI 恰好落在 REF 区间。held-out 检验当初选 5 是怕过拟合，本轮用
    # 【末卷实测 MI】对齐 REF 把粒度重新定在 3（见 [[survivorship-calibration]]：feed 变则实测变）。
    try:
        from ref_device_vocab import cd_cond_register as _cdc
        _CD_COND = _cdc("serial", min_count=int(__import__("os").environ.get("CD_MC","3")))
        _CD_COND_PAR = _cdc("parallel", min_count=int(__import__("os").environ.get("CD_MC","3")))
    except Exception as _e:
        print(f"[CD cond register] 不可用（{_e}）")
        _CD_COND = ({}, {}, {})
        _CD_COND_PAR = ({}, {}, {})
    _load_lead_dev_play(provides_playback)
    _DC_KEYS = list(_DC_REG)
    _DC_W = [_DC_REG[k][0] for k in _DC_KEYS]
    _DC_KEYS_PAR = list(_DC_REG_PAR)
    _DC_W_PAR = [_DC_REG_PAR[k][0] for k in _DC_KEYS_PAR]
    _CDX_KEYS = list(_CDX_REG)
    _CDX_W = [_CDX_REG[k][0] for k in _CDX_KEYS]
    _CDX_KEYS_PAR = list(_CDX_REG_PAR)
    _CDX_W_PAR = [_CDX_REG_PAR[k][0] for k in _CDX_KEYS_PAR]
    _CC_KEYS = list(_CC_REG)
    _CC_W = [_CC_REG[k][0] for k in _CC_KEYS]
    _CC_KEYS_PAR = list(_CC_REG_PAR)
    _CC_W_PAR = [_CC_REG_PAR[k][0] for k in _CC_KEYS_PAR]
    _DD_KEYS = list(_DD_REG)
    _DD_W = [_DD_REG[k][0] for k in _DD_KEYS]
    _DD_KEYS_PAR = list(_DD_REG_PAR)
    _DD_W_PAR = [_DD_REG_PAR[k][0] for k in _DD_KEYS_PAR]
    print(f"DC 联合语域：串行 {len(_DC_KEYS)} 对 / {sum(_DC_W)} 行；"
          f"并行 {len(_DC_KEYS_PAR)} 对 / {sum(_DC_W_PAR)} 行；"
          f"lead-dev-before-play {dict(_LEAD_DEV_PLAY)}")
    print(f"CD 跨标语域：串行 {len(_CDX_KEYS)} 对 / {sum(_CDX_W)} 行；"
          f"并行 {len(_CDX_KEYS_PAR)} 对 / {sum(_CDX_W_PAR)} 行")
    print(f"内容+内容语域：串行 {len(_CC_KEYS)} 对 / {sum(_CC_W)} 行；"
          f"并行 {len(_CC_KEYS_PAR)} 对 / {sum(_CC_W_PAR)} 行")
    print(f"设备+设备语域：串行 {len(_DD_KEYS)} 对 / {sum(_DD_W)} 行；"
          f"并行 {len(_DD_KEYS_PAR)} 对 / {sum(_DD_W_PAR)} 行")
    print(f"CD 条件设备分布：串行 {len(_CD_COND[0])} 个内容说法桶 / "
          f"{len(_CD_COND[1])} 个工具桶；并行 {len(_CD_COND_PAR[0])} / "
          f"{len(_CD_COND_PAR[1])}")
    # 【三步句整行语域】(2026-11-11，第十八轮)。三步句是封闭集（REF 243 行、
    # 串行 51 三元组 / 并行 62），此前是"两步句 + 独立 append 一条设备句"拼出来的，
    # 实测把内容域摊平到六个域（REF 80% 影视）、设备工具对 top 只 11%（REF 82%
    # numeric×numeric）、签名说法（亮度高一点/屏幕亮度提高/点亮度低点）整类产不出。
    # 整行逐字照搬即无偏。见 ref_device_vocab.triple_register。
    try:
        from ref_device_vocab import triple_register as _tr
        _TRI_REG = _tr("serial")
        _TRI_REG_PAR = _tr("parallel")
    except Exception as _e:
        print(f"[triple register] 不可用（{_e}）")
        _TRI_REG = {}
        _TRI_REG_PAR = {}
    _TRI_KEYS = list(_TRI_REG)
    _TRI_W = [_TRI_REG[k][0] for k in _TRI_KEYS]
    _TRI_KEYS_PAR = list(_TRI_REG_PAR)
    _TRI_W_PAR = [_TRI_REG_PAR[k][0] for k in _TRI_KEYS_PAR]
    print(f"三步句整行语域：串行 {len(_TRI_KEYS)} 三元组 / {sum(_TRI_W)} 行；"
          f"并行 {len(_TRI_KEYS_PAR)} / {sum(_TRI_W_PAR)} 行")

    def pick_triple(mode: str = "serial") -> list[dict] | None:
        """抽一条三步句【整行语域】（逐字照搬 REF 的三格工具/说法/参数/金标域）。"""
        keys = _TRI_KEYS_PAR if mode == "parallel" else _TRI_KEYS
        reg = _TRI_REG_PAR if mode == "parallel" else _TRI_REG
        wts = _TRI_W_PAR if mode == "parallel" else _TRI_W
        if not keys:
            return None
        k = rng.choices(keys, weights=wts)[0]
        n, p0, p1, p2, d0, d1, d2 = reg[k]
        return [{"q": k[1], "d": d0, "tool": k[0], "params": p0},
                {"q": k[3], "d": d1, "tool": k[2], "params": p1},
                {"q": k[5], "d": d2, "tool": k[4], "params": p2}]

    def _cd_cond_device(mode: str, csurf: str, ctool: str) -> dict | None:
        """给定内容子句的首位说法，从参考集的【条件设备分布】里抽一条设备子句。

        按内容说法分桶（open set，故非逐字联合语域）：桶存在（该说法在 REF 的
        CD 行里出现 ≥ min_count 次）时按桶内频次抽；否则退回【内容tool】桶；
        再不行返回 None（调用方退回独立抽）。min_count 与『过拟合』的取舍见
        ref_device_vocab.cd_cond_register 的说明（held-out 检验定在 5）。
        """
        by_c, by_t, _ = _CD_COND_PAR if mode == "parallel" else _CD_COND
        rows = by_c.get(csurf)
        if not rows:
            rows = by_t.get(ctool)
        if not rows:
            return None
        it = rng.choices(rows, weights=[r[0] for r in rows])[0]
        _n, dtool, dsurf, dparams = it
        return {"q": dsurf, "d": "device", "tool": dtool, "params": dparams}

    def pick_cc_pair(mode: str = "serial") -> list[dict] | None:
        """抽一条【内容+内容】语域对（逐字照搬 REF）。两个位置都是内容域。"""
        keys = _CC_KEYS_PAR if mode == "parallel" else _CC_KEYS
        reg = _CC_REG_PAR if mode == "parallel" else _CC_REG
        wts = _CC_W_PAR if mode == "parallel" else _CC_W
        if not keys:
            return None
        t0, q0, t1, q1 = rng.choices(keys, weights=wts)[0]
        _n, p0, p1, d0, d1 = reg[(t0, q0, t1, q1)]
        return [{"q": q0, "d": d0, "tool": t0, "params": p0},
                {"q": q1, "d": d1, "tool": t1, "params": p1}]

    def pick_cdx_pair(mode: str = "serial") -> list[dict] | None:
        """抽一条【内容金标·设备工具在前、设备句后随】的跨标语域对（逐字照搬 REF）。

        与 pick_dc_pair 互为镜像：这是 CD 形态，但首位子句的 d 是【内容域】
        （影视/音乐…），工具却是设备工具（source_switch/playback_control/
        screen_lift_rotation/power_control）。d 必须取语域里记的 gold 域，不能由
        工具名反推——否则 join_domains 与金标都会错。

        参数按对留白（REF 里该对从未带过非空参数就留白，与设备/内容单侧同口径）。
        """
        keys = _CDX_KEYS_PAR if mode == "parallel" else _CDX_KEYS
        reg = _CDX_REG_PAR if mode == "parallel" else _CDX_REG
        wts = _CDX_W_PAR if mode == "parallel" else _CDX_W
        if not keys:
            return None
        lt, lq, ft, fq = rng.choices(keys, weights=wts)[0]
        _n, lpar, fpar, ldom = reg[(lt, lq, ft, fq)]
        a = {"q": lq, "d": ldom, "tool": lt, "params": lpar}
        b = {"q": fq, "d": "device", "tool": ft, "params": fpar}
        return [a, b]

    def pick_dc_pair(mode: str = "serial") -> list[dict] | None:
        """抽一条【设备先行·内容后随】的串行对（逐字照搬 REF 的 (tool,说法,参数)）。

        参数按对留白：REF 里该对若从未带过非空参数，就不要凭空编一个（与设备/内容
        单侧 nullrate 同口径）。

        mode="parallel" 走并行封闭表——并行 DC 与串行不是同一批语料，工具构成不同。
        """
        keys = _DC_KEYS_PAR if mode == "parallel" else _DC_KEYS
        reg = _DC_REG_PAR if mode == "parallel" else _DC_REG
        wts = _DC_W_PAR if mode == "parallel" else _DC_W
        if not keys:
            return None
        dt, dq, ct, cq = rng.choices(keys, weights=wts)[0]
        _n, dpar, cpar, cdom = reg[(dt, dq, ct, cq)]
        a = {"q": dq, "d": "device", "tool": dt, "params": dpar}
        b = {"q": cq, "d": cdom, "tool": ct, "params": cpar}
        return [a, b]

    # ---------------------------------------------------------------------------
    # 内容子句权重 = 该说法自己的【参考集频次】（2026-11-10）。
    # ---------------------------------------------------------------------------
    # 走过两代弯路，记在这里免得回退：
    #   1. 最初按【动词】赋权再以 0.72 概率随机剥前缀，凭空造出参考集里罕见的光杆
    #      "看/听"祈使句（实测 3.9-5.0 倍），把"我要看/我想看"压到 0.14 倍——
    #      "通篇祈使句"，正是用户说的"不像真实用户"。请求语与动词在参考集里是强相关的
    #      （"我要看"4.3% + "我想看"3.5% vs 光杆"看"1.0%），不能当两个独立维度。
    #   2. 随后按 (请求语,动词|工具) 条件占比配权。这在 lw 只是长度启发式时是对的，
    #      但 lw 统一成参考频次之后就成了重复计算（详见 pick_content 的说明）。
    # 现值：pick_content 直接以 lw 为句内权重，分布自然落在参考集形状上。
    # 内容位的 null 频率（参考集口径），用于决定该说法是否留白。
    try:
        from ref_device_vocab import content_nullrate as _cnr
        _CONTENT_NULL = _cnr()
    except Exception as _e:
        print(f"[content nullrate] 参考集统计不可用（{_e}）")
        _CONTENT_NULL = {}

    # 权重即参考实测占比，不做解析校正。真正的偏差来源是【存活偏差】：
    # 各工具的子句在 compatible()/order_ok() 里的通过率不同（vod_search_all 的子句
    # 常配合任意设备子句，educ 的子句常与内容域冲突），被拒概率不同，最终占比就会
    # 偏离投入占比。这类偏差无法从池内统计推出，只能"实测→反馈"迭代收敛：
    # 先按实测占比投放，量出产出占比，再用 target/realized 反乘一次。
    # 下面的 _CAL 就是一次这样量出来的修正系数。
    # 系数口径：CAL_t = target_t / realized_t（在 CAL 全为 1 的基线产出上量的），
    # 即"产出比目标低就加权、高就减权"。上一版我把这个乘反了（乘成 realized/target），
    # 一迭代就把 audio_search 从 12.4% 砸到 4.0%、再迭代又反弹，来回振荡。
    # 这些数来自 CAL=1 那一轮：vod 15.2/17.3、audio 12.4/10.1、educ_search_all 10.2/13.1、
    # music 9.5/7.6、edu_fuzzy 6.6/6.5、fan 6.5/4.0、vod_fuzzy 6.1/5.6、edu_search 5.4/4.9、
    # recommend 4.9/3.5、sports_vod 3.4/2.9。
    # 【2026-11-11 第十九轮重校】：改了并行/串行形状 feed 与三步句语域后，各内容工具的
    # 合成占比发生了整体漂移（survivorship-calibration：feed 一变，各分支的存活率跟着变，
    # 旧的 CAL 就不准了）。在 5 seed 均值上量 REF/MINE 比值 r，新 CAL = 旧 CAL / r。
    # 校准前偏差最大：sports_team_search 1.29×、vod_search_all 1.15×、
    # tvchannel_music_search 0.68×、music_ksong_search 0.75×、music_song_mv_search 0.69×、
    # vod_person_search 0.76×、educ_history 3.85×（n 极小，忽略）。
    # ⚠️ 小 n 工具（<1.5%）的 CAL 噪声大，别按单 seed 调。
    _CAL = {
        "audio_search": 0.894, "vod_search_all": 0.748, "educ_search_all": 0.878,
        "music_song_search": 0.789, "edu_fuzzy_search": 0.913, "vod_fuzzy_search": 0.964,
        "edu_search": 0.899, "fan_knowledge_agent": 0.394, "music_song_recommend": 0.663,
        "sports_vod_search": 0.909, "sports_match_search": 1.021, "sports_team_search": 0.893,
        "music_song_history": 0.978, "tvchannel_music_search": 0.895, "sports_rank_search": 1.030,
        "sports_match_reservation": 1.060, "music_ksong_search": 0.920,
        "music_song_favorite_search": 0.998, "music_song_mv_search": 0.943,
        "vod_personalized_search": 0.989,
        "music_song_qqmusic_search": 0.941, "vod_person_search": 0.724, "vod_relate_search": 1.063,
    }
    _ctool_pool = [t for t, w in _CONTENT_TOOL_W.items() if t in _ctool
                   for _ in range(max(1, round(w * _CAL.get(t, 1.0))))]

    def pick_content(pos: str = "") -> dict:
        if pos == "follow":
            # 【跟随位是一个封闭语域，不是"首位分布的一个切片"】（2026-11-11）
            #
            # 参考集跟随位的内容位（gold∈内容域）总共 438 例、只由 23 种说法构成，
            # 且高度集中：「播放第1集」一种就占 10.5%。这是一套【固定的说法表】——
            # 真人不会在"关电视"之后随机甩一个片名，他会说"播放第1集""只看音乐"
            # "放个汪汪队"这类短点播句。
            #
            # 旧实现是"先按 _CONTENT_TOOL_W 抽工具、再在工具内按 lw 抽说法"，
            # 两个环节都用了【全体口径】的分布，于跟随位产出【九十种各出现一次】的
            # 长尾说法（"播放都挺好第一集""学习基本功形体训练""来点高二英语"…），
            # 参考集里这 23 种封闭表一种都没落到应有的份额上。
            #
            # 实测：周转位(总频次)配权下「播放第1集」份额 0.026，参考真值 0.105——
            # 差 4 倍；而我产出的跟随位内容位 106 例里，最常见的一种只占 2.8%。
            # 换成【跟随位频次(n1)】配权后，"播放第1集"份额 = 0.105，与参考逐位吻合。
            #
            # 故跟随位【完全绕开】全局工具池与总频次：工具按注册表实测跟随质量抽
            # （educ 180 / vod 123 / recommend 76 / music_search 37 / playback 22），
            # 说法按 n1 抽。注册表只有 23 种，是【参考集里确实跟随过】的说法，
            # 不在表内的一律不得进入跟随位（这才是"封闭语域"的字面含义）。
            _fq = [q for q in _C_FOLLOW_TABLE if q in _CONTENT_ITEM]
            if _fq:
                _tool = rng.choices(_C_FOLLOW_TOOLS,
                                    weights=[_C_FOLLOW_TOOLW[t] for t in _C_FOLLOW_TOOLS])[0]
                _cand = [q for q in _fq if _CONTENT_ITEM[q]["tool"] == _tool]
                if not _cand:                      # 该工具在表内的说法已用尽 → 退回全表
                    _cand = _fq
                _q = rng.choices(_cand, weights=[_C_FOLLOW_TABLE[q] for q in _cand])[0]
                s = dict(_CONTENT_ITEM[_q])
                if _CONTENT_NULL.get(s["q"]) and rng.random() < _CONTENT_NULL[s["q"]]:
                    s = {**s, "params": None}
                return s
        tool = rng.choice(_ctool_pool)
        lst = _ctool[tool]
        # 位置许可（2026-11-10，与设备侧同因）：先按目标位置筛说法，再按位置专属权重抽。
        # 【首位】：说法级证据（content_may_lead）——参考集里 2662/2684 种说法在首位出现过，
        #   闸门只收掉极少数"从不句首"的（"只听音乐""播放第1集"…），实测我首位违反率 3.3%。
        # 【跟随位】：见上面 2026-11-11 的说明——改为封闭注册表直抽，不再走此处。
        # 筛空时退回原池（不硬性失败）——宁可放行也不让生成器无句可用。
        if pos == "lead":
            _lp = [s for s in lst if content_may_lead(s["q"])]
            if _lp:
                lst = _lp
            # 【首位按 n0 配权】（2026-11-11）——见 _c_lead_w 的说明。
            _w = [_c_lead_w(s) for s in lst]
            # 同句内路径：整桶被语域覆盖（权重全零）时退回全额频次，保证有句可用。
            if not any(_w):
                _w = [s.get("lw_full", _c_len_w(s["q"])) for s in lst]
            s = rng.choices(lst, weights=_w)[0]
        else:
            # 句内权重 = lw【就是】参考集频次，不再叠加框架层。
            #
            # 曾经这里是两级：lw × P((请求语,动词)|tool) / 该档 lw 之和。当时 lw 只是
            # 长度启发式（≤1.0），框架层负责表达"该工具里哪种句式常见"。但后来 lw 统一
            # 成【该子句自己的参考集频次】之后，框架层就变成了【重复计算】：
            # 一个 (请求语,动词) 档的 lw 之和，本来就等于该档在参考集里的全部质量，
            # 再乘一次 P(档|tool) 等于把"档占比"平方了一次。
            #
            # 更要命的是它除以【池内】lw 之和，于是某些档若有成员被池子漏掉，
            # 剩下的成员会把整档的目标质量【吃掉】——实测 vod 的"打开电影长安的荔枝"
            # 被放大 2.57 倍、"播放成何体统" 1.76 倍，正是这些"幸存的兄弟"。
            # 离线对比：只按 lw 配权 totalvar 22.2%、叠加框架层 26.8%。
            # 请求语与动词的分布无需框架层表达——它由 lw 对档内求和自然给出，
            # 且天然带上"请求语与动词强相关"（"我要看"和"看"是两个不同的档）。
            _w = [s.get("lw", _c_len_w(s["q"])) for s in lst]
            # 【残差全零的兜底】本桶若整桶都被语域覆盖（lw 扣到 0），主池无候选可抽。
            # 退回长度权重——但只在【命令】这一级有效；本桶若无候选可退，直接用 lw_full
            # （全额参考频次），保证生成器永远有句可用（宁可轻微双计也不失败）。
            if not any(_w):
                _w = [s.get("lw_full", _c_len_w(s["q"])) for s in lst]
            s = rng.choices(lst, weights=_w)[0]
        # 请求语不再随机剥掉。原先以 0.72 的概率剥前缀，是为了把旧种子池
        # 38.7% 的请求语压到参考集的 7.3%；但参考集里请求语与动词是【强相关】的
        # （"我要看/我想看" 各 4.3/3.5%，光杆"看"仅 1.0%），随机剥前缀等于凭空造出
        # 参考集里罕见的光杆祈使句（实测光杆"看" 3.9 倍、光杆"听" 5.0 倍）。
        # 相关性已由上面的联合权重表达，这里不再二次加工。
        #
        # 参数未标注（null）对齐：与设备侧同理。参考集内容位 26.5% 的 tool_param 是
        # null，且按说法系统性；judge() 对 null 步不计参数分。按该说法的参考 null 频率
        # 抽查一次决定是否留白——必须 copy，池里的 dict 是共享的。
        if _CONTENT_NULL.get(s["q"]) and rng.random() < _CONTENT_NULL[s["q"]]:
            s = {**s, "params": None}
        return s

    rows: list[dict] = []
    half = args.n // 2
    plan = [("parallel", half), ("serial", args.n - half)]

    # 统计用
    used_q = set()

    for mode, cnt in plan:
        got = 0
        guard = 0
        while got < cnt and guard < cnt * 60:
            guard += 1
            # 三步句占比按参考集【分模式】实测配平：全体 2.4%（243/10000），
            # 其中 parallel 2.80%、serial 2.06%。两步句是绝对多数。
            # 我原来一律给 10%（实测 8.5%），是参考的 3.5 倍，句长也被撑到 26 字。
            # 【2026-11-11 第十八轮重校（三步句整行语域上线后）】：整行语域抽样
            # 抽中即通过（_from_reg 豁免 compatible/order_ok），拒绝率与旧的
            # "两步句 + 独立 append" 不同——实测 feed(ser .0206/par .028) →
            # realized(ser .0240/par .0228)：存活 ser 1.165、par 0.814。
            # 目标 REF realized(ser .0206/par .0280) 反推 feed：
            #   ser .0206/1.198=.0172  par .0280/.907=.0309。
            n_steps = 2 if rng.random() >= ((_PAR3_F if mode == "parallel" else _SER3_F)) else 3
            segs: list[dict] = []
            # 结构：内容域 + 设备（占多数）；设备+设备；内容域+内容域。
            # 参考集结构实测：内容+设备 63%、设备+设备 32%、内容+内容 ~0.2%。
            # 我上一版内容+内容给了 15%（整卷 8%），而"两件事都是点播内容"的句子在参考里
            # 几乎不存在，是最容易穿帮的一类。
            # 形状配比按参考集【分模式】实测，而不是全体一个比例：
            #   parallel: 内容+设备 68.6%、设备+设备 29.2%、设备+内容 1.7%、内容+内容 0.6%
            #   serial  : 内容+设备 59.2%、设备+设备 34.5%、设备+内容 6.0%、内容+内容 0.3%
            # 我原来用全体口径 66/32/2，产出 CD 43.2%、DC 24.1%——DC 超了 6 倍，
            # 一半的行是「先调设置、再说我要看什么」，正是用户说的"生硬"。
            _ser = mode == "serial"

            def _pick_dev_pair(serial: bool) -> list[dict]:
                # 【设备+设备按联合语域整对抽】（2026-11-11，第十六轮）。原来先用
                # _DEV_PAIR 抽一对【工具】、再各自在工具内按 lw 抽说法——工具对联合、
                # 说法层独立，实测 REF 的 DD 说法对互信息超自身零模型 +0.956 bit、
                # 我只有 +0.0003（几乎完全独立）。现改为逐字照搬 REF 的
                # (tool,说法,参数) 四元组，与 DC/CDX/CC 三张表同法。
                # 耦合实例：`打开下一集>>关掉电视`(1.00)、`小屏播放>>大点声`(1.00)、
                # `提高亮度>>关闭护眼模式`(1.00)——"两件事本就是一件"的搭配。
                keys = _DD_KEYS_PAR if not serial else _DD_KEYS
                reg = _DD_REG_PAR if not serial else _DD_REG
                wts = _DD_W_PAR if not serial else _DD_W
                if not keys:
                    return [pick_device(serial, pos="lead"),
                            pick_device(serial, pos="follow")]
                at, aq, bt, bq = rng.choices(keys, weights=wts)[0]
                _n, ap, bp = reg[(at, aq, bt, bq)]
                a = {"q": aq, "d": "device", "tool": at, "params": ap}
                b = {"q": bq, "d": "device", "tool": bt, "params": bp}
                return [a, b]

            r = rng.random()
            _from_reg = False                  # 本行是否逐字照搬 DC 联合语域（保序，不重排）
            _from_triple = False               # 本行是否逐字照搬三步句整行语域
            if n_steps == 3:
                # 【三步句整行抽】(2026-11-11，第十八轮)。三步句是封闭语域，且
                # 它的性质与两步句【正交】：REF 243 条三步句 80% 是影视+两个调节
                # （"我想看长安的荔枝#声音大点#亮度高一点"），内容说法只有 10 种、
                # 设备工具对 82% numeric×numeric。原来"先抽两步句再独立 append 一条
                # 设备句"把这三层结构全打散（域摊平、工具对 top 11%、签名说法产不出）。
                # 整行语域逐字照搬即无偏（含单例：单例以 1/103 的比率被抽中）。
                _trip = pick_triple(mode)
                if _trip is not None:
                    segs = _trip
                    _from_triple = True
                    _from_reg = True           # 保序、豁免启发式闸门（与 DC 语域同）
            if _from_triple:
                pass                           # 整行已定，跳过下面的形状分支
            elif mode == "parallel":
                # 【2026-11-11 第十二轮：并行形状按 gold 口径重定】REF 并行 gold 形状
                # CD 0.6894 / DD 0.2874 / DC 0.0232（含跨标行——跨标行 gold 是内容域，
                # 故归 CD）。上一轮 feed(.690/.291/.019) → realized(.6956/.2852/.0192)，
                # 存活 CD 1.008、DD 0.980、DC 1.01。反推目标 feed：
                #   .6894/1.008=.684   .2874/.980=.293   .0232/1.01=.023  （合计 1.000）
                # 【2026-11-11 重校（CD 槽 power 权重修正后）】：上表把并行 CD 槽里
                # power/timer（各 ~5.7%/3.2%，REF 并行 CD 槽里恒 0%）去掉后，CD 行的
                # 拒绝率下降、存活率上升，realized CD 从 .6894 抬到 ~.696。feed 必须
                # 重新实测（survivorship-calibration）：0.675/0.978 → CD .3333（REF .3332）、
                # DD .1440（REF .1418）、DC .0096（REF .0081），三格 |Δ|≤0.002。
                # 【2026-11-11 第十七轮重校（CD 条件设备抽上线后）】：CD 行的设备子句
                # 现由【参考集条件分布】抽（内容说法→设备说法），撞上 order_ok/compatible
                # 被拒的概率下降 ⇒ CD 存活率上升。实测 feed .675/.303/.022 →
                # realized CD .6948 / DD .2860 / DC .0175（存活 1.029 / .944 / .796）。
                # 目标 REF 并行 CD .6856 / DD .2918 / DC .0167 反推 feed：
                #   .6856/1.029=.666  .2918/.944=.309  .0167/.796=.021  （合计 .996）。
                if r < _PAR_CD_B:
                    # CD 分支里混有两种封闭子语域：
                    #  (a) 跨标（内容金标 + 设备工具在首位）：REF 并行 84/5000=0.0168，
                    #      占 CD 分支 0.0168/0.684≈0.0245；实测 feed 0.0340→realized
                    #      0.0220、feed 0.0243→0.0116（斜率≈1.07），目标 0.0168⇒feed 0.029。
                    #  (b) 内容+内容：REF 并行 29/5000=0.0058，占 CD 分支 0.0085。
                    # 顺序：先掷 CC（更稀有），再掷 CDX，余下走普通 CD。
                    # （survivorship-calibration：feed≠realized，只能实测迭代。）
                    _u = rng.random()
                    if _u < _PAR_CC:
                        segs = pick_cc_pair("parallel")
                        _from_reg = segs is not None
                    elif _u < _PAR_CC + _PAR_CDX:
                        segs = pick_cdx_pair("parallel")
                        _from_reg = segs is not None
                    else:
                        # 【CD 条件设备抽】(2026-11-11，第十七轮)。原来这里是
                        # "内容从首位池独立抽 + 设备从单槽池独立抽"，两边独立 ⇒
                        # 内容说法与设备说法近乎无关（实测 MI excess +0.029 vs REF +0.570）。
                        # 改为：先抽内容句，再【以内容说法为条件】从参考集条件分布里
                        # 抽设备句（`播放抖音歌曲`→`大点声`、`换一首歌`→`音量小点`）。
                        # 条件桶缺该说法时退回内容工具桶，再不行退回独立抽。
                        _cs = pick_content(pos="lead")
                        _dv = _cd_cond_device("parallel", _cs["q"], _cs["tool"])
                        if _dv is None:
                            _dv = pick_device_single(False, pos="follow")
                        segs = [_cs, _dv]
                elif r < _PAR_DD_B:
                    segs = _pick_dev_pair(False)
                else:
                    # 并行 DC 目标：REF 并行 gold 口径 DC 0.0232。上一轮 feed 0.019 →
                    # realized 0.0192（存活 1.01）⇒ feed ≈ 0.023。原来这 1.9% 里有一大半
                    # 来自那句 2.4% 倒序硬币（把 CD 行翻成设备先行），本轮把硬币关掉后，
                    # 语域必须自己承担全部（语域抽中即通过，见 [[survivorship-calibration]]）。
                    # 并行 DC 同样是封闭语域（REF 96 行 / 17 对），且工具构成与串行
                    # 不同——独立抽设备/内容会产出参考集从未出现的组合
                    # （实测内容位错成 educ 0.42，REF 是 playback 0.33/vod 0.27）。
                    segs = pick_dc_pair("parallel")
                    _from_reg = segs is not None
            else:
                # 串行的形状配比（2026-11-11 第二轮）：参考集 ser 实测
                # CD 0.580 / DD 0.338 / DC 0.059。旧值（CD 0.600、DC 0.032）是按
                # 「全体口径」定的，DC 因此长期只有 0.022（参考 0.059，差 2.7 倍）。
                # 参考集串行的 DC 正是"先关掉电视、再起播下一件"那一类
                # （"关掉电视随后播放第1集""打开主页然后把歌关掉吧"），
                # 与跟随位注册表是【同一批语料】，故必须一并补上。
                # 投入概率按实测反推：DD 因 compatible/order_ok 通过率低于投入，
                # DC 因注册表刚建立、抽中即通过，故 DC 投入略高于目标值。
                # 投入概率按【实测反馈】定（survivorship-calibration：feed 与 realized
                # 不相等，只能实测迭代）。
                # 【2026-11-11 第十一轮重新校准】：DC 语域修好工具错位后从 196 行涨到 300 行
                # （+53%），语域变大 ⇒ 同一 feed 下"抽中即通过"的绝对量上升，但
                # CD/DD 的相对占比也随之变化。实测 feed(CD .596/DD .353/DC .051)
                # → realized(CD .614/DD .346/DC .040)：存活率 CD 1.03、DD 0.98、
                # DC 0.78（DC 损失来自 _from_reg 行仍要过 compatible/order_ok）。
                # 目标 REF realized(CD .600/DD .340/DC .060) 反推 feed：
                #   .600/1.03=.583   .340/0.98=.347   .060/0.78=.077   （合计 1.007）
                # 【2026-11-11 第十二轮：口径校准】
                # 形状必须按【gold 列】分类，且 gold 永不参与工具重排（gold[i] 与
                # segs[i] 逐位对应；只有 tools 列会与 seg/gold 错位，见 _pair_tools）。
                # 我一度把 gold 也一起重排，得出了错误的"REF ser CD .620/DC .040"，
                # 据此把 feed 改到 .608/.956，结果 realized CD .622/DC .040 偏离真值。
                # 正确口径（tools 只重排、gold 不动）：REF 串行 CD .6004/DD .3396/DC .0600。
                # 本轮实测 feed(CD .608/DD .348/DC .044) → realized(CD .6216/DD .3380/DC .0404)：
                # 存活 CD 1.022、DD 0.971、DC 0.918。目标反推 feed：
                #   .6004/1.022=.587  .3396/.971=.350  .0600/.918=.065  （合计 1.002）
                # 【2026-11-11 第十七轮重校（CD 条件设备抽上线后）】：同并行侧——
                # CD 设备子句改由参考集条件分布抽，存活率上升。实测 feed .585/.354/.061
                # → realized CD .5973 / DD .3415 / DC .0592（存活 1.021 / .965 / .970）。
                # 目标 REF 串行 CD .5922 / DD .3447 / DC .0604 反推 feed：
                #   .5922/1.021=.580  .3447/.965=.357  .0604/.970=.062  （合计 .999）。
                if r < _SER_CD_B:
                    # CD 分支里混有两种封闭子语域：
                    #  (a) 跨标：REF 串行 102/5000=0.0204，占 CD 分支 0.0204/0.600≈0.034；
                    #      实测 feed 0.031→realized 0.0200（斜率≈0.65），目标⇒feed 0.032。
                    #  (b) 内容+内容：REF 串行 13/5000=0.0026，占 CD 分支 0.0043。
                    _u = rng.random()
                    if _u < _SER_CC:
                        segs = pick_cc_pair("serial")
                        _from_reg = segs is not None
                    elif _u < _SER_CC + _SER_CDX:
                        segs = pick_cdx_pair("serial")
                        _from_reg = segs is not None
                    else:
                        # 【CD 条件设备抽】(2026-11-11，第十七轮)——见并行侧同处说明。
                        _cs = pick_content(pos="lead")
                        _dv = _cd_cond_device("serial", _cs["q"], _cs["tool"])
                        if _dv is None:
                            _dv = pick_device_single(True, pos="follow")
                        segs = [_cs, _dv]
                elif r < _SER_DD_B:
                    segs = _pick_dev_pair(True)
                else:
                    segs = pick_dc_pair()
                    # DC 联合语域抽出的对是【有序的】（REF 里就是设备先说、内容后随），
                    # 与 _DEV_PAIR 同理，下面的串行重排/换序不得再打散它。
                    _from_reg = segs is not None
            if segs is None or any(s is None for s in segs):
                continue                       # pick_device 位置闸门无解 → 重抽整行
            # 记住本行期望的句序（内容是否在前），供下面在串行重排时沿用。
            # 串行原来无条件把内容子句提到最前——那是为「播控依赖起播」写的，
            # 但它会把参考集里真实存在的 9.3% 设备先行串行（"关掉电视随后播放强军战歌~"
            # "打开主页然后把歌关掉吧"）全部改写成内容先行，使 DC 变成 0。
            _want_c_first = segs[0]["d"] != "device"
            if _from_reg:
                _want_c_first = False          # 语域对已定序（设备先），不参与重排
            # 问句开头（"怎么预习新课才有效？，同时进入wifi强度测试"）：参考集 205 行，
            # 形状固定——问句在第 0 位、第二步必为设备子句、全部两步。占参考集 2.05%。
            # 问句必须【先于】下面的串行重排逻辑放在首位，否则会被挪走。
            _is_q = False
            if _qpool and n_steps == 2 and rng.random() < 0.021:
                _s = rng.choices(_qpool, weights=[s["w"] for s in _qpool])[0]
                _d = pick_device()
                if _s["q"] != _d["q"] and not (
                        _s["tool"] == _d["tool"] and _s["q"] == _d["q"]):
                    segs = [_s, _d]
                    _is_q = True
            if n_steps == 3 and not _from_triple:
                # 三步句【兜底】：整行语域不可用时（抽到空表）才走这里——仍是"两步句 +
                # 追加一条设备句"。正常情况下 _from_triple 已在上方整行抽定，不走此处。
                segs.append(pick_device(mode == "serial"))
            # 串行：内容起播必须在设备控制之前（跳转/循环/下一集都依赖起播），
            # 故把第一个内容域子句排到最前，其余顺序不动。
            # 问句行：语序固定（问句首位、设备随后），不参与下面的换序/挪位。
            if mode == "serial" and not _is_q and not _from_reg:
                # 只有当【本行期望内容先行】时，才把内容子句提到最前。
                # 原来无条件提，把所有「设备先、内容后」的串行行都改成了内容先行。
                if _want_c_first:
                    ci = next((i for i, s in enumerate(segs) if s["d"] != "device"), None)
                    if ci and ci > 0:
                        segs.insert(0, segs.pop(ci))
                # 定时类（timer_control）挪到最末：参考集 281 行全部在末位（"…45分钟后关机"）。
                gi = next((i for i, s in enumerate(segs)
                           if is_global_like(s) and s["tool"] == "timer_control"), None)
                if gi is not None and gi != len(segs) - 1:
                    segs.append(segs.pop(gi))
                # 关机/开机/重启（power_control）：参考集里首位 443 / 末位 497，近乎对半。
                # "关掉电视随后播放第1集"（先关再说下一件）与"播放X，随后关机"（说完就关）
                # 都是真人说法。原来一律钉到末位，使 power_control 作为首子句的占比 = 0%
                # （参考 47%），也整类删掉了"先关后播"的说法。这里按参考配比随机定序。
                pi = next((i for i, s in enumerate(segs) if s["tool"] == "power_control"), None)
                if pi is not None:
                    # 0.60 是【实测反馈】定的，不是参考的 0.47：投入概率 p 与产出占比不是
                    # 恒等——首位已被内容子句占住时（发函后 pi!=0 也挪不动），且三步句里
                    # 关机只能落到第 1/2 位。实测 p=0.47→产出 37%、p=0.60→45%、p=0.70→45%
                    # （见 [[survivorship-calibration]]：这类偏差只能实测迭代，不能解析推出）。
                    # 取 0.60，产出 45% 贴近参考的 47%。
                    # 【2026-11-11 修 bug】原来写成
                    #     if rng<0.60 and pi!=0: 提到首位
                    #     elif pi < len-1:       挪到末位
                    # 当 power 已经【本来就在首位】(pi==0) 时：第一支因 `pi!=0` 为假而落空，
                    # 于是掉进 elif 被【一律挪到末位】——即"起手就在首位的行"被无条件罚到末尾，
                    # 与那枚 p=0.60 的硬币无关。实测 D+D 行 lead 占比 0.24（参考 0.55），
                    # 正是这条：初始 50% 在首位的行全被推翻。修法是两支各自只在"当前位置确实
                    # 需要移动"时才动，pi==0 想留首位就原样不动、pi==末位想留末尾也原样不动。
                    # 【2026-11-11 按行形分段】参考集里 power 的首位率不是常数，取决于本行
                    # 有没有内容子句：
                    #   · 纯设备行（无内容子句）：首位 294 / 末位 244 = 0.55
                    #   · 含内容行：power 在内容【之前】的比例随内容句的类型再分
                    #       - 内容是【播放请求】句：0.52（"关掉电视随后播放强军战歌"——
                    #         先把当前画面关掉、再起播下一件事，是自然的串行安排）
                    #       - 内容是【检索/条件】句：0.19（"45分钟后关机，播放量高的电影"）
                    # 一个常数 p 命中不了这几档。**两处易错，都已实测钉死**：
                    # (a) **必须按 gold=设备 限制统计**。参考集含内容行里 power 的全量首位率
                    #     0.34 是**含跨域标注**（97/402 是 `放开免费的宝宝巴士` 这类
                    #     gold=少儿/tool=power 的位置，power 首位 0）。我的设备池只产
                    #     gold=设备，正确目标是 **0.446**，不是 0.34——按 0.34 会欠发。
                    # (b) **还要看内容句是不是播放句**：0.52/0.19 两档差 2.7 倍。
                    # 判据用文本层 `looks_like_play`（与 order_ok 同口径），不看金标 action。
                    # p 与产出占比在此处近似恒等（换序是最后一步，无后续拒绝偏置）。
                    # (c) **内容是「检索/条件」句时 power 从不起头**（0/55）。参考集里
                    #     "…关掉电视然后打开音乐" 的 power 首位率 0.54，而
                    #     "…45分钟后关机，播放量高的电影" 这类检索句 power 首位率 **0.00**
                    #     ——"先把电视关了，再帮我找…"没人这么说；关机是让电视别放了，
                    #     后面跟一条检索条件句，那是把它当"下一件事"而不是"要关的东西"。
                    #     【早期误判警示】我一度把这个桶打成 0.19，是把 `打开音乐` 这类
                    #     **播放**说法错划进了"检索"桶——判据（见下 `_is_play`）必须含
                    #     `打开`，且不能直接复用 order_ok 里的 `looks_like_play`
                    #     （它没含 `打开`，改动它会波及"最多一个起播动作"那条规则）。
                    #
                    # (d) **纯设备行【不】重排**（2026-11-11，本轮最有价值的一条）。
                    #     `_DEV_PAIR` 是【有序】工具对（从参考集收割），本来就编码了正确
                    #     语序：`power→numeric_adjust` 181 vs 反向 89、`common→power` 116
                    #     vs 反向 68、`mode→numeric` 29 vs 反向 38。按表权算出的首位率
                    #     与参考集**逐工具吻合**（common 0.605/0.608、mode 0.387/0.387、
                    #     power 0.544/0.546）。我这枚随机换序硬币等于把表里已有的正确顺序
                    #     重新掷一遍——实测把 common_control 首位率从 0.605 打散到 0.456、
                    #     mode_control 0.387→0.323。**故 D+D 行直接保留构造顺序**，换序只
                    #     作用于【有内容子句】的行（表的键是设备对，对 CD 行无约束）。
                    #     注意这解释了为何"聚合 OK 但逐工具偏"：power 自身首位率凑巧对，
                    #     被它换序打散的却是【搭档工具】的首位率。
                    # 【2026-11-11 第三轮：起头硬币关掉】DC 联合语域已建立（pick_dc_pair），
                    # 参考集的【全部】196 条设备先行行 = 该语域本身（实测语域 196 = 全量 DC 196）。
                    # 也就是说"power 能起头"是【按内容说法】的，只有语域里那 12 个内容说法
                    # （播放第1集/强军战歌/最强大脑/只听音乐…）能跟在起头的 power 之后；
                    # `关电视#播放睡衣小电视`、`关闭电视#播放海底小纵队第三季` 这类
                    # 参考集【一次都没有】。起头硬币对语域外的内容说法也会把 power 提到句首，
                    # 实测泄漏 29 条全 power 的"伪 DC"（gold 上是设备+内容，语域里没有），
                    # 把 DC 的 power 占比从语域的 0.694 抬到 0.821。
                    # 故这里对"内容+设备"行【不再硬币起头】，power 一律落到内容之后；
                    # 设备先行完全交给语域。elif 分支（power 挪末位）保留——"播放X，随后关机"
                    # 是参考集里 CD 方向的真实语序。
                    _ci = next((i for i, s in enumerate(segs) if s["d"] != "device"), None)
                    if _ci is None:
                        _p = None                               # 纯设备行：保留 _DEV_PAIR 顺序
                    else:
                        _p = 0.0                                # 不硬币起头（起头由 DC 语域负责）
                    # 【必须把 `_p is not None` 也包住 elif】——这是与前一个 bug 同类的
                    # 陷阱：写成 `if _p is not None and rng< _p: … elif pi<len-1: 挪末位`
                    # 时，`_p is None`（纯设备行）会让首条件为假，于是【elif 照样触发】把
                    # power 一律罚到末位——修完实测 power D-row 首位率 0.549→**0.000**。
                    # 正确写法是整体套一层 `if _p is not None:`。
                    if _p is not None:
                        if rng.random() < _p:
                            if pi != 0:
                                segs.insert(0, segs.pop(pi))
                        elif pi < len(segs) - 1:
                            segs.append(segs.pop(pi))
            elif not _is_q and not _from_reg:
                # 并行也【不是】可以随便换序的——这是上一版最自信也最错的一条。
                # 我当时以为"并行无语义先后，所以句序自由"，于是等概率换序，产出里
                # 设备先说的（DC）冲到 48.8%。但参考集并行里「内容+设备」那 3413 行，
                # 97.6% 是内容在前（"播放X同时调音量"），设备在前只占 2.4%。串行同理 90.7%。
                # 也就是说「先说要看的，再说要调的」不是模板味，恰恰是真人语序：
                # 先交代内容（有信息量、要慢慢说），再顺口补一句设置。
                # 换序保留但压到参考实测的 2.4%，其余保持内容在前。
                #
                # 【2026-11-11 第三轮：并行 DC 也是封闭语域，这枚换序硬币【整枚删掉】】——
                # 上一轮我把参考集并行里那 2.4% 设备先行理解成"把内容先行行倒过来"，于是
                # 让这枚硬币作用在 CD 行上。但 REF 并行 DC（设备先行）只有 96 行、17 个
                # 不同的对，其内容位工具构成是 playback 32/vod 26/music_search 23/
                # music_recommend 15，而并行 CD 内容位的 playback 只有 43/3438 = 1.2%——
                # **两者不是同一批语料**。若 DC 真是 CD 倒过来，两边内容工具分布必须一致；
                # 实测差 27 倍，说明并行 DC 是它自己的封闭语域。故这枚硬币只会产出语域
                # 【之外】的组合（实测泄漏 `开启护眼模式#我要看小猪佩奇` 这类
                # tool=educ_search_all 的 CD 侧工具到 DC 内容位，把分布打偏）。DC 已由
                # pick_dc_pair("parallel") 逐字负责，这里连硬币都不再掷——参考集里
                # "设备句在前"的并行行【无一例外】落在语域那 17 个对里。
                pass
            # 去重 & 可组合校验
            # 【语域行豁免】（2026-11-11）：_from_reg 行是逐字照搬 REF 的 (tool,说法,参数)
            # 联合语域，语序与搭配本就是 REF 原文，compatible/order_ok 这些我自造的启发式
            # 不得再裁决它——否则会整族误杀：内容+内容语域 42 行（"播放音乐>>我要看汪汪队"）
            # 会被 compatible 的"两内容域同域禁配"与 order_ok 的"最多一个起播动作"双双拒掉，
            # 产出 0（REF 真有这 0.42%）。只保留【去重】这一条（同一行内两个说法不得同形）。
            if len({s["q"] for s in segs}) != len(segs):
                continue
            if not _from_reg:
                if any(not compatible(segs[i], segs[j], mode)
                       for i in range(len(segs)) for j in range(i + 1, len(segs))):
                    continue
            if not order_ok(segs, mode, _from_reg):
                continue
            # 位置许可【落位】：上面 pick_device(pos=...) 只是"意图"——句序在随后的
            # 串行重排/并行 2.4% 换序里还会动，所以这里按【最终句序】修正：走到首位的
            # 设备子句换成"可起头"的说法、跟在后面的换成"可跟随"的说法。
            #
            # 用【同工具内改说法】而不是【整行作废重抽】：重抽会按"该说法在最终位置上
            # 的许可概率"筛掉整行，于是那些只在非首位出现的工具（power_control 在串行
            # 里被随机提到首位时必挂、screensaver 永远不能起头）连带整行被拒，形状分布
            # 被系统性拉偏——实测改成整行 reject 后 CD 从 61.6% 涨到 67.8%（参考 58.1%）、
            # DD 从 34.9% 掉到 30.8%，等于拿形状换语序。改说法只动表面形式、不动工具与
            # 工具搭配，形状分布原样保留。找不到同工具的可用说法时才整行重抽。
            # 内容子句同理（2026-11-10）：首位用说法级证据、跟随位用语域闸门。
            # 内容侧【不整行重抽】的理由同样成立，且更强：内容池 2684 种里 2662 种
            # 从不在跟随位出现，整行 reject 会把"设备先行"的形状几乎清零。
            # DC 联合语域的条目是从 REF 逐字照搬的 (tool,说法,参数)，位置本就正确，
            # 不做落位改写（改写会换成另一个说法，破坏"逐字照搬"）。
            _pos_ok = True
            _pos_iter = [] if _from_reg else list(enumerate(segs))
            for _i, _s in _pos_iter:
                if _s["d"] == "device":
                    # 位置证据是【工具条件】的（2026-11-11）：同一个说法在不同工具下
                    # 位置条件可以相反——`关闭屏幕` 全池聚合 (41,64) 判"可起头"，但
                    # 拆开看是【作为 screensaver 从不句首】(0,64)、【作为
                    # screen_lift_rotation/mode_control 才句首】(18,0)/(23,0)。
                    # 聚合证据会让"关闭屏幕"以 screensaver 身份被换序顶到首位
                    # （实测 6 例，参考集 screensaver 首位 0.0%）。故一律带上 tool 查。
                    _want = may_lead if _i == 0 else may_follow
                    if _want(_s["q"].strip(), _s["tool"]):
                        continue
                    _alt = [(d, w) for d, w in _dev_by_tool.get(_s["tool"], [])
                            if _want(d["q"].strip(), _s["tool"]) and d["q"] != _s["q"]]
                    if not _alt:
                        _pos_ok = False
                        break
                    _ns = rng.choices([d for d, _ in _alt], weights=[w for _, w in _alt])[0]
                    if NULL_P.get(_ns["q"]) and rng.random() < NULL_P[_ns["q"]]:
                        _ns = {**_ns, "params": None}
                    segs[_i] = _ns
                else:
                    # 跟随位的落位【改用封闭注册表】（2026-11-11）：语域闸门
                    # content_may_follow 只是"什么形态能进跟随位"（保留 609 种），
                    # 而参考集跟随位实际只由 23 种说法构成。换序把一个【首位抽取】的
                    # 内容子句顶到非首位时，必须换成注册表内的说法，否则跟随位又长回
                    # 九十种一次性长尾（"关掉电视播放我收藏的华语经典"这类）。
                    if _i == 0:
                        if content_may_lead(_s["q"].strip()):
                            continue
                        _alt = [(d, d.get("lw", _c_len_w(d["q"])))
                                for d in _ctool.get(_s["tool"], [])
                                if content_may_lead(d["q"].strip()) and d["q"] != _s["q"]]
                    else:
                        # 跟随位的落位【整条重抽自注册表】（2026-11-11 第二轮）：
                        # 语域闸门只在"说法"维度收口（609 种），但参考集跟随位
                        # 【连工具都是封闭的】——只有 educ_search_all 180 / vod_search_all 123 /
                        # music_song_recommend 76 / music_song_search 37 / playback_control 22
                        # 五个工具出现过（其余 24 个内容工具 0 例）。当换序把一个
                        # audio_search/sports_*/edu_search 的子句顶到非首位时，同工具内
                        # 找注册表说法必然落空，退回语域闸门就又长出一批一次性长尾
                        # （"听本周新上线的推荐有声书""播放中超联赛的比赛视频"…）。
                        # 故此处【连工具一起换】——直接调 pick_content(pos="follow")，
                        # 它按注册表的工具质量与说法频次整条重抽，与"抽取时就是跟随位"
                        # 的那条路径完全同源。
                        if _s["q"] in CONTENT_FOLLOW_REG:
                            continue       # 已在注册表内，无需换说法
                        _ns = pick_content(pos="follow")
                        if _ns["q"] == _s["q"]:
                            continue
                        segs[_i] = _ns
                        continue
                    if not _alt:
                        _pos_ok = False
                        break
                    _ns = rng.choices([d for d, _ in _alt], weights=[w for _, w in _alt])[0]
                    if _CONTENT_NULL.get(_ns["q"]) and rng.random() < _CONTENT_NULL[_ns["q"]]:
                        _ns = {**_ns, "params": None}
                    segs[_i] = _ns
            if not _pos_ok:
                continue
            # 改说法后可能撞同形、或破坏全局约束（如把 timer 换成了别的说法的搭档）→ 重抽
            if len({s["q"] for s in segs}) != len(segs):
                continue
            # 【同目标复核】（2026-11-11）：上面位置闸门的 `_alt` 换说法只按位置许可挑，
            # 可能把某个 numeric_adjust 子句换到与搭档【同一个调节目标】上——同目标约束
            # 只在 `_pick_dev_pair` 里查过（闸门之前），闸门之后没再查。参考集里
            # numeric_adjust 双设备行 490 例【全部】是两个不同目标（音量+亮度），
            # 同目标 0 例；一次说「音量减小」再一次说「音量大一点」是自相矛盾，
            # 真人不会这么说。实测泄漏 6 例（`音量减小一起音量大一点~`/
            # `声音小点同时大点声`/`给我保持音量顺便音量减1~`）。这里按最终句序复核。
            _na = [_dev_target(s) for s in segs if s["tool"] == "numeric_adjust"]
            _na = [t for t in _na if t]
            if len(_na) != len(set(_na)):
                continue
            # 首位换成别【工具】后，工具对/相容性可能被破坏（上面 2026-11-11 的
            # NONLEAD 顶位修复会整条换工具）→ 与初始构造同口径复核一次。
            #
            # 【语域行同豁免】（2026-11-11，第二十一轮）。上一处初始构造的 compatible
            # 已按 `_from_reg` 豁免（见上方"语域行豁免"注释），但这里漏了同样的豁免，
            # 于是 CC 语域的 `少儿+少儿`（`播放免费的小猪佩奇>>播放免费的宝宝巴士`）
            # 被 compatible 的"两内容域同域禁配"（`a["d"]==b["d"]` 一律 False）整族拒掉：
            # 实测 1500 行里 10 次抽中、10 次死在这里，产出 0（REF 真有 21 行）。
            # 而同一张表的 `音乐+少儿`（两个 d 不同）活着——正是"漏豁免"而非"没抽到"的
            # 铁证。`_from_reg` 行的语序/搭配本就是 REF 原文，且此行的落位改写
            # （`_pos_iter`）对语域行是空的、无从"破坏"任何东西，复核纯属多余。
            if not _from_reg:
                if any(not compatible(segs[i], segs[j], mode)
                       for i in range(len(segs)) for j in range(i + 1, len(segs))):
                    continue
            if not order_ok(segs, mode, _from_reg):
                continue

            query = make_query(segs, mode, rng)
            if query in used_q:
                continue
            used_q.add(query)

            # 串行：第 2..n 步带 dep 指向第 1 步；并行：不带
            dep_steps = [i for i in range(2, len(segs) + 1)] if mode == "serial" else []
            gold = join_domains(segs)
            tools = "+".join(s["tool"] for s in segs)
            tool_param = json.dumps([s["params"] for s in segs], ensure_ascii=False,
                                    separators=(",", ":"))
            rows.append({
                "id": "",
                "query": query,
                "seg": "#".join(s["q"] for s in segs),
                "gold": gold,
                "tools": tools,
                "tool_param": tool_param,
                "exec_mode": mode,
                "n_steps": len(segs),
                "source": "synth",
                "tools_ok": True,
                "param_src": "|".join("gold" if s["params"] else "none" for s in segs),
                "dep_steps": "|".join(map(str, dep_steps)),
            })
            got += 1

    # 交错打散 serial/parallel，顺序编号
    rng.shuffle(rows)
    for i, r in enumerate(rows, start=1):
        r["id"] = f"dp{i:05d}"

    out = Path(args.out)
    cols = ["id", "query", "seg", "gold", "tools", "tool_param", "exec_mode", "n_steps",
            "source", "tools_ok", "param_src", "dep_steps"]
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print(f"\n写入 {out}  {len(rows)} 行")
    print("exec_mode:", collections.Counter(r["exec_mode"] for r in rows))
    print("n_steps:", collections.Counter(r["n_steps"] for r in rows))
    print("gold:", collections.Counter(r["gold"] for r in rows).most_common(12))
    print("dep 步分布:", collections.Counter(r["dep_steps"] for r in rows if r["exec_mode"] == "serial").most_common(5))


if __name__ == "__main__":
    main()
