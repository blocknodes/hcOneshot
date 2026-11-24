#!/usr/bin/env python3
"""从参考集 bench_serial_train_10k.csv 直接收割设备子句，作为短语库的第二来源。

为什么要从参考集收割，而不是继续手写：
  手写库覆盖的是我"想得到"的说法，而参考集是 10000 行真人会话/标注，覆盖的是
  真实分布。按子句去重后，参考集在若干工具上的用词量是我的 3-10 倍——
  playback_control 317 vs 89、screensaver_control 255 vs 39、common_control 191 vs 10、
  display_control 118 vs 2。差距不是"缺几条"，而是我整块整块没有覆盖，
  这正是用户反复说"表达多样性不够"的直接来源。

收割规则（宁少勿滥，坏句子拼进多意图会整行作废）：
  1. 只用参考集里 tool_param 非空的子句——参数是标注好的，不是我猜的。
     参考集自身有 25% 的子句 tool_param 是 null（numeric_adjust 里高达 48.5%），
     那些无法复用。
  2. 长度 3-13 字。过短的（"主页"）不成句，过长的（"打开量子点四维真实还原系统"）
     是产品特性名，拼进去像念参数表。
  3. 剔除产品特性名词（量子点/环境控光/疾速屏…）与一格里粘了多个子句的污染项
     （参考集 seg 切分把"关机关掉电视关闭电视"算成一条）。
  4. 剔除自指名式残句（"设置""音乐功能""播放模式"这类能力项标签）。
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

REF = Path(__file__).resolve().parent.parent / "testset" / "feishu" / "bench_serial_train_10k.csv"

# ⚠️ 收割【不】按 tools_ok 过滤整行（2026-11-10 实测否决，勿再回退到行级闸）。
# 曾想当然地认为 tools_ok=False 的 641 行是"标注坏了、该整行丢弃"，并把这道闸加进
# 全部 10 个收割循环。A/B 实测是错的，两层原因：
#   1. 那一列是【行级】标志，却常由【兄弟位置】触发。实测 641 行里 465 行
#      (72.5%) 的每个位置 gold↔tool 都自洽——"图像模式调为标准" 所在位置
#      gold=设备/tool=mode_control 完全正确，被整行丢弃纯属被邻居连坐。
#   2. 判"生硬"必须对【完整参考集】等比缩放（见 memory 的口径），不是对
#      tools_ok=True 子集。加闸后 ref_all 里真实高频的说法被清零：
#      播放依娜和恰恰(ref26→0)、播放开心锤锤(20→0)、播放抖音歌曲(21→0)、
#      播放吞噬星空(25→0)、我想看帕丁顿熊三(19→0)、图像模式调为标准(19→0)——
#      它们按 full-ref 缩放【本该各出 ~9-13 条】。聚合指标也同向变差
#      （distinct 3220→3202、熵 7.368→7.356）。
# 真正的对齐在【位置级】，且已就位：设备收割只取 gold==设备 的位置、
# 内容收割只取 gold∈内容域 的位置；跨域错配的位置自然落不到这两个口径里。
# 只收割口语化程度尚可的设备工具；其余工具（numeric_adjust / mode_control 等）
# 我的手写库已经覆盖且质量可控，不再引入噪声。
HARVEST_TOOLS = {
    "common_control", "display_control", "screensaver_control", "playback_control",
    "timer_control", "source_switch", "power_control", "screen_lift_rotation",
    "network_control", "smart_camera", "audio_control", "demo_control",
}

# 产品特性/技术名词：单看是合法指令，拼进多意图像念参数表。
LABELISH = re.compile(
    r"(量子点|环境控光|疾速屏|色域|布光|叠屏|原生肤色|全时图像优化|"
    r"动态清晰度增强|立体防抖|梯形校正|几何校正|运动侦测|防射眼|"
    r"智能运动|感知画质|臻彩|精晰|黑耀屏|墨晶屏|双生|芯片|微透镜|"
    r"自发光技术|光色同控|超广视角|超解像|超画质引擎|控光差异|背光|分区|"
    r"映画|原音复刻|火箭炮|空间声场|声学|驱动发声|低音管理系统|"
    r"低音炮|超重低音|多声道|发声区域|发声效果|屏幕发声|"
    r"独立驱动|全景声音量|音频增强|无线杜比|双音效|像素|光谱|红光|"
    r"可信度|动态范围|刷新率|HZ屏|Hz屏|演示系统|测试图卡)")

# 参考集 seg 切分把多个子句粘成一条的污染项（"关机关掉电视关闭电视"）。
CONTAM = re.compile(r"(关电视|关机|关闭电视|把电视关了|关掉电视|关闭电源|关掉电源|"
                    r"打开主页|返回主页|返回首页|回到主页|打开首页|返回主页面|回到首页)")

# 能力项标签式残句：名词短语，缺少动词，拼进去语法断裂。
BARE_NOUN = re.compile(r"^(音乐功能|外接设备|外部设备|播放模式|播放顺序|HDMI信号|"
                       r"HDMI接口|HDMI端口|HDMI输入|HDMI设置|HDMI切换|"
                       r"信号源设置|信号源切换|输入源切换|输入源选择|"
                       r"无线网络|无线网|有线网络|有线连接|网线连接|以太网连接|"
                       r"外接设备|HDMI信号源|HDMI输入源|HDMI输入选择|HDMI接口选择|"
                       r"HDMI端口|投影设置|外部设备)$")

# 必须能以动词/请求语起头（或参考集里确实作独立小句用的固定说法）。
GOOD_START = re.compile(
    r"^(打开|关闭|关上|关掉|关|启动|开启|进入|进到|切换|切到|切回|选择|选|返回|回到|"
    r"退下|把|将|调出|退出|停止|暂停|继续|播放|来|看|要|换|跳过|倒回|倒退|退回|"
    r"往|使用|设置|能不能|可以|帮我|给我|去|办|续费|放|调|快进|快退|单曲|循环|"
    r"下一个|上一|下一|先|确定|展示|运行|结束|取消|直接|快点|唤|开|无线|有线|"
    r"WiFi|wifi|以太网|网线|HDMI|延时|定时|过|半小时|一个小时后|一小时后|"
    r"\d+分钟|\d+小时|\d+秒钟|关电视|电视|音乐|声音|这首|这首歌|刚才|列表|顺序|随机|"
    r"我|不|别|太|节奏|听|这首|换个|换回|收回|拉回|立刻|马上)")


# 参考集里出现过的【设备子句】全集（含参数为空的），用作「表面形式准入闸门」。
# 它的用途与上面的 harvest 不同：harvest 是【取材】（参数必须可复用），
# 这里是【准入】——判我手写库/变体机产出的说法在参考集里究竟存不存在。
#
# 为什么要单独做这道闸：实测把产出按工具逐条比对参考集，未命中率差异极大——
#   mode_control 89.5%、screen_layout 50.7%、source_switch 40.9%，
#   而 common_control 2.3%、display_control 1.1%、network_control 0%。
# 未命中率低的工具正好都是【已纳入 harvest】的，未命中率高的三个恰好都是【被排除在
# harvest 之外】的——排除理由写在 HARVEST_TOOLS 注释里（"手写库已覆盖且质量可控"），
# 而实测恰恰相反：这三个工具的手写库是整批里最像说明书、最不像真人的部分，
# 变体机（_TERSE_SOFTEN / _variants）又在它们上面二次派生。
# 结论：准入不设白名单，凡参考集出现过的设备子句一律准入；未出现的一律不许进池。
# 参考集设备子句里能带 tool 出现的全部工具（准入闸门的工具口径，比 HARVEST_TOOLS 宽）。
DEVICE_TOOLS = {
    "numeric_adjust", "mode_control", "screen_layout", "playback_control",
    "screensaver_control", "common_control", "power_control", "display_control",
    "source_switch", "timer_control", "network_control", "smart_camera",
    "audio_control", "demo_control", "screen_lift_rotation",
    "solve_picture_sound_problem_control",
}


def attested_device_clauses(path: Path = REF) -> dict[str, int]:
    """返回 {设备子句: 参考集频次}，含 tool_param 为空的子句（准入只判存在性）。"""
    out: dict[str, int] = {}
    for (q, _t), n in attested_device_freq(path).items():
        out[q] = out.get(q, 0) + n
    return out


def attested_device_freq(path: Path = REF) -> dict[tuple[str, str], int]:
    """返回 {(设备子句, tool): 参考集频次}——只统计 gold 为【设备】的子句。

    两个口径都必须卡死，否则权重会明显偏大：

    1. 按 tool 分。同一个说法在不同工具下频次差很多（"暂停"在 playback_control 是
       常态、在别处几乎没有），有 12 条说法跨工具（关电视/关机/暂停/全屏播放…）。
    2. 按 gold=设备 分。这条更要紧：同一个字符串在参考集里可能以【内容角色】出现，
       频次与本池无关。实测「换一首歌」全集 24 次，但 22 次 gold=音乐（作为"点歌"
       的内容子句），只有 2 次 gold=设备；按全集赋权会让它在设备位被放大 12 倍。
       我的设备池只会落在 gold=设备 的位置上，所以权重必须只数那个位置。
    """
    out: dict[tuple[str, str], int] = {}
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            segs = r["seg"].split("#")
            tools = r["tools"].split("+")
            gold = r["gold"].split("+")
            for i, t in enumerate(tools):
                if t not in DEVICE_TOOLS or i >= len(segs) or i >= len(gold):
                    continue
                if gold[i] != "设备":
                    continue                      # 只数设备位，内容位的频次不适用
                q = segs[i].strip()
                if q:
                    out[(q, t)] = out.get((q, t), 0) + 1
    return out


def attested_device_leadfreq(path: Path = REF) -> dict[tuple[str, str], tuple[int, int]]:
    """返回 {(设备子句, tool): (首位次数, 非首位次数)}——只统计 gold 为【设备】的位置。

    为什么要按【位置】再分一层：参考集的设备说法是【强位置条件】的，不是"都能放句首、
    语序随便挑"。把一个说法的出现位置拆成「首位 / 非首位」两栏后，分布是强烈双峰的：

      · 205 个频次≥10 的说法里，83 个(40.5%)【从不】出现在首位——"亮度降低一点"11 次、
        "声音调到一"10 次、"把屏保关掉"10 次、"换一首"10 次全部落在第二位及以后；
        另有 50 个【永远】在首位——"接着播放"28 次、"重启"27 次、"播放hdmi1"26 次。
      · 过散布检验 chi²/df = 20.6（df=205），87 个说法的 |z|>3（零假设下只该有 0.6 个）。
      · 置换检验（在每行内部随机指定哪个设备子句当首位，200 次）：
        真实「高频说法落在首位」的次数 2718，零分布 [2429, 2524]——真实值远在零分布之外。

    也就是说「哪个说法能起头」是参考集里真人语序的固有属性，不是我该重新掷骰子的事。
    我的生成器原先按位置无关的权重抽说法，于是产出的设备首位子句里 39.4% 用了参考集
    【从不】放首位的那种说法（"启动u盘…""切换到HDMI4…""音量22…"）——真人不会这么起头。
    这类"每个词都认识、但没人这样开头"的句子，正是用户反复说的「生硬」最具体的一层。

    非首位那栏同样有用，只是弱一些：参考里有 111 个频次≥3 的说法【只】在首位出现
    （"接着播放""退下""看中央五台"），我产出里 6.9% 的非首位设备子句用了这类。
    """
    out: dict[tuple[str, str], list[int]] = {}
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            segs = r["seg"].split("#")
            tools = r["tools"].split("+")
            gold = r["gold"].split("+")
            for i, t in enumerate(tools):
                if t not in DEVICE_TOOLS or i >= len(segs) or i >= len(gold):
                    continue
                if gold[i] != "设备":
                    continue
                q = segs[i].strip()
                if not q:
                    continue
                c = out.setdefault((q, t), [0, 0])
                c[0 if i == 0 else 1] += 1
    return {k: (v[0], v[1]) for k, v in out.items()}

# 参考集里【确实是完整指令】但 GOOD_START 认不出的说法：形态是「名词短语 + 动词」倒装
# （"图像模式调到影院"）、或本身就作独立小句用的光杆名词（"全屏""护眼""分屏管理"）。
# 实测这两类在参考集里高频且自然（"全屏"407 次、"大屏播放"337 次），
# GOOD_START 是给 harvest 取材写的（要防参数表式残句），不该用来判准入。
# 准入只额外要求「含动词」或「参考集频次≥3」——低频的名词残句仍挡在外面。
# 注："图像模式调为标准" 是本类的真实例子（参考集 19 次）——它全落在 tools_ok=False
# 行，但那一列是行级标志，该位置本身 gold=设备/tool=mode_control 完全自洽，照常准入。
_ADMIT_VERB = re.compile(
    r"(打开|关闭|关上|关掉|关|启动|开启|进入|切到|切回|切换|调成|调到|调为|调至|"
    r"调整|调节|设置为|设为|设置|换成|切成|退出|展开|放大|缩小|播放|观看|显示|"
    r"管理|选择|解除|暂停|继续|停止|快进|快退|倒回|返回|回到|加强|提高|降低|"
    r"唤醒|结束|校准|恢复|切换)")


def attested_device_nullrate(path: Path = REF) -> dict[str, float]:
    """返回 {设备子句: 该说法在参考集里 tool_param 为 null 的比例}（gold=设备 口径）。

    为什么要有这一张表：参考集里【24.5% 的设备位 tool_param 是 null】，且这个状态
    不是随机缺失而是【按说法系统性】的——分布强烈双峰：1110 条从不 null、241 条永远
    null、只有 53 条混合。恒 null 的是最口语的一档（"大点声" 90 次、"快进到三分钟" 31、
    "音量调到二十" 32、"播放第1集" 40），参考集把这类短促指令标为"该步未标"。
    eval_multi.judge() 对 null 步不计参数分，所以照搬 null 与参考集口径一致。

    我方池子里也有这些说法（手写库写的），但一律带了参数——参考集里它们的参数恒为 null。
    不覆盖就等于凭空给它们编了金标，既偏离参考集，也让"参数已标"的占比虚高
    （实测参考设备位 26.2% null，我 0.0%）。
    """
    num: dict[str, int] = {}
    den: dict[str, int] = {}
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            segs = r["seg"].split("#")
            tools = r["tools"].split("+")
            gold = r["gold"].split("+")
            try:
                tps = json.loads(r["tool_param"])
            except json.JSONDecodeError:
                tps = []
            if not isinstance(tps, list):
                tps = []
            for i, t in enumerate(tools):
                if t not in DEVICE_TOOLS or i >= len(segs):
                    continue
                if i < len(gold) and gold[i] != "设备":
                    continue
                q = segs[i].strip()
                if not q:
                    continue
                den[q] = den.get(q, 0) + 1
                v = tps[i] if i < len(tps) else None
                if not isinstance(v, dict) or not v:
                    num[q] = num.get(q, 0) + 1
    return {q: num.get(q, 0) / n for q, n in den.items() if n}


def dc_pair_register(mode: str = "serial", path: Path = REF) -> dict[tuple, tuple]:
    """返回 {(设备tool,设备说法,内容tool,内容说法): (次数, 设备参数, 内容参数, 内容域)}——
    【设备先行·内容后随】(DC) 行的联合工具×说法语域。内容域是 REF 的 gold 标签
    （vod/music/…），供生成器填内容子句的 d 字段（不能由工具名反推）。

    带参数是因为 DC 语域里的说法有些【不在】DEVICE_PHRASES 池内（`关闭屏幕` 作为
    mode_control 是标注噪声、`关掉电视关电视` 是参考集 seg 切分把两小句粘成一格），
    这些恰恰是 REF 真实用的（top 对 `关闭屏幕>>只听音乐` 12 次全落在这里）。语域是
    封闭的，就该逐字照搬 REF 的 (工具,说法,参数) 三元组，不因"池里没有"而丢。

    为什么必须【联合】抽，而不是"设备按设备池抽、内容按内容注册表抽"分别抽：
    REF 串行里 DC 只有 196 行、62 个不同的 (设备说法, 内容说法) 对，且是【封闭】的——
    `关闭屏幕 >> 只听音乐`(12)、`大点声 >> 最强大脑`(9)、`关掉电视 >> 播放强军战歌`(7)…
    这是"先说把当前画面/声音弄成什么样，再说下一件要看什么"的固定搭配表。
    独立抽两边会产出参考集从未出现过的组合（设备先说了"大点声"、内容却是"有哪些
    免费的电视剧"这种搜索句），且因内容注册表以播放句为主、非 power 设备句又会触发
    语序规则，非 power 设备抽到的行几乎全被拒——实测我产出的 DC 里 power 占 0.994
    （REF 0.683）。联合抽则抽中即合规。

    同时返回设备/内容分列的工具质量，供"先定工具对、再各自选说法"的调用方使用。

    mode：语域按执行关系各成一张封闭表。并行侧同样封闭——REF 并行 DC 只有 96 行、
    17 个不同对（`进入音箱模式>>打开音乐对面的女孩看过来` 15、`关闭屏幕>>播放音乐` 12、
    `关闭屏幕>>只听音乐` 11、`大点声>>最强大脑` 10、`全屏播放>>播放第1集` 9…），
    且工具构成与串行【不同】：并行内容位 playback_control 占 32/96（串行以 vod 搜索为主）。
    因此串行表不能拿来抽并行，必须分 mode 各存一张。
    """
    CONTENT = {"影视", "音乐", "有声", "教育", "少儿", "体育"}
    cnt: dict[tuple[str, str, str, str], int] = {}
    par: dict[tuple[str, str, str, str], list] = {}
    dom_of_gold = {"影视": "vod", "音乐": "music", "教育": "education",
                   "少儿": "children", "体育": "sports", "有声": "audio"}
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r.get("exec_mode") != mode:
                continue
            segs = r["seg"].split("#")
            tools = r["tools"].split("+")
            gold = r["gold"].split("+")
            try:
                tps = json.loads(r["tool_param"])
            except json.JSONDecodeError:
                tps = []
            if not isinstance(tps, list):
                tps = []
            # 【必须恰好两步——第二十二轮】原来是 `< 2`，于是三步句的前两格也被当 DC 对收了。
            # REF 的 19 行 `设备+音乐+设备`（`进入音箱模式#打开音乐对面的女孩看过来#声音放到
            # 二十二`）会把 `进入音箱模式>>打开音乐对面的女孩看过来` 注册成一个"两步 DC 对"，
            # 而参考集里这个组合【只存在于三步句的中间位置】、两步句里零例。生成器照此抽两步句，
            # 就产出了参考集没有的 `进入音箱模式同时打开音乐对面的女孩看过来`（实测 12 行）。
            # 与 rest 三张对表（cc/cdx/dd，都用 `!= 2`）统一口径。
            if len(segs) != 2 or len(tools) != 2 or len(gold) != 2:
                continue
            # 【必须按域重排工具数组】（2026-11-11）。参考集的 tools 列与 seg/gold 列
            # 有系统性错位：`关电视然后放开免费的宝宝巴士呢` 的 seg=['关电视','放开免费的
            # 宝宝巴士']、gold=['设备','少儿']，但 tools=['educ_search_all','power_control']
            # ——两个工具与两个位置【整好反了】。这不是个例：按 _pair_tools 重排后，
            # 串行 DC 行从 196 涨到 **300**（+104，多出 53%），多出的全是
            # `关电视>>放开免费的宝宝巴士`、`把电视关了>>播放海底小纵队第三季`、
            # `关闭电视>>播放睡衣小电视` 这一族——正是我产出里整块缺失的真人说法。
            # harvest_content_counts 早就做了这个重排（第 746 行），dc_pair_register 漏了。
            # 不重排时 tools[0] 常是内容工具，于是 `tools[0] not in DEVICE_TOOLS` 直接把
            # 这些行判成非 DC 丢掉。
            _pr = _pair_tools(segs, tools, tps, gold)
            if _pr != list(range(len(_pr))):
                tools = [tools[j] for j in _pr]
                tps = [tps[j] if j < len(tps) else None for j in _pr]
            if tools[0] not in DEVICE_TOOLS or gold[0] != "设备":
                continue
            if gold[1] not in CONTENT:
                continue
            dq, cq = segs[0].strip(), segs[1].strip()
            if not dq or not cq:
                continue
            k = (tools[0], dq, tools[1], cq)
            cnt[k] = cnt.get(k, 0) + 1
            # 记该对出现过的参数（首见非空为准）：REF 里 tool_param 常为 null，
            # 生成时按对决定是否留白（与设备/内容单侧的 nullrate 同口径）。
            dp = tps[0] if 0 < len(tps) else None
            cp = tps[1] if 1 < len(tps) else None
            slot = par.setdefault(k, [None, None, dom_of_gold.get(gold[1], "vod")])
            if slot[0] is None and isinstance(dp, dict) and dp:
                slot[0] = dp
            if slot[1] is None and isinstance(cp, dict) and cp:
                slot[1] = cp
    # 第三列是内容的 gold 域（"影视"/"音乐"…）：内容子句的 d 字段在生成器里就是域，
    # 不能由工具名反推（playback_control 也能出现在内容位，域仍是标注给的）。
    return {k: (n, par[k][0], par[k][1], par[k][2]) for k, n in cnt.items()}


def cd_cross_register(mode: str = "serial", path: Path = REF) -> dict[tuple, tuple]:
    """返回 {(首位tool,首位说法,次位tool,次位说法): (次数, 首位参数, 次位参数, 首位内容域)}。

    【内容金标 + 设备工具】(CD 方向的跨标语域，2026-11-11)。这是 dc_pair_register 的
    【镜像】：那边的首位子句是 gold=设备 的设备句；这边首位子句的 gold 是【内容域】
    （影视/音乐…），可它的【工具】却是设备工具。

    为什么这不是标注错误，而是参考集一种真实的说法方式：用户说「播放云视听小电视」
    「看中央五台」「播放短视频」「把音乐关闭」「播放上一首歌」时，他说的其实是一个
    切源/屏转/播控动作（工具 source_switch / screen_lift_rotation / playback_control），
    但金标把这个位置标成了他要看的【内容域】——因为句子的信息主体是"我要看的东西"。
    这类行【全部】是 CD 形态（内容域先说、设备句后随），且跟随句一定是设备工具。

    实测（按 _pair_tools 重排后）：串行 102 行 / 37 对、并行 84 行 / 31 对，是封闭表：
      [source_switch]我要看云视听小电视 >> [numeric_adjust]把声音调高点
      [playback_control]把音乐关闭   >> [common_control]返回主页
      [screen_lift_rotation]播放短视频 >> [numeric_adjust]音量调到二
      [playback_control]播放上一首歌  >> [playback_control]循环播放
      [power_control]打开木兰妈妈    >> [screen_layout]全屏

    为什么必须【联合】抽：这是跨标语域，机构造与普通 CD 行不同——首位工具根本不是
    内容工具，在内容池里【永远抽不到】（_ctool_pool 只装内容工具），所以这些真人的
    说法（"云视听小电视" REF 42 行、"把音乐关闭" 41 行、"中央五台" 21 行）在我产出里
    【一条都没有】。独立抽两边做不到：首位的 tool 是设备工具、d 却是内容域，两个字段
    本就来自不同归类口径，必须逐字照搬 REF 的 (tool, 说法, d 域) 三元组。

    返回的四元组第 4 项是首位的【内容 gold 域】——生成器要用它填首位子句的 d 字段
    （决定 join_domains 与金标），不能由工具名反推（源切换工具正常是 d=device）。
    """
    CONTENT = CONTENT_GOLD
    cnt: dict[tuple[str, str, str, str], int] = {}
    par: dict[tuple[str, str, str, str], list] = {}
    dom_of_gold = {"影视": "vod", "音乐": "music", "教育": "education",
                   "少儿": "children", "体育": "sports", "有声": "audio"}
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r.get("exec_mode") != mode:
                continue
            segs = r["seg"].split("#")
            tools = r["tools"].split("+")
            gold = r["gold"].split("+")
            try:
                tps = json.loads(r["tool_param"])
            except json.JSONDecodeError:
                tps = []
            if not isinstance(tps, list):
                tps = []
            # 【必须恰好两步——第二十二轮】与 dc_pair_register 同因：原来是 `< 2`，
            # 于是三步句的前两格被当跨标对收了——REF 的 195 行 `影视+设备+设备`/`音乐+设备+设备`
            # （`我要看外交风云第25集第5分钟#把音量调到二十#亮度调到二十`）会把
            # `我要看外交风云第25集第5分钟>>把音量调到二十` 注册成一个"两步跨标对"，
            # 而这些前两格在参考集两步句里并不以该组合出现。统一改成 `!= 2`。
            if len(segs) != 2 or len(tools) != 2 or len(gold) != 2:
                continue
            # 与 dc_pair_register 同因：tools 列与 seg/gold 列会系统性错位，先按域重排。
            _pr = _pair_tools(segs, tools, tps, gold)
            if _pr != list(range(len(_pr))):
                tools = [tools[j] for j in _pr]
                tps = [tps[j] if j < len(tps) else None for j in _pr]
            # 首位：gold 是【内容域】、tool 却是【设备工具】；次位：gold=设备、tool=设备工具。
            if gold[0] not in CONTENT or tools[0] not in DEVICE_TOOLS:
                continue
            if gold[1] != "设备" or tools[1] not in DEVICE_TOOLS:
                continue
            lq, fq = segs[0].strip(), segs[1].strip()
            if not lq or not fq:
                continue
            k = (tools[0], lq, tools[1], fq)
            cnt[k] = cnt.get(k, 0) + 1
            lp = tps[0] if 0 < len(tps) else None
            fp = tps[1] if 1 < len(tps) else None
            slot = par.setdefault(k, [None, None, dom_of_gold.get(gold[0], "vod")])
            if slot[0] is None and isinstance(lp, dict) and lp:
                slot[0] = lp
            if slot[1] is None and isinstance(fp, dict) and fp:
                slot[1] = fp
    return {k: (n, par[k][0], par[k][1], par[k][2]) for k, n in cnt.items()}


def cc_pair_register(mode: str = "serial", path: Path = REF) -> dict[tuple, tuple]:
    """返回 {(首tool,首说法,次tool,次说法): (次数, 首参数, 次参数, 首域, 次域)}——
    【内容+内容】(CC) 行的联合语域。

    REF 里两个位置都是内容域的行极少（审查口径），两步句里串行 13 行 / 并行 29 行，
    且是【封闭】的两种搭配：
      `播放音乐 >> 我要看汪汪队`（音乐→少儿）
      `播放免费的小猪佩奇 >> 播放免费的宝宝巴士`（少儿→少儿，两段点播）
    生成器此前把"两条内容域"当穿帮一律禁掉（形状配比里 CC ≈0），产出 0 行；但 REF
    真有这 0.42%，且是两个固定搭配——正是"先放上一件、再点下一件"的说法。按联合
    语域逐字照搬即可，机器不会自造"播放A再播放B"的清单句。

    返回的四元组含两个 gold 域——两个位置都要填 d 字段。
    """
    CONTENT = CONTENT_GOLD
    cnt: dict[tuple, int] = {}
    par: dict[tuple, list] = {}
    dom_of_gold = {"影视": "vod", "音乐": "music", "教育": "education",
                   "少儿": "children", "体育": "sports", "有声": "audio"}
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r.get("exec_mode") != mode:
                continue
            segs = r["seg"].split("#")
            tools = r["tools"].split("+")
            gold = r["gold"].split("+")
            try:
                tps = json.loads(r["tool_param"])
            except json.JSONDecodeError:
                tps = []
            if not isinstance(tps, list):
                tps = []
            if len(segs) != 2 or len(tools) != 2 or len(gold) != 2:
                continue
            if gold[0] not in CONTENT or gold[1] not in CONTENT:
                continue
            a, b = segs[0].strip(), segs[1].strip()
            if not a or not b:
                continue
            k = (tools[0], a, tools[1], b)
            cnt[k] = cnt.get(k, 0) + 1
            p0 = tps[0] if tps else None
            p1 = tps[1] if len(tps) > 1 else None
            slot = par.setdefault(k, [None, None, dom_of_gold.get(gold[0], "vod"),
                                      dom_of_gold.get(gold[1], "vod")])
            if slot[0] is None and isinstance(p0, dict) and p0:
                slot[0] = p0
            if slot[1] is None and isinstance(p1, dict) and p1:
                slot[1] = p1
    return {k: (n, par[k][0], par[k][1], par[k][2], par[k][3]) for k, n in cnt.items()}


def dd_pair_register(mode: str = "serial", path: Path = REF) -> dict[tuple, tuple]:
    """返回 {(首tool,首说法,次tool,次说法): (次数, 首参数, 次参数)}——
    【设备+设备】(DD) 行的联合工具×说法语域。

    （2026-11-11，第十六轮「生硬」）设备+设备行占整卷约 30%（REF 串行 3096/5000、
    并行同量级），是最后一块【仍在独立抽两边】的槽位。生成器此前先用 `_DEV_PAIR`
    抽一对【工具】，再在每个工具内【各自】按 lw 抽说法——工具对是联合的，
    可**说法层是独立的**。实测后果：REF 的 DD 说法对互信息超过自身独立零模型
    **+0.956 bit**（强耦合），而我只有 **+0.0003 bit**（几乎完全独立）。

    耦合长什么样（REF 实测，括号内 P(次说法|首说法)）：
      `打开下一集` >> `关掉电视`/`关闭电视`/`关电视`（1.00——看完这集就关电视）
      `小屏播放`   >> `大点声`（1.00）
      `亮度15`     >> `打开护眼`/`开启护眼模式`（1.00——亮度调到15就该开护眼）
      `提高亮度`   >> `关闭护眼模式`（1.00——亮度和护眼是同一件事的两面）
      `下一曲`     >> `音量减`/`声音减`（1.00）
      `从第一集播放` >> `全屏播放`/`大屏播放`（1.00）
      `暂停`       >> `关闭静音`/`把声音小一点`（0.28/0.14）
    这些是"两件事在真人脑子里本就是一件"的搭配（护眼↔亮度、暂停↔静音、
    切集↔全屏），独立抽两边永远产不出——读起来就是"两句不相干的话被逗号连起来"，
    正是用户说的"生硬"。故与其余三个跨标/DC/CC 语域同法：建【联合表】逐字照搬。

    为什么不在 `_DEV_PAIR` 上直接加说法层：那张表是我按工具聚合出来的（30 个工具对），
    说法对则有一千多个、长尾极薄。逐字照搬 REF 的 (tool,说法,参数) 四元组，
    与 dc/cdx/cc 三张表口径一致，也让参数直接继承 REF（不再按池内 nullrate 另抽）。

    mode 分串行/并行两张（执行关系不同、搭配分布不同，与 dc_pair_register 同因）。
    """
    cnt: dict[tuple, int] = {}
    par: dict[tuple, list] = {}
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r.get("exec_mode") != mode:
                continue
            segs = r["seg"].split("#")
            tools = r["tools"].split("+")
            gold = r["gold"].split("+")
            try:
                tps = json.loads(r["tool_param"])
            except json.JSONDecodeError:
                tps = []
            if not isinstance(tps, list):
                tps = []
            if len(segs) != 2 or len(tools) != 2 or len(gold) != 2:
                continue
            # 与 dc/cdx/cc 同因：tools 列会与 seg/gold 列系统性错位，先按域重排。
            _pr = _pair_tools(segs, tools, tps, gold)
            if _pr != list(range(len(_pr))):
                tools = [tools[j] for j in _pr]
                tps = [tps[j] if j < len(tps) else None for j in _pr]
            if gold[0] != "设备" or gold[1] != "设备":
                continue
            # 不要求两端都是设备工具：REF 里还有 61 行「两个位置 gold=设备、工具却是
            # 内容工具」（vod_search_all 36 / music_song_recommend 16 / music_song_history 9）
            # ——与 cd_cross_register 是同一类跨标噪声的镜像。它们同样是真人当设备句说的
            # （"播放第一集" vod_search_all），逐字照搬即可；工具名照带，d 两格都是 device。
            a, b = segs[0].strip(), segs[1].strip()
            if not a or not b or a == b:
                continue
            k = (tools[0], a, tools[1], b)
            cnt[k] = cnt.get(k, 0) + 1
            p0 = tps[0] if tps else None
            p1 = tps[1] if len(tps) > 1 else None
            slot = par.setdefault(k, [None, None])
            if slot[0] is None and isinstance(p0, dict) and p0:
                slot[0] = p0
            if slot[1] is None and isinstance(p1, dict) and p1:
                slot[1] = p1
    return {k: (n, par[k][0], par[k][1]) for k, n in cnt.items()}


def cd_cond_register(mode: str = "serial", path: Path = REF, min_count: int = 5
                     ) -> dict:
    """返回 CD（内容先行、设备后随）行的【条件设备分布】——按内容说法分桶。

    返回 (by_csurf, by_ctool, dtool_of)：
      by_csurf: {内容说法: [(次数, 设备tool, 设备说法, 设备参数), …]}（按次数降序）
      by_ctool : {内容tool: [(次数, 设备tool, 设备说法, 设备参数), …]}（兜底桶）
      dtool_of : {设备说法: 设备tool}（从全参考集收割，供反查）

    【为什么需要它——第十七轮「生硬」】CD 行（内容+设备）占整卷约 62%，是最后一块
    【两边独立抽】的槽位。实测 REF 的内容说法×设备说法互信息超自身独立零模型
    **+0.570 bit**，而生成器只有 **+0.029**（近乎完全独立）。耦合长这样（P(设备|内容)）：
      `播放抖音歌曲` >> `大点声`(1.00)
      `播放开心锤锤` >> `音量减`/`声音减`(1.00)
      `播放吞噬星空` >> `大屏播放`/`全屏播放`(0.92)
      `只听音乐`     >> `关闭屏幕`(1.00)
      `换一首歌`     >> `音量小点`/`声音小点`(1.00)
      `我要看中央一台` >> `音量调到十五`/`声音调到十五`(0.95)
    这些是"看完这个东西顺手要做的下一件事"——真人脑子里两件事本就是一件，独立抽
    永远产不出（读起来就是"两句不相干的话被逗号连起来"，正是用户说的"生硬"）。

    【为什么不是逐字联合语域】DD/DC/CC 三张表都是【封闭】四元组表，但 CD 不是：
    内容先行 CD 6218 行里有 5386 个不同的 (内容说法,设备说法) 二元组，83% 是单例，
    逐字照搬只能覆盖 16.7% 的行——剩余的会退回独立抽，等于没修。CD 是【开放集】，
    故改为【条件分布】：给定内容说法，抽设备说法。

    【按什么分桶——实测定的】三种分桶的 held-out 泛化检验（REF 对半切，A 建表、B 验）：
      按【内容tool】分桶（32 键）：excess 0.07 ——几乎无耦合，因为同一个内容工具下
        设备分布几乎一样（点播工具都点向 numeric/screen_layout）。
      按【内容说法】分桶、单例退回：【excess 0.42~0.46 vs REF held-out 0.37~0.43】——
        逐位吻合。说明参考集的内容说法→设备说法耦合是【可外推的真人规律】，不是
        小样本噪声（若为噪声，held-out 上会回落到 0）。
    释放方向也实测钉死：`min_count=5` 时 excess 0.44、`=3` 0.46、`=2` 0.52——越小越
    过拟合（单例桶把训练集噪声照搬）。取 5（覆盖串行 28%、并行 18% 的行；未覆盖的
    按 by_ctool 兜底，实测总量仍逐位吻合，因为耦合质量本就集中在高频内容说法上）。

    与 dc/dd/cc 同理：tools 列会与 seg/gold 列系统性错位，先按域重排（_pair_tools）。
    只收【内容先行】的行——设备先行的 CD 形态由 dc_pair_register 逐字负责，两桶不重叠。
    """
    CONTENT = CONTENT_GOLD
    # (内容说法, 内容tool) -> Counter[(设备tool, 设备说法, 设备参数)]
    by_c: dict[tuple, dict] = {}
    tot_c: dict[tuple, int] = {}
    by_t: dict[str, dict] = {}
    tot_t: dict[str, int] = {}
    dtool_of: dict[str, str] = {}
    # 【兜底桶必须在"未被说法桶接住"的行上收割】（2026-11-11，第二十轮）。
    # 原来 by_t 是按内容【工具】把【全部】行聚起来的，于是把"自己有说法桶"的那些
    # 内容说法也算了进去——而那些说法恰是设备分布最偏 numeric 的一群。结果：
    # 只要一个内容说法没有自己的桶（REF 里 84% 的行都如此），兜底桶喂给它的
    # 设备分布带上了覆盖行的偏斜。实测（喂 REF 自己的内容说法分布）：
    #   兜底桶 numeric 0.19 vs 该群行真实 0.12 ⇒ 整卷 CD 设备位 numeric 0.26
    #   vs REF 0.194（1.41×）、screen_layout 1.61×，其余工具 0.80×。
    # 正确口径：by_t[ct] 只收"内容说法【没有】上榜说法桶"的行。分两遍收——
    # 第一遍把所有行收进 by_c（含未达阈值的），据 min_count 定出"上榜说法"集合，
    # 第二遍只把未上榜说法的行收进 by_t。这样兜底桶与它真正服务的行同分布，
    # 逐字喂 REF 内容分布即精确复原 REF 设备边际（实测各工具 1.00×）。
    _rows: list = []
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r.get("exec_mode") != mode:
                continue
            segs = r["seg"].split("#")
            tools = r["tools"].split("+")
            gold = r["gold"].split("+")
            try:
                tps = json.loads(r["tool_param"])
            except json.JSONDecodeError:
                tps = []
            if not isinstance(tps, list):
                tps = []
            if len(segs) != 2 or len(tools) != 2 or len(gold) != 2:
                continue
            _pr = _pair_tools(segs, tools, tps, gold)
            if _pr != list(range(len(_pr))):
                tools = [tools[j] for j in _pr]
                tps = [tps[j] if j < len(tps) else None for j in _pr]
            # 只要【内容先行】的 CD 形态：首格 gold 是内容域、次格 gold=设备。
            if gold[0] not in CONTENT or gold[1] != "设备":
                continue
            cq, dq = segs[0].strip(), segs[1].strip()
            if not cq or not dq:
                continue
            ct, dt = tools[0], tools[1]
            # 内容工具归一到目录现名，与生成器查询键逐字一致（见 TOOL_ALIAS 注释）。
            ct = TOOL_ALIAS.get(ct, ct)
            dp = tps[1] if len(tps) > 1 else None
            dtool_of.setdefault(dq, dt)                    # 设备说法的工具（全表收割）
            _rows.append((cq, ct, dt, dq, dp))
            d = by_c.setdefault(cq, {})
            d[(dt, dq, json.dumps(dp, ensure_ascii=False)
               if isinstance(dp, dict) else None)] = \
                d.get((dt, dq, json.dumps(dp, ensure_ascii=False)
                       if isinstance(dp, dict) else None), 0) + 1
            tot_c[cq] = tot_c.get(cq, 0) + 1

    # 上榜说法桶 = 频次达阈值的那些内容说法（_emit 口径一致）。
    _covered_surf = {cs for cs, t in tot_c.items() if t >= min_count}
    for cq, ct, dt, dq, dp in _rows:
        if cq in _covered_surf:
            continue                       # 已有说法桶接住，不再污染兜底分布
        d = by_t.setdefault(ct, {})
        d[(dt, dq, json.dumps(dp, ensure_ascii=False)
           if isinstance(dp, dict) else None)] = \
            d.get((dt, dq, json.dumps(dp, ensure_ascii=False)
                   if isinstance(dp, dict) else None), 0) + 1
        tot_t[ct] = tot_t.get(ct, 0) + 1

    def _emit(bucket, thresh):
        out = {}
        for k, d in bucket.items():
            tot = sum(d.values())
            if thresh > 1 and tot < thresh:
                continue                    # 证据不足 → 交给调用方退回上一级桶
            rows = sorted(((n, t, q, json.loads(p) if p else None)
                           for (t, q, p), n in d.items()),
                          key=lambda x: -x[0])
            out[k] = rows
        return out

    # 【被内容"锁死"的设备说法必须从工具兜底桶里剔除】（2026-11-11，第十八轮）。
    # 有些设备说法在参考集里【只跟一个内容说法搭配】：`免费大点声` 21 行【全部】是
    # `播放超级宝贝jo#免费大点声`、`音量小点` 13/13 跟 `换一首歌`、`音量调到十五`
    # 10/10 跟 `我要看中央一台`。它们本该【只能】经由自己的内容说法桶到达。
    # 但工具兜底桶（按内容【工具】聚合）会把它们摊到该工具的【所有】内容说法上——
    # 实测 MINE 把 `免费大点声` 产了 27~35 行（REF 21），且散在任意少儿句后
    # （"播放小马宝莉顺便免费大点声"），`免费大点声` 在 educ 工具下的条件概率
    # 被抬到 0.061（REF 0.022），并造出参考集零例的 `顺便免费大点声` 4-gram。
    # 判据：该设备说法在 CD 行里的内容说法分布，top 内容占比 ≥0.8 且总频 ≥5
    # （实测 399 个有足够频次的设备说法里 39 个被锁死）。剔除后，非锁定说法
    # （`关掉电视`/`大屏播放`/`音量减`…）照旧进兜底桶，覆盖不受影响。
    dev_to_content: dict[str, dict] = {}
    for k, d in by_c.items():
        for (_t, q, _p), n in d.items():
            dev_to_content.setdefault(q, {})[k] = \
                dev_to_content.setdefault(q, {}).get(k, 0) + n

    def _locked(q: str) -> bool:
        c = dev_to_content.get(q)
        if not c:
            return False
        tot = sum(c.values())
        return tot >= 5 and max(c.values()) / tot >= 0.8

    locked_dev = {q for q in dev_to_content if _locked(q)}
    for k, d in by_t.items():
        for key in [key for key in d if key[1] in locked_dev]:
            del d[key]

    # 【阈值必须真的生效】——首版把 min_count 收下却从不使用，于是每个内容说法都成了桶
    # （等效 min_count=1），单例桶把训练集噪声照搬，正是 held-out 检验否掉的过拟合档。
    # 工具桶不设阈（它是兜底，需要覆盖所有内容工具），但去掉被锁死的设备说法。
    return _emit(by_c, min_count), _emit(by_t, 1), dtool_of


_DOM_OF_GOLD = {"影视": "vod", "音乐": "music", "教育": "education",
                "少儿": "children", "体育": "sports", "有声": "audio"}


def triple_register(mode: str = "serial", path: Path = REF) -> dict:
    """返回三步句的【整行联合语域】——{(t0,q0,t1,q1,t2,q2): (n, p0,p1,p2, d0,d1,d2)}。

    【为什么三步句也要建整行表（第十八轮「生硬」）】三步句只占 2.4%（REF 243 行），
    我此前是【独立拼】：先抽一条两步句，再 append 一条独立抽的设备句。后果实测：
      · 内容域分布：REF 三步句 **80% 影视 + 20% 音乐**（就这俩域！），我均匀摊在
        六个域上（音乐 23/影视 20/体育 17/教育 17/…）——大步幅偏。
      · 设备工具对：REF **82% numeric_adjust×numeric_adjust**（"声音大点 + 亮度高一点"
        这类"调音量顺手调亮度"），我 top 只有 11%（playback×numeric）。
      · 签名说法：REF 的 `亮度高一点`61/`屏幕亮度提高`25/`点亮度低点`23 **只**出现在
        三步句里（两步句 0 例——它是"内容句 + 两个调节"的第三格），独立拼法几乎
        产不出（我 `屏幕亮度提高` 1 行、`点亮度低点` 0 行）。
    它是【封闭】的：串行 103 行 / 51 个不同三元组、并行 140 / 62，内容说法每种模式
    只有 **10 种**。按整行表（权重=频次）抽样即无偏复现 REF 自己的三元组分布
    （含单例——单例以 1/103 的比率被抽到，正是 REF 的比例）。故与 DC/DD/CC 同法建整行表。

    返回每格带【gold 域】——生成器要用它填 d 字段（决定 join_domains 与金标），
    不能由工具名反推（内容格的 d 是内容域，设备格的 d 是 device）。
    """
    CONTENT = CONTENT_GOLD
    cnt: dict[tuple, int] = {}
    par: dict[tuple, list] = {}
    golds: dict[tuple, tuple] = {}
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r.get("exec_mode") != mode:
                continue
            segs = r["seg"].split("#")
            tools = r["tools"].split("+")
            gold = r["gold"].split("+")
            try:
                tps = json.loads(r["tool_param"])
            except json.JSONDecodeError:
                tps = []
            if not isinstance(tps, list):
                tps = []
            if len(segs) != 3 or len(tools) != 3 or len(gold) != 3:
                continue
            # 与其余语域同因：tools 列会与 seg/gold 错位，先按域重排。
            _pr = _pair_tools(segs, tools, tps, gold)
            if _pr != list(range(len(_pr))):
                tools = [tools[j] for j in _pr]
                tps = [tps[j] if j < len(tps) else None for j in _pr]
            # 只要"最多一格内容"（REF 三步句内容域个数 {1:214, 0:29}，从无两格内容）。
            # 内容格不限于首位——并行有 `进入音箱模式#打开音乐对面的女孩看过来#…`
            # 这类内容夹在中间的真人说法（gold=设备+音乐+设备）。
            ncontent = sum(1 for g in gold if g in CONTENT)
            if ncontent > 1:
                continue
            a, b, c = (s.strip() for s in segs)
            if not a or not b or not c:
                continue
            k = (tools[0], a, tools[1], b, tools[2], c)
            cnt[k] = cnt.get(k, 0) + 1
            slot = par.setdefault(k, [None, None, None])
            for i in range(3):
                if slot[i] is None and i < len(tps) and isinstance(tps[i], dict) and tps[i]:
                    slot[i] = tps[i]
            golds.setdefault(k, tuple(
                "device" if g == "设备" else _DOM_OF_GOLD.get(g, "device")
                for g in gold))
    return {k: (n, par[k][0], par[k][1], par[k][2], *golds[k])
            for k, n in cnt.items()}


def lead_dev_before_play(path: Path = REF, provides_playback=None
                         ) -> dict[str, tuple[int, int]]:
    """返回 {设备tool: (位于播放内容句【之前】的次数, 作为首位设备句的总次数)}。

    用途：替换生成器 order_ok 里那条【我自造】的"串行中非 power 设备句不得先于起播句"
    规则。该规则对参考集自身的 5000 行串行实测【误拒 146 行，全部是 DC 形态】——
    REF 真的会写"关闭屏幕随后播放音乐""退出广告看西游记""大屏播放第1集吧"。
    正确口径不是"非 power 一律禁止"，而是【按工具的位置条件性】：power 0.316、
    mode 0.356、common 0.047、screen_layout 0.017、playback 0.028、numeric 0.000…
    即"哪些设备工具能当起头句、先于一条播放内容句"本身是参考集的语序事实。

    `provides_playback` 必须传入生成器里【同一个】判据——否则两边会漂移：生成器的
    provides_playback 还看 params.action 与 PLAY_TOOLS 兜底（"最强大脑" tool=vod 也算
    起播），纯文本正则会把它漏判。默认退回纯文本版仅为独立调用方便。
    """
    import re as _re

    def _default_play(q: str, tool: str, par) -> bool:
        return bool(_re.search(r"[播放听看]|来一首|来一段|进入|打开|给我唱|我想唱|我要唱", q))

    pb = provides_playback or (lambda s: _default_play(s["q"], s["tool"], s.get("params")))
    CONTENT = CONTENT_GOLD
    out: dict[str, list[int]] = {}
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r.get("exec_mode") != "serial":
                continue
            segs = r["seg"].split("#")
            tools = r["tools"].split("+")
            gold = r["gold"].split("+")
            try:
                tps = json.loads(r["tool_param"])
            except json.JSONDecodeError:
                tps = []
            if not isinstance(tps, list):
                tps = []
            if not segs or not tools or gold[0] != "设备" or tools[0] not in DEVICE_TOOLS:
                continue
            t = tools[0]
            c = out.setdefault(t, [0, 0])
            c[1] += 1                       # 作为首位设备句
            fp = None
            for i, g in enumerate(gold):
                if g in CONTENT and i < len(segs):
                    s = {"q": segs[i], "tool": tools[i] if i < len(tools) else "",
                         "d": "content", "params": tps[i] if i < len(tps) else None}
                    if pb(s):
                        fp = i
                        break
            if fp is not None and fp > 0:
                c[0] += 1                   # 先于一条播放内容句
    return {k: (v[0], v[1]) for k, v in out.items()}


def harvest_relaxed(path: Path = REF) -> list[tuple[str, str, dict | None, int]]:
    """按【准入口径】取材，覆盖 DEVICE_TOOLS 全量：字面逐字照搬，允许参数为 null。

    与 harvest() 的差别在准入正则。harvest 用 GOOD_START（防参数表残句），
    但 GOOD_START 有几处把真实指令也挡掉了，实测造成池子缺口：
      · solve_picture_sound_problem_control 根本不在 HARVEST_TOOLS 里（12 条 vs 参考 44）；
      · display_control 产品特性名开头（"打开墨晶屏""打开XDR极致动态范围"）被 LABELISH
        整类剔除，但参考集里这类说法【带动词出现】时就是真指令（60 条 vs 参考 118）。
    这里改用「含动词 or 频次≥3」，并把 LABELISH 的拦截放宽为「仅在不是动词开头时拦截」。

    LABELISH 为什么不能整类删：参考集里 "超解像"/"疾速屏"/"量子点" 这类词【不带动词】
    出现时（"超解像"）确实是能力项残句，拼进多意图会语法断裂——那部分仍要挡。
    带动词的（"打开超解像演示"）才是真指令。
    timer_control 另加时间量词约束，理由见 harvest() 里的同名判断。

    【本次改动：参数非空不再是准入条件】。参考集里 24.5% 的设备位 tool_param 为 null，
    且这一状态是【按子句系统性】的——"大点声"(90 次)、"快进到三分钟"(31)、
    "音量调到二十"(32)、"播放第1集"(40) 每一次都是 null；而"全屏播放""返回主页"
    从不为 null。这 251 条 always-null 说法占设备子句总频次的 14.3%，是整池最大的一块
    缺口（设备缺失质量 17.1% 中的 84%）。它们恰恰是最口语的一档，也是"表达多样性不够"
    的主要来源。参考集把这类短促指令标为未标注状态（eval_multi.judge() 对 null 步
    不计参数分，视为"该步未标"），所以照搬 null 才与参考集口径一致，不是丢标签。

    返回值比 harvest() 多一列【参考集频次】。必须带频次：设备子句的参考分布极度倾斜
    （screen_layout：全屏播放 448 / 全屏 407 / 大屏播放 337，占该工具 75%），
    若按 uniform 入池会抽出一堆尾部说法（"全屏设置" 参考 4 次却被放大 12.5 倍），
    等于把"生硬"换成了"分布不像人话"。
    """
    cnt: dict[tuple[str, str], int] = {}
    par: dict[tuple[str, str], dict] = {}          # 该子句在参考集里出现过的非空参数
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            segs = r["seg"].split("#")
            tools = r["tools"].split("+")
            try:
                tps = json.loads(r["tool_param"])
            except json.JSONDecodeError:
                tps = []
            if not isinstance(tps, list):
                tps = []
            gold = r["gold"].split("+")
            for i, t in enumerate(tools):
                if t not in DEVICE_TOOLS:
                    continue
                if i >= len(segs):
                    continue
                if i < len(gold) and gold[i] != "设备":
                    continue                      # 只取设备位（同 attested_device_freq 的口径）
                q = segs[i].strip()
                # 【长度上限从 13 放到 15】。参考集设备位的合法子句长度 p99=13、max=20，
                # 但 p99 之外的恰是一整族真实说法：「画面X了帮我调节一下」（14-15 字，
                # solve_picture_sound_problem_control 的口语抱怨式指令，参考集 40+ 条）。
                # 13 字上限把它们整族砍掉，该工具的内容位因此几乎全靠我手写库兜底。
                if not (2 <= len(q) <= 15):
                    continue
                # 【删除 BARE_NOUN 拦截】。BARE_NOUN 是为【手写库/变体机】写的——
                # 防变体机从「打开HDMI接口」派生出光杆名词短语残句。但本函数只扫
                # 【gold=设备 的参考集子句】，它们本就逐字出现在真人语料里、且已是
                # 准入闸门 _ATTESTED 的成员，不存在"残句"一说。实测这条拦截把
                # HDMI切换/HDMI接口/无线网/以太网连接/外部设备/播放模式/音乐功能
                # 共 21 种真实说法整类删掉（参考集里它们是标准的第二个设备子句：
                # "唱个白毛女，同时HDMI切换"）。手写库那条路径（_variants）仍受
                # BARE_NOUN 约束，不受此改动影响。
                # LABELISH 只在【不带动词】时拦截——"超解像"是残句，
                # "打开超解像演示"是真指令（参考集里两者都有）。
                if LABELISH.search(q) and not _ADMIT_VERB.search(q):
                    continue
                if len(CONTAM.findall(q)) >= 2:
                    continue
                # timer_control：切分错位粘进来的开机/关机要挡（见 harvest()）。
                # 原有判据要求明面时间词，但「剩余关机时长/关机时间设置/关机时间查询」
                # 这类【查询式】定时指令不带时间词，被整族删掉。改为：明面时间词
                # 【或】带定时语义词根（剩余关机|关机时间|定时|时长）。注意不能只加
                # 时长|时间——那会放进 "音量开小些"（参考集里被错标成 timer_control
                # 的音量句），拼进产出就是"音量开小些"配 timer 工具的金标错位。
                if t == "timer_control" and not re.search(
                        r"(\d+|一个?|半|三|十|两|分钟|小时|点|睡前|延时|定时|"
                        r"剩余关机|关机时间|关机时长|定时关机)", q):
                    continue
                key = (q, t)
                cnt[key] = cnt.get(key, 0) + 1
                tp = tps[i] if i < len(tps) else None
                if isinstance(tp, dict) and tp:
                    par.setdefault(key, tp)       # 记住可复用的非空参数（同一子句参数一致）
    out: list[tuple[str, str, dict | None, int]] = []
    for (q, t), n in sorted(cnt.items(), key=lambda x: -x[1]):
        # 频次门槛从 n>=3 放到 n>=2。这里的候选【全部】是 gold=设备 的参考集子句，
        # 不存在"编造"的可能，门槛唯一的副作用是删掉参考集里真实出现 2 次的说法——
        # 实测「音量放到十」「跳到上一首」「全屏模式」「后退10秒」「音频模式」
        # 等 75 种（124 次）都是这种被误删的。freq=1 仍挡（那批里混有 seg 切分残渣，
        # 且单次说法对整卷 distinct 的贡献远小于它带来的噪声风险）。
        if _ADMIT_VERB.search(q) or n >= 2:
            out.append((q, t, par.get((q, t)), n))
    return out


def harvest(path: Path = REF) -> list[tuple[str, str, dict]]:
    """返回 (子句, tool, params) 列表，params 直接取自参考集标注。"""
    out: list[tuple[str, str, dict]] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            segs = r["seg"].split("#")
            tools = r["tools"].split("+")
            try:
                tps = json.loads(r["tool_param"])
            except json.JSONDecodeError:
                continue
            for i, t in enumerate(tools):
                if t not in HARVEST_TOOLS:
                    continue
                if i >= len(segs) or i >= len(tps) or not tps[i]:
                    continue                      # 只收标注完整的
                q = segs[i].strip()
                if not (3 <= len(q) <= 13):
                    continue
                if LABELISH.search(q) or BARE_NOUN.match(q):
                    continue
                if len(CONTAM.findall(q)) >= 2:
                    continue
                # timer_control 里有些 seg 是被切错的（"设置10分钟把电视关了" 实际是
                # 定时关机 + 关机两条粘在一起；"关电视" 根本不是定时）。定时类必须自带
                # 时间量词才算数。
                if t == "timer_control" and not re.search(
                        r"(\d+|一个?|半|三|十|两|分钟|小时|点|睡前|延时|定时)", q):
                    continue
                if not GOOD_START.match(q):
                    continue
                if q in seen:
                    continue
                seen.add(q)
                out.append((q, t, tps[i]))
    return out


# 参考集 numeric_adjust 的程度档有一处系统性标注毛病：字面是「减/小/低/降」的句子
# 被标成 operation=提高（「音量减」62 次、「声音减」59 次，121/121 全部如此），
# 「声音再小点」也被标成提高。这是参考集自身的标注错误（大概率把「减」读成了
# 「加减档位」），不是我该复刻的「惯例」——复刻会让同一份数据里
# 「音量减→提高」和「音量减30→降低」自相矛盾，模型学到的是噪声。
# 处理：保留参考集的【词表与频次】，但按字面方向过滤掉方向倒置的条目。
_UP_W = re.compile(r"(大|高|亮|提高|加|增|强|浓|深|鲜艳|锐|开大)")
_DOWN_W = re.compile(r"(小|低|暗|降|减|弱|淡|浅|关小|调暗)")


def _direction_inverted(q: str, p: dict) -> bool:
    """字面方向与金标 operation 相反时返回 True。只对程度档（value 为空）判定。"""
    op = p.get("operation")
    if op not in ("提高", "降低"):
        return False
    v = str(p.get("value") or "")
    if v.isdigit() or v.endswith("%"):
        return False                          # 设值类不看方向词
    up, down = bool(_UP_W.search(q)), bool(_DOWN_W.search(q))
    if up and not down:
        return op == "降低"
    if down and not up:
        return op == "提高"
    return False                              # 双向词或无方向词：不作判定


# numeric_adjust 的收割口径与上面不同：它要的是【频次】而非清单。
# 原因：参考集的 numeric_adjust 程度档是典型 Zipf 分布——180 个不同说法里，
# 「亮度高一点」89、「声音大点」78、「音量大点」76，而尾部大量说法只出现 1 次。
# 我手写的 368 条程度说法里 74% 在参考集里根本不存在，且权重是手设的均匀值，
# 于是抽出来既没有高频的头部说法，又充斥参考集没有的自造说法。
# 按参考集真实频次加权，头部自然吃掉大部分概率，尾部低频说法只占极小份额——
# 这才是「表达多样性」的正确形态（多样但倾斜），而不是均匀铺开。
_NUM_START = re.compile(
    r"^(音量|声音|亮度|屏幕|对比度|清晰度|色度|色彩|颜色|画面|大|小|高|低|亮|暗|"
    r"浓|淡|深|浅|把|让|调|提|降|增|减|加|开|关|多|少|点)")


# 参考集里 tool_param 恒为空的内容工具。这些工具的金标参数本来就是空对象——
# 不是「标注缺失」，而是「这个工具不需要参数」（模糊搜索/推荐/知识问答/历史/频道
# 音乐检索都属于此类，参考集里 param 非空率 0%）。
#
# 为什么必须单独收这一类：我原来的收割口径要求 tool_param 非空，于是恰好把这几个
# 【参考集中说法最多】的内容工具整个漏掉了——edu_fuzzy_search 239 种、vod_fuzzy_search
# 199 种、fan_knowledge_agent 128 种、music_song_history 61 种、tvchannel_music_search
# 56 种。结果我的内容子句池在这几个工具上只有 7-16 条候选，抽 313 次只用出 13-73 种
# 说法（distinct 率 0.05-0.23，参考是 0.21-0.53）。这是内容子句 distinct 只有参考
# 六成、4-gram 熵落后的结构性原因，不是「再多写几句」能补的。
EMPTY_PARAM_CONTENT_TOOLS = {
    "edu_fuzzy_search", "educ_fuzzy_search", "vod_fuzzy_search", "fan_knowledge_agent",
    "music_song_history", "tvchannel_music_search", "music_song_recommend",
    "music_song_favorite_search",
}

# 空参内容句里必须剔除的：纯能力项标签（"播放列表""音乐"）——它们不成句，
# 与设备侧 BARE_NOUN 同理。带动词的完整句（"播放音乐""来一首歌呢~"）保留。
CONTENT_BARE = re.compile(
    r"^(播放列表|播放记录|播放历史|歌单|音乐|歌曲|歌曲列表|收藏|收藏列表|历史|"
    r"听歌历史|音乐功能|音乐频道|音乐台|电视台|频道|视频|影片|动画|动画片|课程|"
    r"知识|问答|搜索|查找|播放|推荐)$")


def content_clause_freq(path: Path = REF) -> dict[str, int]:
    """返回 {内容子句: 参考集频次}（gold∈内容域 口径，不计工具）。

    用途与设备侧的 attested_device_clauses 同构：给【内容位】的每个表面形式一个
    参考集实测频次，作为唯一权重来源。此前只有收割来的内容子句带频次，种子库那
    1182 条（80% 也在参考集里、平均 1.7 次）却按长度启发式 _c_len_w(≤1.0) 配权，
    两源差两个数量级，产出高度集中在少数高频收割项上——实测 top-1000 类型占总质量
    87%，而参考集同口径只有 68%。统一到参考频次后两源可比。
    """
    out: dict[str, int] = {}
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            segs = r["seg"].split("#")
            gold = r["gold"].split("+")
            tools = r["tools"].split("+")
            for i, g in enumerate(gold):
                if i >= len(segs):
                    continue
                # 口径与 harvest_content_counts 同步放宽到「gold∈内容域 or tool 属内容域」：
                # 参考集有 306 个位置是 gold=设备 但挂内容工具（2.3%，见那里的说明），
                # 其中 4 个说法（播放第一集/播放第一个/播放第二个视频/播放列表）只存在于
                # 这一档。若这里仍按 gold 过滤，harvest 收进来了、却拿不到频次，lw 退回
                # _c_len_w(≈1.0)，等于把参考集 20-38 次的说法按 1 次配权——收割等于白做。
                _t = tools[i] if i < len(tools) else ""
                if g in CONTENT_GOLD or _tool_dom(_t) in _CONTENT_DOM:
                    q = segs[i].strip()
                    if q:
                        out[q] = out.get(q, 0) + 1
    return out


def content_clause_freq_nonreg(path: Path = REF) -> dict[str, int]:
    """返回 {内容子句: 参考集频次【扣除语域覆盖行】}——主池应配的残差质量。

    【为什么必须扣（2026-11-12，第二十四轮）】语域表（cc_pair / cd_cross / dc_pair /
    dd_pair / triple）是【逐字照搬 REF 整行】的：凡是落在这些行里的说法，已经由
    【语域路径】按 REF 自己的比例产出了（实测语域覆盖率 39.4% vs REF 38.5%，逐位吻合）。
    可主池的权重仍是该说法的【全额参考频次】，于是这些说法被【计了两次】——
    实测（gen 探针，content 说法、target≥4）：语域∩池 组的 realized/expected 中位
    **1.63×**（池 0.58 + 语域 1.13），纯池组 0.78、纯语域组 1.05。超产的正是
    `播放免费的小猪佩奇`(2.5×)、`播放第一集`(1.6×)、`播放第12集`(2.0×)、`我要看云视听小电视`
    这一族——也就是每一枚 n-gram 探针和用户那一眼都指向的"翻来覆去那几句"。
    主池只应补足【非语域行】的那部分质量，故这里把语域覆盖行里的出现次数扣掉。
    """
    full = content_clause_freq(path)
    if not full:
        return full
    tabs: dict[str, set] = {}
    for mode in ("serial", "parallel"):
        ks: set = set()
        for fn in (cc_pair_register, cd_cross_register, dc_pair_register,
                   dd_pair_register, triple_register):
            try:
                ks.update(fn(mode))
            except Exception:
                continue
        tabs[mode] = ks
    covered: dict[str, int] = {}
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            segs = [s.strip() for s in r["seg"].split("#")]
            gold = r["gold"].split("+")
            tools = r["tools"].split("+")
            if len(segs) != len(tools):
                continue
            k = tuple(x for pair in zip(tools, segs) for x in pair)
            if k not in tabs.get(r.get("exec_mode", ""), set()):
                continue
            # 口径必须与 content_clause_freq 逐位一致（gold∈内容域 or 工具属内容域），
            # 否则会把设备位上的同名说法也算进 covered，扣除量 > 实际计入量（出现负残差）。
            for i, s in enumerate(segs):
                if not s or i >= len(gold):
                    continue
                _t = tools[i] if i < len(tools) else ""
                if gold[i] in CONTENT_GOLD or _tool_dom(_t) in _CONTENT_DOM:
                    covered[s] = covered.get(s, 0) + 1
    return {q: max(0, n - covered.get(q, 0)) for q, n in full.items()}


def attested_content_leadfreq(path: Path = REF) -> dict[str, tuple[int, int]]:
    """返回 {(内容子句, gold): (首位次数, 非首位次数)}——只统计 gold∈内容域 的位置。

    与设备侧 attested_device_leadfreq 同因同构。参考集内容子句的位置条件性【更强】：
    χ²/df=96.3，freq≥10 的 61 个表面里 19 个（31.1%）【从不】在首位出现，
    41 个【总是】在首位。且非首位的内容位是一个【封闭语域】——只有 20 个表面出现过，
    其中 86% 是"播放/放开/放个+具体片名"式祈使句，【0%】是搜索式（"有哪些/推荐/适合…"）。
    我的产出把首位的内容混比（33% PLAY、11% SEARCH）原样带进非首位，于是
    204 个非首位内容位里只有 3.4% 落在参考集的封闭集里，2 字前缀命中率也仅 25.5%。
    """
    out: dict[tuple[str, str], list[int]] = {}
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            segs = r["seg"].split("#")
            gold = r["gold"].split("+")
            for i, g in enumerate(gold):
                if g not in CONTENT_GOLD or i >= len(segs):
                    continue
                q = segs[i].strip()
                if not q:
                    continue
                c = out.setdefault((q, g), [0, 0])
                c[0 if i == 0 else 1] += 1
    return {k: (v[0], v[1]) for k, v in out.items()}


def content_nullrate(path: Path = REF) -> dict[str, float]:
    """返回 {内容子句: 该说法在参考集里 tool_param 为 null 的比例}（gold∈内容域 口径）。

    与设备侧 attested_device_nullrate 同因同构：参考集内容位 26.5% 的 tool_param 是
    null（另有 21.4% 是空 dict、52.2% 有内容），且同样是【按说法系统性】的。
    关键：null 与 {}/空参在评测里【不是一回事】——eval_multi.judge() 对 null 步直接
    不计参数分，而 {} 会被当成"有参数但为空"去比对。我的收割把白名单工具的空参
    一律存成 {}，于是 null 与 {} 被合并：实测我的内容位 null 0.0% / {} 25.7%，
    参考集是 26.5% / 21.4%。这不只是占比问题，是把参考集刻意留白的步当成了有金标的步。
    """
    num: dict[str, int] = {}
    den: dict[str, int] = {}
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            segs = r["seg"].split("#")
            tools = r["tools"].split("+")
            gold = r["gold"].split("+")
            try:
                tps = json.loads(r["tool_param"])
            except json.JSONDecodeError:
                tps = []
            if not isinstance(tps, list):
                tps = []
            for i, t in enumerate(tools):
                if i >= len(segs) or i >= len(gold) or gold[i] not in CONTENT_GOLD:
                    continue
                q = segs[i].strip()
                if not q:
                    continue
                den[q] = den.get(q, 0) + 1
                v = tps[i] if i < len(tps) else None
                if v is None:                     # 只数真 null，空 dict 不算
                    num[q] = num.get(q, 0) + 1
    return {q: num.get(q, 0) / n for q, n in den.items() if n}


_CATALOG_DOM: dict[str, str] | None = None
_CONTENT_DOM = {"vod", "music", "education", "children", "sports", "audio"}


def _tool_dom(t: str) -> str | None:
    """tool 名 → catalog 权威域。catalog 缺失时返回 None（退化为纯位置配对）。"""
    global _CATALOG_DOM
    if _CATALOG_DOM is None:
        try:
            cat = json.load(open(Path(__file__).resolve().parent.parent / "catalog.full.json",
                                  encoding="utf-8"))
            _CATALOG_DOM = {x["tool"]: x["domain"] for x in cat["tools"]}
        except (OSError, KeyError, json.JSONDecodeError):
            _CATALOG_DOM = {}
    return _CATALOG_DOM.get(t)


def _pair_tools(segs: list[str], tools: list[str], tps: list, gold: list[str]) -> list[int]:
    """把 gole 每格配到一个 tool 下标，返回配对下标列表（长度 = len(segs)）。

    参考集有 1.42% 的可靠行把 tools 整体写反了（与 seg/gold 位置错位），
    典型形态：seg=[关电视, 播放海底小纵队第三季] gold=[设备, 少儿]
    tools=[educ_search_all, power_control]——内容工具落在设备位上。
    按 gold 的域类别（内容/设备）与 catalog 域做贪心配对，即可把这类行纠正回来，
    使内容子句不再被误配成设备工具（进而被生成器的 dom 闸门整条丢掉）。
    """
    n = len(segs)
    if len(tools) != n or len(gold) != n or n == 0:
        return list(range(n))
    want_c = [(g in CONTENT_GOLD) for g in gold]
    have = [(_tool_dom(tools[j]) in _CONTENT_DOM) for j in range(n)]
    if want_c == have:
        return list(range(n))                       # 位置本就一致，零成本直通
    done = [False] * n                              # 该输出格已定
    taken = [False] * n                             # 该 tool 已被占
    out = list(range(n))
    # 先满足位置本就一致的格（贪心会给它们留出唯一的归属）
    for i in range(n):
        if want_c[i] == have[i]:
            out[i] = i
            done[i] = True
            taken[i] = True
    for i in range(n):
        if done[i]:
            continue
        for j in range(n):
            if not taken[j] and have[j] == want_c[i]:
                out[i] = j
                done[i] = True
                taken[j] = True
                break
    return out


def harvest_content_counts(path: Path = REF) -> dict[str, tuple[str, dict, int]]:
    """收割内容域子句——{子句: (tool, params, 频次)}。

    动机同设备侧：我的内容子句池来自 all_eval.csv（1182 条去重），而参考集里
    内容域子句有 6900+ 条、去重后 2684 种说法，其中绝大多数不在我的池子里。
    这部分正是内容子句表面形式的缺口——产出里内容子句 distinct 只有参考的六成，
    直接拉低整卷 4-gram 熵（type/gram 0.247 vs 参考 0.371）。

    收割口径两类：
      A. tool_param 非空的子句——参数是标注好的，直接复用（严格）。
      B. 参数为 null 的子句——参考集刻意留白（judge() 对 null 步不计参数分），
         照搬 None 即可（宽松，但仍是逐字照搬，不编造参数）。
      C. EMPTY_PARAM_CONTENT_TOOLS 里参数为空 dict 的子句——这些工具参考集参数恒为空，
         复用时空参即可，与线上同口径。
    三类都要剔除残句/自指名项/超长句。

    【本次改动：null 与空 dict 不再合并】。参考集内容位里 null 占 26.5%、空 dict 占 21.4%，
    两者在 eval_multi.judge() 里【不等价】：null 步不计参数分，{} 会被当作"有参数"去比。
    旧版把白名单工具的空参一律存成 {}，且把非白名单的 null 整条丢掉，于是我的内容位
    变成 null 0.0% / {} 25.7%（参考 26.5% / 21.4%），凭空给留白步编了金标。
    """
    out: dict[str, tuple[str, dict | None, int]] = {}
    par: dict[str, dict | None] = {}          # 该子句出现过的最有信息量的参数（优先非空）
    # 同一内容位说法在参考集里可能挂过【内容域】与【设备域】两种 tool
    # （"播放第1集" 内容位 21 次 vod_search_all、25 次 playback_control——后者是标注噪声）。
    # 生成器要求内容步必须用内容域工具，取到 device 的那一份会整条作废。
    # 故这里记下每个说法见过的【内容域】tool，最后优先采用它。
    cdom: dict[str, str] = {}                 # 子句 -> 见过的内容域 tool（首次为准）
    # 【内容说法只出现在几步句里——第二十二轮「生硬」的根因之一】。
    # 参考集里有 8 个内容说法【只在三步句里出现过】：`打开电影长安的荔枝`(26)、
    # `我想看长安的荔枝`(25)、`我想看沈腾的喜剧电影西虹市首富`(23)、
    # `我要看外交风云第25集第5分钟`(22)、`播放外交风云第1集`(21)、
    # `播放狂飙第20集第35分钟`(20)、`打开音乐对面的女孩看过来`(19)、
    # `我想看帕丁顿熊三`(19)——共 175 个实例，两步句里【一次都没有】。
    # 它们本就是 `triple_register` 的三步句内容位（"内容句 + 两个调节"，
    # 如 `我想看沈腾的喜剧电影西虹市首富#音量大#点亮度低点`）。但收割是
    # 【按位置不分步数】收的，于是它们也进了两步句内容池；两步句抽到后
    # `_cd_cond_device` 查不到说法桶（这些说法在 CD 行里不存在→无桶），
    # 退回工具兜底桶，接上一个【随意的】设备尾巴：
    #   `我想看沈腾的喜剧电影西虹市首富，快退1分钟` / `…进入主题`
    #   `打开电影长安的荔枝，wifi功能` / `…半小时后关机` / `…开启小窗`
    # 参考集里这些搭配【零例】——真人说这句时后面跟着的必是"调音量+调亮度"，
    # 那正是它被写进三步句的原因。实测 MINE 两步句里带这类说法的行有 86 条。
    # 判据：该说法在参考集内容位的 n_steps 分布里，三步实例 >0 且两步实例 ==0。
    # 剔除后这些说法只经 triple_register 到达（三步句路径，搭配正确）。
    nsteps: dict[str, set[int]] = {}          # 子句 -> 出现过的 n_steps 集合
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            segs = r["seg"].split("#")
            tools = r["tools"].split("+")
            gold = r["gold"].split("+")
            try:
                tps = json.loads(r["tool_param"])
            except json.JSONDecodeError:
                continue
            if not isinstance(tps, list):
                continue
            pair = _pair_tools(segs, tools, tps, gold)
            if pair != list(range(len(pair))):
                tools = [tools[j] for j in pair]          # 按域重排，纠位置错位
                tps = [tps[j] if j < len(tps) else None for j in pair]
            for i, t in enumerate(tools):
                if i >= len(segs) or i >= len(tps):
                    continue
                # 只收割【内容位】（gold 属内容域）。参考集里同一个说法可以两种身份出现：
                # 「播放第1集」内容位 46 次（vod_search_all/playback_control）、设备位 38 次；
                # 「播放短视频」内容位 21 次、设备位 21 次。旧版按【所有位置】收割，谁先出现
                # 谁的 tool 就定下来，于是这类说法常被钉上设备工具，再被生成器的 dom 闸门
                # 整条丢掉——它是参考集里第 3 高频的内容子句（84 次），我却只产出 8 条。
                # 按 gold 过滤后，每个说法取的是它在【内容位】的真实工具与参数。
                #
                # 【2026-11-11 补漏：gold=设备 但 tool 属内容域 的位置】。参考集的跨域标注
                # 是双向的：除了「内容位挂设备工具」（19.1%，上一条说明），还有反过来的
                # 「设备位挂内容工具」306 例（2.3%）。后者里 6 个说法只有 4 个是我池子缺的
                # ——`播放第一集`38/`播放第二个视频`28/`播放第一个`27/`播放列表`20，共 113
                # 例，且**每例都是不同 query**（参考集里 38 个「播放第一集」是 38 条不同的行）。
                # 它们两头不靠：设备收割要求 tool∈DEVICE_TOOLS（命中不了），内容收割要求
                # gold∈内容域（也命中不了），于是整族消失——我产出里「第X集」只有阿拉伯数字
                # 一种写法（`播放第1集`），完全没有 `播放第一集/播放第一个/播放第二个视频`
                # 这一族，而参考集里它的数量与阿拉伯写法相当。这正是一条"专用词都对、
                # 但就是没有真人会用的那种说法"的多样性缺口（standing instruction）。
                # 判据放宽为「gold∈内容域 or tool 属内容域」：实测只新增这 4 个说法、
                # 对已有说法【零工具变更】（同一说法在两种 gate 下取到的 tool 一致），
                # 也不会把任何设备说法错误收进来（设备位里 tool 属内容域的都是内容播放句）。
                # 语义上它们本就是内容点播（"播放第一集"＝放第 1 集），归内容位是对的；
                # 参考集把 gold 标成"设备"才是那份噪声。
                if i >= len(gold):
                    continue
                if gold[i] not in CONTENT_GOLD and _tool_dom(t) not in _CONTENT_DOM:
                    continue
                tp = tps[i]
                if tp is None:
                    p_here: dict | None = None       # 留白，照搬
                elif isinstance(tp, dict) and tp:
                    p_here = tp
                elif t in EMPTY_PARAM_CONTENT_TOOLS:
                    p_here = {}                       # 白名单工具的空参
                else:
                    continue                          # 空 dict 且不在白名单，不可复用
                q = segs[i].strip()
                # 下限从 5 放到 2。参考集内容子句里最短的那批恰恰是最高频的一批：
                # len<=4 占内容子句【实例】7.8%（538/6907），其中「播放音乐」80 次
                # （全卷最高频内容子句）、「播放儿歌」19、「最强大脑」19、「看西游记」13、
                # 「综艺节目」6，还有球队/艺人裸名（河床、罗马、邵峰、王宝强）。它们不是残句，
                # 而是真人点播时的最短说法，参考集自己就把它们并进多意图
                # （"播放音乐，同时音量小一点"）。5 字下限把这一档整块删了，
                # 换来的正是内容侧"翻来覆去那几句长标题"的僵硬感。
                # 空参工具的能力项标签（"播放""音乐""频道"）另由下面 CONTENT_BARE 拦掉。
                if not (2 <= len(q) <= 26):
                    continue
                if LABELISH.search(q) or BARE_NOUN.match(q):
                    continue
                if p_here == {} and CONTENT_BARE.match(q):
                    continue                      # 空参工具里更要剔能力项标签
                if len(CONTAM.findall(q)) >= 2:
                    continue
                prev = out.get(q)
                if prev:
                    out[q] = (prev[0], prev[1], prev[2] + 1)
                else:
                    out[q] = (t, p_here, 1)
                # 同一子句既有 null 又有非空参数时，保留信息量更大的那份
                cur = par.get(q)
                if p_here and not cur:
                    par[q] = p_here
                elif q not in par:
                    par[q] = p_here
                # 记下该说法在参考集内容位【确实用过】的内容域工具（供下面纠正标注噪声）
                if _tool_dom(t) in _CONTENT_DOM and q not in cdom:
                    cdom[q] = t
                # 记下该说法在参考集里出现过的 n_steps（供下面剔"只在三步句出现"的说法）
                try:
                    nsteps.setdefault(q, set()).add(int(r.get("n_steps") or 2))
                except (TypeError, ValueError):
                    nsteps.setdefault(q, set()).add(2)
    # 剔除【只在三步句出现】的内容说法（见上方 nsteps 说明）：这类说法的正确搭配
    # 是"内容句 + 两个调节"，属 triple_register；放进两步句池只会被接上随意的设备尾巴。
    _only3 = {q for q, ns in nsteps.items() if 3 in ns and 2 not in ns}
    for q in _only3:
        out.pop(q, None)
        par.pop(q, None)
        cdom.pop(q, None)
    for q, (t, _p, n) in list(out.items()):
        # 标注噪声纠正：内容位里挂设备工具的（参考集里占 19.1%），若同一说法在其它内容位
        # 用过内容域工具，就改用那个——否则生成器 dom 闸门会把整条丢掉。
        if _tool_dom(t) not in _CONTENT_DOM and q in cdom:
            t = cdom[q]
        out[q] = (t, par.get(q), n)
    return out


# 内容子句的句首框架「请求语 + 动词」。两段都要能对上参考集的切法，
# 故正则与生成器里的 _REQ_HEAD / _FRAME_HEAD_RE 保持同一份交替表。
_REQ_HEAD = re.compile(r"^(请你|请|帮我|给我|麻烦|我要|我想|我很想)")
_FRAME_HEAD = re.compile(
    r"^(搜一下|放一下|听一下|看一下|查一下|唱一下|播一下|学一下|找一下|挑一下|看下|"
    r"来一首|来点|搜索|查找|播放|推荐|学习|放个|播|放|听|看|唱|搜|找|查|来|打开|进入|学|用|挑)")

# 参考集里可作为【内容子句】的域（与生成器 CN 一致；设备类不算内容）
CONTENT_GOLD = {"影视", "音乐", "有声", "教育", "少儿", "体育"}

# 旧工具名 → 当前目录工具名（与 tools/gen_serial_parallel_5k.py 的 TOOL_ALIAS 同表）。
# 【为什么 cd_cond_register 也要归它——第二十一轮】参考集用的是【旧名】
# （`educ_search` 192 行、`vod_search` 168 行、`educ_relate_recommend`…），
# 而生成器已经按目录现名（`educ_search_all`/`vod_search_all`/`educ_relate_search`）
# 查询 by_t 兜底桶。若不归一，两个名字各自成桶：含糊的旧名桶装着参考集的
# 【干货】（192 行、分布贴近真实边际），而生成器查的现名桶只剩零星行
# （`educ_search_all` 仅 13 行、且 13/13 是单例）——抽出来近乎均匀撒稀有说法。
# 实测（seed 87, pos=1）：`打开高刷新率` REF 1→MINE 12、`现在亮度多少` 5→17、
# `打开月光模式` 4→14，z 高达 9~11.5，全部经由这条 by_t 兜底路径。
# 归并后 by_t 只留【每内容工具的单一桶】，与生成器查询键逐字一致。
TOOL_ALIAS = {"vod_search": "vod_search_all", "educ_search": "educ_search_all",
              "educ_relate_recommend": "educ_relate_search",
              "sports_match_data_search": "sports_rank_search"}


def content_frame_stats(path: Path = REF, k: float = 5.0
                        ) -> tuple[dict[str, dict[tuple[str, str], float]], dict]:
    """返回内容子句的【按工具的 (请求语,动词) 条件分布】与全局分布。

    动机：生成器是两级抽样（先抽工具、再在该工具内抽句子），所以句内分布必须是
    「给定工具」的条件分布 P(请求语,动词 | tool)，不能用全局 P(请求语,动词)。

    实测这个区别很大：audio_search 里「无请求语+播放」占 19.1%，而 vod_fuzzy_search
    里「裸标题」占 75.5%。用全局分布配权，会让每个工具都被拉向同一个平均句式，
    实测 audio_search 的「我要听」被压到 0.33 倍、vod_fuzzy_search 的光杆「找」
    被放大到 1.8 倍。

    平滑：k=5 的加性平滑（先验取全局份额），仅用于给小工具一个非零兜底
    （如 educ_relate_search 全工具只出现 2 次内容子句），不改变中大型工具的分布。

    实测 k 不能大：加权 L1 误差 k=2 时 0.9%、k=20 时 7.4%、k=200 时 34.7%。
    因为【经验条件分布本身就是目标】（它就是从参考集数出来的），平滑只是防零。
    早先取 k=200 属于过度平滑，把各工具都拉向全局均值——小工具的「播放」档
    被抬高到 2-4 倍（vod_fuzzy_search 实测 2.8% 被拉成 11.8%）。

    返回值：(per_tool, global)。per_tool[tool][(req,verb)] 是千分比量级的分值，
    只保证相对大小有意义（生成时按档内 lw 之和再归一）。
    """
    gl: dict[tuple[str, str], int] = {}
    pt: dict[str, dict[tuple[str, str], int]] = {}
    tot: dict[str, int] = {}
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            segs = r["seg"].split("#")
            tools = r["tools"].split("+")
            gold = r["gold"].split("+")
            for i, t in enumerate(tools):
                if i >= len(segs) or i >= len(gold) or gold[i] not in CONTENT_GOLD:
                    continue
                q = segs[i].strip()
                if not q:
                    continue
                m = _REQ_HEAD.match(q)
                rq = m.group(1) if m else ""
                rest = q[m.end():] if m else q
                mv = _FRAME_HEAD.match(rest)
                key = (rq, mv.group(1) if mv else "")
                gl[key] = gl.get(key, 0) + 1
                pt.setdefault(t, {})
                pt[t][key] = pt[t].get(key, 0) + 1
                tot[t] = tot.get(t, 0) + 1
    N = sum(gl.values()) or 1
    keys = set(gl)
    per_tool: dict[str, dict[tuple[str, str], float]] = {}
    for t, cnts in pt.items():
        n = tot[t]
        per_tool[t] = {
            key: (cnts.get(key, 0) + k * gl[key] / N) / (n + k) * 1000.0
            for key in keys
        }
    glob = {key: gl[key] / N for key in keys}
    return per_tool, glob


def harvest_question_clauses(path: Path = REF) -> dict[str, tuple[str, dict, int]]:
    """收割【问句开头】的子句——{子句(含？): (tool, params, 频次)}。

    参考集里问句型子句有 205 行，形状高度一致，是一个独立的句式族：
      · 205/205 全部位于【第 0 位】（问句永远先说）；
      · 205/205 第二步都是【设备】子句（"怎么预习新课才有效？，同时进入wifi强度测试"）；
      · 205/205 都是两步句；
      · parallel 109 / serial 96（几乎对半）。
    工具集中在 fan_knowledge_agent(150)、audio_search(20)、vod_fuzzy_search(16)——
    都是「我不知道要什么，先问一句」的检索/问答类，这正是真人问电视的典型开场。

    我的生成器把这类子句 100% 过滤掉了：ASK_Q 正则（"[吗么呢吧]？$|？|怎么|为什么|哪些…"）
    是给 all_eval 种子池写的——在那个池子里一行问句就是纯 chat、没有可执行的工具。
    但参考集里的问句子句自带工具与金标（fan_knowledge_agent 就是干这个的），
    拼接后完全可执行。用同一把尺子量两个池子，把一整块真实分布误删了。

    单独成池、单独成路：问句要保留句尾「？」，而 clean()/TAIL 会把标点剥掉，
    走普通内容池就会被改写成陈述句，语义与金标都变。
    """
    out: dict[str, tuple[str, dict, int]] = {}
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            segs = [s for s in r["seg"].split("#") if s]
            tools = r["tools"].split("+")
            gold = r["gold"].split("+")
            if len(segs) < 2 or len(tools) < 2 or len(gold) < 2:
                continue
            if not segs[0].endswith(("？", "?")):
                continue
            if gold[1] != "设备":
                continue                          # 参考集里无例外，硬约束
            try:
                tps = json.loads(r["tool_param"])
            except json.JSONDecodeError:
                continue
            if not isinstance(tps, list) or len(tps) < 1:
                continue
            q = segs[0].strip()
            if not (6 <= len(q) <= 30):
                continue
            if LABELISH.search(q) or BARE_NOUN.match(q):
                continue
            p = tps[0] if tps[0] else {}
            prev = out.get(q)
            if prev:
                out[q] = (prev[0], prev[1], prev[2] + 1)
            else:
                out[q] = (tools[0], p, 1)
    return out


def harvest_numeric_counts(path: Path = REF) -> dict[str, tuple[str, dict, int]]:
    """返回 {子句: (tool, params, 频次)}，只收 numeric_adjust 且参数完整的子句。"""
    out: dict[str, tuple[str, dict, int]] = {}
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            segs = r["seg"].split("#")
            tools = r["tools"].split("+")
            try:
                tps = json.loads(r["tool_param"])
            except json.JSONDecodeError:
                continue
            if not isinstance(tps, list):
                continue
            for i, t in enumerate(tools):
                if t != "numeric_adjust":
                    continue
                if i >= len(segs) or i >= len(tps) or not tps[i]:
                    continue                      # 参数为空则不可复用（参考集里占 48.5%）
                q = segs[i].strip()
                if not (3 <= len(q) <= 13):
                    continue
                if LABELISH.search(q) or BARE_NOUN.match(q):
                    continue
                if len(CONTAM.findall(q)) >= 2:
                    continue
                if not _NUM_START.match(q):
                    continue
                if _direction_inverted(q, tps[i]):
                    continue                      # 参考集方向标注倒置的条目，不复刻
                prev = out.get(q)
                if prev:
                    out[q] = (prev[0], prev[1], prev[2] + 1)
                else:
                    out[q] = (t, tps[i], 1)
    return out


if __name__ == "__main__":
    import collections
    v = harvest()
    c = collections.Counter(t for _, t, _ in v)
    print(f"收割设备子句 {len(v)} 条")
    for t, n in c.most_common():
        print(f"  {t:38s} {n}")
    nv = harvest_numeric_counts()
    inv = [q for q, (t, p, n) in nv.items() if _direction_inverted(q, p)]
    print(f"\nnumeric_adjust 带频次子句 {len(nv)} 条，总频次 {sum(x[2] for x in nv.values())}")
    print(f"其中方向倒置（将被过滤）：{len(inv)} 条 -> {inv[:12]}")
    for q, (t, p, n) in sorted(nv.items(), key=lambda x: -x[1][2])[:25]:
        print(f"  {n:4d} {q}")
