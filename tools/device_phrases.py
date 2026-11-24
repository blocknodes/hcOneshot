#!/usr/bin/env python3
"""设备控制「自然口语短语库」：真人会说的说法 → 精确金标 (tool, params)。

来源口径：与 all_eval.csv 的 tv_control 标注完全一致（operation/object/value/date_time 字段名、
枚举取值都照抄），但 query 文本是重写的自然口语句，不用评测集里"能力项标签"式的残句
（"全屏模式""HDMI接口选择""量子点四维真实还原系统"这类拼进多意图必然断裂）。

每个条目：(自然说法, tool, params)。params 用空串占位 value/device/location，与线上标注同形。
"""
from __future__ import annotations

import re

# 表面形式相对权重（默认 1.0）。DEVICE_PHRASES 是按 surface 去重的，所以「同一说法重复
# 入池」并不能加权；需要加权的说法在这里显式登记，由生成器
# `_len_w(q) * SURFACE_W.get(q, 1.0)` 取用。必须在下面各短语档构造前先声明。
SURFACE_W: dict[str, float] = {}

# ---------------------------------------------------------------------------
# numeric_adjust：音量/亮度/对比度/清晰度/色度/静音
# ---------------------------------------------------------------------------
def _num(op: str, obj: str, value: str = "") -> dict:
    return {"operation": op, "object": obj, "value": value, "device": "", "location": ""}


def _pc(op: str, obj: str = "", value: str = "") -> dict:
    return {"operation": op, "object": obj, "value": value, "device": "", "location": ""}


def _screen(op: str, obj: str, value: str = "") -> dict:
    return {"operation": op, "object": obj, "value": value, "device": "", "location": ""}


NUMERIC: list[tuple[str, str, dict]] = []
for obj, ups, downs in [("音量", ["声音大一点", "声音调大些", "把音量调高一点", "声音太小了，调大点"],
                         ["声音小一点", "声音调小些", "太吵了，声音小点"]),
                        ("亮度", ["屏幕亮一点", "把亮度调高些", "画面太暗了，亮一点"],
                         ["屏幕暗一点", "亮度调低些", "画面太亮了，调暗点"]),
                        ("对比度", ["对比度调高一点", "对比度再强一点"],
                         ["对比度调低一点", "对比度弱一点"]),
                        ("清晰度", ["清晰度调高一点", "画面再锐利一点"], []),
                        ("色度", ["颜色调浓一点", "色彩再鲜艳一点"],
                         ["颜色调淡一点", "色彩淡一点"])]:
    for q in ups:
        NUMERIC.append((q, "numeric_adjust", _num("提高", obj)))
    for q in downs:
        NUMERIC.append((q, "numeric_adjust", _num("降低", obj)))
# 设值句模板只用参考集真实存在的「调到」；「设置成」在参考集里 0 次（见 _numeric_set_variants
# 的实测记录），这里不再作为独立源句生成。
for obj, tmpl in [("音量", ["音量调到{n}", "声音调到{n}"]),
                  ("亮度", ["亮度调到{n}"]),
                  ("对比度", ["对比度调到{n}"])]:
    for n in ["20", "30", "50", "70"]:
        for t in tmpl:
            NUMERIC.append((t.format(n=n), "numeric_adjust", _num("设置", obj, n)))
# 参考集设值类的另外几族说法，我原来一条没有（实测：X数值N 33 次→设置、X加N 69 次→提高、
# X减N 36 次→降低、X设置为N% 多次，都带明确金标）。这几族恰好把覆盖面从音量/亮度/对比度
# 扩到清晰度/色度，也是在补「表达多样性」。金标按实测：数值/百分号 → 设置；加/减 → 提高/降低。
_SET_OBJS = ["音量", "亮度", "对比度", "清晰度", "色度"]
for _o in _SET_OBJS:
    for _n in ["30", "50", "70"]:
        NUMERIC.append((f"{_o}数值{_n}", "numeric_adjust", _num("设置", _o, _n)))
    for _n in ["30", "50", "70"]:
        NUMERIC.append((f"{_o}加{_n}", "numeric_adjust", _num("提高", _o, _n)))
        NUMERIC.append((f"{_o}减{_n}", "numeric_adjust", _num("降低", _o, _n)))
        NUMERIC.append((f"{_o}增加{_n}", "numeric_adjust", _num("提高", _o, _n)))
        NUMERIC.append((f"{_o}减少{_n}", "numeric_adjust", _num("降低", _o, _n)))
    # 百分号设值（参考集清晰度/色度/对比度用得多，音量/亮度用汉字数值）
    for _p in ["30", "50", "70"]:
        NUMERIC.append((f"{_o}设置为{_p}%", "numeric_adjust", _num("设置", _o, f"{_p}%")))
    # 极值档：调到最高/最低 的 operation 参考集里两种都有（提高 41 / 设置 16），
    # value 有值的是 100%/0%，无值的是纯程度义。这里只用与字面一致的设置义。
    NUMERIC.append((f"{_o}调到最高", "numeric_adjust", _num("设置", _o, "100%")))
    NUMERIC.append((f"{_o}调到最低", "numeric_adjust", _num("设置", _o, "0%")))
    NUMERIC.append((f"把{_o}调到最大", "numeric_adjust", _num("设置", _o, "100%")))
for q, obj in [("静音", "静音"), ("把声音关掉", "静音"), ("别出声了", "静音")]:
    NUMERIC.append((q, "numeric_adjust", _num("打开", obj)))
for q in ["取消静音", "恢复声音", "解除静音"]:
    NUMERIC.append((q, "numeric_adjust", _num("关闭", "静音")))

# ---------------------------------------------------------------------------
# playback_control：上下首/集/部、跳转、快进快退、倍速、暂停继续、循环
# ---------------------------------------------------------------------------
PLAYBACK: list[tuple[str, str, dict]] = []
for q, op, val in [("下一首", "下", "首"), ("上一首", "上", "首"), ("换一首", "切换", "首"),
                   ("下一集", "下", "集"), ("上一集", "上", "集"), ("下一部", "下", "部"),
                   ("换一部", "切换", "部"), ("换一个", "切换", "个")]:
    PLAYBACK.append((q, "playback_control", _pc(op, "播放控制", val)))
# 注：「继续播放/接着放」语义上要求「已有内容在播」，与前面刚起的播冲突（"播放A，随后继续播放"），
#    故不收入库；「暂停/退出/重播」保留（重播=当前这首再放一遍）。
for q, op in [("暂停", "停止"), ("暂停一下", "停止"),
              ("退出播放", "退出"), ("不听了", "退出")]:
    PLAYBACK.append((q, "playback_control", _pc(op, "播放控制")))
for val, qs in [("10秒", ["快进10秒", "快进十秒"]), ("30秒", ["快进30秒"]),
                ("1分钟", ["快进1分钟"]), ("5分钟", ["快进5分钟"])]:
    for q in qs:
        PLAYBACK.append((q, "playback_control", _pc("快进", "", val)))
for val, qs in [("10秒", ["快退10秒", "倒退10秒"]), ("30秒", ["快退30秒", "倒退30秒"]),
                ("1分钟", ["快退1分钟", "倒退一分钟"])]:
    for q in qs:
        PLAYBACK.append((q, "playback_control", _pc("快退", "", val)))
for val, qs in [("5:00", ["跳到5分钟", "跳到05:00"]), ("10:00", ["跳到10分钟", "直接跳到10:00"]),
                ("1:30", ["跳到1分半"])]:
    for q in qs:
        PLAYBACK.append((q, "playback_control", _pc("跳转", "", val)))
for spd in ["1.25", "1.5", "2", "0.5", "0.75"]:
    PLAYBACK.append((f"调成{spd}倍速", "playback_control", _pc("设置", "倍速", spd)))
    PLAYBACK.append((f"用{spd}倍速放", "playback_control", _pc("设置", "倍速", spd)))
for q, val in [("单曲循环", "单曲播放"), ("列表循环", "循环播放"), ("随机播放", "随机播放"),
               ("顺序播放", "顺序播放")]:
    PLAYBACK.append((q, "playback_control", _pc(val, "播放列表")))
PLAYBACK.append(("再听一遍", "playback_control", _pc("重播", "", "")))

# ---------------------------------------------------------------------------
# screensaver_control：屏保 / 壁纸 / 壁画 / 屏保画册
# ---------------------------------------------------------------------------
SSAVER: list[tuple[str, str, dict]] = []
for q, op, obj in [("打开屏保", "打开", "屏保"), ("关了屏保吧", "关闭", "屏保"),
                   ("不要显示屏保", "关闭", "屏保"), ("打开壁纸", "打开", "壁纸"),
                   ("打开壁画", "打开", "壁画"), ("关闭壁画", "关闭", "壁画"),
                   ("打开屏保画册", "打开", "屏保画册"), ("启动屏保画册", "打开", "屏保画册")]:
    SSAVER.append((q, "screensaver_control", _screen(op, obj)))
# 注：「把第N个设成屏保」的「第N个」指的是当前列表里的第 N 项，脱离列表语境（并进多意图）
#     会失去所指，故不收入库。改用不依赖列表的说法（value 留空，同线上"设置屏保"标注）。
for obj in ["屏保", "壁纸", "壁画"]:
    SSAVER.append((f"换成{obj}", "screensaver_control", _screen("设置", obj)))
    SSAVER.append((f"把{obj}换了", "screensaver_control", _screen("设置", obj)))

# ---------------------------------------------------------------------------
# power_control / timer_control
# ---------------------------------------------------------------------------
POWER: list[tuple[str, str, dict]] = [
    ("关机", "power_control", _screen("打开", "关机")),
    ("把电视关了", "power_control", _screen("打开", "关机")),
    ("关闭电视", "power_control", _screen("打开", "关机")),
    ("开机", "power_control", _screen("打开", "开机")),
    ("把电视打开", "power_control", _screen("打开", "开机")),
    ("重启电视", "power_control", _screen("打开", "重启")),
]
TIMER: list[tuple[str, str, dict]] = []
for dt, qs in [("30分钟", ["30分钟后关机", "半小时后关机"]),
               ("1小时", ["1小时后关机", "一个小时后关机"]),
               ("45分钟", ["45分钟后关机"]),
               ("22:00", ["晚上10点关机", "22点关机"]),
               ("23:00", ["晚上11点关机"])]:
    for q in qs:
        TIMER.append((q, "timer_control",
                      {"operation": "打开", "object": "关机", "device": "", "location": "",
                       "date_time": dt}))
TIMER.append(("还有多久关机", "timer_control",
              {"operation": "查询", "object": "剩余关机时长", "value": "", "device": "", "location": ""}))

# ---------------------------------------------------------------------------
# screen_layout / mode_control / source_switch / display_control
# ---------------------------------------------------------------------------
LAYOUT: list[tuple[str, str, dict]] = [
    ("全屏显示", "screen_layout", _screen("打开", "全屏")),
    ("打开全屏", "screen_layout", _screen("打开", "全屏")),
    ("退出全屏", "screen_layout", _screen("关闭", "全屏")),
    ("打开分屏", "screen_layout", _screen("打开", "分屏")),
    ("关闭分屏", "screen_layout", _screen("关闭", "分屏")),
    ("画面缩小一点", "screen_layout", _screen("缩小", "画面")),
]
MODE: list[tuple[str, str, dict]] = []
# 取值严格对齐 catalog.full.json 对 mode_control.value 的三组枚举，不自造档位名：
#   图像模式：标准/鲜艳/儿童/体育/影院/游戏
#   声音模式：AI音质/标准/影院/音乐/体育/新闻/儿童/长辈/游戏
#   音效模式：杜比/流行/舞曲/蓝调/古典/爵士/慢歌/电子乐/摇滚/乡村/自定义
for obj, base_vals in [("图像模式", ["标准", "鲜艳", "儿童", "体育", "影院", "游戏"]),
                       ("声音模式", ["AI音质", "标准", "影院", "音乐", "体育", "新闻",
                                   "儿童", "长辈", "游戏"]),
                       ("音效模式", ["杜比", "流行", "舞曲", "蓝调", "古典", "爵士",
                                   "慢歌", "电子乐", "摇滚", "乡村", "自定义"])]:
    suffix = "音效" if obj == "音效模式" else "模式"
    short = "画面" if obj == "图像模式" else obj
    for b in base_vals:
        # AI音质 是唯一的例外：catalog 值就写作 "AI音质"，后面不加「模式」。
        v = b if b == "AI音质" else b + suffix
        # 「图像模式」在口语里常被说成「画面」。值后缀也常有省略（"图像模式调到影院"）。
        for h in [obj, short]:
            MODE.append((f"{h}切换到{v}", "mode_control", _screen("设置", obj, v)))
            MODE.append((f"{h}调成{v}", "mode_control", _screen("设置", obj, v)))
        MODE.append((f"{obj}换成{v}", "mode_control", _screen("设置", obj, v)))
        MODE.append((f"{obj}切成{v}", "mode_control", _screen("设置", obj, v)))
        MODE.append((f"{obj}给我调成{v}", "mode_control", _screen("设置", obj, v)))
        MODE.append((f"把{obj}调成{v}", "mode_control", _screen("设置", obj, v)))
        MODE.append((f"把{obj}换成{v}", "mode_control", _screen("设置", obj, v)))
        # 省略值后缀（"图像模式调到影院"）：口语里极常见。
        # 注意 AI音质 本身不带「模式」后缀（catalog 值就是 "AI音质"），不能写成 "AI音质模式"。
        if b != "AI音质":
            MODE.append((f"{obj}调到{b}", "mode_control", _screen("设置", obj, v)))
            MODE.append((f"{short}调到{b}", "mode_control", _screen("设置", obj, v)))
# 开关型模式（catalog 的 operation=打开、value 空）：真人说「打开/启动/进入/切到X模式」。
# 这些在 all_eval 里只有「打开/启动/进入」三种模板，句面重复度高，这里补足口语变体。
# 动词逐项指定：不是所有搭配都成立——「打开熄屏」（屏都关了还打开）不通，
# 缩写也常歧义（「打开音箱」是开音箱设备，不是音箱模式）。
# 每项：(catalog object, [可用简称…], [可搭配的动词…])
_TOGGLE_MODES = [
    ("商场模式", [], ["打开", "进入", "启动", "切到", "换成", "帮我打开", "我想开"]),
    ("护眼模式", ["护眼"], ["打开", "进入", "启动", "切到", "换成", "帮我打开", "我想开"]),
    ("混响模式", ["混响"], ["打开", "进入", "启动", "切到", "换成", "帮我打开", "我想开"]),
    ("会议模式", ["会议"], ["打开", "进入", "启动", "切到", "换成", "帮我打开", "我想开"]),
    ("观赛模式", ["观赛"], ["打开", "进入", "启动", "切到", "换成", "帮我打开", "我想开"]),
    ("聊球模式", ["聊球"], ["打开", "进入", "启动", "切到", "换成", "帮我打开", "我想开"]),
    ("省电模式", ["省电"], ["打开", "进入", "启动", "切到", "换成", "帮我打开", "我想开"]),
    ("熄屏模式", ["熄屏"], ["进入", "启动", "切到", "换成"]),
    ("音箱模式开关", [], ["打开", "进入", "启动", "切到", "换成", "帮我打开"]),
    ("音箱模式", [], ["打开", "进入", "启动", "切到", "换成", "帮我打开"]),
    ("电视模式", [], ["打开", "进入", "启动", "切到", "换成", "帮我打开"]),
    ("月光模式", [], ["打开", "进入", "启动", "切到", "换成", "帮我打开"]),
    ("ai环境模式", ["AI环境模式"], ["打开", "进入", "启动", "切到", "换成", "帮我打开", "我想开"]),
]
for obj, alias, verbs in _TOGGLE_MODES:
    # catalog 里写作 "ai环境模式"（小写 ai），但口语书写应是 "AI环境模式"。
    full = obj.replace("ai环境模式", "AI环境模式")
    f = _screen("打开", obj)
    short = [a.replace("ai环境模式", "AI环境模式") for a in alias]
    for v in verbs:
        MODE.append((f"{v}{full}", "mode_control", f))
        # 简称只在「打开/进入/启动/帮我打开」这类明确开启语义里用，避免歧义。
        if v in ("打开", "进入", "启动", "帮我打开", "切到", "换成", "我想开"):
            for nm in short:
                MODE.append((f"{v}{nm}", "mode_control", f))
    MODE.append((f"{full}打开一下", "mode_control", f))
    MODE.append((f"把{full}打开", "mode_control", f))
SOURCE: list[tuple[str, str, dict]] = []
for v, lab in [("hdmi1", "HDMI1"), ("hdmi2", "HDMI2"), ("hdmi3", "HDMI3"), ("usb", "USB")]:
    SOURCE.append((f"切换到{lab}", "source_switch", _screen("设置", "信号源", v)))
    SOURCE.append((f"信号源切到{lab}", "source_switch", _screen("设置", "信号源", v)))
DISPLAY: list[tuple[str, str, dict]] = [
    ("打开运动补偿", "display_control", _screen("打开", "运动补偿")),
    ("关闭运动补偿", "display_control", _screen("关闭", "运动补偿")),
]

# 画质/音质问题 → solve_picture_sound_problem_control
PICPROB: list[tuple[str, str, dict]] = [
    ("画面颜色太鲜艳了帮我调一下", "solve_picture_sound_problem_control", {"intent": "colors_oversaturated"}),
    ("画面颜色有点淡帮我调一下", "solve_picture_sound_problem_control", {"intent": "colors_washed_out"}),
    ("画面整体偏蓝帮我调一下", "solve_picture_sound_problem_control", {"intent": "cold_color_cast"}),
    ("画面整体发红帮我调整一下", "solve_picture_sound_problem_control", {"intent": "red_color_cast"}),
    ("画面整体发黄帮我调整一下", "solve_picture_sound_problem_control", {"intent": "warm_color_cast"}),
    ("屏幕暗处什么都看不见帮我设置一下", "solve_picture_sound_problem_control", {"intent": "dark_areas_crushed"}),
    ("屏幕亮处过曝了帮我修改一下", "solve_picture_sound_problem_control", {"intent": "highlights_overexposed"}),
    ("屏幕有点糊帮我修改一下", "solve_picture_sound_problem_control", {"intent": "lack_of_clarity"}),
    ("屏幕有噪点帮我修改一下", "solve_picture_sound_problem_control", {"intent": "image_noise_obvious"}),
    ("运动画面有拖尾帮我调节一下", "solve_picture_sound_problem_control", {"intent": "motion_not_sharp"}),
    ("画面忽明忽暗的帮我调节一下", "solve_picture_sound_problem_control", {"intent": "flickering_brightness"}),
    ("人声听不清", "solve_picture_sound_problem_control", {"intent": "voice_not_clear"}),
    ("声音没有立体感", "solve_picture_sound_problem_control", {"intent": "lack_stereo_surround"}),
    ("太刺眼帮我调节一下", "solve_picture_sound_problem_control", {"intent": "poor_eye_comfort"}),
]

# ---------------------------------------------------------------------------
# 「短促口语」档：对标 bench_serial_train_10k.csv 的真实语感。
#
# 该参考集（用户指定为格式与语感基准）的设备子句 58% 不超过 5 个字，
# 高频形态是「全屏播放/关电视/大点声/音量减/返回主页」这类秃动词短语，
# 而不是上面按 catalog 枚举拼出的 13 字长句（"音效模式给我调成自定义音效"）。
# 上面那套长句读起来像说明书，正是整批数据「生硬」的主因。
# 这一档补短说法，金标与长句完全一致（同 tool、同 params），只是把话说短。
# ---------------------------------------------------------------------------
TERSE: list[tuple[str, str, dict]] = []
# —— numeric_adjust 的短促档 ——
# 参考集里这一类几乎全是 3-5 字的秃短语（"大点声""声音大点""音量减""亮度高一点"），
# 我原来只有「把音量弄到30」这种 8 字说法，是设备子句偏长的主因。
for q in ["大点声", "大声点", "声音大点", "音量大点", "声音大些", "音量大些",
          "声音大一点", "音量大一点", "声音调大点", "加音量", "提高音量"]:
    TERSE.append((q, "numeric_adjust", _num("提高", "音量")))
# 注意：「音量减」「声音减」参考集里标的是 operation=提高（121/121 行一致），跟字面义
# 正好相反——那是参考标注的一个系统性毛病（「减」被当成了「加减档位」的意思）。我不跟这个
# 惯例，宁可不用这两个词，只用不会歧义的「小点声/声音小点/降低音量」。
for q in ["小点声", "小声点", "声音小点", "音量小点", "声音小些", "音量小些",
          "声音小一点", "音量小一点", "声音调小点", "降低音量"]:
    TERSE.append((q, "numeric_adjust", _num("降低", "音量")))
for q in ["亮度高一点", "亮度高点", "屏幕亮一点", "亮一点", "调亮一点", "提高亮度"]:
    TERSE.append((q, "numeric_adjust", _num("提高", "亮度")))
# 「降低亮度」「亮度降低」在参考集里操作值自相矛盾（降低/提高都有），不用带「降」的短式。
for q in ["亮度低一点", "亮度低点", "屏幕暗一点", "暗一点", "调暗一点"]:
    TERSE.append((q, "numeric_adjust", _num("降低", "亮度")))
for q in ["对比度高一点", "对比度强一点", "对比度高点"]:
    TERSE.append((q, "numeric_adjust", _num("提高", "对比度")))
for q in ["对比度低一点", "对比度弱一点", "对比度淡一点"]:
    TERSE.append((q, "numeric_adjust", _num("降低", "对比度")))
for q in ["清晰度高一点", "清晰度高些", "画面锐一点"]:
    TERSE.append((q, "numeric_adjust", _num("提高", "清晰度")))
for q in ["颜色深一点", "色彩浓一点", "色度高点"]:
    TERSE.append((q, "numeric_adjust", _num("提高", "色度")))
for q in ["颜色淡一点", "色彩淡一点", "色度低点"]:
    TERSE.append((q, "numeric_adjust", _num("降低", "色度")))
for q in ["静音", "关上声音", "别出声", "把声音关掉"]:
    TERSE.append((q, "numeric_adjust", _num("打开", "静音")))
for q in ["取消静音", "打开声音", "恢复声音", "解除静音", "关闭静音"]:
    TERSE.append((q, "numeric_adjust", _num("关闭", "静音")))
# 中文数字的短设置句（参考集"声音调到二十""音量调到二十"高频）
_CN_NUM = {"10": "十", "20": "二十", "30": "三十", "40": "四十",
           "50": "五十", "60": "六十", "70": "七十", "80": "八十"}
for obj in ["音量", "声音", "亮度", "对比度"]:
    for n, cn in _CN_NUM.items():
        std = "音量" if obj == "声音" else obj
        TERSE.append((f"{obj}调到{cn}", "numeric_adjust", _num("设置", std, n)))
        # 「X到N」这种省略「调」的极简式，参考集里只用在音量/声音上
        # （音量到5、声音到五）；「亮度到三十」「对比度到五十」参考集 0 次，
        # 是我自己类推出来的说法，不加。
        if obj in ("音量", "声音"):
            TERSE.append((f"{obj}到{cn}", "numeric_adjust", _num("设置", std, n)))

# —— numeric_adjust 的相对/程度档（参考集里占比最大的一档）——
# 参考集 numeric_adjust 的构成实测：程度/相对 43.9% / 显式设值 25.4% / 裸数值 25.1% /
# 其他 5.6%。我的库正好倒过来——显式设值 78%——于是「音量调到50」被放大，真人最常说的
# 「音量大点」「亮度调高点」「小点声」严重不足。这一块按参考集的对象×方向词实测清单补出，
# 金标为 (提高/降低, 对象)。方向词按对象分族：不同对象的说法不通用，
# 「色度亮一点」「亮度浓一点」都是病句。
_DEG_FAM = [
    # (对象词候选, 金标 object, 升说法, 降说法)
    (("音量", "声音"), "音量",
     ["大点", "大些", "大一点", "大一些", "调大点", "调大些", "调大一点", "加大",
      "加大点", "增大", "提高", "调高", "调高点", "开大点", "再大点"],
     ["小点", "小些", "小一点", "小一些", "调小点", "调小些", "调小一点", "减小",
      "降低", "调低", "调低点", "关小点", "再小点"]),
    (("亮度", "屏幕"), "亮度",
     ["亮一点", "亮些", "亮一点儿", "调亮", "调亮些", "调高", "调高点", "调高一点",
      "高一点", "提高", "再亮点"],
     ["暗一点", "暗些", "调暗", "调暗些", "调低", "调低点", "调低一点", "低一点",
      "降低", "再暗点"]),
    (("对比度",), "对比度",
     ["调高", "调高点", "调高一点", "高一点", "高点", "高一些", "强一点", "加强点"],
     ["调低", "调低点", "调低一点", "低一点", "低点", "低一些", "弱一点", "减弱点"]),
    (("清晰度",), "清晰度",
     ["调高", "调高点", "调高一点", "高一点", "高一些", "提高", "清晰点"],
     ["调低", "调低点", "调低一点", "低一点", "降低"]),
    (("色度", "色彩", "颜色"), "色度",
     ["调高", "调浓点", "浓一点", "浓些", "深一点", "深些", "鲜艳一点", "高一点"],
     ["调低", "调淡点", "淡一点", "淡些", "淡一点儿", "浅一点", "低点"]),
]
# 程度档的相对权重：短句（≤5 字）吃满权重，本档说法普遍比显式设值句更短，
# 若再给 1.0 会把整档推到 55%+；故主体说法给 0.55，话题后置/把字句/让字句更轻。
_DEG_W_MAIN, _DEG_W_BA, _DEG_W_RANG = 0.55, 0.14, 0.06
for _objs, _std_obj, _ups, _downs in _DEG_FAM:
    for _o in _objs:
        for _sfx, _op in [(s, "提高") for s in _ups] + [(s, "降低") for s in _downs]:
            TERSE.append((f"{_o}{_sfx}", "numeric_adjust", _num(_op, _std_obj)))
            SURFACE_W[f"{_o}{_sfx}"] = _DEG_W_MAIN
            if _sfx in ("调高点", "调低点", "调大点", "调小点", "调高一点", "调低一点"):
                SURFACE_W[f"把{_o}{_sfx}"] = _DEG_W_BA
                TERSE.append((f"把{_o}{_sfx}", "numeric_adjust", _num(_op, _std_obj)))
            if _sfx in ("大点", "小点", "大些", "小些", "高点", "低点"):
                SURFACE_W[f"让{_o}{_sfx}"] = _DEG_W_RANG
                TERSE.append((f"让{_o}{_sfx}", "numeric_adjust", _num(_op, _std_obj)))
# 话题后置（"大点音量""小点声音""大些声音"）——参考集实测这一族【只出现在音量】上：
# 大点声音 50 / 小点音量 47 / 小点声音 42 / 大点音量 39 / 大一点音量 13 / 大些声音 11 /
# 大些音量 9 / 小些声音 5 / 小一点声音 3 / 小些音量 3（合计 222），
# 亮度类总共只有 8 次（大点亮度 3、大一点亮度 2、小点亮度 1、大些亮度 1、大些屏幕 1），
# 色彩/色度/对比度/清晰度 0 次。我上一版对所有对象一视同仁地生成，做出「浓一点色彩」
# 「低点色彩」「高一点色度」这类参考集里根本不存在的说法，是自造。这里只留音量族，
# 亮度族按实测频次给极低权重。
_DEG_FLIP_VOL = (("音量", "音量"), ("声音", "音量"), ("声", "音量"))
for _o, _std in _DEG_FLIP_VOL:
    for _sfx, _op in [("大点", "提高"), ("大些", "提高"), ("大一点", "提高"), ("大一些", "提高"),
                      ("小点", "降低"), ("小些", "降低"), ("小一点", "降低"), ("小一些", "降低")]:
        _q = f"{_sfx}{_o}"
        TERSE.append((_q, "numeric_adjust", _num(_op, _std)))
        SURFACE_W[_q] = 0.30
# 亮度族后置：参考集只有零星几次，权重压到近乎噪声
for _q, _op in [("大点亮度", "提高"), ("大一点亮度", "提高"), ("小点亮度", "降低"), ("大些亮度", "提高")]:
    TERSE.append((_q, "numeric_adjust", _num(_op, "亮度")))
    SURFACE_W[_q] = 0.03
# 「大点声/小点声」这类省略对象词的音量档：参考集里 大点声 90+、小点声 24+，
# 是这一族里最高频的说法之一，权重不能低。
for _q, _op in [("大点声", "提高"), ("大声点", "提高"), ("声音大点声", "提高"),
                ("大点声音", "提高"), ("小点声", "降低"), ("小声点", "降低")]:
    TERSE.append((_q, "numeric_adjust", _num(_op, "音量")))
    SURFACE_W[_q] = 0.85
# —— screen_layout：全屏/大屏/小屏/分屏 ——
# 「看全屏」参考集 0 次（他们只用 全屏播放/播放全屏/大屏播放），删掉。
for q in ["全屏", "全屏播放", "大屏播放", "播放全屏", "打开全屏", "打开全屏播放",
          "开启全屏", "全屏吧", "整成全屏"]:
    TERSE.append((q, "screen_layout", _screen("打开", "全屏")))
for q in ["小屏播放", "小屏", "换成小屏"]:
    TERSE.append((q, "screen_layout", _screen("打开", "小屏")))
for q in ["分屏观看", "分屏", "开分屏", "打开分屏"]:
    TERSE.append((q, "screen_layout", _screen("打开", "分屏")))
for q in ["屏幕加大", "屏幕调大", "放大屏幕", "屏幕放大", "画面放大"]:
    TERSE.append((q, "screen_layout", _screen("放大", "画面")))
# —— power_control：关机/开机/重启（"关闭电视"= 关机，同线上标注） ——
for q in ["关闭电视", "关掉电视", "关电视", "关机", "把电视关了", "电视关掉",
          "直接关机", "关闭电源", "关掉电源", "把电视关掉", "关一下电视", "电视关了"]:
    TERSE.append((q, "power_control", _screen("打开", "关机")))
for q in ["开机", "电视开机", "打开电视", "把电视打开"]:
    TERSE.append((q, "power_control", _screen("打开", "开机")))
for q in ["重启", "重启电视", "电视重启"]:
    TERSE.append((q, "power_control", _screen("打开", "重启")))
# —— common_control：主页/返回（参考集里 954 条，我原来一条都没有） ——
for q in ["返回主页", "打开主页", "返回主页面", "回到主页", "返回首页", "打开首页",
          "回到首页", "回主页", "退下", "回到主页面"]:
    TERSE.append((q, "common_control", _screen("打开", "主页")))
# —— playback_control：秃动词 ——
for q in ["暂停", "暂停播放", "停一下", "先暂停"]:
    TERSE.append((q, "playback_control", _pc("暂停", "播放控制")))
for q in ["继续播放", "继续播", "接着播放", "接着播", "恢复播放", "继续"]:
    TERSE.append((q, "playback_control", _pc("播放", "播放控制")))
for q in ["停止播放", "停止", "别放了", "不听了"]:
    TERSE.append((q, "playback_control", _pc("停止", "播放控制")))
for q in ["下一集", "播放下一集", "切到下一集", "下一曲", "播放下一个视频", "下一个"]:
    TERSE.append((q, "playback_control", _pc("下", "播放控制", "集") if "集" in q
                  else _pc("下", "", "首")))
for q in ["上一集", "上一首", "上一曲", "回到上一集", "上一个"]:
    TERSE.append((q, "playback_control", _pc("上", "播放控制", "集") if "集" in q
                  else _pc("上", "", "首")))
for q in ["下一首", "下一曲", "换一首", "切歌", "换歌"]:
    TERSE.append((q, "playback_control", _pc("切换", "", "首")))
for q in ["快进到三分钟", "跳到三分钟", "快进到3分钟"]:
    TERSE.append((q, "playback_control", _pc("快进", "", "3分钟")))
for q in ["往后退", "倒退", "退回去"]:
    TERSE.append((q, "playback_control", _pc("快退", "", "")))
# —— screensaver_control：屏幕/屏保的开关 ——
for q in ["关闭屏幕", "关上屏幕", "关屏幕", "息屏", "熄屏"]:
    TERSE.append((q, "screensaver_control", _screen("关闭", "屏保")))
for q in ["关闭屏保", "把屏保关掉", "退出屏保", "关屏保", "别开屏保"]:
    TERSE.append((q, "screensaver_control", _screen("关闭", "屏保")))
# —— source_switch：短切换 ——
for lab, v in [("hdmi1", "hdmi1"), ("HDMI1", "hdmi1"), ("hdmi2", "hdmi2"),
               ("HDMI2", "hdmi2"), ("hdmi3", "hdmi3"), ("HDMI3", "hdmi3"),
               ("USB", "usb"), ("usb", "usb"), ("switch", "switch"), ("typec", "typec")]:
    # 「播放+X」只用于 hdmi1 和「电视」（参考集：播放hdmi1 x26、播放电视 x22，
    # 其余「播放USB/播放switch/播放typec」0 次）；其余信号源一律用「切到/切换到/选择…信号源」。
    if lab.lower() == "hdmi1":
        TERSE.append((f"播放{lab}", "source_switch", _screen("设置", "信号源", v)))
    TERSE.append((f"切到{lab}", "source_switch", _screen("设置", "信号源", v)))
    TERSE.append((f"切换到{lab}", "source_switch", _screen("设置", "信号源", v)))
    TERSE.append((f"选择{lab.upper()}信号源", "source_switch",
                  _screen("设置", "信号源", v)))
for q in ["信号源设置", "信号源切换", "输入源切换", "信号源", "切信号源"]:
    TERSE.append((q, "source_switch", _screen("设置", "信号源", "")))
for lab in ["模拟电视", "数字电视", "地面数字电视", "直播电视", "机顶盒"]:
    for pre in ["打开", "进入", "启动"]:
        TERSE.append((f"{pre}{lab}", "source_switch", _screen("打开", lab, "")))

# —— 参考集出现而我原先完全没有的设备工具 ——
# network_control：无线/有线网络、测速、信道、设置
NETWORK: list[tuple[str, str, dict]] = []
for q in ["网线连接", "有线网络连接", "有线网络设置", "有线网络配置", "打开以太网"]:
    NETWORK.append((q, "network_control", _screen("打开", "有线网络", "")))
for q in ["wifi功能", "打开无线网络", "无线网络设置", "无线网设置", "wifi连接",
          "WiFi设置", "打开wifi"]:
    NETWORK.append((q, "network_control", _screen("打开", "无线网络", "")))
for q, obj in [("打开网络设置", "网络设置"), ("启动网络设置", "网络设置"),
               ("进入网络设置", "网络设置"), ("打开网络连接测试", "网络连接测试"),
               ("启动网络连接测试", "网络连接测试"), ("进入wifi强度测试", "wifi强度测试"),
               ("打开网络测速", "网络测速"), ("进入WIFI信道", "wifi信道"),
               ("打开无线热点", "无线热点")]:
    NETWORK.append((q, "network_control", _screen("打开", obj, "")))
# smart_camera：摄像头 / 雷达 / AI 视觉
CAMERA: list[tuple[str, str, dict]] = []
for obj in ["摄像头", "摄像头设置", "摄像头控制开关", "一键入幕", "自动入幕",
            "照镜子", "摄像头锁", "智能设置", "坐姿检测", "手势控制",
            "全时云守护", "手动调整摄像头", "自动调整摄像头"]:
    for pre in ["打开", "启动", "进入"]:
        CAMERA.append((f"{pre}{obj}", "smart_camera", _screen("打开", obj, "")))
# audio_control：音质音效项（注意与 mode_control 分工：这里是具体声学功能开关）
AUDIO: list[tuple[str, str, dict]] = []
for obj in ["中置声场", "人声中置声场", "立体音效", "杜比全景声", "空间声场自适应",
            "声音设置", "声音主题定制", "切换声音", "音乐氛围墙", "全景声音量控制",
            "多声道音响", "影院音效演示", "赛场音效", "按键音", "全场景麦克设置"]:
    for pre in ["打开", "启动", "进入"]:
        AUDIO.append((f"{pre}{obj}", "audio_control", _screen("打开", obj, "")))
# demo_control：效果演示
DEMO: list[tuple[str, str, dict]] = []
for obj in ["demo演示", "图卡demo", "产品展示", "屏幕展示", "自动景深优化演示",
            "亮暗精控演示", "智能暂停演示", "ULED超画质演示", "2K转4K演示",
            "超高清演示", "杜比全景声演示", "高清晰度演示", "高对比度演示",
            "晶彩色控演示", "超解像演示", "全维ai画质演示", "极速屏演示",
            "智能节能演示", "家庭私有云演示"]:
    for pre in ["打开", "启动", "进入"]:
        DEMO.append((f"{pre}{obj}", "demo_control", _screen("打开", obj, "")))
# screen_lift_rotation：旋转/升降
LIFT: list[tuple[str, str, dict]] = []
for obj in ["电视旋转", "自动旋转", "竖屏短视频", "支架"]:
    for pre in ["打开", "启动", "进入"]:
        LIFT.append((f"{pre}{obj}", "screen_lift_rotation", _screen("打开", obj, "")))

_BASE: list[tuple[str, str, dict]] = (
    NUMERIC + PLAYBACK + SSAVER + POWER + TIMER + LAYOUT + MODE + SOURCE
    + DISPLAY + PICPROB + TERSE + NETWORK + CAMERA + AUDIO + DEMO + LIFT
)


# ---------------------------------------------------------------------------
# 同义口语变体：只改说法，不动金标 (tool, params)。
# 基础库平均每条被复用 ~30 次，表面形式单一会显得像模板；这里按规则做「保义改写」，
# 生成的说法与原句指向完全相同的操作，因此 tool/params 原样沿用。
# ---------------------------------------------------------------------------
# 数值设置的「把字句 / 话题前置」形态：真人常说「把音量调到50」「音量给我调到50」
_NUM_SET_RE = re.compile(r"^(音量|声音|亮度|对比度)调到(\d+)$")
_OBJ_STD = {"声音": "音量"}


# 光杆短动词（"静音""暂停""下一首""单曲循环""全屏显示"）单独说是自然的，
# 但接在逗号后面串起来会显得生硬（"播放X，接着静音"）。补一个软化的口语形式，
# 含义与金标完全一致，只是把话说完。
_TERSE_SOFTEN = {
    # 参考集里表达静音的说法（含取消静音）：关闭静音21/取消静音19/打开静音9/静音开关6/
    # 静音管理5/开启静音5/静音模式4/退出静音4。「把声音静了」「静一下音」参考集 0 次，
    # 是我自造的形式，删掉。
    "静音": ["静音吧", "打开静音", "开启静音", "静音模式"],
    "暂停": ["暂停一下", "给我暂停"],
    "关机": ["关机吧", "把电视关了"],
    "开机": ["开机吧", "把电视打开"],
    "下一首": ["换下一首", "放下一首"],
    "上一首": ["回到上一首", "听上一首"],
    "下一集": ["放下一集", "看下一集"],
    "上一集": ["回到上一集", "看上一集"],
    "下一部": ["放下一部", "看下一部"],
    "换一部": ["换一部吧", "给我换一部"],
    "换一首": ["换一首吧", "给我换一首"],
    "换一个": ["换一个吧", "给我换一个"],
    "不听了": ["不听了", "别放了"],
    "取消静音": ["取消静音吧", "把静音取消"],
    "恢复声音": ["把声音恢复", "恢复一下声音"],
    "手动循环": [],
    "单曲循环": ["单曲循环吧", "给我单曲循环"],
    "列表循环": ["列表循环吧", "给我列表循环"],
    "随机播放": ["随机播放吧", "给我随机播"],
    "顺序播放": ["顺序播放吧", "给我顺序播"],
    "再听一遍": ["再听一遍吧", "给我再放一遍"],
    # 「把画面全屏」「把全屏退了」是我自己造的，参考集 0 次，删掉。
    "全屏显示": ["全屏显示一下吧", "全屏显示一下"],
    "打开全屏": ["打开全屏吧", "把全屏打开"],
    "退出全屏": ["退出全屏吧", "退出一下全屏"],
    "打开分屏": ["打开分屏吧", "把分屏打开"],
    "关闭分屏": ["关闭分屏吧", "把分屏关了"],
    "打开屏保": ["把屏保打开", "屏保开一下"],
    "打开壁纸": ["把壁纸打开", "壁纸开一下"],
    "打开壁画": ["把壁画打开", "壁画开一下"],
    "关闭壁画": ["把壁画关了", "关掉壁画"],
    "换成屏保": ["换成屏保吧", "给我换成屏保"],
    "换成壁纸": ["换成壁纸吧", "给我换成壁纸"],
    "换成壁画": ["换成壁画吧", "给我换成壁画"],
    "关闭电视": ["把电视关了", "关一下电视"],
    "重启电视": ["把电视重启", "重启一下电视"],
    "快进十秒": ["快进十秒吧", "往前快进十秒"],
    "调成2倍速": ["调成2倍速吧", "用2倍速放"],
    "用2倍速放": ["用2倍速放吧", "给我用2倍速放"],
    "切到USB": ["切到USB吧", "把信号源切到USB"],
}

# 程度副词：诊断显示原语料 0% 出现（稍微/一点点/尽量），但口语里极常见。
# 只加在「提高/降低」这类相对幅值调整上，不改变金标 value（本来就是空串）。
# 注意「稍微/尽量」不能与自身已强调程度义的「太…了」连用（"声音稍微太小了"不通），
# 但与「一点/些」可以共存（"声音稍微大一点"很自然），故只排除「太」。
def _degree_variants(q: str, tool: str, p: dict) -> list[str]:
    """给相对调整句加程度副词，如「声音大一点」→「声音稍微大一点」。语义不变。"""
    if tool != "numeric_adjust" or p.get("operation") not in ("提高", "降低"):
        return []
    if "太" in q or any(d in q for d in ("稍微", "尽量", "一点点")):
        return []
    m = re.match(r"^把(音量|亮度|对比度|清晰度|色度)(调[高低强弱]+|降|提)(.+)$", q)
    if m:
        return [f"把{m.group(1)}稍微{m.group(2)}{m.group(3)}"]
    m = re.match(r"^(声音|屏幕|亮度|色彩|颜色|对比度|画面|清晰度|色度)(调[高低强弱]+|降|提)(.+)$", q)
    if m:
        return [f"{m.group(1)}稍微{m.group(2)}{m.group(3)}"]
    return []


def _numeral_variants(q: str, tool: str, p: dict) -> list[str]:
    """把「音量调到50」写成「音量调到五十」——真人语音输入常出中文数字，且金标 value 不变。"""
    if tool != "numeric_adjust" or p.get("operation") != "设置":
        return []
    val = p.get("value")
    if not val or not str(val).isdigit():
        return []
    cn = {"20": "二十", "30": "三十", "50": "五十", "70": "七十"}.get(str(val))
    if not cn or cn in q:
        return []
    return [q.replace(str(val), cn)]


def _numeric_set_variants(obj: str, n: str) -> list[str]:
    """数值设置的多种口语形态。objective（音量/亮度/对比度）× 说法 是组合式的，
    与其手列，不如按句型生成。全部指向同一 (设置, obj, n)，金标不变。

    参考集里「设值类动词」只有五个，把它们在带数字的 numeric_adjust 子句里逐一点数
    （n=2234，含 719 条无动词的「音量到三十」式）：调到 40.2% / 【无动词】32.2% /
    「到」17.5% / 调 4.0% / 设置为 2.3% / 调至 2.0% / 调为 1.9%。
    换算到「有动词的设值句」内部（n=1515）：调到 65.6% / 调 6.5% / 设置为 3.7% /
    调至 3.3% / 调为 3.1%。
    关键实测结论：设置成 / 设为 / 设成 / 调成 / 单独「设置」在参考集里全部为 0，
    定在 / 弄到 也为 0。上一版这几个词合计被我放大到 ~20%，是本档最大的失真源；
    这一版全部删掉，只留参考集真实存在的五个动词。权重写进 SURFACE_W（去重不影响它）。
    """
    b = f"把{obj}"
    # (形态, 相对权重)：调到占绝对多数；「到」式在下面单独走裸值档，不在此处重复。
    forms = [
        (f"{obj}调到{n}", 2.2), (f"{b}调到{n}", 2.0),
        (f"{obj}调到{n}吧", 0.16), (f"{b}调到{n}吧", 0.14),
        (f"{obj}调到{n}就行", 0.08), (f"{obj}给我调到{n}", 0.10),
        (f"{obj}帮我调到{n}", 0.08), (f"帮我把{obj}调到{n}", 0.09),
        (f"麻烦把{obj}调到{n}", 0.05), (f"给我把{obj}调到{n}", 0.05),
        (f"我想把{obj}调到{n}", 0.05), (f"我要把{obj}调到{n}", 0.05),
        (f"{obj}调为{n}", 0.20), (f"把{obj}调为{n}", 0.20),
        (f"把{obj}调至{n}", 0.20), (f"{obj}调至{n}", 0.20),
        (f"{obj}调{n}", 0.42), (f"把{obj}调{n}", 0.40),
        (f"{obj}设置为{n}", 0.24), (f"把{obj}设置为{n}", 0.24),
    ]
    out = []
    for q, w in forms:
        SURFACE_W[q] = w
        if w > 0:
            out.append(q)
    return out


def _variants(q: str, tool: str, p: dict) -> list[str]:
    out: list[str] = []
    if tool == "numeric_adjust":
        m = _NUM_SET_RE.match(q)
        if m:
            obj, n = _OBJ_STD.get(m.group(1), m.group(1)), m.group(2)
            out += _numeric_set_variants(obj, n)
    elif tool == "mode_control":
        # 「图像模式调成鲜艳模式」→「画面切成鲜艳模式」「图像模式换成鲜艳模式」
        # 注意 调成/调到 二选一，不能写成 调到? 否则 value 会把「成」吃进去，生成「调到成音乐模式」。
        m = re.match(r"^(图像模式|声音模式|音效模式)(?:切换到|调成|调到)(.+)$", q)
        if m:
            obj, val = m.group(1), m.group(2)
            short = "画面" if obj == "图像模式" else obj
            out += [f"{short}切成{val}", f"{obj}换成{val}", f"{obj}给我调成{val}"]
    elif tool == "source_switch":
        if q.startswith("切换到"):
            out += [f"信号源换成{q[3:]}", f"切到{q[3:]}"]
        elif q.startswith("信号源切到"):
            out += [f"切到{q[5:]}", f"信号源换成{q[5:]}"]
    elif tool == "screensaver_control":
        if q == "打开屏保":
            out += ["把屏保打开", "屏保开一下"]
        elif q == "不要显示屏保":
            out += ["屏保别显示了", "别开屏保"]
    elif tool == "numeric_adjust" and q.endswith("吧"):
        out.append(q.rstrip("吧"))
    # 光杆短动词补软化形式
    out += _TERSE_SOFTEN.get(q, [])
    # 程度副词变体已移除：它把「声音大一点」改写成「声音稍微大一点」，
    # 动机是"稍微很口语"，但参考集 10000 行里「稍微/尽量/一点点」出现【0 次】。
    # 我给了 0.2%（10 行），是整批里少数几个仍能一眼看出的自造说法。
    # 参考集表达程度只用后缀一点/点/些/一些（与我的分布一致：41/36/20/3% vs 44/35/18/2%），
    # 不再叠前置副词。_degree_variants 保留函数体但不再调用（收割项里若有真实用例会原样进池）。
    out += _numeral_variants(q, tool, p)
    return [v for v in out if v != q]


# ---- 第二来源：直接从参考集收割的设备子句 ----
# 手写库只能覆盖我"想得到"的说法；参考集是 10000 行真实标注，用词量在若干工具上是
# 我的数倍（playback_control 232 vs 89、screensaver_control 197 vs 39、
# common_control 184 vs 10、display_control 60 vs 2）。这正是用户反复指出的
# "表达多样性不够"。收割项的金标参数直接取自参考集，不经我转录。
try:
    from ref_device_vocab import (harvest as _harvest_ref,
                                  harvest_numeric_counts as _harvest_num,
                                  harvest_relaxed as _harvest_relaxed,
                                  attested_device_clauses as _attested,
                                  attested_device_freq as _attested_freq,
                                  attested_device_nullrate as _attested_null,
                                  attested_device_leadfreq as _attested_lead)
    _REF_VOCAB = _harvest_ref()
    _REF_NUM = _harvest_num()
    # mode_control / screen_layout 原来被 HARVEST_TOOLS 排除（注释说"手写库已覆盖"），
    # 实测恰恰相反：这两个工具的产出未命中参考集的比例最高（89.5% / 50.7%），
    # 因为手写库是照 catalog 枚举拼的长句，变体机又在上面二次派生。
    _REF_RELAXED = _harvest_relaxed()          # [(子句, tool, params|None, 参考集频次)]
    _REF_VOCAB += [(q, t, p) for q, t, p, _ in _REF_RELAXED]
    _ATTESTED = _attested()
    _ATTESTED_FREQ = _attested_freq()
    _ATTESTED_NULL = _attested_null()
    _ATTESTED_LEAD = _attested_lead()          # {(子句, tool): (首位次数, 非首位次数)}
except Exception as _e:                                  # 参考集不在时退化为纯手写库
    print(f"[device_phrases] 参考集收割不可用（{_e}），仅用手写库")
    _REF_VOCAB = []
    _REF_NUM = {}
    _REF_RELAXED = []
    _ATTESTED = {}
    _ATTESTED_FREQ = {}
    _ATTESTED_NULL = {}
    _ATTESTED_LEAD = {}

# 汇总：手写库 + 保义变体 + 参考集收割（三源同一形去重，手写优先）。
# 变体可能从不同源句收敛到同一说法（"调成"/"切换到"两种源都会生成"…给我调成…"），去重。
# （SURFACE_W 已在文件开头声明，此处不再重复定义。）
DEVICE_PHRASES: list[tuple[str, str, dict]] = []
_seen_surface: set[str] = set()
for _q, _t, _p in list(_BASE) + list(_REF_VOCAB) + [
    (_v, _t, _p) for _q, _t, _p in _BASE for _v in _variants(_q, _t, _p)
]:
    if _q in _seen_surface:
        continue
    _seen_surface.add(_q)
    DEVICE_PHRASES.append((_q, _t, _p))

# 参考集 numeric_adjust 收割：按【真实频次】入池。手写库里的同形说法在此处被频次覆盖
# （手写库的自设权重换成参考集实测频次），参考集有而我手写库没有的说法补进来。
#
# 关键细节：生成器实际取的是 `_len_w(q) * SURFACE_W[q]`，而 _len_w 是为修正【手写库】
# 的长度偏置发明的（我的长句太多）。参考集频次本身已经包含真实的长度分布，再乘一次
# _len_w 等于双重修正——实测会把我这一档压到平均 4.61 字（参考 4.93），4 字句占 43%
# （参考 30%）。故这里把频次【除以】_len_w 存进去，使有效权重恰好等于参考集频次。
# 处理只作用于收割项；手写库的权重仍走 _len_w 修正。
def _len_w_phrase(s: str) -> float:
    n = len(s)
    return 1.0 if n <= 5 else (0.34 if n <= 7 else 0.13)


for _q, (_t, _p, _n) in _REF_NUM.items():
    if _q not in _seen_surface:
        _seen_surface.add(_q)
        DEVICE_PHRASES.append((_q, _t, _p))
    _eff = max(1.0, float(_n))                # 下限 1.0，避免只出现 1 次的说法被压没
    SURFACE_W[_q] = _eff / _len_w_phrase(_q)

# ---------------------------------------------------------------------------
# 准入闸门：凡不在参考集里的设备说法，一律不许进池。
# ---------------------------------------------------------------------------
# 这一条是整批数据「生硬」的最后一块结构性病因。上面每一批短语都各自有理由相信
# 自己是"口语的"（手写对照 catalog 枚举、变体机做保义改写、收割直接取自参考集），
# 但把它们合起来按工具逐条比对参考集，未命中率是这样的：
#   mode_control 89.5%、screen_layout 50.7%、source_switch 40.9%、audio_control 27.8%、
#   power_control 21.2%、playback_control 18.2%、screensaver_control 17.8%、
#   而 common_control 2.3%、display_control 1.1%、network_control 0%。
# 也就是说：整池里【一半的设备说法是参考集里根本不存在的】，且集中在少数几个工具上。
# 这些说法单看都通顺，问题恰恰在此——它们是"通顺但没人这么说"的话，
# 读起来就是模板味（"全屏吧""整成全屏""换成小屏""图像模式给我调成体育模式"）。
#
# 判据只有一条：这个字符串是否在参考集的设备子句里出现过。不修词、不改写、不近似匹配。
# 代价是池子变小（2520 → ~1400），但那是纯度的代价，值得：变小时靠频次权重倾斜，
# 头部真实说法自然吃掉大部分概率，尾部只留真实存在的长尾——这正是「多样但倾斜」。
# 注：闸门只作用于表面形式，不作用于 (tool, params)——收割项参数原样保留。
_ATTESTED_FOR_GATE = _ATTESTED if _ATTESTED else None
if _ATTESTED_FOR_GATE is not None:
    _before = len(DEVICE_PHRASES)
    DEVICE_PHRASES = [(q, t, p) for q, t, p in DEVICE_PHRASES if q in _ATTESTED_FOR_GATE]
    print(f"[device_phrases] 准入闸门：{_before} → {len(DEVICE_PHRASES)}"
          f"（参考集未出现的说法已剔除）")

    # ------------------------------------------------------------------
    # 全池按【参考集频次】赋权——这是本档最后一块结构性修正。
    # ------------------------------------------------------------------
    # 之前只有 numeric_adjust 走频次权重，其余工具（无论手写还是收割）一律 uniform。
    # 后果实测：每个工具的 Zipf 尾都被抬高、头都被压低。power_control 最典型——
    # 参考集里「关闭电视」124 次、「关掉电视」119 次，而「打开重启」「启动重启」
    # 各 1 次；uniform 抽样下尾部的"打开重启"被抽了 19 次（ref 1 次，17 倍），
    # 预计头部反被压到 0.05 倍。语感上这就是"专挑犄角旮旯的说法说"，
    # 用户读到的"不像真人"很大一部分来自这里。
    #
    # 判据与准入闸门同源：该说法在参考集里出现过几次。参考集是 10000 行的真人语料，
    # 其频次就是"真人这么说多少次"的直接度量，没有比它更好的权重来源。
    # 注意：权重按 (子句, tool) 取，不按子句取——"暂停"在 playback_control 是常态、
    # 在别处几乎没有，合并计数会把权重摊错工具（有 12 条说法跨工具）。
    if _ATTESTED_FREQ:
        # 同一说法在参考集里可能只挂在【另一个】tool 下。手写库的 tool 是我对照 catalog
        # 判的，与参考集标注不一定一致——实测「关闭静音」我挂在 numeric_adjust，而参考集
        # 21 次【全部】标 playback_control。这时 (子句,tool) 查不到，旧代码退回 1.0，
        # 把参考集频次 21 的说法压成 1（16 倍抑制），产出里整类消失——而它是参考集
        # 第 21 高频的设备说法之一。故查不到时退回【该说法在所有 tool 下的频次最大值】，
        # 拿不到 0 权重。真正在参考集里没出现过的说法已被上面的准入闸门剔除，
        # 这里不会把噪声放进来。
        _by_surface: dict[str, int] = {}
        for (_q, _t), _n in _ATTESTED_FREQ.items():
            if _n > _by_surface.get(_q, 0):
                _by_surface[_q] = _n
        _n_freq = 0
        _n_fallback = 0
        for _q, _t, _p in DEVICE_PHRASES:
            _f = _ATTESTED_FREQ.get((_q, _t))
            if _f:
                _n_freq += 1
            else:
                # 参考集里出现过（过了闸门）但不在这个 tool 下——跨工具说法，
                # 按该说法的跨工具最大频次赋权。
                _f = _by_surface.get(_q, 1)
                _n_fallback += 1
            # 除以 _len_w_phrase：生成器实际取 _len_w(q) * SURFACE_W[q]，
            # 而参考频次本身已含真实长度分布，再乘一次等于双重修正。
            SURFACE_W[_q] = _f / _len_w_phrase(_q)
        print(f"[device_phrases] 按参考集频次赋权 {_n_freq}/{len(DEVICE_PHRASES)} 条"
              f"（{_n_fallback} 条走跨工具最大频次回退）")

# ---------------------------------------------------------------------------
# 参数未标注（null）对齐：参考集里 24.5% 的设备位 tool_param 为 null，且【按说法系统性】
# ---------------------------------------------------------------------------
# 分布强烈双峰：1110 条从不 null、241 条永远 null、只有 53 条混合。恒 null 的正是最口语
# 的一档（"大点声"、"快进到三分钟"、"音量调到二十"、"播放第1集"）。参考集把这类短促指令
# 标为"该步未标"（eval_multi.judge() 对 null 步不计参数分），这是合法的标注状态而非缺失。
#
# 我手写库里这些说法一律带了参数——若不对齐，等于给它们凭空编金标，且"参数已标"占比虚高
# （实测参考设备位 null 26.2%、我 0.0%）。此处按【该说法在参考集里的 null 比例】对齐：
# NULL_P[q] 就是回收站里该说法的 null 频率，生成时按此概率决定该步是否置 null。
NULL_P: dict[str, float] = {}
if _ATTESTED_NULL:
    for _q, _t, _p in DEVICE_PHRASES:
        NULL_P[_q] = _ATTESTED_NULL.get(_q, 0.0)
    _n_null = sum(1 for v in NULL_P.values() if v >= 0.5)
    print(f"[device_phrases] 参数未标注对齐：{_n_null} 条恒 null 说法（参考集口径）")

# ---------------------------------------------------------------------------
# numeric_adjust 的族间配平（把「调到50」这类显式设值句压到参考集占比）
# ---------------------------------------------------------------------------
# 参考集 numeric_adjust 里 48.5% 的子句 tool_param 为 null，那些句子的参数不可复用，
# 所以配平的目标口径必须限定在「带参数的子句」上，不能拿全体算。实测（n=2280，
# 带参数）：程度/相对 61.2% / 裸数值 20.6% / 显式设值 9.5% / 其他 8.7%。
# 而我的池子天然偏向显式设值（短句少、句式长），按 _len_w 加权后 degree 只有 ~44%，
# num_set 却 ~19%。与其手调一堆魔法权重（每加一批短语就得重算），不如在这里按实测
# 目标族占比自动求一个族内缩放系数——加短语不影响配平结果。
#
# 族判定直接看金标，不看字面——参考集统计口径就是这么定的（value 有值=设值类，
# operation 为 提高/降低 且 value 空=程度类）。用字面正则判会被「一点/一些」里的
# 「一」误判成数字，把「音量调高一点」算进设值类，是上一版的错误来源。
def _num_family(q: str, p: dict) -> str:
    v = str(p.get("value") or "")
    if v.isdigit() or v.endswith("%"):
        return "num_set"
    if p.get("operation") in ("提高", "降低"):
        return "degree"
    return "other"


# 目标占比（参考集「带参数」子句实测，n=2280）：程度 68.9% / 其他 18.9% / 设值 12.1%。
# 注意口径：参考集 numeric_adjust 有 48.5% 子句 tool_param 为 null，那些复用不了，
# 所以配平只对齐「带参数」这一层，不拿全体算。
_NUM_FAM_TARGET = {"degree": 0.689, "other": 0.189, "num_set": 0.121}


def _rebalance_numeric() -> None:
    """按族实测占比重算 SURFACE_W：先求各族当前权重和，再令每族权重和正比于目标占比。"""
    def _len(s: str) -> float:
        n = len(s)
        return 1.0 if n <= 5 else (0.34 if n <= 7 else 0.13)

    fam_sum: dict[str, float] = {}
    for q, t, p in DEVICE_PHRASES:
        if t != "numeric_adjust" or not isinstance(p, dict):
            continue                              # 参数为 null 的不可复用，不参与配平
        f = _num_family(q, p)
        fam_sum[f] = fam_sum.get(f, 0.0) + _len(q) * SURFACE_W.get(q, 1.0)
    # 各族缩放系数 = target_f / 当前权重和。全体同乘常数不影响比例，
    # 但每个族各自归一后总量级会差很多，故再整体乘一个常数把量级拉回 1 附近。
    scale = {f: (tgt / fam_sum[f]) if fam_sum.get(f) else 0.0
             for f, tgt in _NUM_FAM_TARGET.items()}
    # 【保总量】配平只该改族间【比例】，不该改这一档的【总量】。
    # 加入 null 参数档之后这一条成了硬约束：null 档带的是原始参考频次权重（numeric
    # 合计 ~1492），而这里若不守恒会把「带参数」档整体压到 ~1.0，两档相差三个数量级，
    # 于是 numeric_adjust 的 null 率从参考的 48.1% 冲到 100%。守恒后，「带参数 vs null」
    # 的占比直接由参考频次决定（=参考集的 48.1%），配平只管「带参数档内部」的族构成。
    _orig = sum(fam_sum.values())
    _new = sum(scale[f] * fam_sum[f] for f in fam_sum if scale[f])
    _conserve = (_orig / _new) if _new else 0.0
    for q, t, p in DEVICE_PHRASES:
        if t != "numeric_adjust" or not isinstance(p, dict):
            continue
        s = scale.get(_num_family(q, p), 0.0) * _conserve
        if s:
            SURFACE_W[q] = SURFACE_W.get(q, 1.0) * s


_rebalance_numeric()

# ---------------------------------------------------------------------------
# 位置闸门：哪些设备说法【可以】起头 / 【可以】跟在后面（2026-11-10）
# ---------------------------------------------------------------------------
# 这是整批数据「生硬」的最后一块结构性病因，此前一直没被发现——因为它在【行级、
# 词级】指标上完全隐形：用词全都在参考集里（准入闸门已保证），频次也对，唯独
# 【哪个词能出现在哪个位置】没对。
#
# 参考集的设备说法是【强位置条件】的（见 ref_device_vocab.attested_device_leadfreq
# 的实测：过散布 chi²/df=20.6、置换检验真实值 2718 远在零分布 [2429,2524] 之外）：
#   · 205 个频次≥10 的说法里，83 个(40.5%)【从不】出现在句首
#     ——"亮度降低一点"11 次、"把屏保关掉"10 次、"换一首"10 次全在第二位及以后；
#   · 另 50 个【永远】在句首——"接着播放"28、"重启"27、"播放hdmi1"26。
#   · 工具层面还有 8/16 个设备工具【从不】起头：screensaver_control、timer_control、
#     network_control、smart_camera、audio_control、display_control、demo_control、
#     solve_picture_sound_problem_control——它们在参考集里 100% 出现在第二位及以后。
#
# 我的生成器按位置无关的权重抽说法，产出里设备首位子句 39.4% 用了参考集【从不】放
# 首位的说法（"启动u盘…""切换到HDMI4…""音量22…"）。每个词都认识、但没人这样开头——
# 这正是用户反复说的「生硬」最具体的一层。
#
# 闸门口径（宁少勿滥，且要能用）：
#   · 有证据就用证据。某说法在参考集设备位上首位了 n0 次、非首位 n1 次：
#       可起头  ⟺ n0 > 0
#       可跟随  ⟺ n1 > 0
#   · 【单边证据是证据，不是噪声】（2026-11-11 修正）。上一版把 n0+n1<K(=3) 一律当
#     "样本不足"豁免，于是 (n0=0, n1=1) 这种【见过一次、且只在跟随位】的说法也被放行
#     去起头。实测后果：我的设备首位子句 **10.92%** 落在这一档（"进入支架""全屏视窗模式"
#     "继续""进入会议模式"…），且在小工具里被放大成显眼的重复——`screen_lift_rotation`
#     全池只有两条豁免项（`进入支架`/`打开竖屏短视频`，各 n1=1），谁被抽到谁就一直起头，
#     产出里 `进入支架` 出现在首位 11 次（参考集 **0 次**），整卷 24 倍于参考。
#     判据改为：【只要有一侧观测、另一侧为零，就按方向性证据处理】——(0,1)/(0,2) 与
#     (0,500) 同判"只能跟随"。豁免只留给【两侧都是 0】（参考集里根本没见过）的说法，
#     那才是真正的"无证据"，此时回退到 2 字词头族。
#     安全性已实测：新判据在参考集自身返回 0 违规（lead 0 / follow 0）。
_MIN_POS_SUPPORT = 3
_POS_EVIDENCE: dict[str, list[int]] = {}
for (_q, _t), (_n0, _n1) in _ATTESTED_LEAD.items():
    _c = _POS_EVIDENCE.setdefault(_q, [0, 0])
    _c[0] += _n0
    _c[1] += _n1

# 【工具条件】位置证据（2026-11-11）：上面按说法聚合是【跨工具求和】，会把
# 方向相反的证据对冲掉。实测有 6 个说法跨工具且位置条件相反，全部落在设备侧：
#   `关闭屏幕` (screensaver 0/64) vs (lift 18/0) vs (mode 23/0)
#   `关掉电视` (power 89/43) vs (timer 0/25)、(关机 power 61/18 vs timer 0/23)…
# 聚合后 `关闭屏幕`=(41,64) 判"可起头"，于是它【以 screensaver 身份】被换序顶到首位
# ——而参考集里 screensaver 句首 0.0%。故查证据时【带 tool】，同一说法按它当前的
# 工具取证据；工具下查不到再退回聚合值。
_POS_BY_QT: dict[tuple[str, str], tuple[int, int]] = {
    (_q, _t): (_n0, _n1) for (_q, _t), (_n0, _n1) in _ATTESTED_LEAD.items()
}

# 供生成器按【位置专属频次】加权：起头位按首位频次、跟随位按非首位频次，
# 与参考集的「位置 × 说法」联合分布同口径（不是"先按全体频次抽、再筛位置"）。
POS_EVIDENCE: dict[str, tuple[int, int]] = {k: (v[0], v[1]) for k, v in _POS_EVIDENCE.items()}

# 工具层面的位置证据：参考集里 8/16 个设备工具【从不】出现在首位
# （display 358 次、screensaver 1018、timer 258、network 215、smart_camera 197、
#  audio 164、demo 70、solve 139——各自总量都不小，n0=0 是真实语序而非样本不足）。
# 另 8 个（numeric/screen_layout/playback/common/mode/source_switch/power/lift）才起头。
_TOOL_LEAD_MASS: dict[str, int] = {}
for (_q, _t), (_n0, _n1) in _ATTESTED_LEAD.items():
    _TOOL_LEAD_MASS[_t] = _TOOL_LEAD_MASS.get(_t, 0) + _n0
# 只有【参考集里见过】的设备工具才有位置证据。判据用"有证据且 n0==0"——手写库独有、
# 参考集里查不到的工具不会被误判成"不能起头"。
LEAD_CAPABLE_TOOLS: set[str] = {t for t, n0 in _TOOL_LEAD_MASS.items() if n0 > 0}
NONLEAD_TOOLS: set[str] = {t for t, n0 in _TOOL_LEAD_MASS.items() if n0 == 0}

# 前缀族回退：某说法的【2 字词头】在参考集里从不句首、且总量可观时，即使该说法自身
# 样本太少（n0=n1<K）也按"不能起头"处理。K=3 的小样本豁免是为了不误伤冷僻说法，
# 但它同时漏掉了【词头证据】这一层更强的信号：参考集里以「启动」开头的设备说法共 446
# 次、以「切换」开头的 105 次、以「图像」开头的 85 次……【一次都没有出现在句首】。
# 我库里那些冷僻的「启动u盘」(0lead/2follow)、「切换到ps4」正是踩在这个豁免上，
# 加上词头回退后能再收掉 223 条"没人会这样开头"的说法。
# 只用 2 字词头、且要求词头族总量≥10（防止拿一两个巧合样本否定整族）。
_PREFIX_EVIDENCE: dict[str, list[int]] = {}
for _q, (_n0, _n1) in _POS_EVIDENCE.items():
    _c = _PREFIX_EVIDENCE.setdefault(_q[:2], [0, 0])
    _c[0] += _n0
    _c[1] += _n1
_PREFIX_MIN_MASS = 10
_NEVER_LEAD_PREFIX = {p for p, (n0, n1) in _PREFIX_EVIDENCE.items()
                      if n0 == 0 and n0 + n1 >= _PREFIX_MIN_MASS}


_POS_ALPHA = 0.5


def pos_weight(q: str, tool: str, pos: str) -> float:
    """位置【软】权重：把 (首位次数, 非首位次数) 平滑成该位置的概率。

    (n0+α)/(n0+n1+2α)（α=0.5，Laplace）。用它当抽样权重、而不是【硬过滤】，
    是为了同时满足两件事：
      · 位置条件性：`大点声`(44/46) 首位权重≈0.49、`把音量调到最大`(7/0)≈0.94、
        而 screensaver 的 `关闭屏幕`(0/64)≈0.008——重头说法自然吃满，尾部按概率出现。
      · 说法多样性：硬过滤会把只在一侧出现过的说法【整条】逐出该位置，
        首位池因此缩水、头部被动集中（实测 seg distinct 1589→1549、LM |Δ| 0.048→0.060）。
        软权重保留了全部说法，只是把违例概率压到参考集的水准。
    无证据（说法不在参考集任何位置）→ 返回 1.0（不压）。证据取 (说法,工具)，
    查不到退回跨工具聚合值——同一说法在不同工具下位置条件可相反（见 _POS_BY_QT）。
    """
    e = _POS_BY_QT.get((q, tool)) if tool else None
    if e is None:
        e = _POS_EVIDENCE.get(q)
    if e is None:
        return 1.0
    n0, n1 = e
    if pos == "lead":
        return (n0 + _POS_ALPHA) / (n0 + n1 + 2 * _POS_ALPHA)
    return (n1 + _POS_ALPHA) / (n0 + n1 + 2 * _POS_ALPHA)


def may_lead(q: str, tool: str = "") -> bool:
    """该设备说法能否作整句的第一个子句（按参考集位置证据）。

    判据（2026-11-11 修正为【单边证据即方向性证据】+【工具条件】）：
      · n0 > 0                 → 能起头
      · n0 == 0 且 n1 > 0      → 不能起头（见过、且只在跟随位——这是证据，不是小样本噪声）
      · n0 == n1 == 0          → 无证据，回退到 2 字词头族（词头从不句首且量足 → 不能起头）
    带 tool 时按 (说法, 工具) 取证据——同一说法在不同工具下位置条件可相反
    （见 _POS_BY_QT 的说明）；工具下查不到再退回聚合值。
    """
    e = _POS_BY_QT.get((q, tool)) if tool else None
    if e is None:
        e = _POS_EVIDENCE.get(q)
    n0, n1 = (e if e is not None else (0, 0))
    if n0 > 0:
        return True
    if n1 > 0:
        return False                          # 单边跟随证据 → 不能起头
    # 两侧都无证据：退回词头族。词头在参考集里从不句首（且总量可观）→ 不能起头。
    if q[:2] in _NEVER_LEAD_PREFIX:
        return False
    return True


def may_follow(q: str, tool: str = "") -> bool:
    """该设备说法能否出现在非首位（按参考集位置证据），判据与 may_lead 对称。"""
    e = _POS_BY_QT.get((q, tool)) if tool else None
    if e is None:
        e = _POS_EVIDENCE.get(q)
    n0, n1 = (e if e is not None else (0, 0))
    if n1 > 0:
        return True
    if n0 > 0:
        return False                          # 单边首位证据 → 不能跟随
    return True                               # 两侧都无证据 → 允许


# ===========================================================================
# 内容子句的位置条件性（2026-11-10，与设备侧同因但【更强】）
# ===========================================================================
# 参考集内容子句的位置条件性比设备更强：χ²/df=96.3；freq≥10 的 61 个表面里 31.1%
# 【从不】在首位出现。但内容侧的"跟随位"不是【说法级】现象，而是【语域级】现象——
# 参考集内容子句的全部 6442 个实例里，只有 254 个落在非首位（3.9%），且这 254 个
# 只由 14 种说法构成（"播放音乐""播放第1集""播放强军战歌"…），
# 清一色是【短的、点播式的】祈使句；【0%】是搜索/问句式（"有哪些/推荐/适合…"）。
# 实测口径（只看真内容工具，剔除标注噪声）：
#   ref 跟随位 254 例：100% 满足 len≤12 且非搜索式；mean 6.2 字、median 5、max 12
#   我的跟随位 204 例：只有 72.5% 满足（mean 9.6 字），27.5% 是长搜索句——
#     "把电视关了，适合炎热晴天看的电影有哪些吧"
#     "关闭电视然后查免费高分综艺吧！"
#     "关掉电视，七年级语文，要苏教版或北师大版，但不要免费的啊"
# 一个真人不会在"关电视"之后接着甩一串条件搜索——这是最扎眼的"生硬"。
#
# 故内容侧【跟随位】用【语域闸门】（长度+句式）而非说法闸门：参考集跟随位只有 14 种
# 说法，说法级判据会把内容池几乎清空（2684 种里 2662 种从不在跟随位出现）。
# 语域闸门保留 65.5% 的池子，且在参考集上【零误伤】（14/14 通过）。
# 【首位】仍用说法级证据（与设备侧同构）——那侧 2662/2684 在首位出现过，闸门只收极少数。
try:
    from ref_device_vocab import attested_content_leadfreq as _attested_clead
    _ATTESTED_CLEAD = _attested_clead()        # {(内容子句, gold): (首位次数, 非首位次数)}
except Exception:                              # 参考集不在时退化为无闸门
    _ATTESTED_CLEAD = {}

_CONTENT_POS: dict[str, list[int]] = {}
for (_q, _g), (_n0, _n1) in _ATTESTED_CLEAD.items():
    _c = _CONTENT_POS.setdefault(_q, [0, 0])
    _c[0] += _n0
    _c[1] += _n1

CONTENT_POS_EVIDENCE: dict[str, tuple[int, int]] = {k: (v[0], v[1]) for k, v in _CONTENT_POS.items()}

# 参考集【内容域跟随位】的封闭注册表 {说法: 非首位出现次数}（2026-11-11）。
# 参考集内容位共 6907 例，落在非首位的只有 438 例，且【只由 23 种说法构成】
# （「播放第1集」46 次占 10.5%）。这是一套固定说法表，不是"首位分布的一个切片"：
# 真人在"关电视"之后会说"播放第1集""只听音乐""放个汪汪队"，不会随机甩一个片名。
# 与 content_may_follow（语域闸门，保留 609 种）是【两道不同的闸】：语域闸决定
# "什么形态能进跟随位"，本表决定"跟随位实际用哪几句"。生成器两个落点位都用本表。
CONTENT_FOLLOW_REG: dict[str, int] = {q: v[1] for q, v in _CONTENT_POS.items() if v[1] > 0}


def content_follow_register() -> dict[str, int]:
    """返回 {说法: 参考集跟随位频次}——见 CONTENT_FOLLOW_REG 的说明。"""
    return CONTENT_FOLLOW_REG

# 首位侧的词头族回退，与设备侧同构（"只看/只听"这类词头若在参考集里从不句首，则冷僻说法也拦住）。
_CPREFIX_EVIDENCE: dict[str, list[int]] = {}
for _q, (_n0, _n1) in _CONTENT_POS.items():
    _c = _CPREFIX_EVIDENCE.setdefault(_q[:2], [0, 0])
    _c[0] += _n0
    _c[1] += _n1
_C_NEVER_LEAD_PREFIX = {p for p, (n0, n1) in _CPREFIX_EVIDENCE.items()
                        if n0 == 0 and n0 + n1 >= _PREFIX_MIN_MASS}


def content_may_lead(q: str) -> bool:
    """该内容说法能否作整句第一个子句（按参考集位置证据，同设备侧两级判据）。"""
    e = _CONTENT_POS.get(q.strip())
    n0, n1 = (e if e is not None else (0, 0))
    if n0 > 0:
        return True
    if n0 + n1 >= _MIN_POS_SUPPORT:
        return False                          # 有足够样本且从不句首 → 不能起头
    if q.strip()[:2] in _C_NEVER_LEAD_PREFIX:
        return False
    return True


# 跟随位语域闸门：长度上限与句式。12 是参考集跟随位的【实测上界】（max=12，100% 落在内）。
_CONTENT_FOLLOW_MAXLEN = 12
_C_SEARCH_PREFIXES = (
    "查", "搜索", "搜", "找", "有哪些", "推荐", "适合", "有没有", "什么",
    "帮我", "怎样", "怎么", "如何", "哪", "多少", "想看", "想听",
)


def content_is_searchlike(q: str) -> bool:
    """搜索/问句式：真人不会把它挂在"关电视"之后，参考集跟随位 0% 出现。"""
    s = q.strip()
    return any(s.startswith(p) or p in s[:7] for p in _C_SEARCH_PREFIXES)


# 跟随位的【点播祈使】词头。参考集跟随位是一个【封闭语域】：438 例只由 23 种说法
# 构成，其中 21/23 以点播/请求动词起头（播放第1集/我要看汪汪队/打开音乐…），
# 剩下 2 种是 ≤5 字、不含「的」的裸标题（"最强大脑"19、"只听音乐"23）。我原来的
# 闸门只卡 len≤12 + 非搜索式，于是长描述性名词（"北京卫视的养生堂片头曲"、
# "王鹤棣演的电影"、"射雕英雄传八三版的"）整批漏进跟随位——实测我的跟随位内容
# 36.8% 是【无动词的裸名词】（60 种），参考集只有 8.2%（2 种）。真人不会在
# "关电视"之后甩一个光秃秃的片名描述，要么带点播动词（"放个汪汪队"），要么是
# 极短的节目名（"最强大脑"）。判据：以点播/请求动词起头，或（len≤5 且不含「的」）。
_C_FOLLOW_VERB = re.compile(
    r"^(播放|放|放开|放一|放个|放首|看|听|打开音乐|打开|我要看|我要听|我想看|我想听|"
    r"要看|要听|只看|只听|来|搜|查|找|推荐|给我|帮我|学|唱|讲|挑|选|把|将|收|预约|"
    r"继续|开始|选择|来一|来首|来点|要点|换个|换一|点播|播|换|放点)")


def content_is_follow_register(q: str) -> bool:
    """参考集跟随位的点播语域：点播祈使句，或 ≤5 字且不含「的」的短标题。"""
    s = q.strip()
    if _C_FOLLOW_VERB.match(s):
        return True
    return len(s) <= 5 and "的" not in s


def content_may_follow(q: str) -> bool:
    """该内容说法能否出现在非首位（参考集跟随位的语域闸门）。

    与设备侧同法用【参考集位置证据】，并叠一层语域：参考集跟随位是一个【封闭语域】
    ——438 例只由 23 种说法构成，全部以点播/请求动词起头（"播放第1集""我要看汪汪队"
    "打开音乐"）或为 ≤5 字裸标题（"最强大脑"）。我原来只卡 len≤12 + 非搜索式，
    于是长描述性名词（"北京卫视的养生堂片头曲""王鹤棣演的电影"）整批漏进跟随位：
    实测我跟随位内容 36.8% 是无动词裸名词（60 种），参考集只有 8.2%（2 种）。
    判据取二者并集（任一成立即可）：
      · 点播/请求动词起头（`_C_FOLLOW_VERB`）——覆盖参考集 21/23 种；
      · 该说法在参考集里【确有】跟随位出现（`_CONTENT_POS` 的 n1>0）——覆盖
        "最强大脑"（pos=(0,19)）这类无动词但确实跟随的短标题。
    "真人在'关电视'后甩片名描述"这种句式参考集一次没有，故长裸名词一律排除。
    并集在参考集上零误伤（23/23 通过）、保留 609 条说法、27 个工具无一被抽空。
    """
    s = q.strip()
    if len(s) > _CONTENT_FOLLOW_MAXLEN or content_is_searchlike(s):
        return False
    if _C_FOLLOW_VERB.match(s):
        return True
    e = _CONTENT_POS.get(s)
    return bool(e) and e[1] > 0


# 去掉空 value/device/location 的键（与 all_eval 标注一致：device/location 保留空串）
def clean_device_params(p: dict) -> dict:
    """保留 operation/object/value/date_time/intent；device/location 空串照留（同线上标注）。"""
    keep = ("operation", "object", "value", "date_time", "intent")
    return {k: v for k, v in p.items() if k in keep}


if __name__ == "__main__":
    import collections
    print("设备自然子句条数:", len(DEVICE_PHRASES))
    c = collections.Counter(t for _, t, _ in DEVICE_PHRASES)
    for t, n in c.most_common():
        print(f"  {t:40s} {n}")
    print()
    for q, t, p in DEVICE_PHRASES[:12]:
        print(f"  {q:28s} → {t} {p}")
