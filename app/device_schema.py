"""设备域「虚拟 schema」：给 LLM 看的参数视图 ⇄ 真实工具参数（operation/object）的确定性互转。

为什么
  真实 schema 全用 `operation + object` 表达，很多地方要模型背约定：
    关机 = 打开+关机   暂停 = 停止+播放控制   下一集 = 下+播放控制   全屏 = 打开+全屏
    定时关机 = 打开+关机(timer)   返回 = 返回+返回   静音 = 打开+静音
  用户说的是动词，参数却是「打开 + 名词」，模型要先把动词翻成约定再填，且 operation 在多数工具里
  没有枚举。虚拟 schema 让 action 直接等于用户要做的事，由本模块查表还原成真实参数。

设计
  · 工具名不变（device:power_control 等），路由与选工具口径不受影响；只换参数视图。
  · 虚拟参数：action（必填）、target（对象，可选）、value、time（= date_time）。
    device / location 几乎恒空，不再展示（非空时原样透传）。
  · 两类 action：
      固定动作  一个词直接对应一个真实 (operation, object)，无需 target，如 关机、暂停、全屏、回到主页；
      通用动作  就是真实 operation 词（打开/关闭/设置/提高/降低…），配 target 指明对象，如 打开 护眼模式。
    固定动作的 object 恒定，所以不出现 target；带 target 时一律按通用动作处理。
  · 互转只查表，不含语言规则：compile_call（虚拟→真实，线上用）、decompile_call（真实→虚拟，造训练数据用）。

标注里同一概念写法不一致的（如 playback_control 的 object 有时为空），decompile 归一到规范写法，
由 tools/check_device_schema.py 统计「无损 / 归一 / 无法表示」。
"""
from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# 固定动作：real_tool → {action: (operation, object)}
# ---------------------------------------------------------------------------
_PB = "播放控制"
_PL = "播放列表"
FIXED: dict[str, dict[str, tuple[str, str]]] = {
    "power_control": {
        "开机": ("打开", "开机"), "关机": ("打开", "关机"), "重启": ("打开", "重启"),
    },
    "timer_control": {
        "定时关机": ("打开", "关机"), "取消定时关机": ("关闭", "关机"),
        "查询剩余关机时长": ("查询", "剩余关机时长"),
        "开始倒计时": ("打开", "剩余关机时长"), "取消倒计时": ("关闭", "剩余关机时长"),
        "设置闹钟": ("打开", "闹钟"),
    },
    "numeric_adjust": {
        "静音": ("打开", "静音"), "取消静音": ("关闭", "静音"),
    },
    "playback_control": {
        # object 取金标多数写法：播放/暂停带「播放控制」，其余为空（all_eval.csv 统计）
        "播放": ("播放", _PB), "暂停": ("停止", _PB), "退出播放": ("退出", ""), "重播": ("重播", ""),
        "快进": ("快进", ""), "快退": ("快退", ""), "快进到": ("快进到", ""), "快退到": ("快退到", ""),
        "跳转": ("跳转", ""),
        "上一个": ("上", ""), "下一个": ("下", ""), "换一个": ("切换", ""),
        "单曲循环": ("单曲循环", _PL), "列表循环": ("循环播放", _PL), "随机播放": ("随机播放", _PL),
        "顺序播放": ("顺序播放", _PL), "列表播放": ("列表播放", _PL),
    },
    "screen_layout": {
        "全屏": ("打开", "全屏"), "退出全屏": ("关闭", "全屏"),
        "小屏": ("打开", "小屏"), "退出小屏": ("关闭", "小屏"),
        "分屏": ("打开", "分屏"), "退出分屏": ("关闭", "分屏"),
        "隐藏分屏": ("打开", "隐藏分屏"), "显示分屏": ("打开", "显示分屏"),
        "放大画面": ("放大", "画面"), "缩小画面": ("缩小", "画面"),
    },
    "screensaver_control": {
        "关屏": ("关闭", "屏幕"), "亮屏": ("打开", "屏幕"),
    },
    "common_control": {
        "回到主页": ("打开", "主页"), "返回": ("返回", "返回"),
        "退下": ("打开", "退下"), "退出语音": ("关闭", "退下"),
    },
    "video_picture_save_share": {
        "保存": ("保存", "[视频/图片]"), "删除": ("删除", "[视频/图片]"), "分享": ("分享", "[视频/图片]"),
    },
}

# 播放控制的 object 在标注里「播放控制」与空同义：这些 operation 下两种写法互相归一到 FIXED 里的写法
_PB_OPS = {op: obj for op, obj in FIXED["playback_control"].values() if obj in (_PB, "")}

# 通用动作（= 真实 operation 词）：real_tool → 可用 operation；用于 prompt 展示
GENERIC_OPS: dict[str, list[str]] = {
    "numeric_adjust": ["提高", "降低", "设置", "查询", "打开", "关闭"],
    "source_switch": ["打开", "关闭", "设置"],
    "mode_control": ["设置", "打开", "关闭"],
    "screen_lift_rotation": ["设置", "打开", "关闭", "提高", "降低"],
    "demo_control": ["打开", "关闭", "演示", "展示", "体验"],
    "display_control": ["打开", "关闭"],
    "audio_control": ["打开", "关闭"],
    "smart_camera": ["打开", "关闭", "设置", "调整"],
    "network_control": ["打开", "关闭", "查询", "设置", "测试"],
    "screensaver_control": ["打开", "关闭", "设置", "切换"],
    "common_control": ["打开", "关闭", "设置", "查询", "切换", "导入", "更新", "恢复"],
}
# 注：标注里 playback_control 下偶有 (设置, 倍速) 等写法，应走 numeric_adjust；decompile 仍能往返，
# 但不向模型展示，避免两个工具都能调倍速。

# 原样透传的真实工具（参数本就友好）
PASSTHROUGH = {"solve_picture_sound_problem_control", "fan_knowledge_agent"}

_REV: dict[str, dict[tuple[str, str], str]] = {
    t: {v: k for k, v in acts.items()} for t, acts in FIXED.items()
}


def _prune(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v not in ("", None, [], {})}


# ---------------------------------------------------------------------------
# 虚拟 → 真实
# ---------------------------------------------------------------------------
def compile_call(tool: str, params: dict[str, Any]) -> dict[str, Any]:
    """虚拟参数 → 真实参数（只含非空字段）。action 缺失或固定动作未知时抛 ValueError。"""
    if tool in PASSTHROUGH:
        return _prune(dict(params))
    p = dict(params)
    action = p.pop("action", "")
    target = p.pop("target", "")
    if not action:
        raise ValueError(f"{tool}: 缺少 action")
    out: dict[str, Any] = {}
    fixed = FIXED.get(tool, {})
    if not target and action in fixed:
        out["operation"], out["object"] = fixed[action]
    else:
        out["operation"], out["object"] = action, target
    if "time" in p:
        out["date_time"] = p.pop("time")
    out.update(p)                                  # value / device / location 原样带过
    return _prune(out)


# ---------------------------------------------------------------------------
# 真实 → 虚拟（造训练数据 / 校验用）
# ---------------------------------------------------------------------------
def decompile_call(tool: str, params: dict[str, Any]) -> dict[str, Any]:
    """真实参数 → 虚拟参数。不可表示时抛 ValueError。
    播放控制 object 为空视同「播放控制」（标注写法不一致，归一）。"""
    if tool in PASSTHROUGH:
        return _prune(dict(params))
    p = _prune(dict(params))
    op, obj = p.pop("operation", ""), p.pop("object", "")
    if not op:
        raise ValueError(f"{tool}: 缺少 operation")
    out: dict[str, Any] = {}
    key_obj = obj
    if tool == "playback_control" and obj in ("", _PB) and op in _PB_OPS:
        key_obj = _PB_OPS[op]
    act = _REV.get(tool, {}).get((op, key_obj))
    if act:
        out["action"] = act
    else:
        out["action"] = op
        if obj:
            out["target"] = obj
        # 通用 action 与同工具固定动作重名且无 target 时会被 compile 误当固定动作
        if not obj and op in FIXED.get(tool, {}):
            raise ValueError(f"{tool}: ({op!r}, 空对象) 与固定动作重名，无法无歧义表示")
    if "date_time" in p:
        out["time"] = p.pop("date_time")
    out.update(p)
    return out


# ---------------------------------------------------------------------------
# 接入辅助
# ---------------------------------------------------------------------------
VIRTUAL_PARAM_NAMES = frozenset({"action", "target", "value", "time", "device"})


def is_virtual_tool(tool: str) -> bool:
    """该设备工具对 LLM 展示的是虚拟参数（透传类工具除外）。"""
    return tool in FIXED or tool in GENERIC_OPS or tool in _DESC


def compile_tolerant(tool: str, params: dict[str, Any]) -> dict[str, Any]:
    """后处理入口：虚拟参数 → 真实参数。
    · 不展示虚拟参数的工具（透传类 / 未知工具）原样返回；
    · 模型仍按旧习惯写了真实参数（有 operation、无 action）→ 视为已是真实写法，原样返回；
    · 其余按 compile_call 还原，缺 action 抛 ValueError（由调用方降级）。"""
    if not is_virtual_tool(tool) or tool in PASSTHROUGH:
        return dict(params)
    if "action" not in params and "operation" in params:
        return dict(params)
    return compile_call(tool, params)


def virtualize_params(tool: str, params: dict[str, Any]) -> dict[str, Any] | None:
    """库内示例（真实参数）→ 虚拟参数；不可表示返回 None（该示例不应进 few-shot）。"""
    p = {k: v for k, v in params.items() if k != "retext"}
    if not is_virtual_tool(tool) or tool in PASSTHROUGH:
        return p
    try:
        return decompile_call(tool, p)
    except ValueError:
        return None


# 静态锚例里的设备参数（prompts._ANCHORS）→ 虚拟写法
ANCHOR_REWRITES: list[tuple[str, str]] = [
    ('"params":{"object":"volume","value":"30"}',
     '"params":{"action":"设置","target":"音量","value":"30"}'),
    ('"params":{"operation":"跳转","object":"播放控制","value":"15分钟"}',
     '"params":{"action":"跳转","value":"15分钟"}'),
    ('"params":{"operation":"设置","object":"音量","value":"8"}',
     '"params":{"action":"设置","target":"音量","value":"8"}'),
]


# ---------------------------------------------------------------------------
# prompt 目录块（与 catalog.render_oneshot_block 同风格，设备域部分）
# ---------------------------------------------------------------------------
_DESC = {
    "numeric_adjust": "数值调节：音量、亮度、对比度、色度、清晰度、分辨率、倍速、刷新率（target 为调节对象，value 为数值）；"
                      "静音/取消静音",
    "power_control": "开机、关机、重启（立即执行；定时关机用 timer_control）",
    "timer_control": "定时关机、取消定时关机、倒计时、闹钟、查询剩余关机时长；time 为时间表达（30分钟/22:00）",
    "source_switch": "信号源切换（HDMI、USB、机顶盒、数字电视等）：设置 + target=信号源 + value=HDMI1",
    "playback_control": "播放控制：播放/暂停/退出/重播、快进快退/跳转（value=时间）、上一个/下一个/换一个"
                        "（value=集/部/首）、单曲/列表循环、随机/顺序播放",
    "video_picture_save_share": "视频/图片保存、删除、分享（value 为分享渠道等）",
    "mode_control": "模式：图像/声音/音效/护眼/混响/商场等；设置 + target=图像模式 + value=标准",
    "screen_layout": "画面布局：全屏、小屏、分屏、放大缩小画面",
    "screen_lift_rotation": "屏幕旋转、支架升降",
    "demo_control": "效果演示/展示/demo",
    "display_control": "画质与显示功能开关（HDR、背光、运动补偿、色彩、护眼等）",
    "audio_control": "音质音效功能开关（杜比全景声、声场、按键音等）",
    "smart_camera": "摄像头、雷达、AI 视觉功能",
    "network_control": "网络功能（WiFi、有线、热点、测速）",
    "screensaver_control": "屏幕开关（关屏/亮屏）、屏保/壁纸/壁画的打开关闭与设置、待机",
    "common_control": "回到主页/返回/退下；其他设备功能兜底（系统设置、输入法、蓝牙、存储、遥控器等，打开/关闭 + target）",
}
_HAS_TIME = {"timer_control", "video_picture_save_share"}
_NO_VALUE = {"power_control", "display_control", "audio_control", "smart_camera", "network_control"}
# 标注里 device 非空（如「帮电视机打开屏保」→ device=电视）的工具：目录里必须展示，否则模型不会填
_HAS_DEVICE = {"screensaver_control"}


def render_block(tools: list[str] | None = None) -> str:
    """设备域虚拟目录块。solve_picture_sound_problem_control / fan_knowledge_agent 原样（无需改写）。

    target 的必填性按工具类型写清：纯固定动作的工具没有 target；只有通用动作的工具 target 必填；
    两者都有的工具（numeric_adjust / screensaver_control / common_control）注明
    「通用动作必须带 target，固定动作不带」，否则模型会漏写对象（真实 schema 里 object 是必填）。"""
    lines = []
    for t, desc in _DESC.items():
        if tools is not None and t not in tools:
            continue
        fixed = list(FIXED.get(t, {}))
        generic = [o for o in GENERIC_OPS.get(t, []) if o not in FIXED.get(t, {})]
        sig = f"必填 action[{'|'.join(fixed + generic)}]"
        opt: list[str] = []
        if generic and not fixed:
            sig += " target"
        elif generic and fixed:
            opt.append("target")
            desc += f"；{'/'.join(generic)} 必须带 target，{'/'.join(fixed)} 不带 target"
        if t not in _NO_VALUE:
            opt.append("value")
        if t in _HAS_TIME:
            opt.append("time")
        if t in _HAS_DEVICE:
            opt.append("device")
            desc += "；用户点名设备时填 device（如 电视）"
        if opt:
            sig += f"；可选 {' '.join(opt)}"
        lines.append(f"- device:{t} | {sig} | {desc}")
    return "\n".join(lines)
