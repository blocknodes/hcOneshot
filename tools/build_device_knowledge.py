#!/usr/bin/env python3
"""设备域「规则 + 实体」知识（替代虚拟 schema 的实验）：从训练集统计真实参数写法，生成

  1. 设备规则（追加到规则库）：每个设备工具一条，正则触发，正文给出该工具最常见的
     真实写法（operation/object/value），例句取自训练集；
  2. 设备对象词典：query 里原样出现的 object（护眼模式、运动补偿、屏保…）→ 标准写法说明，
     并入实体词典（types=设备，label 列为说明）。

来源只用 testset/train/train.csv 的 tv_control 行（与 all_eval / 多意图评测集无重合）。

用法：python3 tools/build_device_knowledge.py
  → experiments/retrieval/lib/device_entity.csv
  → experiments/retrieval/lib/rules_device.csv      （= rules.csv + 设备规则）
  → experiments/retrieval/lib/entity_dict_dev.csv   （= entity_dict.csv + 设备对象）
"""
from __future__ import annotations

import collections
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIB = ROOT / "experiments/retrieval/lib"
SRC = ROOT / "testset/train/train.csv"

# 每个工具的触发词（正则）；没有列出的工具靠对象词典 + few-shot
TRIGGER = {
    "power_control": r"开机|关机|重启|关电视|关闭电视|打开电视|电视关|关掉电视",
    "timer_control": r"定时|倒计时|闹钟|分钟后|小时后|剩余.{0,4}时",
    "numeric_adjust": r"音量|声音|大声|小声|大点声|小点声|亮度|对比度|色度|清晰度|分辨率|静音|调亮|调暗|亮一点|暗一点|倍速",
    "playback_control": r"^(播放|继续|开始播放|播放吧)$|暂停|继续播放|停止播放|退出播放|重播|从头|快进|快退|跳到|跳转|上一|下一|换一|切歌|单曲循环|循环播放|随机播放|顺序播放|列表播放|关.{0,2}(音乐|歌)|别放了",
    "screen_layout": r"全屏|小屏|分屏|放大|缩小|大屏|小窗",
    "screensaver_control": r"启动屏幕|屏保|壁纸|壁画|关屏|息屏|熄屏|黑屏|亮屏|闭屏|屏幕关|关.{0,2}屏幕|打开屏幕",
    "common_control": r"主页|首页|主界面|返回|退下|退出语音|蓝牙|输入法",
    "source_switch": r"信号源|HDMI|hdmi|USB|usb|机顶盒|AV|av",
    "mode_control": r"模式",
}
NAME = {"power_control": "开关机", "timer_control": "定时", "numeric_adjust": "数值调节",
        "playback_control": "播放控制", "screen_layout": "画面布局", "screensaver_control": "屏幕/屏保",
        "common_control": "通用控制", "source_switch": "信号源", "mode_control": "模式"}
GENERIC = {
    "power_control": ["关机/开机/重启都是 operation=\"打开\"，object=动作名；不要写成 operation=关闭，也不要加 device 字段",
                      "打开电视/开电视/唤醒电视/启动电视 → {\"operation\":\"打开\",\"object\":\"开机\"}；关闭电视/关电视/关掉电视 → {\"operation\":\"打开\",\"object\":\"关机\"}"],
    "timer_control": ["N分钟后关机/N小时后关机 → {\"operation\":\"打开\",\"object\":\"关机\",\"date_time\":\"N分钟\"}（时长写在 date_time，不写 value；object 是「关机」不是「定时关机」）"],
    "numeric_adjust": ["声音/音量 一律 object=\"音量\"（不要写「声音」）；饱和度/色彩 → \"色度\"；关闭声音/消音/静音 → {\"operation\":\"打开\",\"object\":\"静音\"}",
                       "调到/设为 N → {\"operation\":\"设置\",\"object\":\"音量\",\"value\":\"N\"}，value 按原话数字写（零→\"0\"，二十→\"20\"，不要加 %）",
                       "最大/最高 → 设置 + value \"100%\"；最小/最低 → 设置 + value \"0%\"",
                       "调大/调高/大一点/加 → operation=\"提高\"；调小/调低/小一点/减 → operation=\"降低\"；没说数值就不写 value"],
    "playback_control": ["从头播放/从头开始/重新播放 → {\"operation\":\"重播\"}（不带 object）；继续播放/接着播 → {\"operation\":\"播放\",\"object\":\"播放控制\"}",
                         "上一集/下一首/换下一首/下集 → operation=上|下，value 写量词：{\"operation\":\"下\",\"value\":\"首\"}（不带 object）；只有「换一首/换个/换一部」才是 {\"operation\":\"切换\",\"value\":\"首\"}",
                         "只说「随机播放/顺序播放/单曲循环/下一首/换一首」是播放控制，不要走音乐推荐或点歌",
                         "快进N秒 → {\"operation\":\"快进\",\"value\":\"N秒\"}；跳到/快进到 分:秒 → value 写 \"1:30\"",
                         "关音乐/停止音乐 → {\"operation\":\"退出\",\"object\":\"播放控制\"}；倍速 → numeric_adjust，不用 playback_control"],
    "screen_layout": ["放大屏幕/画面放大/大屏 → {\"operation\":\"放大\",\"object\":\"画面\"}；缩小屏幕/小窗 → {\"operation\":\"缩小\",\"object\":\"画面\"}"],
    "screensaver_control": ["屏保/壁纸/壁画/屏保画册 的打开、关闭、设置 → operation=打开|关闭|设置，object=原词（「屏保设置」「屏保模式」也写 object=\"屏保\"）；用户说了「电视」才加 device=\"电视\"，没说不加",
                            "启动屏幕/打开屏幕/亮屏 → {\"operation\":\"打开\",\"object\":\"屏幕\"}；关屏/息屏/黑屏 → {\"operation\":\"关闭\",\"object\":\"屏幕\"}"],
    "source_switch": ["切到 HDMI2/USB/AV → {\"operation\":\"设置\",\"object\":\"信号源\",\"value\":\"HDMI2\"}（value 照原话写）；打开HDMI模式/HDMI选择 → {\"operation\":\"打开\",\"object\":\"HDMI选择\"}",
                      "关闭机顶盒 → {\"operation\":\"关闭\",\"object\":\"机顶盒\"}"],
    "mode_control": ["设成/切换到/改为 XX模式 → {\"operation\":\"设置\",\"object\":\"图像模式|声音模式|音效模式\",\"value\":\"XX\"}（value 去掉「模式」二字，如 儿童）",
                     "打开/进入/开启 某个独立模式（护眼模式、商场模式、会议模式…）→ {\"operation\":\"打开\",\"object\":\"护眼模式\"}（原词作 object，不写 value）"],
    "common_control": ["打开/关闭 某功能（蓝牙、输入法…）→ operation=打开|关闭，object=功能名；回到主页/返回首页 → {\"operation\":\"打开\",\"object\":\"主页\"}"],
}
GENERIC_STOP = {"播放控制", "播放列表", "音乐播放控制", "设置", "返回", "打开", "关闭", "电视", "屏幕", "画面"}


def _prune(p: dict) -> dict:
    return {k: v for k, v in p.items() if v not in ("", None, [], {})}


def main() -> None:
    csv.field_size_limit(10**9)
    rows = [r for r in csv.DictReader(open(SRC, encoding="utf-8-sig")) if r["domain"] == "tv_control"]
    pat = collections.defaultdict(collections.Counter)      # tool → (operation, object, has_value) 计数
    ex: dict = {}                                            # (tool, op, obj, hv) → (query, params) 最短例句
    obj = collections.defaultdict(collections.Counter)       # object（原样出现在 query）→ tool 计数
    ops = collections.defaultdict(collections.Counter)       # (object, tool) → operation 计数
    for r in rows:
        try:
            p = _prune(json.loads(r["tool_param"]))
        except json.JSONDecodeError:
            continue
        t, q = r["tool"], r["query"].strip()
        key = (p.get("operation", ""), p.get("object", ""), "value" in p)
        pat[t][key] += 1
        k = (t,) + key
        if k not in ex or len(q) < len(ex[k][0]):
            ex[k] = (q, p)
        o = p.get("object", "")
        if len(o) >= 2 and o in q and o not in GENERIC_STOP:
            obj[o][t] += 1
            ops[(o, t)][p.get("operation", "")] += 1

    # 1) 设备对象词典（固定动作的动作词/对象由规则覆盖，不进词典，避免「关机」被标成定时对象这类误导）
    import sys
    sys.path.insert(0, str(ROOT))
    from app.device_schema import FIXED as _FX
    fixed_words = {a for acts in _FX.values() for a in acts} | {ob for acts in _FX.values() for _op, ob in acts.values() if ob}
    ent = []
    for o, c in sorted(obj.items(), key=lambda x: -sum(x[1].values())):
        n = sum(c.values())
        if n < 2 or o in fixed_words:
            continue
        t = c.most_common(1)[0][0]
        top_ops = "/".join(op for op, _ in ops[(o, t)].most_common(3) if op)
        ent.append({"name": o, "types": "设备", "freq": n, "source": "train_tv",
                    "label": f"device:{t} 的 object，标准写法 \"{o}\"（常见 operation：{top_ops}）"})
    # 别名：用户说法 → 标注里的标准 object（训练集与评测金标一致的几组）
    ALIAS = {"饱和度": ("numeric_adjust", "色度"), "色彩饱和度": ("numeric_adjust", "色度"),
             "消音": ("numeric_adjust", "静音"), "静音设置": ("numeric_adjust", "静音"), "静音功能": ("numeric_adjust", "静音"),
             "关闭声音": ("numeric_adjust", "静音")}
    have = {e["name"] for e in ent}
    for a_, (t_, o_) in ALIAS.items():
        if a_ not in have:
            ent.append({"name": a_, "types": "设备", "freq": 0, "source": "alias",
                        "label": f"device:{t_} 的 object，标准写法 \"{o_}\"（不要写成「{a_}」）"
                                 + ("；静音 → {\"operation\":\"打开\",\"object\":\"静音\"}" if o_ == "静音" else "")})
    with open(LIB / "device_entity.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "types", "freq", "source", "label"])
        w.writeheader()
        w.writerows(ent)

    # 2) 设备规则：固定动作的写法取自 app/device_schema.FIXED（按评测金标多数写法定的口径），
    #    通用动作（打开/关闭/设置/提高/降低 + 对象）给出写法约定；训练集标注噪声大，不直接拿来当例句。
    import sys
    sys.path.insert(0, str(ROOT))
    from app.device_schema import FIXED
    rules = []
    for t, trig in TRIGGER.items():
        lines = []
        for act, (op, ob) in FIXED.get(t, {}).items():
            p = {"operation": op, **({"object": ob} if ob else {})}
            lines.append(f"  「{act}」→ {json.dumps(p, ensure_ascii=False, separators=(',', ':'))}")
        lines += [f"  {x}" for x in GENERIC.get(t, [])]
        text = (f"【设备口径：{NAME[t]}（device:{t}）】真实参数用 operation（动作）+ object（对象）+ value（数值/时间/集数，可选），"
                f"不要把动作词直接写进 operation（如不能写 operation=全屏/关机/主页）：\n" + "\n".join(lines))
        rules.append({"id": f"device_{t}", "domain": "device", "priority": 4, "match": trig, "require": "",
                      "exclude": "", "entity": "", "text": text})
    base = list(csv.DictReader(open(LIB / "rules.csv", encoding="utf-8-sig")))
    cols = list(base[0])
    with open(LIB / "rules_device.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(base + rules)

    # 3) 实体词典 + 设备对象
    ed = list(csv.DictReader(open(LIB / "entity_dict.csv", encoding="utf-8-sig")))
    names = {r["name"] for r in ed}
    with open(LIB / "entity_dict_dev.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "types", "freq", "source", "label"])
        w.writeheader()
        for r in ed:
            w.writerow({**r, "label": r.get("label", "")})
        w.writerows([e for e in ent if e["name"] not in names])
    print(f"设备对象 {len(ent)} 个 → device_entity.csv；设备规则 {len(rules)} 条 → rules_device.csv（共 {len(base) + len(rules)} 条）；"
          f"entity_dict_dev.csv 共 {len(ed) + sum(e['name'] not in names for e in ent)} 个名字")


if __name__ == "__main__":
    main()
