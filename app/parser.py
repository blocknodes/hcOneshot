"""解析一把出输出 → [OneIntent]。

容错链：直接 json → 抠 ```json 块 → raw_decode 扫描 → 轻度修复（去尾逗号）。
逐条校验并**降级而非丢弃**：域非法→清空 d；工具不在该域目录→清空 tool；params 非 dict→置空。
"""
from __future__ import annotations

import json
import logging
from typing import Any

from . import catalog
from .models import DOMAINS, OneIntent

logger = logging.getLogger("hcOneshot.parser")

# 条目的「信封」键：即便某工具恰有同名参数，也不从顶层折回 params
# （避免把 q/d/tool 这类元信息误当参数）。别名一并列出。
_RESERVED_TOP = frozenset({
    "q", "query", "d", "domain", "tool", "toolName", "vt",
    "params", "parameters", "dep", "depends_on_previous",
})


def _repair(t: str) -> str:
    """去尾逗号（`,]` / `,}`），**仅在字符串外生效**。

    旧实现 `re.sub(r",\\s*([\\}\\]])", ...)` 会穿透字符串：query/retext 装用户原话，
    一旦含 `,]`/`,}` 就被静默删字符（`{"a":"x,}"}` → `{"a":"x}"}`）——解析"成功"
    但参数已错，比解析失败更危险。这里做括号/引号感知的单遍扫描，只剔真尾逗号。
    """
    s = t.strip()
    out: list[str] = []
    in_str = False
    esc = False
    for i, ch in enumerate(s):
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            continue
        if ch == ",":
            # 前瞻：逗号后（跳过空白）紧跟 ] 或 } → 是尾逗号，丢弃
            j = i + 1
            while j < len(s) and s[j] in " \t\r\n":
                j += 1
            if j < len(s) and s[j] in "]}":
                continue
        out.append(ch)
    return "".join(out)


def parse_json_loose(text: str) -> Any:
    """宽松解析：直接 → 修复 → 抠 ```json 块 → raw_decode 扫描容器。"""
    if not isinstance(text, str) or not text.strip():
        return None
    t = text.strip()
    for cand in (t, _repair(t)):
        try:
            return json.loads(cand)
        except Exception:  # noqa: BLE001
            pass
    # ```json ... ```
    if t.startswith("```"):
        body = t.split("```")[1] if "```" in t[3:] else t[3:]
        # 精确剥语言标记，别用 lstrip("json") —— 那是【字符集】剥离：
        # 'jsonnull'.lstrip('json') == 'ull'（把 null 啃掉两个字符，静默毁值）。
        body = body.strip()
        low = body.lower()
        if low.startswith("json"):
            body = body[4:]
        body = body.strip()
        for cand in (body, _repair(body)):
            try:
                return json.loads(cand)
            except Exception:  # noqa: BLE001
                pass
    # 扫描第一个可解码的 [ 或 { 容器
    dec = json.JSONDecoder()
    for i, ch in enumerate(t):
        if ch not in "[{":
            continue
        try:
            val, _ = dec.raw_decode(t[i:])
            return val
        except json.JSONDecodeError:
            continue
    return None


def _coerce_dep(raw: Any, orig_idx: int) -> int:
    """把模型给的依赖字段归一到【模型原始条序号】(1-start)；无依赖/无法识别返回 0。

    兼容三种写法：
      · 布尔 true（含别名 `depends_on_previous`）→ 依赖【紧邻前一条】=orig_idx-1；
        注意 bool 是 int 子类，必须【先于】int 判，否则 true 会被当成序号 1（错指首条）。
      · 数字（整数、整数型浮点、纯数字串）→ 即该序号。
      · 其余（false/0/空/乱值）→ 0。
    只做归一；越界/错位校验由 parse_oneshot 的序号重映射统一处理。
    """
    if isinstance(raw, bool):
        return (orig_idx - 1) if raw and orig_idx > 1 else 0
    if isinstance(raw, int):
        return raw if raw > 0 else 0
    if isinstance(raw, float) and raw.is_integer():
        return int(raw) if raw > 0 else 0
    if isinstance(raw, str):
        s = raw.strip()
        return int(s) if s.isdigit() else 0
    return 0


def _coerce_list(v: Any) -> list[dict] | None:
    """归一成 list[dict]：兼容 dict 包裹（{"plan":[...]} / 单条 dict）。"""
    if isinstance(v, dict):
        for key in ("plan", "intents", "steps", "data"):
            if isinstance(v.get(key), list):
                v = v[key]
                break
        else:
            if "q" in v or "query" in v:
                v = [v]
    if not isinstance(v, list):
        return None
    return [x for x in v if isinstance(x, dict)]


def parse_oneshot(text: str) -> tuple[list[OneIntent], str]:
    """返回 (intents, error)。error 非空表示整体解析失败。"""
    value = parse_json_loose(text)
    items = _coerce_list(value)
    if items is None:
        return [], "无法解析为 JSON 数组"

    out: list[OneIntent] = []
    model_idx: list[int] = []      # out[k] 对应模型原始输出里的第 model_idx[k] 条（1-start）
    for i, item in enumerate(items, start=1):
        q = item.get("q") or item.get("query")
        if not isinstance(q, str) or not q.strip():
            continue
        dom = str(item.get("d") or item.get("domain") or "").strip()
        dom = dom if dom in DOMAINS else ""
        tool = str(item.get("tool") or item.get("toolName") or "").strip()
        # 目录块按 `域:工具` 渲染，模型常把整串照抄进 tool；兼容 "vod:vod_search_all"
        # 与裸 "vod_search_all"，并让前缀顺带补全 d（vt 含域）。
        if ":" in tool:
            prefix, _, bare = tool.partition(":")
            if prefix in DOMAINS:
                dom = dom or prefix
                tool = bare.strip()
        # 模型漏写 d、但工具名全局唯一时可反推域（49/56 工具如此），
        # 否则已选对的工具会被下面「无域即清空」白白丢掉。
        if tool and not dom:
            dom = catalog.domain_of_tool(tool)
            if dom:
                logger.info("据唯一工具名补域: tool=%s → d=%s", tool, dom)
        # 工具校验：非法（不在该域目录）→ 清空 tool，保留 d 交下游选
        if tool and dom and tool not in catalog.valid_tools_for(dom):
            logger.info("工具不在目录，降级清空: d=%s tool=%s", dom, tool)
            tool = ""
        if tool and not dom:
            tool = ""
        params = item.get("params") or item.get("parameters")
        if not isinstance(params, dict):
            params = {}
        else:
            params = dict(params)
        # 模型常把某个参数【外置到顶层】（与 q/d/tool 同级），如
        # `{...,"params":{...},"sort":{...}}` —— 锚例把 sort 写在 params 内，
        # 但实测模型会提到顶层；只认 params 会静默丢参。此处把「属于本工具参数名」
        # 的顶层键折回 params（内层已有同名者以 params 内为准，不覆盖）。
        if tool:
            known = catalog.param_names_for(dom, tool)
            for k in list(item):
                if k in known and k not in params and k not in _RESERVED_TOP:
                    params[k] = item[k]
                    logger.info("外层参数折回 params: tool=%s key=%s", tool, k)
        # dep 取「模型原始条序号」；别名 depends_on_previous 为布尔 true 时=紧邻前一条。
        # 注意别用 `or`：dep=false/0 会被 or 吞掉再去取别名，幸好二者同义；显式判 None 更稳。
        dep_raw = item.get("dep")
        if dep_raw is None:
            dep_raw = item.get("depends_on_previous")
        dep_on = _coerce_dep(dep_raw, i)
        out.append(OneIntent(
            q=q.strip(), d=dom, tool=tool,
            vt=f"{dom}:{tool}" if dom and tool else "",
            params=params, dep=dep_on,
        ))
        model_idx.append(i)        # 记下本条在模型原始输出里的序号
    # dep 后校验：dep 按【模型原始输出序号】给，而上面 continue 可能丢条（如 q 空），
    # 使 out 序号与模型序号错位。故先把 dep 重映射到 out 序号，再要求它指向更早条目；
    # 越界/自指/前指/指向已丢条目者一律清零（退化为并行），避免下游按 dep 串链路时错配。
    orig_to_out = {orig: k for k, orig in enumerate(model_idx, start=1)}   # 模型序号→out序号
    for k, it in enumerate(out, start=1):
        if not it.dep:
            continue
        mapped = orig_to_out.get(it.dep, 0)          # 被丢弃的依赖条 → 0
        if not (1 <= mapped < k):
            logger.info("dep 无效，清零: out_idx=%d orig_dep=%d", k, it.dep)
            mapped = 0
        it.dep = mapped
    if not out:
        # 区分「模型本就给了空计划」与「给了条目但全被降级丢弃」：
        # 前者是合法结果（纯闲聊 → []，见 prompt 规则 3），error 应为空，
        # 否则下游会把正确的闲聊响应当失败；后者才是真异常（条目全无效），保留 error。
        if items:
            return [], "解析成功但无有效条目"
        return [], ""
    return out, ""
