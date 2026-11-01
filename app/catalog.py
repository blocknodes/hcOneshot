"""工具目录：读本地编译产物 `catalog.full.json`，渲染成【精简参数块】供一把出 prompt 注入。

**自包含**：运行期只读本地文件，**不依赖任何其他项目**（不起 hcTools、不拉端点）。
产物由 `tools/build_catalog.py` 离线编译自 hcSlime/hcTools 的 schema.json。

为什么需要精简
--------------
各工具完整 schema 共 56 个，含散文 description 时约 126k 字符 ≈ 36k token；
一把出要把「所有工具」塞进一个 prompt，必须瘦身。实测只保留
「参数名 / 类型 / 必填 / enum / 子字段」后压到 ~5.3k 字符 ≈ 1.5k token（4%），
再叠加各工具的用途说明（选工具要靠它，约 11k token），一把出 prompt 约 13k token。

渲染形态（每工具一段）：
    vod:vod_search_all — 影视组合式搜索与起播（search/play）
      必填: query, action, retext
      参数: query(布尔树) action[search|play] retext sort系列/集数/起播秒
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any

from . import config

logger = logging.getLogger("hcOneshot.catalog")

_LOCK = threading.Lock()
_CATALOG: list[dict[str, Any]] | None = None   # None=未拉取；[]=拉取失败/空
_TS: float = 0.0


def _load_file(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        logger.warning("目录文件不存在：%s（先跑 tools/build_catalog.py 生成）", path)
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        tools = data.get("tools") if isinstance(data, dict) else data
        # 缺 id/tool/domain 的行丢弃：render_oneshot_block/domain_of_tool 以 domain 分组与判重，
        # 带上无域行会在渲染时 KeyError → 500（schema 变动即触发）。
        return [t for t in (tools or [])
                if isinstance(t, dict) and t.get("id") and t.get("tool") and t.get("domain")]
    except Exception as exc:  # noqa: BLE001
        logger.warning("目录文件读取失败 %s：%s", path, exc)
        return []


def _stale() -> bool:
    if _CATALOG is None:
        return True
    if _CATALOG:
        return False
    return time.time() - _TS > config.get_settings().catalog_empty_ttl


def _load(force: bool = False) -> list[dict[str, Any]]:
    global _CATALOG, _TS
    if not force and not _stale():
        return _CATALOG
    with _LOCK:
        if not force and not _stale():
            return _CATALOG
        _CATALOG = _load_file(Path(config.get_settings().catalog_file))
        _TS = time.time()
        return _CATALOG


def tools() -> list[dict[str, Any]]:
    """当前工具目录（可能为空）。"""
    return _load()


def valid_tools_for(domain: str) -> set[str]:
    """某域下的合法工具名集合（工具校验用）。"""
    return {t["tool"] for t in _load() if t.get("domain") == domain}


def param_names_for(domain: str, tool: str) -> set[str]:
    """某工具的参数名集合（用于把模型外置到顶层的参数折回 params）。"""
    for t in _load():
        if t.get("domain") == domain and t.get("tool") == tool:
            props = (t.get("parameters") or {}).get("properties")
            if isinstance(props, dict):
                return set(props)
            return set(t.get("required") or [])
    return set()


def domain_of_tool(tool: str) -> str:
    """由工具名反推唯一域；跨域重名（如 fan_knowledge_agent）或未知则返回 ""。

    56 工具里 49 个名字全局唯一，parser 可据此在模型漏写 d 时补全，
    避免把已选对的工具白白清空（fan 类 7 域重名，无法唯一判定，仍交下游）。
    """
    if not tool:
        return ""
    doms = {t["domain"] for t in _load() if t.get("tool") == tool}
    return next(iter(doms)) if len(doms) == 1 else ""


def save_catalog_file(path: Path | None = None) -> Path:
    """把当前内存目录落盘（训练数据对齐/手工维护用；常规编译见 tools/build_catalog.py）。"""
    settings = config.get_settings()
    path = path or Path(settings.catalog_file)
    data = {"tools": _load(force=True)}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 精简：JSON Schema → 「名字[枚举]」短记号
# ---------------------------------------------------------------------------
def _unwrap(node: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
    """解 $$ref / allOf 包裹，拿到真实 node。"""
    if not isinstance(node, dict):
        return {}
    ref = node.get("$$ref") or node.get("$ref")
    if isinstance(ref, str):
        name = ref.split("/")[-1]
        target = defs.get(name)
        if isinstance(target, dict):
            merged = dict(target)
            merged.update({k: v for k, v in node.items() if k not in ("$$ref", "$ref")})
            return _unwrap(merged, defs)
    if isinstance(node.get("allOf"), list) and node["allOf"]:
        merged = {}
        for sub in node["allOf"]:
            merged.update(_unwrap(sub, defs))
        merged.update({k: v for k, v in node.items() if k != "allOf"})
        return merged
    return node


def _brief_param(name: str, node: dict[str, Any], defs: dict[str, Any],
                 required: bool, depth: int = 0) -> str:
    """单个参数的短记号：name[enum] / name(type) / name{子字段}。"""
    node = _unwrap(node, defs)
    if not node:
        return name
    enum = node.get("enum")
    if enum:
        return f"{name}[{'|'.join(str(x) for x in enum)}]"
    typ = node.get("type", "")
    if typ == "array":
        items = _unwrap(node.get("items", {}) or {}, defs)
        sub = ",".join(list((items.get("properties") or {}).keys())) if isinstance(items, dict) else ""
        return f"{name}(数组:{sub})" if sub else f"{name}(数组)"
    props = node.get("properties")
    # properties 必须是 dict 才当子键表；上游 schema 若给成 list/str，这里静默降级为
    # 「无子键」而非 .values()/.items() 抛 AttributeError（引擎在 try 之外渲染目录块 → 会 500）。
    if isinstance(props, dict) and props and depth < 2:
        keys = list(props)
        # 若所有子键【同构】（如 sort 的 rate/hot/new/play 均为 {order: asc|desc}），
        # 则把公共形状补进记号 `{k1,k2:形状}` —— 否则内层枚举/结构被吃掉，
        # 模型只从锚例见过一种写法，换子键/取值只能靠猜（sort 正是边界槽位之一）。
        shape = _common_shape([_brief_param("", v, defs, False, depth + 1)
                               for v in props.values()])
        # 曾用 keys[:8] 静默截断。当前全库对象参数最多 4 个子键（sort），从未触发；
        # 但为防未来 schema 膨胀变成静默丢失，宁可显式标 `+N`。顶层参数（如 music 的
        # 12 个）由 _param_line 全量遍历，本就不受影响。
        head = ",".join(str(k) for k in keys[:_PARAM_KEY_LIMIT])
        if len(keys) > _PARAM_KEY_LIMIT:
            head += f",+{len(keys) - _PARAM_KEY_LIMIT}"
        if shape:
            # 「各」标明是【每个子键的共同形状】，避免 `{...,k:{...}}` 被误读成又一个子键
            return f"{name}{{{head}，各{shape}}}"
        return f"{name}{{{head}}}"
    if typ:
        return f"{name}({typ})"
    return name


def _common_shape(rendered: list[str]) -> str:
    """子键渲染串全同 → 返回该公共形状（调用方以空名前缀渲染，故无前导名）；否则 ""。"""
    if not rendered or any(r != rendered[0] for r in rendered):
        return ""
    return rendered[0]


def _param_line(schema: dict[str, Any]) -> tuple[str, str, str]:
    """返回 (必填记号串, 可选记号串, 是否为空)。"""
    if not isinstance(schema, dict):
        return "", "", "空"
    defs = schema.get("definitions") or schema.get("$defs") or {}
    top = _unwrap(schema, defs)
    props = top.get("properties")
    if not isinstance(props, dict):
        props = {}
    req = list(top.get("required") or schema.get("required") or []) if isinstance(schema, dict) else []
    if not props:
        return "", "", "无参数"
    req_set = set(req)
    required = [_brief_param(k, v, defs, True) for k, v in props.items() if k in req_set]
    optional = [_brief_param(k, v, defs, False) for k, v in props.items() if k not in req_set]
    return " ".join(required), " ".join(optional), ""


def _squeeze(text: str, limit: int) -> str:
    return " ".join((text or "").replace("\\n", " ").split())[:limit]


# 选工具需要的只是「这个工具做什么」；hcTools 的 brief 后半句常是「不适用场景/提参约束」
# 等长段，一把出 prompt 里不必全带。但首句偶尔极短（形如「XX知识问答（fan_knowledge_agent）。」），
# 那就续上下一句凑够信息量。
_BRIEF_LIMIT = 90
_BRIEF_MIN = 20            # 阈值按「去掉工具名括号后」的有效字数算
# 对象参数最多列多少个子键。当前全库最多 12（music 三工具），设 32 以容纳之、
# 又不至于让异常膨胀的 schema 撑爆目录块（超出部分标 `+N` 而非静默丢弃）。
_PARAM_KEY_LIMIT = 32


def _brief_of(t: dict[str, Any]) -> str:
    # 目录块裁剪口径只在此处：优先用原始 description（api 源端已不再截断），
    # 旧 catalog 只有 brief 时兼容之。选工具只需「做什么」，故取首句；过短则续句。
    raw = t.get("description") or t.get("brief") or ""
    text = " ".join(raw.replace("\\n", " ").split())
    if not text:
        return ""
    out = ""
    for seg in text.split("。"):
        if not seg.strip():
            # 原文以「。」结尾时 split 会多出一个空段；若照旧追加「。」会拼出「。。」
            # （实测 device 的 power/network/common_control 三条中招）。
            continue
        out = (out + seg + "。") if out else (seg + "。")
        # 工具名括号（如「（fan_knowledge_agent）」）不算信息量
        if len(re.sub(r"（[^）]*）|\([^)]*\)", "", out)) >= _BRIEF_MIN:
            break
    return out[:_BRIEF_LIMIT]


def render_tool_block(t: dict[str, Any]) -> str:
    """单工具的 prompt 段：id — 用途首句 / 必填 / 参数记号。"""
    req, opt, empty = _param_line(t.get("parameters") or {})
    lines = [f"- {t['id']}：{_brief_of(t)}"]
    if empty == "无参数":
        lines.append("    参数: （本工具无需参数）")
    else:
        if req:
            lines.append(f"    必填: {req}")
        if opt:
            lines.append(f"    可选: {opt}")
    return "\n".join(lines)


def render_oneshot_block() -> str:
    """按域分组渲染全部工具（一把出 prompt 的目录段）。空目录返回空串。"""
    items = _load()
    if not items:
        return ""
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for t in items:
        by_domain.setdefault(t["domain"], []).append(t)
    lines: list[str] = []
    for dom in sorted(by_domain):
        lines.append(f"【{dom}】")
        for t in by_domain[dom]:
            lines.append(render_tool_block(t))
    return "\n".join(lines)


def block_stats() -> dict[str, int]:
    """目录块体量统计（字符数），供调试/训练对齐核对 token 预算。"""
    block = render_oneshot_block()
    items = _load()
    total_schema = sum(len(json.dumps(t.get("parameters") or {}, ensure_ascii=False)) for t in items)
    return {
        "tools": len(items),
        "block_chars": len(block),
        "full_schema_chars": total_schema,
        "est_block_tokens": int(len(block) / 2.5),      # 中英混排粗估
        "est_full_schema_tokens": int(total_schema / 3.5),
    }
