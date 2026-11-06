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

# 系统代填参数：不在目录里展示、模型不必填、评测集也不标（与清洗后的 7 域评测集口径一致）。
# retext = 用户原句，由 engine._steps_from_batch 按工具 schema 是否含该参数自动补。
SYSTEM_FILLED_PARAMS = frozenset({"retext"})
# 按工具的系统代填参数：值就是用户原话（自然语言整句透传），模型不填，引擎补用户原句。
#   audio_search.query      —— schema 明确「原样透传用户原始语句」；
#   fan_knowledge_agent.messages —— 引擎 rebuild_fan_params 本就由用户原句重建。
_TOOL_FILLED: dict[str, frozenset[str]] = {
    "audio_search": frozenset({"query"}),
    "fan_knowledge_agent": frozenset({"messages"}),
}


def system_filled(tool: str) -> frozenset[str]:
    """某工具由系统代填（不展示给模型、模型不填、评测不比）的参数名。"""
    return SYSTEM_FILLED_PARAMS | _TOOL_FILLED.get(tool, frozenset())
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


def _param_line(schema: dict[str, Any], drop: frozenset[str] = SYSTEM_FILLED_PARAMS) -> tuple[str, str, str]:
    """返回 (必填记号串, 可选记号串, 是否为空)。"""
    if not isinstance(schema, dict):
        return "", "", "空"
    defs = schema.get("definitions") or schema.get("$defs") or {}
    top = _unwrap(schema, defs)
    props = top.get("properties")
    if not isinstance(props, dict):
        props = {}
    req = list(top.get("required") or schema.get("required") or []) if isinstance(schema, dict) else []
    # 上游 schema 笔误：把 required 写进了 properties（vod_person_search、3 个 sports 卡片工具），
    # 旧渲染成「可选: person_name(数组) required」——必填信息丢失、还多出一个假参数。
    if isinstance(props.get("required"), list):
        req += [x for x in props["required"] if isinstance(x, str)]
        props = {k: v for k, v in props.items() if k != "required"}
    # 系统代填参数（retext=用户原句，由引擎拼 step 时补）不进目录：模型不填、评测不比。
    props = {k: v for k, v in props.items() if k not in drop}
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


def _clean(text: str) -> list[str]:
    """schema 文本 → 行列表（不截断）：字面量 `\\n` 还原为换行、`\\"` 还原为引号，行内空白压成单空格。"""
    raw = (text or "").replace("\\n", "\n").replace('\\"', '"').replace("。。", "。")
    return [" ".join(ln.split()) for ln in raw.splitlines() if ln.strip()]


_DESC_REWRITES: list[tuple[re.Pattern[str], str]] = [
    # 上游描述里对「系统代填参数」的说明与本目录口径冲突（模型不填 retext / fan.messages），
    # 只改这几处措辞，其余原文全量保留。
    (re.compile(r'^参数：\{"messages":[^}]*\}。?$'), ""),            # 7 个 fan
    (re.compile(r"除 ?retext ?外所有业务槽位"), "所有业务槽位"),       # music 三工具
    (re.compile(r"传入结构仅有 ?retext"), "无需传参（系统自动补用户原话）"),  # music_song_semantic_search
]


def _desc_of(t: dict[str, Any]) -> list[str]:
    """目录块用的工具描述：原文全量（适用/不适用场景、提参约束、时间格式都在后半段）。

    _brief_of 只留首句（≤90 字），曾把字段清单、时间格式、工具边界一并截掉
    （如 educ_search_all 断在「二级分类(c」），这里不再截断。
    """
    out = []
    for ln in _clean(t.get("description") or t.get("brief") or ""):
        for pat, rep in _DESC_REWRITES:
            ln = pat.sub(rep, ln)
        if ln.strip():
            out.append(ln)
    return out


def _schema_parts(schema: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    defs = (schema.get("definitions") or schema.get("$defs") or {}) if isinstance(schema, dict) else {}
    top = _unwrap(schema, defs) if isinstance(schema, dict) else {}
    props = top.get("properties")
    return (props if isinstance(props, dict) else {}), defs


_COND_LABEL = {"ExactFieldCondition": "精确", "StatusFieldCondition": "状态",
               "RangeFieldCondition": "范围", "PlaybackControlFieldCondition": "媒资"}


def _field_options(node: dict[str, Any]) -> list[tuple[str, str]]:
    """一个 *FieldCondition 定义里 field 的候选：[(字段名, 说明)]。"""
    out: list[tuple[str, str]] = []
    fld = ((node.get("properties") or {}).get("field") or {}) if isinstance(node, dict) else {}
    for opt in fld.get("oneOf") or []:
        if isinstance(opt, dict) and "const" in opt:
            out.append((str(opt["const"]), opt.get("description") or ""))
    for v in fld.get("enum") or []:
        out.append((str(v), ""))
    if "const" in fld:
        out.append((str(fld["const"]), fld.get("description") or ""))
    for sub in (node.get("oneOf") or []) if isinstance(node, dict) else []:   # PlaybackControl 形态
        f = ((sub.get("properties") or {}).get("field") or {}) if isinstance(sub, dict) else {}
        if "const" in f:
            out.append((str(f["const"]), f.get("description") or ""))
    return out


# 精简版（HC_CATALOG_COMPACT=1）的去重登记表：文本 → 首次出现的工具 id。
# 只合并【逐字相同】的说明，后续出现处写「同 <工具id>」引用，不删任何信息。
Seen = dict[str, str]


def _ref(seen: Seen | None, owner: str, key: str) -> str:
    """登记 key；已被别的工具登记过则返回其 id（调用方据此改写为引用），否则返回空串。"""
    if seen is None:
        return ""
    first = seen.get(key)
    if first and first != owner:
        return first
    seen.setdefault(key, owner)
    return ""


def _query_doc(defs: dict[str, Any], seen: Seen | None = None, owner: str = "") -> list[str]:
    """检索树参数 query 的完整说明：节点形态 + 各类叶子条件规则 + 全部可用 field 及其说明。

    seen 非空（精简版）时：条件总述、单个字段说明与先前工具逐字相同的，改写为「同 <工具id>」。
    """
    conds = [k for k in defs if k.endswith("FieldCondition")]
    if not conds:
        return []
    lines = ['· query：检索树。布尔节点 {"and":[…]} / {"or":[…]} / {"not":[…]} 可任意嵌套；'
             '叶子为下列条件之一（field 只能取列出的字段名）。']
    for k in sorted(conds, key=lambda x: list(_COND_LABEL).index(x) if x in _COND_LABEL else 99):
        node = defs[k] if isinstance(defs[k], dict) else {}
        label = _COND_LABEL.get(k, k)
        shape = {"精确": '{"field","value"} 或 {"field","values":[…],"operator":"and|or"}',
                 "状态": '{"field","value"}', "范围": '{"field","from","to"}',
                 "媒资": '{"field","value"}'}.get(label, "")
        lines.append(f"  [{label}条件 {shape}]")
        cdesc = _clean(node.get("description") or "")
        first = _ref(seen, owner, "cond\x00" + "\n".join(cdesc)) if cdesc else ""
        if first:
            lines.append(f"    （本条件说明同 {first}）")
        else:
            lines += [f"    {ln}" for ln in cdesc]
        for name, desc in _field_options(node):
            d = _clean(desc)
            first = _ref(seen, owner, f"field\x00{name}\x00" + "\n".join(d)) if d else ""
            if first:
                lines.append(f"    - {name}：同 {first}")
                continue
            lines.append(f"    - {name}：{d[0]}" if d else f"    - {name}")
            lines += [f"      {ln}" for ln in d[1:]]
    return lines


def _param_docs(schema: dict[str, Any], drop: frozenset[str],
                seen: Seen | None = None, owner: str = "") -> list[str]:
    """逐参数说明（原文全量）：取值范围/枚举/格式/归一化规则都在这里，如
    common_control.object 的候选对象（「输入法设置」等中文原词）、history 的时间格式。
    seen 非空（精简版）时：与先前工具同名且逐字相同的参数说明改写为「同 <工具id>」。"""
    props, defs = _schema_parts(schema)
    lines: list[str] = []
    for k, v in props.items():
        if k in drop or k == "required" or not isinstance(v, dict):
            continue
        node = _unwrap(v, defs)
        if k == "query" and _query_doc(defs):
            lines += _query_doc(defs, seen, owner)
            continue
        d = _clean(node.get("description") or v.get("description") or "")
        if not d:
            continue
        first = _ref(seen, owner, f"param\x00{k}\x00" + "\n".join(d))
        if first:
            lines.append(f"· {k}：同 {first}")
            continue
        lines.append(f"· {k}：{d[0]}")
        lines += [f"  {ln}" for ln in d[1:]]
    return lines


def _desc_lines(t: dict[str, Any], seen: Seen | None) -> list[str]:
    """工具描述行；精简版下与先前工具逐字相同的【续行】（首行是工具自身用途，恒保留）改写为引用。"""
    desc = _desc_of(t)
    if seen is None or len(desc) < 2:
        return desc
    out = desc[:1]
    for ln in desc[1:]:
        first = _ref(seen, t["id"], "desc\x00" + ln) if len(ln) >= 30 else ""
        out.append(f"{ln.split('：', 1)[0]}：同 {first} 的这一条" if first and "：" in ln[:12]
                   else (f"（此条说明同 {first}）" if first else ln))
    return out


def render_tool_block(t: dict[str, Any], seen: Seen | None = None) -> str:
    """单工具的 prompt 段：id — 完整描述 / 必填 / 可选 / 逐参数说明（均不截断）。

    seen 为精简版的去重登记表（render_oneshot_block 传入）；None=完整版。
    """
    schema = t.get("parameters") or {}
    drop = system_filled(t.get("tool", ""))
    req, opt, empty = _param_line(schema, drop)
    desc = _desc_lines(t, seen)
    lines = [f"- {t['id']}：{desc[0] if desc else ''}"]
    lines += [f"    {ln}" for ln in desc[1:]]
    if empty == "无参数":
        lines.append("    参数: （本工具无需参数）")
    else:
        if req:
            lines.append(f"    必填: {req}")
        if opt:
            lines.append(f"    可选: {opt}")
        lines += [f"    {ln}" for ln in _param_docs(schema, drop, seen, t["id"])]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 极简版（HC_CATALOG_COMPACT=2）：每个工具 = 用途句 + 边界句 + 签名 + 字段名及其短标签/枚举/格式。
# 只按【整句】取舍（以「。」切句），绝不在句中截断；去掉的是例句、辨析长段、后端实现细节。
# ---------------------------------------------------------------------------
_KEEP_SENT = re.compile(r"枚举|可选值|可选范围|格式为|格式化取值|yyyy|固定候选，共|\"0\"=|1=未开始")
_BOUNDARY = re.compile(r"^(不适用场景|禁用场景|边界与禁用场景|工具边界|关联工具|单场 vs)")


def _sents(text: str) -> list[str]:
    out = []
    for seg in re.split(r"(?<=。)", text):
        seg = seg.strip()
        if seg:
            out.append(seg)
    return out


def _min_desc(lines: list[str]) -> list[str]:
    """用途：首行的第 1 整句（过短则续第 2 句）；另保留「不适用/禁用/工具边界」整行（选工具的关键）。"""
    if not lines:
        return []
    ss = _sents(lines[0])
    head = ss[0] if ss else lines[0]
    if len(re.sub(r"（[^）]*）|\([^)]*\)", "", head)) < 20 and len(ss) > 1:
        head += ss[1]
    out = [head]
    rest = lines[1:] + ["".join(ss[1:])]
    for ln in rest:
        for sent in _sents(ln) if not _BOUNDARY.match(ln) else [ln]:
            if _BOUNDARY.match(sent) or "工具边界" in sent[:6]:
                out.append(sent)
    # 首行里内嵌的「工具边界：…」整句
    for sent in ss[1:]:
        if sent.startswith("工具边界"):
            out.append(sent)
    return list(dict.fromkeys(out))


def _min_label(desc: str) -> str:
    """字段/参数短说明：第 1 整句 + 含枚举/可选值/格式的整句。"""
    ss = _sents(" ".join(_clean(desc)))
    if not ss:
        return ""
    keep = [ss[0]] + [x for x in ss[1:] if _KEEP_SENT.search(x)]
    return "".join(dict.fromkeys(keep))


def _min_query_doc(defs: dict[str, Any]) -> list[str]:
    conds = [k for k in defs if k.endswith("FieldCondition")]
    if not conds:
        return []
    lines = ['· query：检索树，布尔节点 {"and"|"or"|"not":[…]} 可嵌套；叶子 field 只能取下列字段。']
    for k in sorted(conds, key=lambda x: list(_COND_LABEL).index(x) if x in _COND_LABEL else 99):
        node = defs[k] if isinstance(defs[k], dict) else {}
        label = _COND_LABEL.get(k, k)
        shape = {"精确": '{"field","value"} 或 {"field","values":[…],"operator":"and|or"}',
                 "状态": '{"field","value"}', "范围": '{"field","from","to"}',
                 "媒资": '{"field","value"}'}.get(label, "")
        items = []
        for name, desc in _field_options(node):
            lb = _min_label(desc)
            items.append(f"{name}（{lb.rstrip('。')}）" if lb else name)
        if label == "范围":                              # 范围条件的格式要点在条件总述里
            fmt = [x for x in _sents(" ".join(_clean(node.get("description") or "")))
                   if re.search(r"格式|yyyy|\"\*\"", x)][:2]
            lines.append(f"  [{label} {shape}] " + "；".join(items) + ("  " + "".join(fmt) if fmt else ""))
        else:
            lines.append(f"  [{label} {shape}] " + "；".join(items))
    return lines


def render_tool_block_min(t: dict[str, Any], seen: Seen | None = None) -> str:
    schema = t.get("parameters") or {}
    drop = system_filled(t.get("tool", ""))
    req, opt, empty = _param_line(schema, drop)
    desc = _min_desc(_desc_of(t))
    lines = [f"- {t['id']}：{desc[0] if desc else ''}"] + [f"    {ln}" for ln in desc[1:]]
    if empty == "无参数":
        lines.append("    参数: （本工具无需参数）")
        return "\n".join(lines)
    if req:
        lines.append(f"    必填: {req}")
    if opt:
        lines.append(f"    可选: {opt}")
    props, defs = _schema_parts(schema)
    for k, v in props.items():
        if k in drop or k == "required" or not isinstance(v, dict):
            continue
        if k == "query" and _min_query_doc(defs):
            body = _min_query_doc(defs)
            first = _ref(seen, t["id"], "minq\x00" + "\n".join(body))
            lines.append(f"    · query：同 {first}" if first else "\n".join(f"    {ln}" for ln in body))
            continue
        lb = _min_label(_unwrap(v, defs).get("description") or v.get("description") or "")
        if not lb or "当前工具不使用" in lb:
            continue
        first = _ref(seen, t["id"], f"minp\x00{k}\x00{lb}")
        lines.append(f"    · {k}：同 {first}" if first else f"    · {k}：{lb}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 超简版（HC_CATALOG_COMPACT=3）：一行签名 + 用途首句 + 只保留「模型猜不到的」枚举/候选清单。
# 去掉工具边界句、字段辨析与例句（这部分改由 few-shot 示例提供）；仍按整句取舍，不截断。
# ---------------------------------------------------------------------------
_ENUM_SENT = re.compile(r"枚举|可选值|可选范围|\"0\"=|1=未开始")
_TIME_FORMATS = ("【时间格式】release_time / rate 等日期范围用 yyyyMMdd；sport_time 用 yyyyMMdd，"
                 "sport_time_range 用 \"yyyyMMdd HH:mm:ss\"；各 history 工具的 time 用 "
                 "\"yyyy-MM-dd HH:mm:ss TO yyyy-MM-dd HH:mm:ss\"；范围条件开放端填 \"*\"。")


def _is_list_sent(x: str) -> bool:
    """候选清单句：含 ≥4 个「、」或「/」或「；」分隔项（如 device 的「控制对象，如主页、设置、…」）。"""
    return max(x.count("、"), x.count("/"), x.count("；")) >= 4


def _enum_part(desc: str, lists: bool = False) -> str:
    """只取含枚举/可选值的整句；lists=True 时另取候选清单句。无则空串。"""
    return "".join(x for x in _sents(" ".join(_clean(desc)))
                   if _ENUM_SENT.search(x) or (lists and _is_list_sent(x)))


def _purpose(t: dict[str, Any]) -> str:
    d = _desc_of(t)
    ss = _sents(d[0]) if d else []
    head = ss[0] if ss else ""
    return re.sub(r"[（(]" + re.escape(t.get("tool", "")) + r"[）)]", "", head)   # 去掉与 id 重复的工具名括号


def render_tool_block_l3(t: dict[str, Any], seen: Seen | None = None) -> str:
    schema = t.get("parameters") or {}
    drop = system_filled(t.get("tool", ""))
    req, opt, empty = _param_line(schema, drop)
    sig = "无参数" if empty == "无参数" else "；".join(x for x in (f"必填 {req}" if req else "",
                                                                f"可选 {opt}" if opt else "") if x)
    lines = [f"- {t['id']}：{_purpose(t)}【{sig}】"]
    if empty == "无参数":
        return "\n".join(lines)
    props, defs = _schema_parts(schema)
    for k, v in props.items():
        if k in drop or k == "required" or not isinstance(v, dict):
            continue
        if k == "query":
            conds = [c for c in defs if c.endswith("FieldCondition")]
            if not conds:
                continue
            parts = []
            for c in sorted(conds, key=lambda x: list(_COND_LABEL).index(x) if x in _COND_LABEL else 99):
                for name, desc in _field_options(defs[c] if isinstance(defs[c], dict) else {}):
                    e = _enum_part(desc)
                    parts.append(f"{name}（{e.rstrip('。')}）" if e else name)
            body = "query 字段：" + "，".join(parts)
            first = _ref(seen, t["id"], "l3q\x00" + body)
            lines.append(f"    query 字段同 {first}" if first else f"    {body}")
            continue
        node = _unwrap(v, defs)
        if node.get("enum"):                       # 枚举已在签名 name[a|b|…] 里，不再重复
            continue
        e = _enum_part(node.get("description") or v.get("description") or "", lists=True)
        if not e:
            continue
        first = _ref(seen, t["id"], f"l3p\x00{k}\x00{e}")
        lines.append(f"    {k}：同 {first}" if first else f"    {k}：{e}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 工具索引版（HC_CATALOG_COMPACT=4）：每工具一行「id | 签名 | 用途短句」+ query 字段名；
# 字段说明/候选值/辨析全部交给动态层（few-shot / 工具 RAG）。用途短句为人工撰写（非截断），
# 单测保证 catalog 每个工具都有条目。
# ---------------------------------------------------------------------------
_PURPOSE: dict[str, str] = {
    "audio:audio_search": "有声内容全文检索/起播（听书、评书、相声、广播剧、儿童故事）",
    "audio:audio_history": "有声收听历史（查看或从历史起播）",
    "audio:fan_knowledge_agent": "有声内容知识问答",
    "children:educ_search_all": "少儿媒资结构化检索（动画/动漫/绘本/儿歌/早教，按片名、角色、分类等）",
    "children:educ_fuzzy_search": "少儿模糊/描述性检索（剧情、台词、画面、复杂描述，提不出明确槽位时）",
    "children:educ_relate_search": "找与某少儿作品相似/同类的内容",
    "children:educ_history": "少儿观看历史",
    "children:fan_knowledge_agent": "少儿知识问答（角色、作品属性、百科）",
    "device:numeric_adjust": "数值调节：音量、亮度、对比度、色度、清晰度、分辨率、倍速、刷新率",
    "device:power_control": "开机/关机/重启",
    "device:timer_control": "定时开关机、定时熄屏、查询剩余关机时长",
    "device:source_switch": "信号源切换（HDMI、USB、机顶盒、数字电视等）",
    "device:playback_control": "播放控制：快进快退、跳转、上/下一个、暂停停止退出、重播、循环/随机/顺序播放",
    "device:video_picture_save_share": "视频/图片保存、删除、分享",
    "device:mode_control": "模式切换：图像/声音/音效/护眼/混响/商场等模式",
    "device:screen_layout": "画面布局：分屏、全屏、小屏、缩放",
    "device:screen_lift_rotation": "屏幕旋转、升降",
    "device:demo_control": "效果演示/展示/demo",
    "device:display_control": "画质与显示功能开关（HDR、背光、运动补偿、色彩、护眼等）",
    "device:audio_control": "音质音效功能开关（杜比全景声、声场、按键音等）",
    "device:smart_camera": "摄像头、雷达、AI 视觉功能",
    "device:network_control": "网络功能（WiFi、有线、热点、测速）",
    "device:screensaver_control": "屏保、壁纸、熄屏亮屏、待机睡眠",
    "device:common_control": "其他设备功能兜底（系统设置、输入法、蓝牙、主页、存储、遥控器等）",
    "device:solve_picture_sound_problem_control": "用户只描述画质/声音问题（太暗、偏色、不清楚、没立体声）时的问题处理",
    "device:fan_knowledge_agent": "设备使用/功能知识问答",
    "education:edu_search": "K12 课程结构化检索（年级、学科、教材版本、学期明确）",
    "education:edu_fuzzy_search": "教育模糊检索（知识点、习题、专题、素质课、学习方法等，超出结构化槽位时）",
    "education:fan_knowledge_agent": "教育知识问答（知识点、解题、学习方法）",
    "music:music_song_search": "歌曲检索/播放（有歌名、歌手、专辑、标签等明确槽位）",
    "music:music_song_semantic_search": "想听歌但提不出任何明确槽位（情绪/场景描述），或「X和Y合作的歌」",
    "music:music_song_mv_search": "MV 检索",
    "music:music_ksong_search": "K歌/唱歌/找伴奏",
    "music:music_song_qqmusic_search": "明确指定在 QQ 音乐里搜歌",
    "music:music_song_recommend": "随心听/随便听（无目标个性化播歌）",
    "music:music_song_history": "音乐播放历史/最近播放",
    "music:tvchannel_music_search": "在某个电视频道/电视台上放音乐",
    "music:music_song_favorite_search": "播放我收藏的歌曲/歌单/专辑",
    "music:fan_knowledge_agent": "音乐知识问答（歌手、歌曲属性）",
    "sports:sports_match_search": "体育赛程/对阵/比赛查询与直播跳转",
    "sports:sports_vod_search": "体育视频：录像、回放、集锦、精华、纪录片",
    "sports:sports_rank_search": "积分榜/射手榜/排行榜",
    "sports:sports_team_search": "球队资料（只点名某支球队）",
    "sports:sports_match_reservation": "预约/订阅比赛",
    "sports:sports_match_forecast": "预测某场比赛胜负",
    "sports:sports_match_lineup_search": "已锁定比赛的阵容/首发",
    "sports:sports_match_event_card_search": "已锁定比赛的赛况",
    "sports:sports_match_statistics_card_search": "已锁定比赛的技术统计",
    "sports:fan_knowledge_agent": "体育知识问答（规则、球队球星背景）",
    "vod:vod_search_all": "影视结构化检索/起播（片名、演员、导演、分类、地区、标签等）",
    "vod:vod_fuzzy_search": "影视模糊/描述性检索（剧情、台词、片段、名场面、情绪主题，提不出明确槽位时）",
    "vod:vod_relate_search": "找与某影视作品/演员/导演相似的内容",
    "vod:vod_personalized_search": "按我的喜好/观看记录推荐影视",
    "vod:vod_history": "影视观看历史",
    "vod:vod_person_search": "只报明星名、查人物介绍",
    "vod:fan_knowledge_agent": "影视知识问答（作品/人物属性、剧情疑问）",
}

_INDEX_HEAD = """【参数语法】query 为检索树：叶子 {"field":F,"value":V} 或 {"field":F,"values":[…],"operator":"and|or"}，
范围叶子 {"field":F,"from":A,"to":B}（开放端填 "*"）；布尔节点 {"and":[…]} / {"or":[…]} / {"not":[…]} 可嵌套；field 只能取该工具列出的字段。
sort 形如 {"hot":{"order":"desc"}}（键 rate/hot/new/play）。状态字段值为字符串，如 is_fee "0"=免费 "1"=付费。
【时间格式】release_time 用 yyyyMMdd；sport_time 用 yyyyMMdd，sport_time_range 用 "yyyyMMdd HH:mm:ss"；
各 history 工具的 time 用 "yyyy-MM-dd HH:mm:ss TO yyyy-MM-dd HH:mm:ss"。字段详细取值见下文「相关工具详情 / 相似示例」（若有）。"""


def _sig_l4(schema: dict[str, Any], drop: frozenset[str]) -> str:
    props, defs = _schema_parts(schema)
    req, opt, empty = _param_line(schema, drop)
    if empty == "无参数":
        return "无参"
    req = re.sub(r"sort\{[^}]*\}\}?\}?", "sort", req)
    opt = re.sub(r"sort\{[^}]*\}\}?\}?", "sort", opt)
    out = "；".join(x for x in (f"必填 {req}" if req else "", f"可选 {opt}" if opt else "") if x)
    out = out.replace("(string)", "").replace("(数组)", "[]")              # 类型记号：string 省略，数组记 []
    return re.sub(r"\s+", " ", out).strip()


def render_tool_block_l4(t: dict[str, Any], seen: Seen | None = None) -> str:
    schema = t.get("parameters") or {}
    drop = system_filled(t.get("tool", ""))
    purpose = _PURPOSE.get(t["id"]) or _purpose(t)
    lines = [f"- {t['id']} | {_sig_l4(schema, drop)} | {purpose}"]
    props, defs = _schema_parts(schema)
    if "query" in props and "query" not in drop:
        names = []
        for c in sorted((c for c in defs if c.endswith("FieldCondition")),
                        key=lambda x: list(_COND_LABEL).index(x) if x in _COND_LABEL else 99):
            names += [n for n, _ in _field_options(defs[c] if isinstance(defs[c], dict) else {})]
        if names:
            body = ",".join(dict.fromkeys(names))
            first = _ref(seen, t["id"], "l4q\x00" + body)
            lines.append(f"    query字段同 {first}" if first else f"    query字段：{body}")
    return "\n".join(lines)


def render_oneshot_block(compact: bool | int | None = None) -> str:
    """按域分组渲染全部工具（一把出 prompt 的目录段）。空目录返回空串。

    compact（None 时取配置 HC_CATALOG_COMPACT，默认 0）：
      0 完整版：描述与参数/字段说明原文全量；
      1 精简版：同完整版，但逐字重复的说明改写为「同 <首次出现的工具id>」（信息不丢）；
      2 极简版：每个工具只留用途句、边界句、签名、字段名及短标签/枚举/格式（按整句取舍，不截断）；
      3 超简版：一行签名 + 用途首句 + 枚举/候选清单 + 统一时间格式行（边界/辨析交给 few-shot）；
      4 工具索引版：每工具一行 id|签名|人工用途短句 + query 字段名，外加参数语法/时间格式头
        （候选值与字段说明全部交给动态层，单用会掉点，配合 few-shot / 工具 RAG 使用）。
    """
    level = int(config.get_settings().catalog_compact if compact is None else compact)
    items = _load()
    if not items:
        return ""
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for t in items:
        by_domain.setdefault(t["domain"], []).append(t)
    seen: Seen | None = {} if level >= 1 else None
    render = {2: render_tool_block_min, 3: render_tool_block_l3,
              4: render_tool_block_l4}.get(min(level, 4), render_tool_block)
    lines: list[str] = [_INDEX_HEAD] if level >= 4 else ([_TIME_FORMATS] if level == 3 else [])
    for dom in sorted(by_domain):
        lines.append(f"【{dom}】")
        for t in by_domain[dom]:
            lines.append(render(t, seen))
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
