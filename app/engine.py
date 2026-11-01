"""hcOneshot 编排：**一个 tick ＝ 恰一次 LLM**，直接产出该批 steps（含 tool+params）。

与 hcAgent 的关系
-----------------
hcOneshot 是 hcAgent 的完整替代品，暴露同一 `POST /slowAgent/{poc_id}` 契约，
但去掉「T0 选工具/域 → hcTools 填参」两跳：模型**一次**调用即给出 `q/d/tool/params`
（首轮整份计划；续跑一轮改依赖步）。编排层零语言规则。

  · 首轮（无 toolHistory、无该 device 状态）：一次 LLM 出整份计划
    [{"q","d","tool","params","dep"?}]；无依赖条目全发（并行），有依赖条目等其依赖项执行。
  · 续跑（带真实 toolHistory）：一次 LLM 把当前待执行条目改写成自包含 query，
    **并同时重填 tool+params**（用上一步结果里的实体换掉指代）；
    保留 ReAct 自适应（plan 锁定不漂移，见 docs/query_pipeline.md A5）。
  · 无规则补丁：不拆连接词、不落域词法、不关键词换工具。LLM 失败/输出非法 →
    降级为单条自包含意图（透传原 query，走 fan 兜底），不注入语言规则。

状态：串行计划按 **deviceId** 缓存（换 request 用 deviceId 识别会话，而非 traceId ——
串行 N 步 = N 个 SSE 请求 = N 个不同 traceId，只有 device 稳定才能跨轮推进依赖链）。
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from typing import Any

from . import catalog, fanparams, llm, slots
from .config import get_settings
from .models import OneIntent
from .parser import parse_json_loose, parse_oneshot
from .prompts import build_oneshot_prompt, build_t1_prompt, MT_REWRITE_PROMPT

logger = logging.getLogger("hcOneshot.engine")

# 域名闭合集（供 LLM 输出校验）。qa 不是候选域：落域失败（空 d）由下游直接走泛知识问答。
_DOMAINS = {"vod", "children", "education", "music", "audio", "sports", "device"}

_TURNSLOW_QA_TOOL = "fan_knowledge_agent"
_TURNSLOW_NO_REASON = "NO_DATA"
# reason -> (按域分流的同域模糊兜底)。其余/未知 → fan_knowledge_agent（链路终点）。
# 注意域命名：children 域的工具前缀是 educ_*、education 域是 edu_*，两者是【不同域】。
# 少儿检索（educ_search_all）属 children，故 children 也须有退路，否则它 turnSlow 后
# 会穿到 fan 终点（本该换同域模糊检索 educ_fuzzy_search）。
_TURNSLOW_FUZZY = {
    "NO_SLOT_MAPPING_VALUE": {
        "vod": "vod_fuzzy_search",
        "children": "educ_fuzzy_search",
        "education": "edu_fuzzy_search",
    },
}


# ---------------------------------------------------------------------------
# 弱设备词提示（从 hcAgent multiintent.py 移植）：只生成提示串，不做确定性拆分。
# ---------------------------------------------------------------------------
_DEV_WEAK = [
    "色彩", "画质", "图像", "色温", "原画", "原音", "音画", "氛围", "场景细节",
    "细节增强", "动态范围", "数字艺术",
]


def weak_device_hint(query: str) -> str:
    """检测 query 是否含弱设备词；命中返回提示串（供首轮 prompt 提示 LLM），否则空串。"""
    q = (query or "").strip()
    if not q:
        return ""
    hits = [w for w in _DEV_WEAK if w in q]
    if not hits:
        return ""
    return (f"含弱设备词({'/'.join(hits)})：可能是设备调节，也可能是剧情/美学描述，"
            "请结合上下文判定是否拆成设备意图。")


# ---------------------------------------------------------------------------
# 多轮上下文改写（MT）：逐逻辑移植自 hcAgent/app/engine.py 的 _mt_* 家族。
# 语义：首轮无任何跨轮上下文 → 原句直通（不改写）；只要存在任一上下文
# （进程内 _MT_CONTEXT 或请求 shortMemory）→ 交 LLM 合并历史成自包含 query。
# 历史 = real_q（本进程真正改写/执行的 query 链） + short_a（请求 shortMemory 的应答与候选）。
# 注意：这里【只改写 query】，不改工具/计划——域判定与选工具仍归 T0（一把出）。
# ---------------------------------------------------------------------------
_MT_CONTEXT: dict[str, dict[str, Any]] = {}
_MT_LOCK = threading.Lock()


def _mt_get(device_id: str) -> list[dict[str, str]]:
    """返回该 device 的有效对话历史（每轮 {"q":…,"a":…}），空或超时返回 []。"""
    if not device_id:
        return []
    now = time.time()
    with _MT_LOCK:
        ent = _MT_CONTEXT.get(device_id)
        if ent is not None and now - ent["ts"] > get_settings().mt_ttl:
            _MT_CONTEXT.pop(device_id, None)
            return []
        return [dict(t) for t in ent["turns"]] if ent else []


def _mt_add(device_id: str, turn: dict[str, str]) -> None:
    """把一轮已完成对话 (q, answer) 追加进 device 历史；answer 为空则不追加。"""
    if not device_id:
        return
    with _MT_LOCK:
        ent = _MT_CONTEXT.get(device_id)
        if ent is None:
            ent = {"turns": [], "ts": time.time()}
            _MT_CONTEXT[device_id] = ent
        turns = ent["turns"]
        # 末轮 query 与当前 q 相同（改写结果是原句直通）→ 视为继续同轮，避免重复累积
        if turns and turns[-1]["q"] == turn.get("q"):
            turns[-1]["answer"] = turn.get("answer", "")
        else:
            new_turn = {"q": turn.get("q", ""), "answer": turn.get("answer", "")}
            dom = turn.get("domain")
            if dom:
                new_turn["domain"] = dom
            turns.append(new_turn)
            if len(turns) > get_settings().mt_max_turns:
                del turns[0]
        ent["ts"] = time.time()


def _mt_serialize(history: list[dict[str, str]]) -> str:
    lines = []
    for idx, t in enumerate(history, 1):
        lines.append(f"{idx}. 用户：{t['q']} → 助手：{t['answer'] or '（应答为空）'}")
    return "\n".join(lines)


def _short_turns(short_memory: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    """把请求里的 shortMemory 解析成『已完成轮次』的 (q, a) 序列。

    每轮形如 {"dialogData":{"query","answer"},"businessData":{"serviceData":[{"data":[...]}]}}。
    - a_i 以 shortMemory 为准：取 answer 文本，并把候选媒资实体铺成可读列表
      （带「第N部」排位），供 LLM 把「第一位/刚才那部/该导演」还原成具体内容名。
    - 只收录已完成轮：无应答也无候选（仅回显当前 query 的空 stub）直接丢弃。
    - 该轮真实 query 不取 dialogData.query（不准确的原始句），由 _mt_merge 用真实链路兜底/对齐。
    """
    out: list[dict[str, str]] = []
    for ent in short_memory or []:
        if not isinstance(ent, dict):
            continue
        dialog = ent.get("dialogData") or {}
        if not isinstance(dialog, dict):
            continue
        a = str(dialog.get("answer") or "").strip() or ""
        candidates: list[dict] = []
        biz = ent.get("businessData") or {}
        if isinstance(biz, dict):
            for svc in biz.get("serviceData") or []:
                if isinstance(svc, dict):
                    for m in svc.get("data") or []:
                        if _candidate_key(m):
                            candidates.append(m)
        pieces = []
        if candidates:
            cap = get_settings().candidate_context_limit
            ranked = [f"第{i+1}部: {_candidate_key(m)}"
                      for i, m in enumerate(candidates[:cap]) if _candidate_key(m)]
            pieces.append("候选媒资:" + "；".join(ranked))
        if a:
            pieces.append(a)
        if not pieces:
            continue                      # 无应答也无候选 → 未完成/空位，跳过
        out.append({
            "q": str(dialog.get("query") or "").strip(),
            "answer": "；".join(pieces),
        })
    return out


def _mt_merge(real_q: list[dict[str, str]],
              short_a: list[dict[str, str]]) -> list[dict[str, str]]:
    """把真实改写链路(real_q.q) 和 shortMemory 应答(short_a.a) 按轮对齐成 q1-a1/q2-a2。

    以轮次多的为准；同一轮优先取真实执行 query(改写链路)，应答取 shortMemory 的应答。
    只有询问无应答的孤立轮可余留为纯 query。
    """
    n = max(len(real_q), len(short_a))
    merged: list[dict[str, str]] = []
    for i in range(n):
        q = ""
        if i < len(real_q):
            q = real_q[i].get("q") or ""
        if not q and i < len(short_a):
            q = short_a[i].get("q") or ""
        a = ""
        if i < len(short_a):
            a = short_a[i].get("answer") or ""
        if not a and i < len(real_q):
            a = real_q[i].get("answer") or real_q[i].get("a") or ""
        if q or a:
            merged.append({"q": q, "answer": a})
    return merged


async def _mt_rewrite(cur_query: str, device_id: str,
                      short_memory: list[dict[str, Any]] | None = None,
                      trace_id: str = "") -> str:
    """把本轮合并历史改写成自包含 query。仅改写、不写回。无有效历史直接用原句。

    历史 = real_q（进程内真实执行 query 链）+ short_a（请求 shortMemory 的应答与候选）；
    两者按轮对齐成 q1-a1/q2-a2（key 取改写链路、应答取 shortMemory，缺一用另一侧补齐）。
    """
    real_q = _mt_get(device_id)
    prev_domain = (real_q[-1].get("domain") or "") if real_q else ""
    short_a = _short_turns(short_memory)
    merged = _mt_merge(real_q, short_a)
    if not merged:
        return cur_query
    dialog = _mt_serialize(merged)
    user = MT_REWRITE_PROMPT.format(dialog=dialog, cur_query=cur_query)
    try:
        message = await llm.chat(
            [{"role": "user", "content": user}],
            metadata={"purpose": "mt_rewrite", "same_turn": False, "op": "rewrite",
                      "domain": prev_domain, "device_id": device_id, "trace_id": trace_id,
                      "has_history": bool(real_q),
                      "has_short_memory": bool(short_memory)})
    except llm.LLMError as exc:
        logger.error("多轮改写 LLM 失败（降级原句）：%s", exc)
        return cur_query
    raw = message.get("content")
    out = (raw if isinstance(raw, str) else str(raw or "")).strip()
    return out if out else cur_query


def _has_mt_history(device_id: str,
                    short_memory: list[dict[str, Any]] | None) -> bool:
    """是否具备需要做多轮改写的上下文。

    只有【没有任何跨轮上下文】才算"首轮单 query"（原句直通、不改写）；只要存在任一上下文
    （进程内 _MT_CONTEXT 或请求 shortMemory）就仍需改写（含只有 shortMemory 无应答/stub）。
    """
    if _mt_get(device_id):
        return True
    return bool(short_memory)


def _mt_append_answer(device_id: str, query: str, answer: str,
                      domain: str = "") -> None:
    """把一轮 已完成 (query, answer) 追加进 device 对话历史。

    answer 取本轮 steps 的 retext 摘要，供下一轮改写还原「指上一环/刚才那部」等历史引用。
    query = 本轮真正改写/执行的自包含句；domain = 本轮落定域。
    """
    if not device_id or not answer.strip():
        return
    turn = {"q": query or "", "answer": answer.strip()}
    if domain:
        turn["domain"] = domain
    _mt_add(device_id, turn)


# ---------------------------------------------------------------------------
# 计划条目（引擎内部态，较 OneIntent 多出 src/依赖执行等续跑所需字段）
# ---------------------------------------------------------------------------
class Intent:
    """一条计划条目。index 为 1-start 计划序号；src 为喂给工具解析的用户原话。"""

    __slots__ = ("query", "domain", "tool", "params", "index", "depends",
                 "dep_on", "src", "include_tool_result", "tool_result")

    def __init__(self, query: str, domain: str = "", tool: str = "",
                 params: dict[str, Any] | None = None, index: int = 0,
                 depends: bool = False, dep_on: int = 0, src: str = "",
                 include_tool_result: bool = False, tool_result: str = "") -> None:
        self.query = query
        self.domain = domain
        self.tool = tool
        self.params = params or {}
        self.index = index
        self.depends = depends
        self.dep_on = dep_on
        self.src = src
        self.include_tool_result = include_tool_result
        self.tool_result = tool_result


class Plan:
    """一份计划（首轮产出，续跑时按 deviceId 缓存复用）。"""
    __slots__ = ("intents",)

    def __init__(self, intents: list[Intent] | None = None) -> None:
        self.intents: list[Intent] = intents or []

    def total(self) -> int:
        return len(self.intents)


# ---------------------------------------------------------------------------
# 首轮计划解析：复用 parser.parse_oneshot（短键/长键/代码块/尾逗号/降级 全兼容），
# 再把 OneIntent 映射成引擎 Intent。
# ---------------------------------------------------------------------------
def parse_plan(text: str) -> Plan:
    """LLM 输出 → Plan。复用一把出 parser 的全部容错；解析失败 → 空 Plan。

    多步架构：首轮只为【可立即执行】条目定 tool/params；依赖条目只保留 q/d/dep，
    其 tool/params 由续跑轮（拿到前序结果、完成指代消解后）再生成。模型若仍给了依赖条目的
    tool/params，这里一律丢弃——否则续跑会把这个"盲猜"的工具当计划工具钉死。
    """
    intents, _err = parse_oneshot(text)
    out: list[Intent] = []
    for i, it in enumerate(intents, start=1):
        dep = bool(it.dep)
        out.append(Intent(
            query=it.q, domain=it.d or "",
            tool="" if dep else (it.tool or ""),
            params={} if dep else it.params,
            index=i, depends=dep, dep_on=it.dep, src=it.q,
        ))
    return Plan(out)


# ---------------------------------------------------------------------------
# 批次派生（契约派生，非语言规则：由 dep 序号确定当前可执行批）
# ---------------------------------------------------------------------------
def _executed_indexes(history: list[dict[str, Any]] | None) -> set[int]:
    """从 toolHistory 提取「已执行过的 step 序号」。"""
    out: set[int] = set()
    for item in history or []:
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        if isinstance(idx, int) and idx > 0:
            out.add(idx)
            continue
        i = item.get("id") or item.get("stepId")
        if isinstance(i, str) and i.startswith("s") and i[1:].isdigit():
            out.add(int(i[1:]))
    return out


def next_batch(plan: Plan, executed: set[int]) -> list[Intent]:
    """下一批：无依赖全发(并行)；依赖意图需其依赖项已执行(串行)。"""
    out: list[Intent] = []
    for it in plan.intents:
        if it.index in executed:
            continue
        if it.depends and it.dep_on not in executed:
            continue
        out.append(it)
        if len(out) >= get_settings().max_plan_steps:
            break
    return out


# ---------------------------------------------------------------------------
# toolHistory 读取（续跑改写要用的实体）
# ---------------------------------------------------------------------------
def _load_maybe_str(v: Any) -> Any:
    """dict/list 原样返回；字符串尝试 JSON 反解（上游 result 可能是 JSON 文本）。"""
    if not isinstance(v, str):
        return v
    s = v.strip()
    if s[:1] in ("{", "["):
        try:
            return json.loads(s)
        except (ValueError, TypeError):
            return v
    return v


def _candidate_key(m: Any) -> str:
    """从一条候选媒资里抽『指代改写』所需的关键字段，control 为平文本。"""
    if not isinstance(m, dict):
        return ""
    title = str(m.get("mediaTitle") or m.get("title") or "").strip().strip("《》 　")
    if not title:
        return ""
    direc = m.get("director") or []
    direc = "/".join(str(x) for x in direc if isinstance(x, str) and x) or "未知"
    rate = m.get("doubanRate") or m.get("rate") or ""
    rate = "" if rate in (None, "", 0, "0", 0.0) else str(rate)
    actor = m.get("actor") or []
    actor = "/".join(str(x) for x in actor[:3] if isinstance(x, str) and x)
    year = str(m.get("pubdate") or "")[:4]
    parts = [f"《{title}》", f"导演:{direc}"]
    if rate:
        parts.append(f"评分:{rate}")
    if actor:
        parts.append(f"主演:{actor}")
    if year:
        parts.append(year)
    return " ".join(parts)


def _buckets_medias(raw: Any) -> list[dict]:
    """从工具结果里取 memoryData[].data[] 候选媒资(跨 bucket 平铺)。"""
    if not isinstance(raw, dict):
        return []
    data = raw.get("data")
    if not isinstance(data, dict):
        return []
    out: list[dict] = []
    for bucket in data.get("memoryData") or []:
        if isinstance(bucket, dict) and isinstance(bucket.get("data"), list):
            out.extend(m for m in bucket["data"] if isinstance(m, dict))
    return out


def _first_text(raw: Any, keys: tuple[str, ...]) -> str:
    if not isinstance(raw, dict):
        return ""
    for k in keys:
        v = raw.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _tool_result_text(history: list[dict[str, Any]] | None) -> str:
    """把『最近一步』的工具结果压缩成给续跑 LLM 的参考文本（保留实体）。

    - 按倒序(最新在前)；对每个 item 独立决定返回什么，绝不越过最新 item 取更早候选，
      否则问答步会被旧的搜索候选覆盖。
    - 最新 item 若是 vod 类(dict + memoryData) → 提候选(片名/导演/评分/主演/年份)，保序，
      带「第N部」序号，让改写 LLM 能把"第二部"准确落到排位号。
    - 若是问答/文本类 → 直接返回该文本。仅当某 item 完全无内容才回退到更早 item。
    """
    limit = get_settings().candidate_context_limit
    for item in reversed(history or []):
        if not isinstance(item, dict):
            continue
        for k in ("result", "answer", "tts"):
            v = item.get(k)
            if v is None:
                continue
            raw = _load_maybe_str(v)
            cands = [c for c in (_candidate_key(m) for m in _buckets_medias(raw)) if c]
            if cands:
                ranked = "\n".join(f"第{i+1}部: {c}" for i, c in enumerate(cands[:limit]))
                return "候选媒资(按搜索排位):\n" + ranked
            text = raw if isinstance(raw, str) else _first_text(raw, ("ttscontent", "answer", "tts"))
            if isinstance(raw, str) and raw.strip() and not raw.startswith(("{", "[")):
                text = raw.strip()
            if text:
                return text.strip()
            if isinstance(raw, (dict, list)) and raw:
                return json.dumps(raw, ensure_ascii=False, separators=(",", ":"))[:4000]
    return ""


# ---------------------------------------------------------------------------
# turnSlow 兜底路由（协议派生，非语言规则：查表，无词表无 LLM）
# ---------------------------------------------------------------------------
def _turnslow_steps(history: list[dict[str, Any]] | None) -> dict[str, str]:
    """从 toolHistory 提取「返回过 turnSlow 的工具」→ reason（同工具保留首次）。"""
    out: dict[str, str] = {}
    for item in history or []:
        if not isinstance(item, dict):
            continue
        raw = _load_maybe_str(item.get("result"))
        if not isinstance(raw, dict):
            continue
        data = raw.get("data")
        if not isinstance(data, dict) or data.get("turnSlow") is not True:
            continue
        tool = item.get("toolName") or item.get("tool") or ""
        if not isinstance(tool, str) or not tool or tool in out:
            continue
        reason = data.get("reason")
        out[tool] = reason if isinstance(reason, str) and reason else _TURNSLOW_NO_REASON
    return out


def _turnslow_route(tool: str, reason: str, domain: str) -> str:
    """按文档表给出兜底工具名；无同域模糊兜底时收口 fan_knowledge_agent。"""
    by_domain = _TURNSLOW_FUZZY.get(reason.upper())
    if by_domain:
        target = by_domain.get(domain)
        if target:
            return target
    return _TURNSLOW_QA_TOOL


# ---------------------------------------------------------------------------
# 赛事阵容：把「第N场」对应的 shortMemory 候选 id 绑到 matchId（契约绑定，非语言规则）
# ---------------------------------------------------------------------------
_CN_ORDINALS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
                "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _short_memory_candidate_id(short_memory: list[dict[str, Any]] | None,
                               query: str) -> Any | None:
    """按本轮序号（"第2场"/"第三场"）从最近一轮 shortMemory 候选中取 id。"""
    import re as _re
    m = _re.search(r"第\s*(\d+|[一二三四五六七八九十])", query or "")
    rank = 1
    if m:
        raw = m.group(1)
        rank = int(raw) if raw.isdigit() else _CN_ORDINALS.get(raw, 1)
    if rank < 1:
        return None
    # shortMemory 末尾可能是当前轮空 stub；倒序找最近一个真正带业务候选的轮次。
    for ent in reversed(short_memory or []):
        if not isinstance(ent, dict):
            continue
        biz = ent.get("businessData")
        if not isinstance(biz, dict):
            continue
        candidates: list[dict[str, Any]] = []
        for svc in biz.get("serviceData") or []:
            if not isinstance(svc, dict):
                continue
            data = svc.get("data")
            if isinstance(data, list):
                candidates.extend(x for x in data if isinstance(x, dict))
            elif isinstance(data, dict):
                candidates.append(data)
        if not candidates:
            continue
        if rank > len(candidates):
            return None
        cid = candidates[rank - 1].get("id")
        if cid in (None, ""):
            return None
        return int(cid) if isinstance(cid, str) and cid.isdigit() else cid
    return None


def _bind_sports_short_memory_match_id(steps: list[dict[str, Any]],
                                       short_memory: list[dict[str, Any]] | None,
                                       query: str) -> None:
    """给赛事阵容工具的 step 补 `matchId`（从 shortMemory 候选 id），就地改 steps。"""
    for step in steps:
        if step.get("toolName") != "sports_match_lineup_search":
            continue
        params = step.get("parameters")
        if not isinstance(params, dict):
            params = {}
            step["parameters"] = params
        legacy = params.pop("id", None)      # 兼容上游旧 id，统一输出 matchId
        if params.get("matchId") not in (None, ""):
            continue
        cid = legacy if legacy not in (None, "") else _short_memory_candidate_id(
            short_memory, str(step.get("retext") or query or ""))
        if cid not in (None, ""):
            params["matchId"] = cid
            logger.info("sports lineup 绑定 shortMemory 比赛 matchId=%r", cid)


def _arm_forecast_serial_plan(state_key: str, batch: list[Intent],
                              steps: list[dict[str, Any]], query: str) -> bool:
    """首步命中赛事预测后，保存依赖计划；本轮仅返回 s1，下一 tick 执行泛知识 s2。

    与 hcAgent 同口径：s2 依赖 s1 结果（include_tool_result），由状态机在 runtime 回传
    toolHistory 后经 _rewrite_step 改写执行。
    """
    if (not state_key or not batch or not steps
            or steps[0].get("toolName") != "sports_match_forecast"):
        return False
    first = batch[0]
    plan = Plan(intents=[
        Intent(query=first.query, domain=first.domain, tool="sports_match_forecast",
               index=1, depends=False, dep_on=0, src=query),
        Intent(query=query, domain="", tool=_TURNSLOW_QA_TOOL,
               index=2, depends=True, dep_on=1, src=query,
               include_tool_result=True),
    ])
    _sm()._store(state_key, plan)
    logger.info("sports forecast 已挂起串行泛知识步骤: s1 -> s2")
    return True


# ---------------------------------------------------------------------------
# step 组装：把引擎 Intent → slowAgent step（含兜底 / turnSlow / fan 契约规范化）
# ---------------------------------------------------------------------------
def _steps_from_batch(batch: list[Intent],
                      history: list[dict[str, Any]] | None = None,
                      short_memory: list[dict[str, Any]] | None = None,
                      query: str = "") -> tuple[list[dict[str, Any]], bool]:
    """组装本轮 steps。返回 (steps, turnslow_force_stop)。

    与 hcAgent `_build_steps` 同构，差异：tool/params 已由 LLM 直接给出（无 hcTools 跳），
    这里只做「兜底判定 + fan 契约规范化 + turnSlow 路由 + step 字段拼装」。
    """
    turnslow_hits = _turnslow_steps(history)
    forced = False
    steps: list[dict[str, Any]] = []
    for it in batch:
        single = it.src and it.src == it.query
        tool_name = it.tool or ""
        params: Any = dict(it.params) if isinstance(it.params, dict) else {}

        # 兜底：无工具（选工具失败 / 目录不可用）→ 泛知识问答。
        # 【不要】在此清空 it.domain —— 域是有效信号，下面 rebuild 用 QA_SCENE_BY_DOMAIN
        # 把它转成 sceneType：d="" 落域失败 → knowledge_qa（通用 qa，同 hcAgent）；
        # d="vod" 选工具失败 → sceneType=video（域内 qa，同 hcAgent 的 execute_fallback_qa）。
        # 若清空域，两种情形都会塌成通用 qa，丢掉域信号。
        if not tool_name:
            fb_q = it.src if single else (it.query or it.src or "")
            tool_name = _TURNSLOW_QA_TOOL
            params = {"messages": fb_q}          # 交给下面 rebuild 统一成契约

        # turnSlow 兜底路由：本步工具刚返回过 turnSlow → 禁止重发，按表换兜底工具。
        if turnslow_hits and tool_name in turnslow_hits:
            reason = turnslow_hits[tool_name]
            target = _turnslow_route(tool_name, reason, it.domain)
            fb_q = it.src if single else (it.query or it.src or "")
            if target == _TURNSLOW_QA_TOOL:
                # qa 是链路终点：走到它就是最后一发，forceStop 置真（runtime 不可 override）。
                tool_name = _TURNSLOW_QA_TOOL
                params = {"messages": fb_q}
                forced = True
            else:
                # 同域模糊搜索兜底：整句原话直给。三个 fuzzy 工具（vod/educ/edu）schema
                # 均为 {retext}（additionalProperties:false），故一律用 retext —— 早期对
                # vod 特判发 {"query":…} 是错的（query 不在 schema 里会被丢弃，兜底参数空转）。
                tool_name = target
                params = {"retext": fb_q}

        # fan_knowledge_agent：目录块把它压成 messages:string，模型照填就不合契约 →
        # 统一经 rebuild 规范化（messages 数组 + riskRespStrategy + sceneTypeIntentList）。
        if tool_name == _TURNSLOW_QA_TOOL:
            fb_q = it.src if single else (it.query or it.src or "")
            params = fanparams.rebuild_fan_params(
                params, scene_hint=fanparams.QA_SCENE_BY_DOMAIN.get(it.domain, ""),
                fallback_query=fb_q or it.query,
            )

        # 串行预测第二步：把上一步结果写入 current_media（与 hcAgent 同口径）。
        if it.include_tool_result and it.tool_result:
            params = dict(params) if isinstance(params, dict) else {}
            params["current_media"] = it.tool_result

        step: dict[str, Any] = {
            "id": f"s{it.index}",
            "toolName": tool_name,
            "parameters": params,
            "plan": None,
            "dependsOn": [f"s{it.dep_on}"] if it.depends and it.dep_on else [],
            "retext": it.src if single else (it.query or it.src),
            "hitQuery": it.query,      # 透传真正发给工具/展示的改写后 query（审计）
        }
        steps.append(step)
    # 赛事阵容：把「第N场」的 shortMemory 候选 id 绑到 matchId（否则 runtime 只能拿到空参数）。
    _bind_sports_short_memory_match_id(steps, short_memory, query)
    return steps, forced


# ---------------------------------------------------------------------------
# 串行状态机：按 deviceId 缓存 plan，跨 tick 推进依赖链
# ---------------------------------------------------------------------------
class TraceStateMachine:
    def __init__(self, ttl: float | None = None) -> None:
        self._lock = threading.Lock()
        self._state: dict[str, dict[str, Any]] = {}
        self._ttl = ttl if ttl is not None else get_settings().plan_trace_ttl

    async def tick(self, key: str, query: str,
                   history: list[dict[str, Any]] | None,
                   weak_hint: str = "") -> tuple[list[Intent], bool]:
        entry = self._load(key)
        if entry is None:
            return await self._first(key, query, history, weak_hint=weak_hint)
        return await self._continue(key, query, history)

    async def _first(self, key: str, query: str,
                     history: list[dict[str, Any]] | None,
                     weak_hint: str = "") -> tuple[list[Intent], bool]:
        plan = await self._plan(query, has_history=bool(history), weak_hint=weak_hint)
        if not plan.intents:
            # LLM 没出方案（空/纯聊天/失败）→ 兜底：原句单意图、空域，走 fan 兜底。
            return [Intent(query=query, domain="", tool="", index=1, src=query)], True
        if not any(it.domain for it in plan.intents):
            # 落域失败（全条目 d 都空）→ 整句交泛知识问答。
            logger.info("首轮落域失败（全空 d），整句转泛知识问答：%r", query)
            return [Intent(query=query, domain="", tool="", index=1, src=query)], True
        _with_source(plan, query)
        executed = _executed_indexes(history)
        batch = next_batch(plan, executed) or [plan.intents[0]]
        stop = len(executed) + len(batch) >= plan.total()
        if not stop:
            self._store(key, plan)
        return batch, stop

    async def _plan(self, query: str, has_history: bool = False,
                    weak_hint: str = "") -> Plan:
        user = f"用户请求：{query}"
        if weak_hint:
            user += f"\n【判定提示】{weak_hint}"
        block = _catalog_block()
        slot_block = await slots.slot_block_for(query)
        try:
            message = await llm.chat(
                [{"role": "system", "content": build_oneshot_prompt(block, slot_block)},
                 {"role": "user", "content": user}])
        except llm.LLMError as exc:
            logger.error("首轮计划 LLM 失败：%s", exc)
            return Plan()
        raw = message.get("content")
        return parse_plan(raw if isinstance(raw, str) else str(raw or ""))

    async def _continue(self, key: str, query: str,
                        history: list[dict[str, Any]] | None) -> tuple[list[Intent], bool]:
        plan: Plan = self._state[key]["plan"]
        executed = _executed_indexes(history)
        batch = next_batch(plan, executed)
        if not batch:
            self._clear(key)
            return [], True
        tool_result = _tool_result_text(history)
        # 批内依赖改写相互独立，并发执行（真并发收益点）；每个仍恰好一次 LLM。
        async def resolve(it: Intent) -> Intent:
            if it.depends:
                q, tool, params = await self._rewrite_step(
                    it.query, tool_result, step=it.index, plan=plan)
                # 计划的 tool 是权威：续跑只允许「改写 query + 重填【同工具】参数」，
                # 不允许擅自换工具。hcAgent 的 T1 本就只改写 query（工具由 T0 计划固定），
                # hcOneshot 因无 hcTools 需在续跑里补参，故多出 params，但工具仍须钉住——
                # 否则模型会把挂起的 fan 终点（预测 s2）换成重发 sports_match_forecast，
                # 导致该收口的链路被无限重发。换工具时其 params 亦不匹配，一并丢弃交给下游兜底。
                if it.tool and tool != it.tool:
                    logger.info("续跑模型擅换工具 %r→%r，钉回计划工具", tool, it.tool)
                    tool, params = it.tool, {}
            else:
                q, tool, params = it.query, it.tool, it.params
            return Intent(query=q, domain=it.domain, tool=tool, params=params,
                          index=it.index, depends=it.depends, dep_on=it.dep_on,
                          src=it.src, include_tool_result=it.include_tool_result,
                          tool_result=tool_result if it.include_tool_result else "")

        resolved = list(await asyncio.gather(*[resolve(it) for it in batch]))
        cursor = max((i.index for i in resolved), default=0)
        stop = not [i for i in plan.intents if i.index > cursor]
        if stop:
            self._clear(key)
        else:
            self._store(key, plan)
        return resolved, stop

    async def _rewrite_step(self, q: str, tool_result: str, step: int = 0,
                            plan: Plan | None = None) -> tuple[str, str, dict[str, Any]]:
        """续跑一次 LLM：拿前序结果做指代消解，生成当前步的 q + tool + params。

        输入：用户原话 + 该步所属域 + 待办子请求 + 前一步工具结果。
        失败降级：原 q + 计划工具（依赖步通常为空）→ 由 _steps_from_batch 走域内 fan 兜底。
        """
        prev = None
        if plan is not None:
            prev = next((i for i in plan.intents if i.index == step), None)
        lines = []
        if prev is not None and prev.src and prev.src != q:
            lines.append(f"用户原话:{prev.src}")
        if prev is not None and prev.domain:
            lines.append(f"所属域:{prev.domain}")
        lines.append(f"待办意图:{q}")
        if tool_result:
            lines.append(f"工具结果:{tool_result}")
        user = "\n".join(lines)
        fb_tool = prev.tool if prev else ""
        fb_params = dict(prev.params) if prev else {}
        try:
            message = await llm.chat(
                [{"role": "system", "content": build_t1_prompt(_catalog_block())},
                 {"role": "user", "content": user}])
        except llm.LLMError as exc:
            logger.error("续跑改写 LLM 失败（降级原样）：%s", exc)
            return q, fb_tool, fb_params
        raw = message.get("content")
        text = raw if isinstance(raw, str) else str(raw or "")
        obj = parse_json_loose(text)
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            obj = obj[0]
        if isinstance(obj, dict):
            # 模型漏写 d 时补上计划里的域：fan_knowledge_agent 在 7 个域都有，
            # 无域就无法反推、会被 parser 当非法工具清空。
            if not (obj.get("d") or obj.get("domain")) and prev is not None and prev.domain:
                obj = {**obj, "d": prev.domain}
            _intents, _err = parse_oneshot(json.dumps([obj], ensure_ascii=False))
            if _intents:
                it = _intents[0]
                return (it.q or q), (it.tool or fb_tool), (it.params or fb_params)
        # 退化：模型只回了改写句（纯文本）→ 用改写句，工具沿用计划（通常为空 → 兜底）
        cleaned = text.strip().strip('"')
        if cleaned.startswith(("{", "[")):
            cleaned = ""
        return (cleaned or q), fb_tool, fb_params

    # -- 状态 --
    def _store(self, key: str, plan: Plan) -> None:
        with self._lock:
            self._state[key] = {"plan": plan, "ts": time.time()}

    def _load(self, key: str) -> dict[str, Any] | None:
        if not key:
            return None
        now = time.time()
        with self._lock:
            st = self._state.get(key)
            if st is not None and now - st["ts"] > self._ttl:
                self._state.pop(key, None)
                return None
            return st

    def _clear(self, key: str) -> None:
        with self._lock:
            self._state.pop(key, None)


def _with_source(plan: Plan, src: str) -> None:
    """给计划内每个意图标注 src=用户原文（就地）。"""
    for it in plan.intents:
        it.src = src


_SM: TraceStateMachine | None = None


def _sm() -> TraceStateMachine:
    """惰性单例（TTL 取自 settings，故不在 import 期建）。"""
    global _SM
    if _SM is None:
        _SM = TraceStateMachine()
    return _SM


def reset() -> None:
    """测试用：清空状态机与多轮上下文。"""
    sm = _sm()
    with sm._lock:
        sm._state.clear()
    with _MT_LOCK:
        _MT_CONTEXT.clear()


def _catalog_block() -> str:
    """目录块渲染（异常不抛出，降级空块 —— 引擎契约是失败回结构化 error，不 500）。"""
    try:
        return catalog.render_oneshot_block()
    except Exception as exc:  # noqa: BLE001
        logger.exception("目录块渲染失败，降级为空目录：%s", exc)
        return ""


# ---------------------------------------------------------------------------
# 对外主入口：slowAgent POC 契约
# ---------------------------------------------------------------------------
def _error(message: str) -> dict[str, Any]:
    return {"code": 400, "message": message, "data": {"steps": []},
            "stop": True, "forceStop": True}


async def build_response(req) -> dict[str, Any]:
    """构建 slowAgent 响应（对齐 juagent orchestrator/server.py 契约）。"""
    if req.data is None:
        return _error("missing data")
    query = (req.data.query or "").strip()
    if not query:
        return _error("empty query")
    # 超长 query 拒绝：一把出把整句塞进 LLM prompt，超长会撑爆上下文/拖垮延迟。
    # max_query_len 是声明式旋钮（HC_MAX_QUERY_LEN），此处收敛为结构化 error（非 500）。
    if len(query) > get_settings().max_query_len:
        return _error(f"query too long (>{get_settings().max_query_len})")
    trace_id = req.traceId or ""
    device_id = req.deviceId or ""

    # badcase 精确覆盖层（debug 专用，默认关）：命中即直出给定响应，跳过 LLM 主干。
    from . import badcase
    hit = badcase.try_hit(query, trace_id, device_id)
    if hit is not None:
        return hit

    history = req.data.toolHistory or []
    short_memory = (req.data.memory.shortMemory if req.data.memory else None) or []

    # 多轮改写分界（与 hcAgent 同口径）：
    #  - history 非空 = 续跑（串行第 2..N 步）：query 交给状态机用 toolHistory 改写依赖步，
    #    不在这里做跨轮 merge——避免把整句误并入上轮检索条件。
    #  - 无 history 且无任何跨轮上下文（进程内 _MT_CONTEXT 与 shortMemory 皆空）→ 纯首轮
    #    单 query 原句直通、强制不改写，避免「想看悬疑的→推荐悬疑」这类改写噪声。
    #  - 否则（有任一上下文）→ 交 LLM 合并历史成自包含 query。
    if history:
        mt_query = query            # 续跑轮：原 query 交给状态机改写依赖步
    elif not _has_mt_history(device_id, short_memory):
        mt_query = query            # 首轮无任何上下文：原句直通
    else:
        mt_query = await _mt_rewrite(query, device_id, short_memory, trace_id=trace_id)
    weak_hint = weak_device_hint(mt_query) if not history else ""

    # 串行状态按 deviceId 存（换 request 用 deviceId 识别会话，而非 traceId）。
    batch, stop = await _sm().tick(device_id, mt_query, history, weak_hint=weak_hint)
    # 双域并行标记：首轮无历史、本批恰好是 children+vod 两个并行意图时标记 parallel，
    # 供客户端并发渲染两个 tab（与 hcAgent 同口径）。
    is_dual_tab = stop and not history and len(batch) == 2 and (
        {i.domain for i in batch} == {"children", "vod"}
    )
    # matchId 绑定 / 兜底里用的 query 取【用户原句】（与 hcAgent 同口径：那里也传原始 query）。
    steps, force_stop = _steps_from_batch(batch, history=history,
                                          short_memory=short_memory, query=query)
    # 胜负预测走真正的串行续跑：首轮只发 s1 并 stop=false；runtime 回传 toolHistory 后，
    # 状态机才发依赖 s1 的 fan_knowledge_agent(s2)。（与 hcAgent 同口径）
    if not history and _arm_forecast_serial_plan(device_id, batch, steps, query):
        stop = False
    # 本轮 answer 摘要（各 step 的 retext）：供下一轮改写引用具体内容/序号。
    turn_answer = "；".join(s.get("retext") or "" for s in steps if s.get("retext"))
    round_domain = next((i.domain for i in batch if i.domain), "")
    # 写回的 q 用【真正改写/执行】的 mt_query（_MT_CONTEXT 语义即"真实 query 链"）。
    _mt_append_answer(device_id, mt_query, turn_answer, domain=round_domain)

    body: dict[str, Any] = {
        "code": 200, "message": "success", "traceId": trace_id, "deviceId": device_id,
        "data": {
            "planId": f"plan_{uuid.uuid4().hex[:4]}",
            "schemaVersion": "1.0", "planType": "execute", "planConfidence": 0.9,
            "steps": steps,
        },
        "stop": stop,
        # forceStop=异常强制终止（runtime 不可 override）：仅 turnSlow 兜底落到 qa 终点时为真。
        "forceStop": force_stop,
    }
    if is_dual_tab:
        body["data"]["parallel"] = True
    if not steps and stop:
        body["data"]["final"] = True
    if req.data.debug is True:
        body["data"]["debug"] = {
            "planId": body["data"]["planId"],
            "intents": [{"index": i.index, "query": i.query, "domain": i.domain,
                         "tool": i.tool, "depends_on_previous": i.depends} for i in batch],
        }
    return body


# ---------------------------------------------------------------------------
# 一把出原生入口（保留给调试 / 训练数据生成；不挂在 slowAgent 上）
# ---------------------------------------------------------------------------
async def run_oneshot(query: str, metadata: dict[str, Any] | None = None):
    """一把出原生格式：一次 LLM 调出 [{q,d,tool,params}]（不做串行续跑）。"""
    from .models import OneResponse
    started = time.perf_counter()
    system = build_oneshot_prompt(_catalog_block(), await slots.slot_block_for(query))
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"用户请求：{query}"},
    ]
    try:
        message = await llm.chat(messages, metadata=metadata)
    except llm.LLMError as exc:
        logger.error("LLM 调用失败：%s", exc)
        return OneResponse(error=f"llm: {exc}", latency_ms=(time.perf_counter() - started) * 1000)
    raw = message.get("content") or ""
    if not isinstance(raw, str):
        raw = str(raw)
    intents, err = parse_oneshot(raw)
    return OneResponse(intents=intents, raw=raw, error=err,
                       latency_ms=(time.perf_counter() - started) * 1000)
