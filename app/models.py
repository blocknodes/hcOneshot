"""请求/响应数据模型。

一把出：单次 LLM 调用产出 [{q, d, tool, params}]，替代 hcSlime 的
「hcAgent T0 选 vt + hcTools pinned 填参」两跳。
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

# 域闭合：只在 7 候选域内采纳；带外域（含 qa）归空 → 下游通用兜底。
DOMAINS = ("vod", "children", "education", "music", "audio", "sports", "device")


class Memory(BaseModel):
    shortMemory: list[dict[str, Any]] | None = Field(default_factory=list)
    longMemory: dict[str, Any] | None = Field(default_factory=dict)


class OneData(BaseModel):
    """兼容 hcAgent SlowData 形状，便于直接替换 /slowAgent/{poc_id}。"""
    query: str | None = ""
    tvMode: str | int | None = None
    memory: Memory | None = Field(default_factory=Memory)
    toolHistory: list[dict[str, Any]] | None = Field(default_factory=list)
    metadata: dict[str, Any] | None = None


class OneRequest(BaseModel):
    """一把出请求。两种入参形状都接受：
      · 扁平：{"query": "...", "metadata": {...}}
      · 嵌套：{"data": {"query": "...", "memory": {...}}}   （hcAgent 兼容）
    """
    query: str | None = None
    metadata: dict[str, Any] | None = None
    data: OneData | None = None

    @field_validator("query")
    @classmethod
    def _strip_query(cls, v: str | None) -> str | None:
        # 只归一空白，不在此抛错：空 query 由端点统一回 {error:"query 不能为空"}（HTTP 200）。
        # 曾在此 raise → 扁平 {"query":""} 走 400、而嵌套 {"data":{"query":""}} 不触发校验器
        # 走 200，同一逻辑请求两种 HTTP 语义。空判定收敛到 resolved_query()。
        return v.strip() if isinstance(v, str) else v

    def resolved_query(self) -> str:
        q = (self.query or "") or ((self.data.query if self.data else "") or "")
        return (q or "").strip()

    def resolved_metadata(self) -> dict[str, Any] | None:
        if self.metadata:
            return self.metadata
        if self.data and self.data.metadata:
            return self.data.metadata
        return None


class OneIntent(BaseModel):
    """一把出的单条计划：改写句 + 域 + 工具 + 参数。"""
    q: str = ""                              # 改写后自包含子请求
    d: str = ""                              # 域（7 候选之一；空=通用 qa 兜底）
    tool: str = ""                           # 域内工具名
    vt: str = ""                             # domain:tool（由 d+tool 派生，便于兼容）
    params: dict[str, Any] = Field(default_factory=dict)
    dep: int = 0                             # 依赖的前序条序号（1-start；0=无）


class OneResponse(BaseModel):
    intents: list[OneIntent] = Field(default_factory=list)
    raw: str = ""                            # 模型原始输出（审计）
    error: str = ""                          # 解析/调用失败原因（空=正常）
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# slowAgent POC 契约（对齐 hcAgent/models.py 与 juagent orchestrator/server.py）
# ---------------------------------------------------------------------------
class SlowData(BaseModel):
    """poc 慢任务 data 段。extra=allow：runtime 会带 segment/toolList 等未声明字段。"""
    query: str | None = ""
    tvMode: str | int | None = None
    memory: Memory | None = Field(default_factory=Memory)
    toolHistory: list[dict[str, Any]] | None = Field(default_factory=list)
    toolList: list[Any] | None = Field(default_factory=list)
    timestamp: str | int | float | None = None
    debug: bool | None = None

    model_config = {"extra": "allow"}


class SlowRequest(BaseModel):
    """poc 慢任务请求：{"traceId","deviceId","deviceType","data":{...}}。"""
    traceId: str | None = ""
    deviceId: str | None = ""
    deviceType: str | None = None
    data: SlowData | None = Field(default_factory=SlowData)

    model_config = {"extra": "allow"}
