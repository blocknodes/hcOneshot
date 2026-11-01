"""slowAgent 侧 badcase 精确覆盖层（debug 专用，与主干解耦）。

用途：命中 badcase（query 归一化后精确匹配）时，直接返回配置里给定的完整响应体，
绕过 LLM 主干（落域/改写/编排）。仅用于 debug 复现，不影响正常链路。

开关默认关：设环境变量 HC_BADCASE_ENABLED=1 才生效（见 config.badcase_enabled）。
配置路径：环境变量 HC_BADCASE_PATH，默认同目录下 badcases_oneshot.json。

配置格式（JSON object，key=query 原文）：
    {
      "我想看悬疑剧": {                       # 简写：steps/stop/forceStop(/parallel/final)
        "steps": [{"id":"s1","toolName":"vod_search","parameters":{"query":"悬疑剧"}}],
        "stop": true
      },
      "另一个 query": {                        # 或直接给完整 body（含 data）
        "data": {"steps": [...], "parallel": true},
        "stop": false, "forceStop": false
      }
    }
命中结果会补全 code/message/planId/schemaVersion/planType/planConfidence，
并把本轮 traceId/deviceId 回填进 body。

与 hcAgent/app/badcase.py 同构（逐逻辑移植）：差别仅在 hash/开关读 env 而非 config 常量、
默认文件名 badcases_oneshot.json。hcOneshot 的 slowAgent 契约与 hcAgent 一致，故可直接复用。
"""
from __future__ import annotations

import copy
import json
import logging
import os
import threading
import uuid
from typing import Any

logger = logging.getLogger("hcOneshot.badcase")

_PATH = os.environ.get("HC_BADCASE_PATH") or os.path.join(
    os.path.dirname(__file__), "badcases_oneshot.json"
)

_LOCK = threading.Lock()
_CACHE: dict[str, Any] | None = None


def _enabled() -> bool:
    """开关默认关：HC_BADCASE_ENABLED=1/true/yes 才生效（对齐 hcAgent 常量语义）。"""
    return os.environ.get("HC_BADCASE_ENABLED", "0").lower() in ("1", "true", "yes")


def _norm(q: str) -> str:
    """归一化：去首尾空白 + 折叠内部空白（与 Tool 侧 badcase key 归一保持一致）。"""
    return " ".join((q or "").split())


def _load() -> dict[str, Any]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    with _LOCK:
        if _CACHE is not None:
            return _CACHE
        table: dict[str, Any] = {}
        try:
            with open(_PATH, encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                for k, v in raw.items():
                    table[_norm(k)] = v
            logger.info("badcase 已加载 %d 条：%s", len(table), _PATH)
        except FileNotFoundError:
            logger.info("badcase 文件不存在，跳过：%s", _PATH)
        except Exception as exc:  # noqa: BLE001 加载失败不阻断服务
            logger.error("badcase 加载失败 %s：%s", _PATH, exc)
        _CACHE = table
        return _CACHE


def reload() -> int:
    """清缓存并重载，返回条数（供调试热更）。"""
    global _CACHE
    with _LOCK:
        _CACHE = None
    return len(_load())


def lookup(query: str) -> Any | None:
    """命中返回配置原始条目，未命中/开关关返回 None。"""
    if not _enabled():
        return None
    return _load().get(_norm(query))


def _render(hit: Any, trace_id: str, device_id: str) -> dict[str, Any] | None:
    if not isinstance(hit, dict):
        return None
    if "data" in hit:
        body = copy.deepcopy(hit)               # 完整 body 写法
    else:                                        # 简写写法
        data: dict[str, Any] = {"steps": hit.get("steps", [])}
        if "parallel" in hit:
            data["parallel"] = hit["parallel"]
        if "final" in hit:
            data["final"] = hit["final"]
        body = {"data": data,
                "stop": hit.get("stop", True),
                "forceStop": hit.get("forceStop", False)}
    body.setdefault("code", 200)
    body.setdefault("message", "success")
    body.setdefault("data", {})
    d = body["data"]
    d.setdefault("planId", f"bc_{uuid.uuid4().hex[:4]}")
    d.setdefault("schemaVersion", "1.0")
    d.setdefault("planType", "execute")
    d.setdefault("planConfidence", 1.0)
    d.setdefault("steps", [])
    body["traceId"] = trace_id
    body["deviceId"] = device_id
    body["stop"] = bool(body.get("stop", True))
    body["forceStop"] = bool(body.get("forceStop", False))
    return body


def try_hit(query: str, trace_id: str, device_id: str) -> dict[str, Any] | None:
    """响应入口：命中即返回可直出的完整 body，否则 None。"""
    hit = lookup(query)
    if hit is None:
        return None
    body = _render(hit, trace_id, device_id)
    if body is not None:
        logger.info("badcase 命中，直出（跳过 LLM 主干）：query=%s", query)
    return body
