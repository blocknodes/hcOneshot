"""OpenAI 兼容网关的异步最小客户端（照搬 hcTools/app/llm.py，精简 select 相关）。

metadata 原样放进请求体 metadata 字段（对齐 hcAgent/hcTools，hcProxy 据此分流与审计）。
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import httpx

from .config import get_settings

logger = logging.getLogger("hcOneshot.llm")

TEMPERATURE = 0.0
REQUEST_TIMEOUT = 120.0
MAX_RETRY = 3


class LLMError(Exception):
    """网关调用失败或返回无法解析。"""


def _extract_message(data: Any) -> dict[str, Any]:
    """从网关响应取 choices[0].message；形状异常抛 ValueError（调用方视作可重试）。

    独立成纯函数便于无网络单测：旧实现直接 `data["choices"][0]["message"]`，
    空 choices → IndexError 不在 except 列表 → 冒泡穿出 chat() → FastAPI 500（实测）。
    """
    choices = data.get("choices") if isinstance(data, dict) else None
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"网关响应无 choices: {str(data)[:200]}")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise ValueError(f"choices[0] 无 message: {str(choices[0])[:200]}")
    return message


async def chat(messages: list[dict[str, Any]],
               metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """调用 /chat/completions，返回 choices[0].message。失败抛 LLMError。"""
    settings = get_settings()
    if not settings.api_base:
        raise LLMError("HC_API_BASE 未配置（网关地址）")

    payload: dict[str, Any] = {"model": settings.model, "messages": messages}
    if metadata:
        payload["metadata"] = metadata
    if not settings.model.startswith("gpt-5"):
        payload["temperature"] = TEMPERATURE
        # 禁用 thinking：baseline 不开会在 content 里只回 " thinking"（与 hcTools 一致）。
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    headers = {"Content-Type": "application/json"}
    if settings.api_key:
        headers["Authorization"] = f"Bearer {settings.api_key}"

    last_error = ""
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        for attempt in range(MAX_RETRY):
            try:
                resp = await client.post(settings.chat_url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                message = _extract_message(data)
                logger.info(
                    "LLM req %s | %s | md=%s\n  >> IN  %s\n  << OUT %s  (%.0fms)",
                    settings.model,
                    " ".join((m.get("role", "?") for m in messages)),
                    json.dumps(metadata, ensure_ascii=False) if metadata else "-",
                    json.dumps(messages, ensure_ascii=False),
                    json.dumps(message, ensure_ascii=False),
                    (time.perf_counter() - started) * 1000,
                )
                return message
            except httpx.HTTPStatusError as exc:
                code = exc.response.status_code
                last_error = f"HTTP {code}: {exc.response.text[:200]}"
                if code != 429 and 400 <= code < 500:
                    break
            # ValueError：响应形状异常（无 choices/message），见上；与传输错误同列可重试。
            except (httpx.HTTPError, KeyError, ValueError, json.JSONDecodeError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            await asyncio.sleep(1.5 * (attempt + 1))

    raise LLMError(last_error or "unknown gateway error")


async def close() -> None:
    """连接池关闭钩子：本模块 client 为每次调用临时创建（async with），无需常驻清理。

    保留空实现以对齐 hcAgent 的 lifespan 调用点，避免 main 里出现 hasattr 分支。
    """
    return None
