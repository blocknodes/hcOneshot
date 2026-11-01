"""槽值检索（slot retrieval）：按本轮 query 查槽值库，渲染成首轮 system prompt 末尾的补充块。

接口：POST {HC_SLOT_RETRIEVE_URL}  {"query": "..."}
  → {"ok": true, "hits": [{"domain","domain_cn","field","value","alias","similarity",...}], ...}

放在 system prompt **最后**：前面的 prompt+目录+锚例对所有请求逐字节相同，可复用 KV cache；
只有这一段随 query 变化。故本块必须确定性渲染（同一响应 → 同一文本）。

失败（超时/网络/形状异常）一律降级为空块，不影响主干。
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from .config import get_settings

logger = logging.getLogger("hcOneshot.slots")

# 检索服务的域名 → 本工程域名。注意 educ=少儿(children)、edu=教育(education)，是两个域。
_DOMAIN_MAP = {
    "vod": "vod",
    "educ": "children",
    "edu": "education",
    "music": "music",
    "audio": "audio",
    "sports": "sports",
    "device": "device",
}
_DOMAIN_ORDER = ("vod", "children", "education", "music", "audio", "sports", "device")

_HEADER = """
【槽位检索参考】（系统按本轮请求检索槽值库得到的命中，仅作补充信息）
用法：
- 只用于把 params 里的槽值归一成库内标准值（如 国语→普通话），字段名与上方目录参数一致；
- 不据此改变拆句、落域、选工具；同一词在多个域都有命中时，只采用你所选域下的那条；
- 没列出的槽值照常按用户原话填写，不要因为检索未命中而丢参。
"""


async def retrieve(query: str) -> dict[str, Any] | None:
    """调用槽值检索接口；未配置或失败返回 None（不抛异常）。"""
    settings = get_settings()
    url = (settings.slot_retrieve_url or "").strip()
    if not url or not query:
        return None
    try:
        async with httpx.AsyncClient(timeout=settings.slot_retrieve_timeout) as client:
            resp = await client.post(url, json={"query": query})
            resp.raise_for_status()
            data = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("槽值检索失败（降级为无补充块）：%s: %s", type(exc).__name__, exc)
        return None
    if not isinstance(data, dict) or data.get("ok") is False:
        logger.warning("槽值检索返回异常（降级）：%s", str(data)[:200])
        return None
    return data


def render_slot_block(data: dict[str, Any] | None, max_hits: int | None = None) -> str:
    """把检索响应渲染成 prompt 补充块；无有效命中返回空串。确定性：同输入同输出。"""
    if not isinstance(data, dict):
        return ""
    hits = data.get("hits")
    if not isinstance(hits, list):
        return ""
    limit = max_hits if max_hits is not None else get_settings().slot_max_hits

    grouped: dict[str, list[str]] = {}
    seen: set[tuple[str, str, str, str]] = set()
    kept = 0
    for h in hits:
        if kept >= limit:
            break
        if not isinstance(h, dict):
            continue
        domain = _DOMAIN_MAP.get(str(h.get("domain") or ""))
        field = str(h.get("field") or "").strip()
        value = str(h.get("value") or "").strip()
        if not domain or not field or not value:
            continue
        alias = str(h.get("alias") or "").strip() or value
        key = (domain, alias, field, value)
        if key in seen:
            continue
        seen.add(key)
        item = f"{alias} → {field}={value}"
        sim = h.get("similarity")
        if isinstance(sim, (int, float)) and not isinstance(sim, bool) and sim < 1.0:
            item += f"（相似度{sim:.2f}）"
        grouped.setdefault(domain, []).append(item)
        kept += 1

    if not grouped:
        return ""
    lines = [f"- {d}：" + "；".join(grouped[d]) for d in _DOMAIN_ORDER if d in grouped]
    return _HEADER + "\n".join(lines) + "\n"


async def slot_block_for(query: str) -> str:
    """检索 + 渲染，一步到位（引擎调用口）。"""
    return render_slot_block(await retrieve(query))
