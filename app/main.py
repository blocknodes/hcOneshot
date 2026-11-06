"""hcOneshot —— juagent poc 慢任务接口的 FastAPI 实现（hcAgent 的完整替代）。

对齐 juagent/orchestrator/server.py 与 hcAgent 的对外契约：

  POST /slowAgent/poc_<id>
    {"traceId":"...","deviceId":"...","deviceType":"...",
     "data":{"query":"...","tvMode":"...","segment":{},
             "memory":{"shortMemory":[...],"longMemory":{}},
             "toolHistory":[...],"toolList":[...],"timestamp":"..."}}
  → {"code":200,"message":"success","traceId":"...","deviceId":"...",
     "data":{"planId","schemaVersion","planType","planConfidence","steps":[...],
             "final"?:true,"parallel"?:true},"stop":bool,"forceStop":bool}

与 hcAgent 的唯一差异：**单步只需一次 LLM 调用**。hcAgent 走
「T0 选 vt → hcTools 填参」两跳；hcOneshot 一次调用即出 q/d/tool/params
（多步仍走续跑状态机，但每个 tick 也恰好一次 LLM）。
"""
from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from . import catalog
from .engine import build_response
from .models import SlowRequest

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("hcOneshot")


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio
    from . import fewshot
    await asyncio.to_thread(fewshot.warmup)     # 开启 few-shot 时预建 BM25 索引（关闭时无操作）
    yield
    from . import llm
    await llm.close()


app = FastAPI(
    title="hcOneshot",
    description="juagent poc 慢任务接口（一把出：单步一次 LLM 出 tool+params）",
    version="0.1.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def on_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """把校验错误打成可读日志 + hcAgent 同款响应体（runtime 回调时字段错位便于定位）。"""
    raw = (await request.body()).decode("utf-8", "replace")
    logger.warning("422 on %s body=%s errors=%s", request.url.path, raw, exc.errors())
    return JSONResponse(
        status_code=422,
        content={
            "code": 422, "message": "validation error", "data": {"steps": []},
            "stop": True, "forceStop": True,
            "errors": [{"loc": list(e.get("loc", [])), "msg": e.get("msg"), "type": e.get("type")}
                       for e in exc.errors()],
        },
    )


@app.get("/")
def index() -> dict:
    return {"ok": True, "service": "hcOneshot",
            "endpoints": {"poc": "POST /slowAgent/poc_<id>"}}


@app.get("/health")
def health() -> dict:
    return {"ok": True, "status": "healthy", "catalog_tools": len(catalog.tools())}


@app.get("/api/catalog")
async def get_catalog(block: int = 0) -> dict:
    """导出工具目录（调试/训练对齐用）。block=1 时附渲染好的目录块。"""
    out: dict = {"tools": catalog.tools(), "stats": catalog.block_stats()}
    if block:
        out["block"] = catalog.render_oneshot_block()
    return out


@app.post("/slowAgent/{poc_id}")
async def slow_agent(poc_id: str, req: SlowRequest, request: Request) -> JSONResponse:
    """唯一对外接口：poc 慢任务请求 → 编排 steps（一把出）。"""
    started = time.perf_counter()
    raw = (await request.body()).decode("utf-8", "replace")
    logger.info("REQ %s raw_body=%s", poc_id, raw)
    body = await build_response(req)
    logger.info("RESP %s cost=%.0fms body=%s", poc_id,
                (time.perf_counter() - started) * 1000, json.dumps(body, ensure_ascii=False))
    return JSONResponse(content=body, status_code=body.get("code", 200))
