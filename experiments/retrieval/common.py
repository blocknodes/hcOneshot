"""实验公共件：评测集载入、LLM 调用（带磁盘缓存）、判分（口径同 tools/eval_stats.py）。

实验性质，不被 app/ 引用。默认走 hcProxy（127.0.0.1:7000，调用会被其落盘捕获）；EXP_LLM_URL 可改。
"""
from __future__ import annotations

import asyncio
import collections
import csv
import hashlib
import json
import logging
import os
import re
import sys
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent                      # hcOneshot/
sys.path.insert(0, str(ROOT))
os.environ["HC_CATALOG_FILE"] = str(ROOT / "catalog.full.json")   # .env 里是相对路径
os.environ["HC_SLOT_RETRIEVE_URL"] = ""                          # 槽值检索由 slot_layer 自己调
sys.path.insert(0, str(ROOT / "tools"))
logging.getLogger("app.parser").setLevel(logging.WARNING)
logging.getLogger("hcOneshot").setLevel(logging.WARNING)

from app.parser import parse_json_loose, parse_oneshot  # noqa: E402
from eval_stats import canon  # noqa: E402
from gen_sft_dpo import TOOL_ALIAS  # noqa: E402

LLM_URL = os.environ.get("EXP_LLM_URL", "http://127.0.0.1:7000/v1/chat/completions")
LLM_MODEL = os.environ.get("EXP_LLM_MODEL", "baseline")
# 评测集相对时间金标多按 2026-08-24（周一）写死；各 arm 统一用这一时间行，保证可比。
NOW_LINE = "当前时间：2026-08-24 18:00 星期一（北京时间）"
CACHE = HERE / "cache" / "llm.jsonl"
DOMAIN_ORDER = ["vod", "educ", "edu", "music", "audio", "sports", "tv_control"]
EVAL_DOMAIN = {"vod": "vod", "educ": "children", "edu": "education", "music": "music",
               "audio": "audio", "sports": "sports", "tv_control": "device"}


def nq(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
def load_gold(path: Path) -> list[dict]:
    """按去空白 query 合并重复行：首行为主答案，同工具的其余行与 tool_param_alt 进 accept。"""
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    by: dict[str, dict] = {}
    for r in rows:
        k = nq(r["query"])
        tool = TOOL_ALIAS.get(r["tool"].strip(), r["tool"].strip())
        tp = (r.get("tool_param") or "").strip()
        p = parse_json_loose(tp) if tp else ({} if tool == "fan_knowledge_agent" else None)
        alts = parse_json_loose(r.get("tool_param_alt") or "")
        alts = [a for a in alts if isinstance(a, dict)] if isinstance(alts, list) else []
        if k in by:
            g = by[k]
            if g["tool"] == tool and isinstance(p, dict):
                g["alts"].append(p)
            g["alts"] += alts
            continue
        by[k] = {"domain": r["domain"].strip(), "query": r["query"].strip(), "tool": tool,
                 "params": p if isinstance(p, dict) else None, "alts": alts,
                 "seed_query": (r.get("seed_query") or "").strip()}
    return list(by.values())


# ---------------------------------------------------------------------------
# 判分（与 eval_stats.judge 同口径；取模型输出第 1 条）
# ---------------------------------------------------------------------------
def judge(raw: str, g: dict) -> dict:
    intents, _err = parse_oneshot(raw or "")
    first = intents[0] if intents else None
    pred_tool = TOOL_ALIAS.get(first.tool, first.tool) if first else ""
    tool_ok = pred_tool == g["tool"]
    scored = g["params"] is not None
    ok = False
    pc = canon(first.params, g["tool"]) if first else ""
    if scored and tool_ok:
        accept = {canon(g["params"], g["tool"])} | {canon(a, g["tool"]) for a in g["alts"]}
        ok = pc in accept
    return {"pred_tool": pred_tool, "pred_d": first.d if first else "", "n_intents": len(intents),
            "tool_ok": tool_ok, "param_scored": scored, "tool_param_ok": ok,
            "gold_canon": canon(g["params"], g["tool"]) if scored else "", "pred_canon": pc}


def summarize(golds: list[dict], js: list[dict]) -> dict:
    st: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for g, j in zip(golds, js):
        for d in (g["domain"], "ALL"):
            s = st[d]
            s["n"] += 1
            s["tool"] += j["tool_ok"]
            s["n_param"] += j["param_scored"]
            s["tp"] += j["tool_param_ok"]
    order = [d for d in DOMAIN_ORDER if d in st] + ["ALL"]
    return {d: {"n": st[d]["n"], "tool": round(st[d]["tool"] / st[d]["n"], 4),
                "tool_param": round(st[d]["tp"] / max(1, st[d]["n_param"]), 4)} for d in order}


def print_summary(name: str, summ: dict, base: dict | None = None) -> None:
    print(f"\n[{name}]  {'域':<11}{'样本':>6}{'tool':>9}{'tool+param':>12}" + ("   Δtool  Δt+p" if base else ""))
    for d, s in summ.items():
        line = f"{'':>{len(name) + 4}}{d:<11}{s['n']:>6}{s['tool'] * 100:8.1f}%{s['tool_param'] * 100:11.1f}%"
        if base and d in base:
            line += f"  {(s['tool'] - base[d]['tool']) * 100:+6.1f}{(s['tool_param'] - base[d]['tool_param']) * 100:+6.1f}"
        print(line)


# ---------------------------------------------------------------------------
# LLM（磁盘缓存：key = sha1(system + user + model)）
# ---------------------------------------------------------------------------
_cache: dict[str, str] | None = None


def _load_cache() -> dict[str, str]:
    global _cache
    if _cache is None:
        _cache = {}
        if CACHE.exists():
            for line in open(CACHE, encoding="utf-8"):
                try:
                    o = json.loads(line)
                    _cache[o["k"]] = o["v"]
                except (json.JSONDecodeError, KeyError):
                    pass
    return _cache


def _key(system: str, user: str) -> str:
    return hashlib.sha1(f"{LLM_MODEL}\x00{system}\x00{user}".encode()).hexdigest()


async def chat_many(pairs: list[tuple[str, str]], concurrency: int = 48) -> list[str]:
    """pairs=[(system, user)] → 原始输出列表（缓存命中不发请求）。"""
    cache = _load_cache()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    out: list[str | None] = [cache.get(_key(s, u)) for s, u in pairs]
    todo = [i for i, v in enumerate(out) if v is None]
    if not todo:
        return out  # type: ignore[return-value]
    sem = asyncio.Semaphore(concurrency)
    fh = open(CACHE, "a", encoding="utf-8")
    done = 0

    async with httpx.AsyncClient(timeout=300, trust_env=False,
                                 limits=httpx.Limits(max_connections=concurrency)) as client:
        async def one(i: int) -> None:
            nonlocal done
            s, u = pairs[i]
            payload = {"model": LLM_MODEL, "temperature": 0.0, "max_tokens": 1024,
                       "chat_template_kwargs": {"enable_thinking": False},
                       "messages": [{"role": "system", "content": s}, {"role": "user", "content": u}]}
            text = ""
            for attempt in range(4):
                try:
                    async with sem:
                        r = await client.post(LLM_URL, json=payload)
                    r.raise_for_status()
                    text = r.json()["choices"][0]["message"]["content"] or ""
                    break
                except Exception:  # noqa: BLE001
                    await asyncio.sleep(2 + 3 * attempt)
            out[i] = text
            k = _key(s, u)
            cache[k] = text
            fh.write(json.dumps({"k": k, "v": text}, ensure_ascii=False) + "\n")
            done += 1
            if done % 500 == 0:
                fh.flush()
                print(f"  llm {done}/{len(todo)}", flush=True)

        await asyncio.gather(*[one(i) for i in todo])
    fh.close()
    return out  # type: ignore[return-value]


def user_msg(query: str) -> str:
    return f"{NOW_LINE}\n用户请求：{query}"
