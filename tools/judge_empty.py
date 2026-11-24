#!/usr/bin/env python3
"""参数为 {} 的训练样本：成对判定「变体与种子是否同一类请求、仍适用同一工具」。

背景：fuzzy / fan / music 透传类工具参数为空，结构校验约束不到，变体会漂成别的意图
（如场景描述 → 裸片名「中国合伙人」、→ 设备指令「切换到超清电视模式」）。
只判意图类别，不判参数（上一版 LLM 判官挑参数刺，误杀约一半）。
结果写回 samples.jsonl 的 pair_judge 字段（1 同类 / 0 漂移），已判的不重判。
用法：python3 tools/judge_empty.py [--limit N 试跑只打印] [--concurrency 64]
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import random
import re
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import catalog  # noqa: E402
from app.catalog import _brief_of  # noqa: E402
from app.config import get_settings  # noqa: E402

CKPT = ROOT / "testset" / "train" / "samples.jsonl"

PROMPT = """你在给电视语音助手的训练数据做质检。每条给出：工具及用途、种子 query（已确认用这个工具是对的）、变体 query。
判断【变体是否仍然是同一类请求、仍然应该用这个工具】：
- 同类（写 1）：换了说法，或换了话题/对象/科目/年级，但请求的类型没变。
  例：工具=模糊搜索，种子「楷书教程」→ 变体「有没有国画入门课」：仍是模糊找内容，1。
- 漂移（写 0）：请求类型变了，换个工具更合适。例如：
  · 模糊搜索/场景描述 → 变成一个具体片名或剧名本身（如「中国合伙人」「唐山」），应走精确检索；
  · 找内容 → 变成设备操作（切换模式/画质/信号源/打开某功能）、下载、闲聊；
  · 知识问答 → 变成播放/搜索；播放/搜索 → 变成知识问答；
  · 历史/收藏/随便听 → 变成点具体某首歌；
  · 语义和种子完全无关、或句子不通顺不像真实用户说的话。
只输出 JSON 数组，长度与条目数相同，例如 [1,0,1]。

{items}"""


def brief(d: str, tool: str, cache: dict = {}) -> str:  # noqa: B006
    key = f"{d}:{tool}"
    if key not in cache:
        t = next((t for t in catalog.tools() if t["domain"] == d and t["tool"] == tool), {})
        cache[key] = _brief_of(t)[:60]
    return cache[key]


async def judge(client, batch):
    s = get_settings()
    items = "\n".join(
        f"{i + 1}. 工具: {r['tool']}（{brief(r['d'], r['tool'])}）\n   种子: {r['seed_query']}\n   变体: {r['query']}"
        for i, r in enumerate(batch))
    payload = {"model": s.model, "temperature": 0.0, "max_tokens": 200,
               "chat_template_kwargs": {"enable_thinking": False},
               "messages": [{"role": "user", "content": PROMPT.replace("{items}", items)}]}
    for attempt in range(3):
        try:
            r = await client.post(s.chat_url, json=payload)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"] or ""
            m = re.search(r"\[[\s\d,]*\]", text)
            v = json.loads(m.group(0)) if m else []
            if len(v) == len(batch) and all(x in (0, 1) for x in v):
                return v
        except Exception:  # noqa: BLE001
            pass
        await asyncio.sleep(1 + attempt)
    return None


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--only", default="", help="试跑：只判 query 命中该正则的样本")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(CKPT, encoding="utf-8") if l.strip()]
    todo = [r for r in rows if not r["params"] and "pair_judge" not in r]
    if args.only:
        todo = [r for r in todo if re.search(args.only, r["query"])]
    if args.limit:
        random.Random(1).shuffle(todo)
        todo = todo[:args.limit]
    sem = asyncio.Semaphore(args.concurrency)
    fails = 0
    async with httpx.AsyncClient(timeout=120,
                                 limits=httpx.Limits(max_connections=args.concurrency)) as client:
        async def run(b):
            nonlocal fails
            async with sem:
                v = await judge(client, b)
            if v is None:
                fails += 1
                return
            for r, x in zip(b, v):
                r["pair_judge"] = x
        bs = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]
        await asyncio.gather(*[run(b) for b in bs])
    judged = [r for r in todo if "pair_judge" in r]
    rej = [r for r in judged if r["pair_judge"] == 0]
    print(f"todo={len(todo)} judged={len(judged)} reject={len(rej)} failed_batches={fails}")
    print("reject by tool:", dict(collections.Counter(r["tool"] for r in rej)))
    if args.limit or args.only:
        for r in rej[:40]:
            print("  0 |", r["tool"], "|", r["seed_query"], "=>", r["query"])
        for r in [r for r in judged if r["pair_judge"] == 1][:15]:
            print("  1 |", r["tool"], "|", r["seed_query"], "=>", r["query"])
        return 0
    tmp = CKPT.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(CKPT)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
