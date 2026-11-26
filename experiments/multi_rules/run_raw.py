#!/usr/bin/env python3
"""实验：对评测 csv 的每条 query 直接调引擎首轮计划（一次 LLM），缓存【原始计划】到 json，供规则层离线反复迭代。
不改动 app/。用法：python3 experiments/multi_rules/run_raw.py [in.csv=3domain_revised.csv] [out.json=experiments/multi_rules/raw_plans.json]"""
import asyncio, csv, json, sys, logging
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
logging.disable(logging.WARNING)
from app import engine  # noqa: E402

async def one(q, sem):
    async with sem:
        try:
            plan = await engine._sm()._plan(q)
        except Exception as e:  # noqa: BLE001
            return {"query": q, "err": repr(e)[:200], "steps": []}
        steps = [{"q": it.query, "d": it.domain, "tool": it.tool, "params": it.params} for it in plan.intents]
        return {"query": q, "steps": steps}

async def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "3domain_revised.csv")
    dst = Path(sys.argv[2] if len(sys.argv) > 2 else ROOT / "experiments/multi_rules/raw_plans.json")
    qs = [r["query"] for r in csv.DictReader(open(src, encoding="utf-8-sig"))]
    sem = asyncio.Semaphore(8)
    res = await asyncio.gather(*[one(q, sem) for q in qs])
    json.dump(res, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("done", len(res), "err", sum(1 for r in res if r.get("err")))
asyncio.run(main())
