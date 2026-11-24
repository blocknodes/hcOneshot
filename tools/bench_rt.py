#!/usr/bin/env python3
"""hcOneshot 接口 RT 压测：POST {url}/slowAgent/{poc_id}。

query 来源：评测集 CSV（默认 all_eval.csv 的 query 列，可 --domain 过滤、--n 抽样），
或 --query 直接指定（可多次）。每个请求用独立随机 deviceId，避免串行状态/多轮上下文干扰，
测的是「首轮一个 tick」的耗时（不带 toolHistory，不跑续跑）。

指标：成功率、avg / p50 / p90 / p95 / p99 / max（毫秒）、吞吐 QPS；按并发档位、按域分别统计。
明细写到 --out（CSV：domain, query, concurrency, status, rt_ms, steps, stop, tools, error）。

用法：
  python3 tools/bench_rt.py --url http://127.0.0.1:4006 --n 200 --concurrency 1,4,16
  python3 tools/bench_rt.py --url http://127.0.0.1:4006 --query "播放战狼2" --repeat 20
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import statistics
import time
import uuid
from collections import defaultdict
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(rs: list[dict], wall: float | None = None) -> dict:
    ok = [r["rt_ms"] for r in rs if r["status"] == 200 and not r["error"]]
    out = {"n": len(rs), "ok": len(ok), "ok_rate": round(len(ok) / max(1, len(rs)), 4)}
    if ok:
        out.update(avg=round(statistics.mean(ok)), p50=round(pct(ok, 50)), p90=round(pct(ok, 90)),
                   p95=round(pct(ok, 95)), p99=round(pct(ok, 99)), max=round(max(ok)),
                   min=round(min(ok)))
    if wall:
        out["qps"] = round(len(rs) / wall, 2)
    return out


def load_queries(args) -> list[tuple[str, str]]:
    if args.query:
        return [("-", q) for q in args.query] * args.repeat
    with open(args.csv, encoding="utf-8-sig") as fh:
        rows = [(r.get("domain", "-"), r["query"]) for r in csv.DictReader(fh) if r.get("query")]
    if args.domain:
        rows = [r for r in rows if r[0] in set(args.domain.split(","))]
    random.Random(args.seed).shuffle(rows)
    return rows[:args.n] if args.n else rows


async def one(client, url, poc, dom, q, timeout):
    body = {"traceId": f"rt-{uuid.uuid4().hex[:12]}", "deviceId": f"rt-{uuid.uuid4().hex}",
            "deviceType": "VIDAA",
            "data": {"query": q, "memory": {"shortMemory": []}, "toolHistory": [],
                     "toolList": [], "tvMode": "0", "timestamp": int(time.time() * 1000)}}
    t0 = time.perf_counter()
    rec = {"domain": dom, "query": q, "status": 0, "rt_ms": 0.0, "steps": 0, "stop": "",
           "tools": "", "error": ""}
    try:
        r = await client.post(f"{url}/slowAgent/{poc}", json=body, timeout=timeout)
        rec["rt_ms"] = (time.perf_counter() - t0) * 1000
        rec["status"] = r.status_code
        d = r.json()
        steps = (d.get("data") or {}).get("steps") or []
        rec["steps"] = len(steps)
        rec["stop"] = d.get("stop")
        rec["tools"] = "|".join(s.get("toolName", "") for s in steps)
        if r.status_code != 200:
            rec["error"] = str(d.get("message", ""))[:200]
    except Exception as exc:  # noqa: BLE001
        rec["rt_ms"] = (time.perf_counter() - t0) * 1000
        rec["error"] = f"{type(exc).__name__}: {exc}"[:200]
    return rec


async def run_level(url, poc, items, conc, timeout):
    sem = asyncio.Semaphore(conc)
    # trust_env=False：内网地址不走 http_proxy
    async with httpx.AsyncClient(trust_env=False,
                                 limits=httpx.Limits(max_connections=conc,
                                                     max_keepalive_connections=conc)) as client:
        async def guarded(dom, q):
            async with sem:
                return await one(client, url, poc, dom, q, timeout)
        t0 = time.perf_counter()
        rs = await asyncio.gather(*[guarded(d, q) for d, q in items])
        return list(rs), time.perf_counter() - t0


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:4006")
    ap.add_argument("--poc", default="poc_rt")
    ap.add_argument("--csv", default=str(ROOT / "all_eval.csv"))
    ap.add_argument("--domain", default="", help="只测这些域，逗号分隔（评测集 domain 列取值）")
    ap.add_argument("--n", type=int, default=200, help="每档并发的请求数（0=全部）")
    ap.add_argument("--query", action="append", default=[], help="直接指定 query，可多次")
    ap.add_argument("--repeat", type=int, default=10, help="--query 模式下每条重复次数")
    ap.add_argument("--concurrency", default="1,4,16", help="并发档位，逗号分隔")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "testset" / "rt_detail.csv"))
    args = ap.parse_args()

    url = args.url.rstrip("/")
    items = load_queries(args)
    if not items:
        raise SystemExit("没有可用的 query")
    async with httpx.AsyncClient(trust_env=False) as c:
        h = await c.get(f"{url}/health", timeout=10)
        print(f"health {h.status_code} {h.text[:120]}")
    if args.warmup:
        await run_level(url, args.poc, items[:args.warmup], 1, args.timeout)

    all_rs = []
    print(f"\nURL {url}/slowAgent/{args.poc}  每档 {len(items)} 请求\n")
    hdr = f"{'conc':>5} {'n':>5} {'ok%':>6} {'avg':>6} {'p50':>6} {'p90':>6} {'p95':>6} {'p99':>6} {'max':>6} {'qps':>7}"
    print(hdr)
    print("-" * len(hdr))
    levels = {}
    for conc in [int(x) for x in args.concurrency.split(",") if x.strip()]:
        rs, wall = await run_level(url, args.poc, items, conc, args.timeout)
        for r in rs:
            r["concurrency"] = conc
        all_rs += rs
        s = summarize(rs, wall)
        levels[conc] = s
        print(f"{conc:>5} {s['n']:>5} {s['ok_rate']*100:>5.1f}% {s.get('avg',0):>6} {s.get('p50',0):>6} "
              f"{s.get('p90',0):>6} {s.get('p95',0):>6} {s.get('p99',0):>6} {s.get('max',0):>6} {s.get('qps',0):>7}")

    # 按域（取最低并发档，反映单请求 RT）
    base = min(levels)
    by = defaultdict(list)
    for r in all_rs:
        if r["concurrency"] == base:
            by[r["domain"]].append(r)
    if len(by) > 1:
        print(f"\n按域（并发={base}）")
        print(f"{'domain':>11} {'n':>5} {'avg':>6} {'p50':>6} {'p90':>6} {'p99':>6} {'max':>6}")
        for d, rs in sorted(by.items()):
            s = summarize(rs)
            print(f"{d:>11} {s['n']:>5} {s.get('avg',0):>6} {s.get('p50',0):>6} {s.get('p90',0):>6} "
                  f"{s.get('p99',0):>6} {s.get('max',0):>6}")
    errs = [r for r in all_rs if r["error"] or r["status"] != 200]
    if errs:
        print(f"\n失败 {len(errs)} 条，例：")
        for r in errs[:5]:
            print(f"   [{r['status']}] {r['query']} -> {r['error']}")
    slow = sorted(all_rs, key=lambda r: -r["rt_ms"])[:5]
    print("\n最慢 5 条：")
    for r in slow:
        print(f"   {r['rt_ms']:.0f}ms  conc={r['concurrency']}  {r['query']}  -> {r['tools']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["domain", "query", "concurrency", "status", "rt_ms",
                                           "steps", "stop", "tools", "error"])
        w.writeheader()
        for r in all_rs:
            w.writerow({**r, "rt_ms": round(r["rt_ms"], 1)})
    print(f"\n明细 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
