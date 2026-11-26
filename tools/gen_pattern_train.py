#!/usr/bin/env python3
"""把 testset/feishu/20261116 七个域的例子聚合成 pattern，每个 pattern 按 1:3 造训练数据。

pattern = (域, 工具, 参数结构签名)。结构签名 = 参数里的键路径 + field/operator/action 等结构取值，
忽略叶子取值（片名/人名/数字…）；设备域的 operation 也计入签名（操作类型不同即不同 pattern）。
每个 pattern 内每条种子生成 3 条变体（种子:生成 = 1:3）；变体校验沿用 gen_train.validate。

产物（testset/train_pattern/）：patterns.csv（pattern 清单）、samples.jsonl（断点）、
train.csv、train.sft.jsonl、report.json
用法：python3 tools/gen_pattern_train.py [--ratio 3] [--concurrency 32] [--limit N]
"""
from __future__ import annotations

import argparse, asyncio, collections, csv, hashlib, json, sys, time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import gen_train as G  # noqa: E402
from app import catalog, slots  # noqa: E402
from app.engine import _now_line  # noqa: E402
from app.prompts import build_oneshot_prompt  # noqa: E402

SRC = ROOT / "testset/feishu/20261116"
OUT = ROOT / "testset/train_pattern"
FILES = {"vod": "vod_eval.csv", "educ": "educ_eval.csv", "edu": "edu_eval.csv", "music": "music_eval.csv",
         "audio": "audio_eval.csv", "sports": "sports_eval.csv", "tv_control": "tv_control_eval.csv"}
SIG_VALUE_KEYS = {"field", "operator", "action", "order", "operation"}


def signature(o, path=""):
    if isinstance(o, dict):
        return "{" + ",".join(f"{k}:{signature(v, k)}" for k, v in sorted(o.items())) + "}"
    if isinstance(o, list):
        parts = [signature(x, path) for x in o]
        if path in ("and", "or"):
            parts = sorted(parts)
        return "[" + ",".join(parts) + "]"
    return json.dumps(o, ensure_ascii=False) if path in SIG_VALUE_KEYS else "_"


def load():
    seeds, skipped = [], collections.Counter()
    for dom, fn in FILES.items():
        for r in csv.DictReader(open(SRC / fn, encoding="utf-8-sig")):
            q, tool = r["query"].strip(), G.TOOL_ALIAS.get(r["期望工具"], r["期望工具"])
            d = G.DOMAIN_MAP[dom]
            if tool not in catalog.valid_tools_for(d):
                skipped["tool_not_in_catalog"] += 1; continue
            raw = r["期望参数"].strip()
            if not raw:
                if tool != "fan_knowledge_agent":
                    skipped["missing_label"] += 1; continue
                params = {}
            else:
                try: params = json.loads(raw)
                except Exception: skipped["bad_json"] += 1; continue
            for k in catalog.system_filled(tool): params.pop(k, None)
            if G.REL_TIME.search(q):
                skipped["relative_time"] += 1; continue
            sig = f"{d}|{tool}|{signature(params)}"
            seeds.append({"src_domain": dom, "d": d, "tool": tool, "query": q, "params": params,
                          "pattern": sig, "pid": "P" + hashlib.md5(sig.encode()).hexdigest()[:8]})
    return seeds, skipped


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max-rounds", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    seeds, skipped = load()
    if a.limit: seeds = seeds[:a.limit]
    OUT.mkdir(parents=True, exist_ok=True)
    pats = collections.defaultdict(list)
    for s in seeds: pats[s["pid"]].append(s)
    with open(OUT / "patterns.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f); w.writerow(["pid", "domain", "tool", "seeds", "target", "signature", "example"])
        for pid, l in sorted(pats.items(), key=lambda x: (x[1][0]["src_domain"], x[1][0]["tool"], -len(x[1]))):
            w.writerow([pid, l[0]["src_domain"], l[0]["tool"], len(l), len(l) * a.ratio,
                        l[0]["pattern"].split("|", 2)[2], l[0]["query"]])
    eval_q = {s["query"] for s in seeds}
    ck = OUT / "samples.jsonl"
    samples = [json.loads(l) for l in open(ck, encoding="utf-8")] if ck.exists() else []
    seen = {s["query"] for s in samples} | eval_q
    per = collections.defaultdict(list)
    for s in samples: per[s["seed_query"] + "\0" + s["tool"]].append(s["query"])
    drop = collections.Counter(); dead = collections.Counter()
    sem = asyncio.Semaphore(a.concurrency); fh = open(ck, "a", encoding="utf-8"); t0 = time.time()
    async with httpx.AsyncClient(timeout=180, limits=httpx.Limits(max_connections=a.concurrency)) as c:
        async def one(sd):
            key = sd["query"] + "\0" + sd["tool"]
            need = a.ratio - len(per[key])
            async with sem:
                vs = await G.gen_variants(c, sd, max(need + 1, 3), per[key])
            got = 0
            for v in vs:
                if len(per[key]) >= a.ratio: break
                q = str(v.get("query") or "").strip(); p = v.get("params")
                if isinstance(p, dict):
                    for kk in catalog.system_filled(sd["tool"]): p.pop(kk, None)
                why = G.validate(sd["query"], sd["params"], q, p) or G.rule_reject({"query": q, "tool": sd["tool"]})
                if not why and q in seen: why = "dup"
                if why: drop[why] += 1; continue
                rec = {"src_domain": sd["src_domain"], "d": sd["d"], "tool": sd["tool"], "pid": sd["pid"],
                       "seed_query": sd["query"], "query": q, "params": G._prune(p)}
                seen.add(q); per[key].append(q); samples.append(rec)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); got += 1
            fh.flush(); dead[key] = 0 if got else dead[key] + 1
        for rnd in range(a.max_rounds):
            batch = [s for s in seeds if len(per[s["query"] + "\0" + s["tool"]]) < a.ratio
                     and dead[s["query"] + "\0" + s["tool"]] < 3]
            if not batch: break
            await asyncio.gather(*[one(s) for s in batch])
            print(f"[round {rnd}] calls={len(batch)} samples={len(samples)}/{len(seeds)*a.ratio} drop={dict(drop)} {time.time()-t0:.0f}s", flush=True)
    fh.close()
    block, now = catalog.render_oneshot_block(), _now_line()
    ss = asyncio.Semaphore(32)
    async def sb(q):
        async with ss: return await slots.slot_block_for(q)
    blocks = await asyncio.gather(*[sb(s["query"]) for s in samples])
    with open(OUT / "train.sft.jsonl", "w", encoding="utf-8") as fo:
        for s, b in zip(samples, blocks):
            asst = [{"q": s["query"], "d": s["d"], "vt": f"{s['d']}:{s['tool']}", "params": s["params"]}]
            fo.write(json.dumps({"messages": [
                {"role": "system", "content": build_oneshot_prompt(block, b)},
                {"role": "user", "content": f"{now}\n用户请求：{s['query']}"},
                {"role": "assistant", "content": json.dumps(asst, ensure_ascii=False)}]}, ensure_ascii=False) + "\n")
    with open(OUT / "train.csv", "w", encoding="utf-8-sig", newline="") as fo:
        w = csv.writer(fo); w.writerow(["domain", "pid", "query", "tool", "tool_param", "seed_query"])
        for s in sorted(samples, key=lambda s: (s["src_domain"], s["pid"], s["seed_query"])):
            w.writerow([s["src_domain"], s["pid"], s["query"], s["tool"],
                        json.dumps(s["params"], ensure_ascii=False, separators=(",", ":")), s["seed_query"]])
    gen = collections.Counter(s["pid"] for s in samples)
    rep = {"seeds": len(seeds), "skipped": dict(skipped), "patterns": len(pats), "samples": len(samples),
           "ratio_actual": round(len(samples) / max(1, len(seeds)), 3), "drop": dict(drop),
           "patterns_full": sum(1 for p, l in pats.items() if gen[p] >= len(l) * a.ratio),
           "by_domain": {d: {"seeds": sum(1 for s in seeds if s["src_domain"] == d),
                             "gen": sum(1 for s in samples if s["src_domain"] == d)} for d in FILES}}
    json.dump(rep, open(OUT / "report.json", "w"), ensure_ascii=False, indent=2)
    print(json.dumps(rep, ensure_ascii=False, indent=2))

asyncio.run(main())
