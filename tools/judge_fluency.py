#!/usr/bin/env python3
"""训练样本 query 通顺度判定：不像真实用户会说的话（语病、搭配不当、拼凑、生造词、中英乱混、
意思不通）判 0，丢出训练集。结果缓存在 <目录>/fluency.jsonl（query -> ok/reason），已判不重判。
判完重写 train.csv / train_<域>.csv / train.sft.jsonl，只保留 ok=1；被丢的写 dropped_fluency.csv。
用法：python3 tools/judge_fluency.py [目录，默认 testset/train_pattern_v2] [--concurrency 64] [--limit N 试跑只打印]
"""
from __future__ import annotations
import argparse, asyncio, collections, csv, json, re, sys
from pathlib import Path
import httpx
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.config import get_settings  # noqa: E402

PROMPT = """你在审核电视语音助手的训练数据。下面是若干条"用户对电视说的话"。逐条判断：这句话是否通顺自然、像真实用户会说出口的中文。
判 0（不通顺）的情况：语病/缺字多字（如"兴趣听下XX"）、搭配不当、几个词生硬拼凑、生造的词或不存在的说法、
中英文莫名混杂（如"老默回家杀李响 footage"）、前后矛盾、意思读不懂。
判 1：口语化、省略、简短（如"湖人队""高三的物理"）、带语气词都没关系，只要真实用户会这么说、意思清楚。
只输出 JSON 数组，与输入顺序一一对应：[{{"i": 序号, "ok": 1或0, "why": "判0时简述原因"}}]

{items}"""

async def judge(c, s, batch):
    items = "\n".join(f"{i}. {q}" for i, q in enumerate(batch))
    payload = {"model": s.model, "temperature": 0, "max_tokens": 4096,
               "chat_template_kwargs": {"enable_thinking": False},
               "messages": [{"role": "user", "content": PROMPT.format(items=items)}]}
    for a in range(3):
        try:
            r = await c.post(s.chat_url, json=payload); r.raise_for_status()
            t = r.json()["choices"][0]["message"]["content"] or ""
            v = json.loads(re.search(r"\[.*\]", t, re.S).group(0))
            out = {int(x["i"]): x for x in v if isinstance(x, dict) and "i" in x}
            if len(out) == len(batch):
                return {batch[i]: {"ok": int(out[i].get("ok", 1)), "why": out[i].get("why", "")} for i in out}
        except Exception:  # noqa: BLE001
            await asyncio.sleep(1 + a)
    return {}

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default=str(ROOT / "testset/train_pattern_v2"))
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); d = Path(a.dir); s = get_settings()
    samples = [json.loads(l) for l in open(d / "samples.jsonl", encoding="utf-8") if l.strip()]
    ck = d / "fluency.jsonl"
    res = {}
    if ck.exists():
        for l in open(ck, encoding="utf-8"):
            x = json.loads(l); res[x["query"]] = x
    todo = sorted({x["query"] for x in samples} - set(res))
    if a.limit: todo = todo[:a.limit]
    sem = asyncio.Semaphore(a.concurrency); fh = open(ck, "a", encoding="utf-8")
    async with httpx.AsyncClient(timeout=180, limits=httpx.Limits(max_connections=a.concurrency)) as c:
        async def one(b):
            async with sem: r = await judge(c, s, b)
            for q, v in r.items():
                rec = {"query": q, **v}; res[q] = rec; fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
        for _ in range(3):
            todo = [q for q in todo if q not in res]
            if not todo: break
            await asyncio.gather(*[one(todo[i:i + a.batch]) for i in range(0, len(todo), a.batch)])
    fh.close()
    if a.limit:
        for q in todo:
            if q in res: print(res[q]["ok"], q, res[q].get("why", ""))
        return
    keep = [x for x in samples if res.get(x["query"], {}).get("ok", 1) == 1]
    drop = [x for x in samples if res.get(x["query"], {}).get("ok", 1) == 0]
    (d / "samples.clean.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in keep), encoding="utf-8")
    with open(d / "dropped_fluency.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f); w.writerow(["domain", "query", "tool", "seed_query", "why"])
        for x in drop: w.writerow([x["src_domain"], x["query"], x["tool"], x["seed_query"], res[x["query"]].get("why", "")])
    # 重写 CSV
    import subprocess
    subprocess.run([sys.executable, str(ROOT / "tools/export_pattern_csv.py"), str(d), "samples.clean.jsonl"], check=True)
    # 过滤 sft
    sft = d / "train.sft.jsonl"
    if sft.exists():
        bad = {x["query"] for x in drop}; out = []
        for l in open(sft, encoding="utf-8"):
            u = json.loads(l)["messages"][1]["content"].split("用户请求：", 1)[-1]
            if u not in bad: out.append(l)
        (d / "train.sft.clean.jsonl").write_text("".join(out), encoding="utf-8")
    print(json.dumps({"total": len(samples), "keep": len(keep), "drop": len(drop),
                      "drop_by_domain": dict(collections.Counter(x["src_domain"] for x in drop))}, ensure_ascii=False))

asyncio.run(main())
