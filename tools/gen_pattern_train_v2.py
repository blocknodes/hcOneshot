#!/usr/bin/env python3
"""把 testset/feishu/20261116 七个域的例子聚合成 pattern，每个 pattern 按 1:3 造训练数据。

pattern = (域, 工具, 参数结构签名)。结构签名 = 参数里的键路径 + field/operator/action 等结构取值，
忽略叶子取值（片名/人名/数字…）；设备域的 operation 也计入签名（操作类型不同即不同 pattern）。
每个 pattern 内每条种子生成 3 条变体（种子:生成 = 1:3）；变体校验沿用 gen_train.validate。

产物（testset/train_pattern/）：patterns.csv（pattern 清单）、samples.jsonl（断点）、
train.csv、train.sft.jsonl、report.json
用法：python3 tools/gen_pattern_train.py [--ratio 3] [--concurrency 32] [--limit N]

v2（本文件）：强制换槽位。种子里来自原话的槽值（片名/人名/歌名/球队/年级/科目/数值/设备对象…）
每条变体都必须换成别的真实取值（LLM 额外输出 replaced={旧:新}，校验旧值不在新 query、新值在新 query）；
种子无可换槽位（如“继续播放”“我收藏的歌”）才允许纯改写。设备域 object 只能换成评测集中同工具同 operation
出现过的 object。产物写 testset/train_pattern_v2/，另按域输出 train_<域>.csv。
"""
from __future__ import annotations

import argparse, asyncio, collections, csv, hashlib, json, re, sys, time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import gen_train as G  # noqa: E402
from app import catalog, slots  # noqa: E402
from app.engine import _now_line  # noqa: E402
from app.prompts import build_oneshot_prompt  # noqa: E402

SRC = ROOT / "testset/feishu/20261116"
OUT = ROOT / ("testset/train_pattern_v2" if "--limit" not in sys.argv else "testset/train_pattern_v2_try")
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


SWAP_PROMPT = """你是电视语音助手的训练数据生成器。给你一条【种子样本】：用户 query、工具、参数 params（金标）。
请生成 {k} 条【新的】用户 query 及对应 params。核心要求：**替换槽位**。
1. 工具不变；params 结构完全不变（键名、field、嵌套、and/or、条件个数一致），只改叶子取值；field/operator/action/order/operation 不许改。
2. 找出种子 query 里所有【可替换槽位】：片名、人名（演员/导演/歌手/角色）、歌名、专辑、IP、球队、球星、赛事、运动项目、
   年份、评分、集数、数值、年级、学科、教材版本、地区、类型标签、设备对象等。每条新 query 都必须把这些槽位
   **全部或大部分换成别的真实存在、合理的取值**（不同条之间取值也要不同），同时换个口语说法；params 里对应叶子同步改成新值。
   不要只改说法而保留原槽值。
3. 新 query 意图、动作与种子一致（播放仍是播放，搜索仍是搜索，问答仍是问答），像真实电视用户的口语。
4. params 里来自原话的取值，新 query 必须原样出现新值；归一化取值（国语→普通话、初一→七年级、is_fee=0 等）保持种子口径。
   种子 query 没提到的条件，新 query 也不要凭空加；种子提到的条件，新 query 也要提到。
5. params 为空 {{}} 时，query 里的槽位（如片名、台词、场景、科目）照样要换，但保持同一类请求（描述仍是描述）。
6. 只有种子确实没有任何可替换槽位（如"继续播放""我收藏的歌""打开历史"）时，才只换说法，此时 replaced 为 {{}}。
7. 不要出现相对时间（今天/明天/最近…）；不要加 retext。{extra}
只输出 JSON 数组：[{{"query": "...", "params": {{...}}, "replaced": {{"原槽值": "新槽值", ...}}}}, ...]

【种子样本】
query: {query}
tool: {tool}
params: {params}"""

DEVICE_OBJ = collections.defaultdict(set)   # (tool, operation) -> 评测集中出现过的 object


def _slot_leaves(seed_q, params):
    return [v for path, v in G._leaves(params) if isinstance(v, str) and v and v in seed_q
            and not (path and path[-1] in SIG_VALUE_KEYS)]


def check_swap(sd, q, p, rep):
    """强制换槽校验；返回丢弃原因，空串=通过。"""
    if not isinstance(rep, dict):
        rep = {}
    rep = {str(k): str(v) for k, v in rep.items() if str(k) and str(v) and str(k) != str(v)}
    slot = _slot_leaves(sd["query"], sd["params"])
    if sd["d"] == "device" and not (DEVICE_OBJ[(sd["tool"], sd["params"].get("operation"))] - {sd["params"].get("object")}):
        slot = [v for path, v in G._leaves(sd["params"]) if path and path[-1] != "object"
                and isinstance(v, str) and v and v in sd["query"] and path[-1] not in SIG_VALUE_KEYS]
    if slot:
        newv = {v for _, v in G._leaves(p)}
        if sum(1 for v in slot if v not in newv) == 0:
            return "slot_not_swapped"
    elif sd.get("swappable") and not rep:
        return "slot_not_swapped"
    for old, new in rep.items():
        if old in sd["query"] and old in q and old not in new:
            return "old_slot_kept"
        if new not in q:
            return "replaced_not_in_query"
    if sd["d"] == "device" and isinstance(p, dict):
        obj, op = p.get("object"), p.get("operation")
        if obj and obj != sd["params"].get("object") and obj not in DEVICE_OBJ[(sd["tool"], op)]:
            return "device_obj_unknown"
    return ""


async def gen_swap(client, sd, k, avoid):
    s = G.get_settings()
    extra = ""
    if sd["d"] == "device":
        objs = sorted(DEVICE_OBJ[(sd["tool"], sd["params"].get("operation"))] - {sd["params"].get("object")})
        extra = ("\n8. 设备对象 object 只能从这些里选（或保持原值）：" + "、".join(objs[:80])) if objs else \
                "\n8. 设备对象 object 保持原值，只换数值/说法。"
    content = SWAP_PROMPT.format(k=k, query=sd["query"], tool=sd["tool"], extra=extra,
                                 params=json.dumps(sd["params"], ensure_ascii=False))
    if avoid:
        content += "\n\n以下已经有了，不要重复，槽值也别再用：\n" + "\n".join(avoid[-20:])
    payload = {"model": s.model, "temperature": 1.0, "max_tokens": 2048,
               "chat_template_kwargs": {"enable_thinking": False},
               "messages": [{"role": "user", "content": content}]}
    for attempt in range(3):
        try:
            r = await client.post(s.chat_url, json=payload); r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"] or ""
            m = re.search(r"\[.*\]", text, re.S)
            v = json.loads(m.group(0)) if m else []
            return [x for x in v if isinstance(x, dict)]
        except Exception:  # noqa: BLE001
            await asyncio.sleep(1 + attempt)
    return []


_NO_SLOT = re.compile(r"^(继续|接着|暂停|停止|返回|退出|上一|下一|重新|打开|关闭|我收藏|收藏的|历史|播放记录|随便)")


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
            if d == "device" and params.get("object"):
                DEVICE_OBJ[(tool, params.get("operation"))].add(params["object"])
            sig = f"{d}|{tool}|{signature(params)}"
            seeds.append({"src_domain": dom, "d": d, "tool": tool, "query": q, "params": params,
                          "pattern": sig, "swappable": bool(_slot_leaves(q, params)) or (not params and d != "device" and len(q) > 6 and not _NO_SLOT.match(q)), "pid": "P" + hashlib.md5(sig.encode()).hexdigest()[:8]})
    return seeds, skipped


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max-rounds", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    seeds, skipped = load()
    if a.limit:
        import random; random.seed(0); bd = collections.defaultdict(list)
        for x in seeds: bd[x["src_domain"]].append(x)
        seeds = [x for l in bd.values() for x in random.sample(l, min(a.limit, len(l)))]
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
                vs = await gen_swap(c, sd, max(need + 2, 4), per[key])
            got = 0
            for v in vs:
                if len(per[key]) >= a.ratio: break
                q = str(v.get("query") or "").strip(); p = v.get("params")
                if isinstance(p, dict):
                    for kk in catalog.system_filled(sd["tool"]): p.pop(kk, None)
                why = G.validate(sd["query"], sd["params"], q, p) or G.rule_reject({"query": q, "tool": sd["tool"]}) \
                    or (check_swap(sd, q, p, v.get("replaced")) if isinstance(p, dict) else "params_not_object")
                if not why and q in seen: why = "dup"
                if why: drop[why] += 1; continue
                rec = {"src_domain": sd["src_domain"], "d": sd["d"], "tool": sd["tool"], "pid": sd["pid"],
                       "seed_query": sd["query"], "query": q, "params": G._prune(p),
                       "replaced": v.get("replaced") if isinstance(v.get("replaced"), dict) else {}}
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
        w = csv.writer(fo); w.writerow(["domain", "pid", "query", "tool", "tool_param", "seed_query", "replaced"])
        for s in sorted(samples, key=lambda s: (s["src_domain"], s["pid"], s["seed_query"])):
            w.writerow([s["src_domain"], s["pid"], s["query"], s["tool"],
                        json.dumps(s["params"], ensure_ascii=False, separators=(",", ":")), s["seed_query"]])
    hdr = ["domain", "pid", "query", "tool", "tool_param", "seed_query", "replaced"]
    byd = collections.defaultdict(list)
    for s in samples: byd[s["src_domain"]].append(s)
    for dom, l in byd.items():
        with open(OUT / f"train_{dom}.csv", "w", encoding="utf-8-sig", newline="") as fo:
            w = csv.writer(fo); w.writerow(hdr)
            for s in sorted(l, key=lambda s: (s["pid"], s["seed_query"])):
                w.writerow([dom, s["pid"], s["query"], s["tool"],
                            json.dumps(s["params"], ensure_ascii=False, separators=(",", ":")), s["seed_query"],
                            json.dumps(s.get("replaced") or {}, ensure_ascii=False)])
    gen = collections.Counter(s["pid"] for s in samples)
    rep = {"seeds": len(seeds), "skipped": dict(skipped), "patterns": len(pats), "samples": len(samples),
           "ratio_actual": round(len(samples) / max(1, len(seeds)), 3), "drop": dict(drop),
           "patterns_full": sum(1 for p, l in pats.items() if gen[p] >= len(l) * a.ratio),
           "by_domain": {d: {"seeds": sum(1 for s in seeds if s["src_domain"] == d),
                             "gen": sum(1 for s in samples if s["src_domain"] == d)} for d in FILES}}
    json.dump(rep, open(OUT / "report.json", "w"), ensure_ascii=False, indent=2)
    print(json.dumps(rep, ensure_ascii=False, indent=2))

asyncio.run(main())
