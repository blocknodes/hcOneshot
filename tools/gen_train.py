#!/usr/bin/env python3
"""由评测集（all_eval.csv）扩增训练数据：同工具、同参数结构，换说法 / 换槽值。

流程
  1. 种子：all_eval.csv 每行（domain/query/tool/tool_param）。
     跳过：漏标（非 fan 且 tool_param 空）、含相对时间词（金标是绝对日期，与训练时的
     「当前时间」行冲突，标签会错）。旧工具名按 TOOL_ALIAS 归一到当前目录。
  2. 每个种子调 LLM 生成 K 条变体 {"query", "params"}。
  3. 严格校验（不过即丢，丢弃原因计数进报告）：
     · JSON 合法、params 为对象；
     · 结构同构：除叶子值外，键路径/嵌套形状与种子完全一致（不许改字段名、增删条件）；
     · 槽值落地：种子里「出现在原 query 中的叶子值」在变体里换成的新值，必须出现在新 query 中；
     · 去重：与评测集任一 query 相同、或与已生成样本相同 → 丢。
  4. 训练样本（SFT messages）：与线上推理输入逐字同构
       system    = build_oneshot_prompt(目录块, 槽值检索块)
       user      = 当前时间行 + "用户请求：<query>"
       assistant = [{"q": query, "d": 域, "vt": "域:工具", "params": 参数（无 retext，去空值）}]
     比例：每个种子同样生成 K 条，故各域/各工具占比随评测集。

用法：python3 tools/gen_train.py [--k 2] [--concurrency 16] [--limit N]
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import csv
import json
import random
import re
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from app import catalog, slots  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.engine import _now_line  # noqa: E402
from app.prompts import build_oneshot_prompt  # noqa: E402

SEED = ROOT / "all_eval.csv"
OUT = ROOT / "testset" / "train"

DOMAIN_MAP = {"vod": "vod", "educ": "children", "edu": "education", "music": "music",
              "audio": "audio", "sports": "sports", "tv_control": "device"}
TOOL_ALIAS = {"vod_search": "vod_search_all", "educ_search": "educ_search_all",
              "educ_relate_recommend": "educ_relate_search",
              "sports_match_data_search": "sports_rank_search"}
REL_TIME = re.compile(r"今天|明天|昨天|前天|后天|今晚|昨晚|明晚|上周|本周|这周|下周|周末|最近|"
                      r"上个月|本月|这个月|下个月|今年|去年|明年|周[一二三四五六日天]|星期|"
                      r"早上|上午|中午|下午|晚上|凌晨|刚才|几天")

GEN_PROMPT = """你是电视语音助手的训练数据生成器。给你一条【种子样本】：用户 query、选中的工具、以及该工具的参数 params（金标）。
请生成 {k} 条【新的】用户 query，并给出每条 query 对应的 params。要求：
1. 工具不变；params 的结构完全不变：键名、字段名(field)、嵌套层级、and/or/not、条件个数都一模一样，只允许改【叶子取值】。
2. 【意图不变】新 query 的意图类型、动作必须与种子完全一致，只换说法或槽值：种子只报一个人名，新 query 也只报人名
   （不要加"演的电影"）；种子是"播放"就不要变成"推荐/查询/问答"；种子问属性，新 query 也问同一类属性。
   新 query 要像真实电视用户的口语说法，和种子的说法明显不同（换句式/换口头禅/增减语气词），
   并且至少有一条要换掉槽位取值（如换一部片名、换一个演员/歌手/球队/年级/科目/数值），新取值要真实存在、合理。
3. 凡是 params 里来自用户原话的取值（片名/人名/歌名/数字等），新 query 里必须原样出现该取值；
   归一化取值（如 国语→普通话、初一→七年级、is_fee=0）保持种子的归一化口径。
4. 若 params 为空 {{}}（模糊搜索/知识问答/随便听/历史/收藏/频道听歌等），新 query 必须保持同一类请求：
   场景/台词/剧情描述仍是描述，不要变成一个具体片名或剧名；找内容不要变成设备操作或下载；问答不要变成播放。
5. 不要出现相对时间（今天/明天/上周/最近…）；不要加 retext。
只输出 JSON 数组，不要解释：[{{"query": "...", "params": {{...}}}}, ...]

【种子样本】
query: {query}
tool: {tool}
params: {params}"""


# ---------------------------------------------------------------------------
# 结构校验
# ---------------------------------------------------------------------------
def _leaves(o, path=()):
    """(路径, 叶子值) 列表；列表下标计入路径（结构同构要求条件顺序与个数一致）。"""
    if isinstance(o, dict):
        for k in sorted(o):
            yield from _leaves(o[k], path + (k,))
    elif isinstance(o, list):
        for i, x in enumerate(o):
            yield from _leaves(x, path + (i,))
    else:
        yield path, o


def _shape(o):
    return [p for p, _ in _leaves(o)]


_STRUCT_KEYS = {"field", "operator", "action", "order"}


def validate(seed_q: str, seed_p: dict, q: str, p: dict) -> str:
    """返回丢弃原因；空串=通过。"""
    if not isinstance(q, str) or not q.strip():
        return "empty_query"
    if not isinstance(p, dict):
        return "params_not_object"
    if "retext" in p:
        p.pop("retext")
    if _shape(seed_p) != _shape(p):
        return "shape_changed"
    sl = dict(_leaves(seed_p))
    changed = False
    for path, v in _leaves(p):
        sv = sl[path]
        if type(v) is not type(sv):
            return "type_changed"                  # "0" 与 0、"3" 与 3 不许互换
        if path and path[-1] in _STRUCT_KEYS and v != sv:
            return "struct_value_changed"          # field 名 / operator / action 不许改
        if isinstance(sv, str) and sv and sv in seed_q:
            # 种子里来自原话的取值 → 新取值必须出现在新 query
            if not (isinstance(v, str) and v and v in q):
                return "slot_not_in_query"
        changed |= v != sv
    return ""


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
async def gen_variants(client: httpx.AsyncClient, seed: dict, k: int,
                       avoid: list[str] | None = None) -> list[dict]:
    s = get_settings()
    content = GEN_PROMPT.format(k=k, query=seed["query"], tool=seed["tool"],
                                params=json.dumps(seed["params"], ensure_ascii=False))
    if avoid:
        content += "\n\n以下说法已经有了，不要重复，也不要只改一两个字：\n" + "\n".join(avoid[-20:])
    payload = {"model": s.model, "temperature": 0.9, "max_tokens": 2048,
               "chat_template_kwargs": {"enable_thinking": False},
               "messages": [{"role": "user", "content": content}]}
    for attempt in range(3):
        try:
            r = await client.post(s.chat_url, json=payload)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"] or ""
            m = re.search(r"\[.*\]", text, re.S)
            v = json.loads(m.group(0)) if m else []
            return [x for x in v if isinstance(x, dict)]
        except Exception:  # noqa: BLE001
            await asyncio.sleep(1 + attempt)
    return []


def _prune(o):
    """去空值（空串/None/空容器），与 prompt「用户没提的可选字段不要出现」口径一致。"""
    if isinstance(o, dict):
        out = {k: _prune(v) for k, v in o.items()}
        return {k: v for k, v in out.items() if v not in ("", None, [], {})}
    if isinstance(o, list):
        return [_prune(x) for x in o]
    return o


def load_seeds():
    rows = list(csv.DictReader(open(SEED, encoding="utf-8-sig")))
    seeds, skipped = [], collections.Counter()
    for r in rows:
        d = DOMAIN_MAP[r["domain"]]
        tool = TOOL_ALIAS.get(r["tool"], r["tool"])
        if tool not in catalog.valid_tools_for(d):
            skipped["tool_not_in_catalog"] += 1
            continue
        if not r["tool_param"].strip():
            if tool != "fan_knowledge_agent":
                skipped["missing_label"] += 1
                continue
            params = {}
        else:
            params = json.loads(r["tool_param"])
        for k in catalog.system_filled(tool):          # 整句透传参数由系统补，不进金标
            params.pop(k, None)
        if REL_TIME.search(r["query"]):
            skipped["relative_time"] += 1
            continue
        seeds.append({"src_domain": r["domain"], "d": d, "tool": tool,
                      "query": r["query"], "params": params})
    return rows, seeds, skipped


def rule_reject(r: dict) -> str:
    from fix_train import rule_reject as _rr   # 单一事实源
    return _rr(r)


def quotas(rows, seeds, total: int) -> dict[str, int]:
    """按评测集各域占比分配目标条数（最大余数法，合计恰为 total）。"""
    ev = collections.Counter(r["domain"] for r in rows)
    raw = {d: total * ev[d] / len(rows) for d in DOMAIN_MAP}
    q = {d: int(v) for d, v in raw.items()}
    for d in sorted(raw, key=lambda d: raw[d] - q[d], reverse=True)[:total - sum(q.values())]:
        q[d] += 1
    return q


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=50000)
    ap.add_argument("--k", type=int, default=8, help="每次 LLM 调用要的变体数")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--max-rounds", type=int, default=40)
    ap.add_argument("--cap", type=int, default=16, help="每个种子最多保留的变体数")
    ap.add_argument("--exclude-seed", action="append", default=[],
                    help="不再用于补量的种子 query（反复生成漂移样本的歧义种子）")
    ap.add_argument("--build-only", action="store_true", help="跳过生成，只由 samples.jsonl 出产物")
    args = ap.parse_args()

    rows, seeds, skipped = load_seeds()
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt = OUT / "samples.jsonl"                    # 生成结果逐条追加（可断点续跑）
    eval_q = {r["query"].strip() for r in rows}
    target = quotas(rows, seeds, args.total)
    samples: list[dict] = []
    if ckpt.exists():
        samples = [json.loads(l) for l in open(ckpt, encoding="utf-8") if l.strip()]
    seen = {s["query"] for s in samples}
    per_seed: dict[str, list[str]] = collections.defaultdict(list)
    for s in samples:
        per_seed[s["seed_query"] + "\x00" + s["tool"]].append(s["query"])
    have = collections.Counter(s["src_domain"] for s in samples)
    drop = collections.Counter()
    dead: collections.Counter = collections.Counter()   # 连续零产出的种子
    t0 = time.time()
    by_dom: dict[str, list[dict]] = collections.defaultdict(list)
    for sd in seeds:
        if sd["query"] in args.exclude_seed:
            continue
        by_dom[sd["src_domain"]].append(sd)

    sem = asyncio.Semaphore(args.concurrency)
    fh = open(ckpt, "a", encoding="utf-8")
    async with httpx.AsyncClient(timeout=180, limits=httpx.Limits(max_connections=args.concurrency)) as client:
        async def one(seed):
            key = seed["query"] + "\x00" + seed["tool"]
            async with sem:
                vs = await gen_variants(client, seed, args.k, per_seed[key])
            got = 0
            for v in vs:
                if (have[seed["src_domain"]] >= target[seed["src_domain"]]
                        or len(per_seed[key]) >= args.cap):
                    break
                q = str(v.get("query") or "").strip()
                p = v.get("params")
                if isinstance(p, dict):
                    for kk in catalog.system_filled(seed["tool"]):
                        p.pop(kk, None)
                why = validate(seed["query"], seed["params"], q, p)
                if not why:
                    why = rule_reject({"query": q, "tool": seed["tool"]})
                if not why and q in eval_q:
                    why = "dup_eval"
                if not why and q in seen:
                    why = "dup_train"
                if why:
                    drop[why] += 1
                    continue
                rec = {"src_domain": seed["src_domain"], "d": seed["d"], "tool": seed["tool"],
                       "seed_query": seed["query"], "query": q, "params": _prune(p)}
                seen.add(q)
                per_seed[key].append(q)
                have[seed["src_domain"]] += 1
                samples.append(rec)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                got += 1
            fh.flush()
            dead[key] = 0 if got else dead[key] + 1

        for rnd in range(0 if args.build_only else args.max_rounds):
            batch = []
            for d, lst in by_dom.items():
                need = target[d] - have[d]
                if need <= 0:
                    continue
                alive = [sd for sd in lst if dead[sd["query"] + "\x00" + sd["tool"]] < 3
                         and len(per_seed[sd["query"] + "\x00" + sd["tool"]]) < args.cap]
                alive.sort(key=lambda sd: len(per_seed[sd["query"] + "\x00" + sd["tool"]]))
                # 本轮调用数 ≈ 缺口/（k×通过率≈0.5），最少 1，最多该域全部活种子
                n_calls = min(len(alive), max(1, -(-need * 2 // args.k)))
                batch += alive[:n_calls]
            if not batch:
                break
            await asyncio.gather(*[one(sd) for sd in batch])
            print(f"[round {rnd}] calls={len(batch)} have={sum(have.values())}/{args.total} "
                  f"{dict(have)} drop={dict(drop)} {time.time() - t0:.0f}s", flush=True)
    fh.close()

    # 产物：train.csv + train.sft.jsonl（system 与线上同构：目录块 + 本 query 的槽值检索块）
    block = catalog.render_oneshot_block()
    now = _now_line()
    ssem = asyncio.Semaphore(32)

    async def slot_of(q):
        async with ssem:
            return await slots.slot_block_for(q)
    slot_blocks = await asyncio.gather(*[slot_of(s["query"]) for s in samples])
    order = {d: i for i, d in enumerate(DOMAIN_MAP)}
    idx = sorted(range(len(samples)), key=lambda i: (order[samples[i]["src_domain"]],
                                                      samples[i]["tool"], samples[i]["seed_query"],
                                                      samples[i]["query"]))
    with open(OUT / "train.sft.jsonl", "w", encoding="utf-8") as fo:
        for i in idx:
            s = samples[i]
            asst = [{"q": s["query"], "d": s["d"], "vt": f"{s['d']}:{s['tool']}", "params": s["params"]}]
            fo.write(json.dumps({"messages": [
                {"role": "system", "content": build_oneshot_prompt(block, slot_blocks[i])},
                {"role": "user", "content": f"{now}\n用户请求：{s['query']}"},
                {"role": "assistant", "content": json.dumps(asst, ensure_ascii=False)},
            ]}, ensure_ascii=False) + "\n")
    with open(OUT / "train.csv", "w", encoding="utf-8-sig", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["domain", "query", "tool", "tool_param", "seed_query"])
        for i in idx:
            s = samples[i]
            w.writerow([s["src_domain"], s["query"], s["tool"],
                        json.dumps(s["params"], ensure_ascii=False, separators=(",", ":")),
                        s["seed_query"]])
    ev = collections.Counter(r["domain"] for r in rows)
    tr = collections.Counter(s["src_domain"] for s in samples)
    report = {"target_total": args.total, "samples": len(samples),
              "seeds_total": len(rows), "seeds_used": len(seeds), "seeds_skipped": dict(skipped),
              "dropped_this_run": dict(drop),
              "with_slot_block": sum(1 for b in slot_blocks if b),
              "by_domain": {d: {"eval": ev[d], "target": target[d], "train": tr[d],
                                "eval_ratio": round(ev[d] / len(rows), 4),
                                "train_ratio": round(tr[d] / max(1, len(samples)), 4)}
                            for d in DOMAIN_MAP},
              "elapsed_s": round(time.time() - t0, 1)}
    json.dump(report, open(OUT / "train_report.json", "w"), ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
