#!/usr/bin/env python3
"""训练数据修复：对 testset/train/samples.jsonl 就地清洗（gen_train.py 再补齐配额）。

  1. 去整句透传参数：audio_search.query、fan_knowledge_agent.messages（同 retext，系统补）。
  2. 删变体里带相对时间词的样本（金标是绝对日期 / 时间语义随当前时间变）。
  3. 规则判一致性：K歌/MV/收藏/历史/相似推荐等工具，用途须在 query 里有字面依据，否则删。
     （LLM 判官试跑 370 条判否 102 条、约一半是误杀，弃用。）
  5. 去掉 edu 默认补的 k12_category=[同步课,同步拓展课]（同 clean_testset 规则 8；query 说了「同步」的保留）。
  4. 每个种子最多保留 --cap 条（缓解简单种子过度集中）。
用法：python3 tools/fix_train.py [--cap 16] [--dry-run]
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
sys.path.insert(0, str(ROOT / "tools"))

from app import catalog  # noqa: E402
from app.catalog import _brief_of  # noqa: E402
from app.config import get_settings  # noqa: E402
from clean_testset import _strip_k12  # noqa: E402
from gen_train import REL_TIME  # noqa: E402

CKPT = ROOT / "testset" / "train" / "samples.jsonl"

JUDGE = """你是电视语音助手的标注质检员。下面每条是：用户 query、系统选的工具(及其用途)、工具参数 params。
逐条判断【工具和参数是否与 query 一致】：
- 工具的用途要匹配 query 的意图（如 query 是"听/播放歌"却用了 K 歌工具、query 没有"历史/看过"却用了历史工具、
  query 不是找相似却用了相似推荐工具、query 是具体检索却用了模糊搜索或反之明显不当 → 不一致）；
- params 里的取值要能在 query 里找到依据，query 里明确提到的关键条件（片名/人名/年级/科目/集数/数值等）
  不能和 params 矛盾；
- 参数为空 {} 的工具只看工具用途是否匹配。
不要因为措辞、口语、轻微归一化（国语→普通话、初一→七年级）判不一致。
只输出 JSON 数组，长度与条目数相同，一致写 1、不一致写 0，例如 [1,0,1]。

{items}"""


def normalize(r: dict) -> dict:
    p = dict(r["params"]) if isinstance(r["params"], dict) else {}
    for k in catalog.system_filled(r["tool"]):
        p.pop(k, None)
    p, _ = _strip_k12(p, r.get("query") or "")
    r["params"] = p
    return r


_BRIEF: dict[str, str] = {}


def brief(d: str, tool: str) -> str:
    key = f"{d}:{tool}"
    if key not in _BRIEF:
        t = next((t for t in catalog.tools() if t["domain"] == d and t["tool"] == tool), {})
        _BRIEF[key] = _brief_of(t)[:70]
    return _BRIEF[key]


async def judge_batch(client, batch: list[dict]) -> list[int] | None:
    s = get_settings()
    items = "\n".join(
        f"{i + 1}. query: {r['query']}\n   工具: {r['tool']}（{brief(r['d'], r['tool'])}）\n"
        f"   params: {json.dumps(r['params'], ensure_ascii=False)}"
        for i, r in enumerate(batch))
    payload = {"model": s.model, "temperature": 0.0, "max_tokens": 200,
               "chat_template_kwargs": {"enable_thinking": False},
               "messages": [{"role": "user", "content": JUDGE.replace("{items}", items)}]}
    for attempt in range(3):
        try:
            resp = await client.post(s.chat_url, json=payload)
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"] or ""
            m = re.search(r"\[[\s\d,]*\]", text)
            v = json.loads(m.group(0)) if m else []
            if len(v) == len(batch) and all(x in (0, 1) for x in v):
                return v
        except Exception:  # noqa: BLE001
            pass
        await asyncio.sleep(1 + attempt)
    return None


# 规则判一致性（LLM 判官试跑误杀约一半，弃用）：这些工具的用途必须在 query 里有字面依据。
TOOL_NEEDS = {
    "music_ksong_search": r"唱|K歌|k歌|伴奏|KTV|ktv|卡拉|点唱|跟唱",
    "music_song_mv_search": r"MV|mv|Mv|视频|音乐电视|画面|看",
    "music_song_favorite_search": r"收藏|喜欢|喜爱|红心|我的歌单|存着|存起来|存下|收着|保存",
    "music_song_history": r"历史|记录|听过|播过|放过|上次|之前|刚才|最近播放|常听|以前|日记|日志|清单|回忆|听了哪些|看了哪些|看过哪些|听过哪些|足迹|播放过|重播",
    "vod_history": r"历史|记录|看过|播过|放过|上次|之前|刚才|追过|没看完|看到哪|以前|日记|日志|清单|回忆|听了哪些|看了哪些|看过哪些|听过哪些|足迹|播放过|重播",
    "audio_history": r"历史|记录|听过|播过|放过|上次|之前|刚才|收听过|没听完|听到哪|以前|日记|日志|清单|回忆|听了哪些|看了哪些|看过哪些|听过哪些|足迹|播放过|重播",
    "educ_history": r"历史|记录|看过|播过|放过|上次|之前|刚才|没看完|看到哪|以前|日记|日志|清单|回忆|听了哪些|看了哪些|看过哪些|听过哪些|足迹|播放过|重播",
    "vod_relate_search": r"类似|相似|相近|接近|像|同类|一类|一样|同款|同风格|同赛道|同题材|同系列|调性|差不多|这类|这种|那种|相关|类型相同|风格|格式|类型|那样|那类|同样|一致|除了",
    "educ_relate_search": r"类似|相似|相近|接近|像|同类|一类|一样|同款|同风格|同赛道|同题材|同系列|调性|差不多|这类|这种|那种|相关|类型相同|风格|格式|类型|那样|那类|同样|一致|除了",
}
# 相对时间（扩充）：变体里出现即删（金标若含日期会随当前时间失效）
REL_EXTRA = re.compile(r"近期|前段时间|前阵子|一周内|这几天|近几|几天前|上一周|过去|之前几")


def rule_reject(r: dict) -> str:
    q = r["query"]
    if REL_TIME.search(q) or REL_EXTRA.search(q):
        return "drop_relative_time"
    pat = TOOL_NEEDS.get(r["tool"])
    if pat and not re.search(pat, q):
        return "drop_tool_intent_mismatch"
    return ""


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = [normalize(json.loads(l)) for l in open(CKPT, encoding="utf-8") if l.strip()]
    stat = collections.Counter(total=len(rows))
    ex = collections.defaultdict(list)
    keep = []
    for r in rows:
        r.pop("judge", None)
        why = rule_reject(r)
        if why:
            stat[why] += 1
            if len(ex[why]) < 6:
                ex[why].append(f"{r['seed_query']} => {r['query']} | {r['tool']}")
            continue
        keep.append(r)
    per = collections.defaultdict(list)
    for r in keep:
        per[(r["seed_query"], r["tool"])].append(r)
    out = []
    rnd = random.Random(0)
    for lst in per.values():
        if len(lst) > args.cap:
            rnd.shuffle(lst)
            stat["drop_over_cap"] += len(lst) - args.cap
            lst = lst[:args.cap]
        out += lst
    stat["kept"] = len(out)
    for k, v in ex.items():
        print(k)
        for e in v:
            print("   ", e)
    print(json.dumps(dict(stat), ensure_ascii=False))
    print(json.dumps(dict(collections.Counter(r["src_domain"] for r in out)), ensure_ascii=False))
    if args.dry_run:
        return 0
    tmp = CKPT.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(CKPT)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
