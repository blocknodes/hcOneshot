#!/usr/bin/env python3
"""用 OpenAI 兼容接口评测 SFT 格式（system/user/assistant）jsonl 的准确率。

金标 = assistant 的 JSON 计划数组；预测 = 模型对 system+user 的回复。
指标（逐条）：
  解析成功   回复能解析为 JSON 数组
  条数对     预测计划条数 = 金标条数
  域对       按位置 d 全部一致
  工具对     按位置 vt 全部一致（含 dep 条目缺 vt 的情况）
  工具+参数  工具对且 params 经 canon_params 规范化后一致
  整条全对   计划数组（q 以外的字段）全部一致：d / vt / params / dep
  完全一致   连 q（改写句）也逐字一致
用法：
  python3 tools/eval_sft_jsonl.py --data test.jsonl --base-url http://host:port/v1 [--model baseline]
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import re
import sys
import urllib.request
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "hcSlime" / "benchmark"))
from gen_sft_dpo import canon_params  # noqa: E402

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # 内网接口，不走代理


def chat(base_url: str, model: str, messages: list, max_tokens: int, timeout: int) -> str:
    body = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with OPENER.open(req, timeout=timeout) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"] or ""
        except Exception as e:  # noqa: BLE001
            err = e
    raise RuntimeError(f"request failed: {err}")


def parse_plan(text: str):
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t).strip()
    try:
        v = json.loads(t)
    except Exception:  # noqa: BLE001
        m = re.search(r"\[.*\]", t, flags=re.S)
        if not m:
            return None
        try:
            v = json.loads(m.group(0))
        except Exception:  # noqa: BLE001
            return None
    return v if isinstance(v, list) else None


def canon_item(x: dict, with_q: bool) -> str:
    if not isinstance(x, dict):
        return json.dumps(x, ensure_ascii=False)
    keep = {"d": x.get("d"), "vt": x.get("vt"), "dep": x.get("dep"),
            "params": canon_params(x.get("params")) if "params" in x else None}
    if with_q:
        keep["q"] = x.get("q")
    return json.dumps(keep, sort_keys=True, ensure_ascii=False)


def judge(gold: list, pred) -> dict:
    r = {"parse_ok": pred is not None}
    pred = pred or []
    n_ok = len(pred) == len(gold)
    pos = lambda f: n_ok and all(isinstance(p, dict) and p.get(f) == g.get(f) for g, p in zip(gold, pred))
    r["n_ok"] = n_ok
    r["d_ok"] = pos("d")
    r["vt_ok"] = pos("vt")
    r["vt_param_ok"] = r["vt_ok"] and all(
        canon_params(g.get("params")) == canon_params(p.get("params")) for g, p in zip(gold, pred))
    r["all_ok"] = n_ok and all(canon_item(g, False) == canon_item(p, False) for g, p in zip(gold, pred))
    r["exact_ok"] = n_ok and all(canon_item(g, True) == canon_item(p, True) for g, p in zip(gold, pred))
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    model = args.model
    if not model:
        with OPENER.open(args.base_url.rstrip("/") + "/models", timeout=20) as r:
            model = json.loads(r.read())["data"][0]["id"]

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]

    def run(i):
        msgs = rows[i]["messages"]
        try:
            return i, chat(args.base_url, model, msgs[:-1], args.max_tokens, args.timeout), ""
        except Exception as e:  # noqa: BLE001
            return i, "", str(e)

    outs = [None] * len(rows)
    with cf.ThreadPoolExecutor(args.workers) as ex:
        for k, (i, text, err) in enumerate(ex.map(run, range(len(rows))), 1):
            outs[i] = (text, err)
            if k % 50 == 0:
                print(f"  {k}/{len(rows)}", file=sys.stderr)

    out_path = Path(args.out) if args.out else Path(args.data).with_suffix(f".{model}.pred.jsonl")
    stats = collections.defaultdict(collections.Counter)
    n_err = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row, (text, err) in zip(rows, outs):
            msgs = row["messages"]
            gold = json.loads(msgs[-1]["content"])
            pred = parse_plan(text)
            j = judge(gold, pred)
            n_err += bool(err)
            dom = gold[0].get("d", "none") if gold else "none"
            for key in (dom, "ALL"):
                stats[key]["n"] += 1
                for k, v in j.items():
                    stats[key][k] += v
            query = msgs[1]["content"].split("用户请求：", 1)[-1]
            f.write(json.dumps({"query": query, "gold": gold, "pred_raw": text, "error": err, **j},
                               ensure_ascii=False) + "\n")

    cols = [("parse_ok", "解析成功"), ("n_ok", "条数对"), ("d_ok", "域对"), ("vt_ok", "工具对"),
            ("vt_param_ok", "工具+参数"), ("all_ok", "整条全对"), ("exact_ok", "含q完全一致")]
    print(f"model={model}  data={args.data}  n={len(rows)}  请求失败={n_err}")
    print(f"{'域':<12}{'样本':>6}" + "".join(f"{c:>10}" for _, c in cols))
    for key in sorted(stats, key=lambda k: (k == "ALL", -stats[k]["n"], k)):
        s = stats[key]
        print(f"{key:<12}{s['n']:>6}" + "".join(f"{s[k] / s['n'] * 100:>9.1f}%" for k, _ in cols))
    print(f"明细: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
