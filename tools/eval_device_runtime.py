#!/usr/bin/env python3
"""评估 runtime_execute.py --csv-in 的输出（设备单步），并可与另一次结果（audit.csv 或另一份 runtime 输出）逐条对比。

判定口径同 tools/gen_sft_dpo_label.py：只比工具和参数（q 不参与；retext/audio.query/fan.messages 不比；
检索树语义归一；仅大小写/空格/标点不同视为一致）。runtime 输出里的参数是后处理之后的真实参数，
所以虚拟 schema 开/关两种配置可以直接比较。

用法：
  python3 tools/eval_device_runtime.py --new /tmp/dev_virtual.csv \
      --base-audit testset/sftdpo/train_new_20261116.audit.csv
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from app import device_schema  # noqa: E402
from app.parser import parse_json_loose, parse_oneshot  # noqa: E402
from gen_sft_dpo import TOOL_ALIAS, canon_params, nq  # noqa: E402
from gen_sft_dpo_label import _loose, _norm, load_single, same_params as _same  # noqa: E402


def judge_plan(raw: str, g: dict) -> str:
    """基线：对旧日志里模型原始输出（真实 schema）重新判定，口径同 judge_runtime。"""
    its, _ = parse_oneshot(raw or "")
    if not its:
        return "empty_plan"
    if len(its) != 1:
        return "step_count"
    it, s = its[0], g["steps"][0]
    if it.dep or not it.tool:
        return "tool_missing"
    if (it.d, TOOL_ALIAS.get(it.tool, it.tool)) != (s["d"], s["tool"]):
        return "tool_wrong"
    return "match" if _same(s["tool"], s["params"], it.params) else "param_diff"


def judge_runtime(row: dict, g: dict) -> str:
    if row.get("err"):
        return "runtime_err"
    tools = [t.strip() for t in (row.get("工具链") or "").split("|") if t.strip()]
    if not tools:
        return "empty_plan"
    if len(tools) != 1:
        return "step_count"
    s = g["steps"][0]
    if TOOL_ALIAS.get(tools[0], tools[0]) != s["tool"]:
        return "tool_wrong"
    plist = parse_json_loose(row.get("参数") or "[]")
    p = plist[0] if isinstance(plist, list) and plist and isinstance(plist[0], dict) else {}
    return "match" if _same(s["tool"], s["params"], p) else "param_diff"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True, help="runtime_execute 输出 csv（本次）")
    ap.add_argument("--base-audit", help="gen_sft_dpo_label 的 audit.csv（旧 prompt，真实 schema）")
    ap.add_argument("--base", help="另一份 runtime_execute 输出 csv（作基线）")
    ap.add_argument("--gold-dir", default=str(ROOT / "testset/feishu/20261116"))
    ap.add_argument("--show", type=int, default=8)
    args = ap.parse_args()

    stats: collections.Counter = collections.Counter()
    gold = {q: g for q, g in load_single(Path(args.gold_dir), stats, []).items() if g["id"] == "tv_control"}

    def runtime_verdicts(path: str) -> dict[str, tuple[str, dict]]:
        out = {}
        for r in csv.DictReader(open(path, encoding="utf-8-sig")):
            q = nq(r["query"])
            if q in gold and q not in out:
                out[q] = (judge_runtime(r, gold[q]), r)
        return out

    new = runtime_verdicts(args.new)
    base: dict[str, tuple[str, dict]] = {}
    if args.base_audit:
        for r in csv.DictReader(open(args.base_audit, encoding="utf-8-sig")):
            q = nq(r["query"])
            if r["kind"] == "single" and r["src"] == "tv_control" and q in gold and q not in base:
                base[q] = (judge_plan(r["model_output"], gold[q]), r)
    elif args.base:
        base = runtime_verdicts(args.base)

    def summary(name: str, d: dict) -> None:
        c = collections.Counter(v for v, _ in d.values())
        n = sum(c.values())
        print(f"{name:10s} n={n}  准确率 {c['match'] / max(1, n):.2%}  " + "  ".join(f"{k}={v}" for k, v in c.most_common()))

    print(f"设备单步标注 {len(gold)} 条")
    summary("本次", new)
    if not base:
        return 0
    summary("基线(全量)", base)
    common = [q for q in new if q in base]
    nb = {q: new[q] for q in common}
    bb = {q: base[q] for q in common}
    print(f"\n== 相同 query 逐条对比（{len(common)} 条）==")
    summary("基线", bb)
    summary("本次", nb)
    better = [q for q in common if base[q][0] != "match" and new[q][0] == "match"]
    worse = [q for q in common if base[q][0] == "match" and new[q][0] != "match"]
    print(f"\n基线错→本次对 {len(better)}   基线对→本次错 {len(worse)}   净增 {len(better) - len(worse)}")

    by_tool: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0, 0])
    for q in common:
        t = gold[q]["steps"][0]["tool"]
        by_tool[t][0] += 1
        by_tool[t][1] += base[q][0] == "match"
        by_tool[t][2] += new[q][0] == "match"
    print("\n按工具（n / 基线对 / 本次对）:")
    for t, (n, b, w) in sorted(by_tool.items(), key=lambda x: -x[1][0]):
        print(f"  {t:38s} {n:5d} {b:5d} {w:5d}  {w - b:+d}")

    def show(title: str, qs: list[str]) -> None:
        print(f"\n{title}（前 {args.show}）")
        for q in qs[: args.show]:
            s = gold[q]["steps"][0]
            r = new[q][1]
            print(f"  {gold[q]['query']}\n     标注 {s['tool']} {json.dumps(s['params'], ensure_ascii=False)}"
                  f"\n     本次 [{new[q][0]}] {r.get('工具链', '')} {r.get('参数', '')[:160]}")

    show("本次新增错误", worse)
    show("仍然错误（本次）", [q for q in common if new[q][0] != "match" and base[q][0] != "match"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
