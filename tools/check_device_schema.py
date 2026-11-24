#!/usr/bin/env python3
"""用设备域标注验证虚拟 schema（app/device_schema.py）能否无损往返：真实 → 虚拟 → 真实。

数据：testset/feishu/<日期>/tv_control_eval.csv（单步）+ multi_intent.csv 里 device 工具的步（打标状态=完整）。
分类：
  exact       往返后参数完全一致
  normalized  仅因标注写法不一致被归一（如 playback_control 的 object 为空 → 播放控制），语义不变
  uncovered   无法表示（缺 operation 等标注问题，或与固定动作重名的歧义），列出明细

用法：python3 tools/check_device_schema.py [--dir testset/feishu/20261116] [--show 12]
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

from app import device_schema as ds  # noqa: E402


def load_steps(d: Path):
    for r in csv.DictReader(open(d / "tv_control_eval.csv", encoding="utf-8-sig")):
        yield r["期望工具"].strip(), json.loads(r["期望参数"] or "{}"), r["query"], "single"
    for r in csv.DictReader(open(d / "multi_intent.csv", encoding="utf-8-sig")):
        if r["打标状态"] != "完整":
            continue
        for i in range(1, 9):
            vt = r.get(f"tool{i}", "")
            if vt.startswith("device:"):
                yield (vt.split(":", 1)[1], json.loads(r[f"tool{i}_param"] or "{}"),
                       r.get(f"tool{i}_q") or r["query"], "multi")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "testset/feishu/20261116"))
    ap.add_argument("--show", type=int, default=10)
    args = ap.parse_args()

    stat = collections.Counter()
    by_tool: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    norm_ex, unc_ex = collections.Counter(), collections.defaultdict(list)
    vocab = collections.Counter()
    for tool, params, q, _src in load_steps(Path(args.dir)):
        real = ds._prune(params)
        try:
            v = ds.decompile_call(tool, real)
            back = ds.compile_call(tool, v)
        except ValueError as e:
            kind = "uncovered"
            unc_ex[(tool, str(e)[:60])].append(q)
        else:
            if back == real:
                kind = "exact"
            elif (tool == "playback_control" and not real.get("object")
                  and {**back, "object": ""} == {**real, "object": ""}):
                kind = "normalized"
                norm_ex[(tool, real.get("operation"))] += 1
            else:
                kind = "uncovered"
                unc_ex[(tool, f"往返不一致 {real} -> {v} -> {back}"[:90])].append(q)
            if kind != "uncovered" and "action" in v:
                vocab[(tool, v["action"])] += 1
        stat[kind] += 1
        by_tool[tool][kind] += 1

    total = sum(stat.values())
    print(f"设备步总数 {total}: " + "  ".join(f"{k}={v} ({v / total:.2%})" for k, v in stat.items()))
    print("\n按工具:")
    for t, c in sorted(by_tool.items(), key=lambda x: -sum(x[1].values())):
        print(f"  {t:38s} {sum(c.values()):5d}  " + " ".join(f"{k}={v}" for k, v in c.items()))
    if norm_ex:
        print("\n归一明细（标注 object 为空，已并入 播放控制）:", dict(norm_ex))
    if unc_ex:
        print(f"\n无法表示（共 {sum(len(v) for v in unc_ex.values())} 步）:")
        for (t, why), qs in sorted(unc_ex.items(), key=lambda x: -len(x[1]))[: args.show]:
            print(f"  {len(qs):4d}  {t}: {why}   e.g. {qs[:2]}")
    print(f"\n虚拟 action 词表规模: {len({a for _, a in vocab})} 个不同 action（含通用 operation 词）")
    return 0 if stat["uncovered"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
