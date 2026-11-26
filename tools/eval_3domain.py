#!/usr/bin/env python3
"""3domain_result.csv（两步多意图：第一步/第二步 的 工具+参数 为金标；工具链/参数 为预测）打分。
口径同 tools/eval_multi.judge：执行关系=并行 → 工具无序配对；顺序/条件 → 有序。参数比对同 eval_stats.canon。
用法：python3 tools/eval_3domain.py [3domain_result.csv]   产物：<输入>.judge.csv"""
import collections, csv, json, logging, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
logging.disable(logging.WARNING)
from eval_multi import judge  # noqa
from app.parser import parse_json_loose  # noqa

src = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "3domain_result.csv")
rows = list(csv.DictReader(open(src, encoding="utf-8-sig")))
out, by = [], collections.defaultdict(collections.Counter)
for r in rows:
    tools, params = [], []
    for k in ("第一步", "第二步"):
        t = (r.get(f"{k}:工具") or "").strip()
        if t:
            tools.append(t)
            p = parse_json_loose(r.get(f"{k}:参数") or "")
            params.append(p if isinstance(p, dict) else None)
    x = {"tools": "+".join(tools), "tool_param": json.dumps(params, ensure_ascii=False), "query": r["query"],
         "exec_mode": "parallel" if r["执行关系"] == "并行" else "serial",
         "工具链": r.get("工具链", ""), "参数": r.get("参数", "")}
    j = judge(x)
    rec = {"query": r["query"], "执行关系": r["执行关系"], "九类模式": r["九类模式"],
           "域组合": f"{r['第一步:域']}+{r['第二步:域']}", **j}
    out.append(rec)
    for key in ["ALL", "执行关系:" + r["执行关系"], "域组合:" + rec["域组合"]] + \
               ["九类:" + m.strip() for m in r["九类模式"].split("；")]:
        s = by[key]; s["n"] += 1; s["steps"] += j["steps_ok"]; s["chain"] += j["chain_ok"]
        s["step_tool"] += j["step_tool"]; s["n_step"] += j["n_gold"]; s["scored"] += j["scored"]
        s["param_ok"] += j["param_ok"]; s["all"] += j["all_ok"]
with open(src.with_suffix(".judge.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
p = lambda a, b: f"{a / b * 100:5.1f}%" if b else "   - "
print(f"{'分组':28s} {'n':>4s} {'步数':>6s} {'工具链':>6s} {'步级工具':>6s} {'步级参数':>6s} {'整条全对':>6s}")
for k in sorted(by, key=lambda k: (k != "ALL", k.split(":")[0], -by[k]["n"])):
    s = by[k]
    print(f"{k:28s} {s['n']:4d} {p(s['steps'], s['n']):>6s} {p(s['chain'], s['n']):>6s} "
          f"{p(s['step_tool'], s['n_step']):>6s} {p(s['param_ok'], s['scored']):>6s} {p(s['all'], s['n']):>6s}")
