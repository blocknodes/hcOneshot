#!/usr/bin/env python3
"""按 runtime 端到端结果生成 SFT + 错例 DPO（ms-swift 格式）。

输入
  runtime  <bench>_runtime_out.csv（列 domain,query,tool,tool_param,…,工具链,参数）：端到端执行结果，
           对错口径 = tools/eval_stats.judge 的 tool_param_ok（与平时报的 tool+param 准确率一致）。
  金标     同一份 benchmark CSV（domain,query,tool,tool_param），清洗口径同 gen_sft_dpo_label.load_csv
           （去系统代填参数、fan_knowledge_agent 置空），但**保留 k12_category**：runtime 口径
           （eval_stats.judge）要求 edu_search 带 k12_category，引擎不会代补，chosen 去掉它就成了错答案。
  日志     该次跑分的 LLM 调用日志（system/user 原样取自日志）。

SFT（benchmark 里每个在日志中有计划调用的 query 一条）
    assistant = 模型某次原始输出，若它按 runtime 口径判对且干净（纯 JSON 数组、无系统代填参数、
                fan_knowledge_agent 不带参数）；否则按金标重建（同 DPO 的 chosen）。

DPO（只看 runtime 判错的 query）
  对该 query 在日志里的每次计划调用：模型原始输出按 eval_stats 口径也判错 → 一条 DPO
    chosen   = 金标计划（q 沿用模型改写；同 gen_sft_dpo_label.rebuild）
    rejected = 模型原始输出（同 query 同错去重）
  不产出 DPO 的错例记入 audit：
    no_log        日志里没有该 query 的计划调用（如被多轮改写成别的 query）
    raw_ok        模型原始输出本身判对，错出在引擎后处理/串行步骤，不是模型的锅
    dup_rejected  同 query 同样的错误输出已收过

产物（--outdir 下）：<prefix>.sft.jsonl / <prefix>.dpo.jsonl（训练用）、
      <prefix>.sft.csv / <prefix>.dpo.csv（同一批数据，一行一条，便于查看）、
      <prefix>.audit.csv（全部错例的去向）、<prefix>.report.json

用法：python3 tools/gen_dpo_from_runtime.py \\
        --runtime testset/train_manual/train_runtime_out.csv --gold testset/train_manual/train.csv \\
        --log '../hcSmith/train_pattern/_raw/*.jsonl' --prefix train_pattern_runtime_wrong
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from app import catalog  # noqa: E402
from app.parser import parse_json_loose, parse_oneshot  # noqa: E402
from eval_stats import judge as rt_judge  # noqa: E402
from gen_sft_dpo import TOOL_ALIAS, compact, nq, prune  # noqa: E402
from gen_sft_dpo_label import _raw_is_clean, iter_plan_calls, load_csv, rebuild  # noqa: E402

logging.getLogger("app.parser").setLevel(logging.WARNING)


def raw_as_runtime(raw: str, row: dict) -> dict:
    """模型原始输出 → eval_stats.judge 的输入形状（工具链 / 参数 列表），金标列沿用 runtime 行。"""
    its, _ = parse_oneshot(raw or "")
    return {"tool": row["tool"], "tool_param": row["tool_param"], "domain": row["domain"],
            "工具链": " | ".join(TOOL_ALIAS.get(i.tool, i.tool) or "" for i in its),
            "参数": json.dumps([i.params for i in its], ensure_ascii=False)}, its


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--outdir", default=str(ROOT / "testset" / "sftdpo"))
    ap.add_argument("--prefix", required=True)
    a = ap.parse_args()

    stats: collections.Counter = collections.Counter()
    gold = load_csv(Path(a.gold), stats, [])
    for r in csv.DictReader(open(a.gold, encoding="utf-8-sig")):      # 恢复 k12_category
        g = gold.get(nq(r["query"]))
        if g and g["steps"][0]["tool"] == "edu_search" and "k12_category" in r["tool_param"]:
            p = parse_json_loose(r["tool_param"])
            g["steps"][0]["params"] = prune({k: v for k, v in p.items()
                                             if k not in catalog.system_filled("edu_search")})
            stats["gold_keep_k12"] += 1
    calls: dict[str, dict] = {}
    for q, msgs, out in iter_plan_calls(a.log, "", stats):
        calls.setdefault(q, {"msgs": msgs, "outs": []})["outs"].append(out)

    dpo, audit, sft, sft_view = [], [], [], []
    for row in csv.DictReader(open(a.runtime, encoding="utf-8-sig")):
        j = rt_judge(row)
        stats["runtime_rows"] += 1
        q = nq(row["query"])
        g, c = gold.get(q), calls.get(q)
        # ---- SFT ----
        if g and c:
            chosen, src = "", "gold_rebuild"
            first_its = []
            for raw in c["outs"]:
                x, its = raw_as_runtime(raw, row)
                first_its = first_its or its
                if (rt_judge(x)["tool_param_ok"] and _raw_is_clean(raw, its)
                        and not any(i.tool == "fan_knowledge_agent" and i.params for i in its)):
                    chosen, src = raw.strip(), "model_raw"
                    break
            chosen = chosen or rebuild(g, first_its)
            sft.append({"messages": c["msgs"] + [{"role": "assistant", "content": chosen}]})
            ci, _ = parse_oneshot(chosen)
            sft_view.append({"domain": row["domain"], "query": row["query"], "source": src,
                             "runtime_ok": int(bool(j["tool_param_ok"])),
                             "tool": " | ".join(f"{i.d}:{i.tool}" for i in ci),
                             "params": " | ".join(json.dumps(i.params, ensure_ascii=False, separators=(",", ":"))
                                                  for i in ci),
                             "assistant": chosen})
            stats[f"sft_{src}"] += 1
        elif g:
            stats["sft_no_log"] += 1
        # ---- DPO ----
        if j["tool_param_ok"]:
            stats["runtime_ok"] += 1
            continue
        stats["runtime_wrong"] += 1
        rec = {"domain": row["domain"], "query": row["query"], "gold_tool": j["gold_tool"],
               "runtime_tool": j["chain"], "gold_canon": j["gold_canon"], "runtime_canon": j["pred_canon"]}
        if not g:
            audit.append({**rec, "status": "no_gold"}); stats["no_gold"] += 1
            continue
        if not c:
            audit.append({**rec, "status": "no_log"}); stats["no_log"] += 1
            continue
        seen: set[str] = set()
        for raw in c["outs"]:
            x, its = raw_as_runtime(raw, row)
            rj = rt_judge(x)
            r2 = {**rec, "model_tool": rj["chain"], "model_canon": rj["pred_canon"], "model_output": raw}
            if rj["tool_param_ok"]:
                audit.append({**r2, "status": "raw_ok"}); stats["raw_ok"] += 1
                continue
            key = compact([[i.d, i.tool, i.params, i.dep] for i in its]) if its else raw
            if key in seen:
                audit.append({**r2, "status": "dup_rejected"}); stats["dup_rejected"] += 1
                continue
            seen.add(key)
            chosen = rebuild(g, its)
            dpo.append({"messages": c["msgs"] + [{"role": "assistant", "content": chosen}],
                        "rejected_response": raw})
            audit.append({**r2, "status": "dpo", "chosen": chosen}); stats["dpo"] += 1
            stats[f"dpo_{row['domain']}"] += 1

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("dpo", dpo), ("sft", sft)):
        with open(out / f"{a.prefix}.{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / f"{a.prefix}.sft.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["domain", "query", "source", "runtime_ok", "tool", "params", "assistant"])
        w.writeheader()
        w.writerows(sft_view)
    cols = ["status", "domain", "query", "gold_tool", "runtime_tool", "model_tool",
            "gold_canon", "runtime_canon", "model_canon", "chosen", "model_output"]
    with open(out / f"{a.prefix}.audit.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(audit)
    # 方便查看的 CSV：一行一对，只放 query / 判错原因 / chosen 与 rejected 的工具和参数，不含长 system prompt
    vcols = ["domain", "query", "gold_tool", "chosen_tool", "chosen_params", "rejected_tool", "rejected_params",
             "runtime_tool", "chosen", "rejected"]
    with open(out / f"{a.prefix}.dpo.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=vcols)
        w.writeheader()
        for r in audit:
            if r["status"] != "dpo":
                continue
            ci, _ = parse_oneshot(r["chosen"])
            ri, _ = parse_oneshot(r["model_output"])
            fmt_t = lambda its: " | ".join(f"{i.d}:{i.tool}" if i.tool else f"{i.d}:dep" for i in its)
            fmt_p = lambda its: " | ".join(json.dumps(i.params, ensure_ascii=False, separators=(",", ":"))
                                           for i in its)
            w.writerow({"domain": r["domain"], "query": r["query"], "gold_tool": r["gold_tool"],
                        "chosen_tool": fmt_t(ci), "chosen_params": fmt_p(ci),
                        "rejected_tool": fmt_t(ri) if ri else "(解析失败/空)", "rejected_params": fmt_p(ri),
                        "runtime_tool": r["runtime_tool"], "chosen": r["chosen"], "rejected": r["model_output"]})
    report = {"runtime": a.runtime, "gold": a.gold, "log": a.log, "sft": len(sft), "dpo": len(dpo),
              "counts": dict(sorted(stats.items()))}
    (out / f"{a.prefix}.report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                                 encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
