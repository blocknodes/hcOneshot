#!/usr/bin/env python3
"""手工训练数据（testset/train_manual/src/<域>.tsv：query\\ttool\\tparams_json）→ CSV。
校验：JSON 合法、工具在该域目录内、去掉与 20261116 评测集重复的 query、域内去重。
用法：python3 tools/build_manual_train.py [输出目录，默认 testset/train_manual]
产物：<目录>/train_<域>.csv、train.csv（列 domain,query,tool,tool_param）、report.json
"""
import collections, csv, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from app import catalog  # noqa: E402
import gen_train as G  # noqa: E402
OUT = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "testset/train_manual"); SRC = OUT / "src"
EV = ROOT / "testset/feishu/20261116"
FILES = {"vod": "vod_eval.csv", "educ": "educ_eval.csv", "edu": "edu_eval.csv", "music": "music_eval.csv",
         "audio": "audio_eval.csv", "sports": "sports_eval.csv", "tv_control": "tv_control_eval.csv"}
evq = {r["query"].strip() for f in FILES.values() for r in csv.DictReader(open(EV / f, encoding="utf-8-sig"))}
allrows, rep = [], {}
for dom in FILES:
    ps = sorted(SRC.glob(f"{dom}.tsv")) + sorted(SRC.glob(f"{dom}_*.tsv"))
    if not ps: continue
    seen, rows, bad = set(), [], collections.Counter()
    lines = [l for p in ps for l in open(p, encoding="utf-8")]
    for i, line in enumerate(lines, 1):
        line = line.rstrip("\n")
        if not line.strip(): continue
        try:
            q, tool, ps = line.split("\t"); params = json.loads(ps)
        except Exception:
            bad["format"] += 1; print(f"[{dom}:{i}] 格式错: {line}"); continue
        if tool not in catalog.valid_tools_for(G.DOMAIN_MAP[dom]):
            bad["tool"] += 1; print(f"[{dom}:{i}] 工具不在目录: {tool}"); continue
        if q in evq: bad["in_eval"] += 1; continue
        if q in seen: bad["dup"] += 1; continue
        seen.add(q); rows.append([dom, q, tool, json.dumps(params, ensure_ascii=False, separators=(",", ":"))])
    with open(OUT / f"train_{dom}.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f); w.writerow(["domain", "query", "tool", "tool_param"]); w.writerows(rows)
    allrows += rows; rep[dom] = {"rows": len(rows), "dropped": dict(bad),
                                 "by_tool": dict(collections.Counter(r[2] for r in rows))}
with open(OUT / "train.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f); w.writerow(["domain", "query", "tool", "tool_param"]); w.writerows(allrows)
json.dump(rep, open(OUT / "report.json", "w"), ensure_ascii=False, indent=2)
print({d: (v["rows"], v["dropped"]) for d, v in rep.items()}, "total", len(allrows))
