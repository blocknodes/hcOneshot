#!/usr/bin/env python3
"""用法：python3 tools/check_manual_coverage.py [目录] [每 pattern 最少条数，默认 1]
手工训练集对 20261116 七域 pattern（域, 工具, 参数结构签名）的覆盖检查。
签名口径与 gen_pattern_train.signature 一致，两侧都先去空值（设备域标注带空 value/device/location）。输出未覆盖 pattern 清单到 testset/train_manual/uncovered.csv。
"""
import collections, csv, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import gen_train as G  # noqa: E402
from app import catalog  # noqa: E402
import importlib.util
OUTD = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "testset/train_manual")
MIN = int(sys.argv[2]) if len(sys.argv) > 2 else 1
spec = importlib.util.spec_from_file_location("gp", ROOT / "tools/gen_pattern_train.py")
src = open(ROOT / "tools/gen_pattern_train.py", encoding="utf-8").read().split("async def main")[0]
gp = {"__file__": str(ROOT / "tools/gen_pattern_train.py"), "__name__": "gp"}; exec(compile(src, "gp", "exec"), gp)
seeds, _ = gp["load"]()
pats = collections.defaultdict(list)
for s in seeds: pats[(s["src_domain"], s["tool"], gp["signature"](G._prune(s["params"])))].append(s["query"])
have = collections.Counter()
for r in csv.DictReader(open(OUTD / "train.csv", encoding="utf-8-sig")):
    p = json.loads(r["tool_param"])
    for k in catalog.system_filled(r["tool"]): p.pop(k, None)
    have[(r["domain"], r["tool"], gp["signature"](G._prune(p)))] += 1
unc = [(k, v) for k, v in pats.items() if have[k] < MIN]
by = collections.Counter(k[0] for k in pats); ub = collections.Counter(k[0] for k, _ in unc)
for d in by: print(f"{d:12s} patterns={by[d]:4d} covered={by[d]-ub[d]:4d} uncovered={ub[d]:4d}")
print("total", len(pats), "uncovered", len(unc), "uncovered_seeds", sum(len(v) for _, v in unc))
with open(OUTD / "uncovered.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f); w.writerow(["domain", "tool", "signature", "seeds", "have", "examples"])
    for (d, t, sig), v in sorted(unc, key=lambda x: (x[0][0], x[0][1], -len(x[1]))):
        w.writerow([d, t, sig, len(v), have[(d, t, sig)], " / ".join(v[:3])])
extra = [k for k in have if k not in pats]
print("manual signatures not in eval patterns:", len(extra))
for k in extra[:40]: print("  EXTRA", k, have[k])
