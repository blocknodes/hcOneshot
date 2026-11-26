#!/usr/bin/env python3
"""手写补数辅助（不调用任何模型）：
  plan  <目录> [MIN]  列出不足 MIN 条的 pattern，给出带编号槽位的参数模板 → <目录>/fill_plan.txt
  apply <目录>        读 <目录>/fill/*.txt（每行：pid<TAB>query<TAB>槽值1|槽值2|…），按模板填回参数，
                      写成 <目录>/src/<域>_fill.tsv；槽值个数不符的行报错丢弃。
模板只替换签名里的 "_" 叶子（取值类），field/operator/action/order/operation 等结构值保持不变。
"""
import collections, csv, hashlib, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import gen_train as G  # noqa: E402
from app import catalog  # noqa: E402
src = open(ROOT / "tools/gen_pattern_train.py", encoding="utf-8").read().split("async def main")[0]
gp = {"__file__": str(ROOT / "tools/gen_pattern_train.py"), "__name__": "gp"}; exec(compile(src, "gp", "exec"), gp)
SVK = gp["SIG_VALUE_KEYS"]
CIRC = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮"

def slots(o, path=()):
    """按 G._leaves 顺序返回可替换叶子路径。"""
    return [p for p, v in G._leaves(o) if not (p and p[-1] in SVK)]

def setp(o, path, v):
    for k in path[:-1]: o = o[k]
    o[path[-1]] = v

def key(d, t, p): return (d, t, gp["signature"](G._prune(p)))
def pid(k): return "Q" + hashlib.md5("|".join(k).encode()).hexdigest()[:7]

def load_pats():
    seeds, _ = gp["load"]()
    pats = collections.OrderedDict()
    for s in seeds:
        p = G._prune(s["params"]); k = key(s["src_domain"], s["tool"], p)
        pats.setdefault(k, {"tmpl": p, "seeds": []})["seeds"].append(s["query"])
    return pats

def cmd_plan(d, mn):
    pats = load_pats(); have = collections.defaultdict(list)
    for r in csv.DictReader(open(d / "train.csv", encoding="utf-8-sig")):
        p = json.loads(r["tool_param"])
        for k in catalog.system_filled(r["tool"]): p.pop(k, None)
        have[key(r["domain"], r["tool"], p)].append((r["query"], G._prune(p)))
    out = []
    for k, v in pats.items():
        n = len(have[k])
        if n >= mn: continue
        t = json.loads(json.dumps(have[k][0][1] if have[k] else v["tmpl"]))
        sl = slots(t); vals = []
        for i, pth in enumerate(sl):
            o = t
            for kk in pth[:-1]: o = o[kk]
            vals.append(o[pth[-1]]); o[pth[-1]] = CIRC[i]
        ex = (have[k][0][0] if have[k] else v["seeds"][0])
        out.append(f"{pid(k)}\t{k[0]}\t{k[1]}\tneed={mn-n}\t{json.dumps(t, ensure_ascii=False, separators=(',',':'))}\t例:{ex} => {'|'.join(map(str, vals))}\t已有:{' / '.join(q for q, _ in have[k][:6])}")
    (d / "fill_plan.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(len(out), "patterns need fill")

def cmd_apply(d):
    pats = load_pats(); byid = {pid(k): (k, v) for k, v in pats.items()}
    # 模板优先用已有手写样本（与 plan 一致）
    have = {}
    for r in csv.DictReader(open(d / "train.csv", encoding="utf-8-sig")):
        p = json.loads(r["tool_param"])
        for kk in catalog.system_filled(r["tool"]): p.pop(kk, None)
        have.setdefault(key(r["domain"], r["tool"], p), G._prune(p))
    rows = collections.defaultdict(list); bad = 0
    for f in sorted((d / "fill").glob("*.txt")):
        for i, line in enumerate(open(f, encoding="utf-8"), 1):
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"): continue
            parts = line.split("\t")
            if len(parts) != 3 or parts[0] not in byid:
                print(f"[{f.name}:{i}] 格式/pid 错: {line}"); bad += 1; continue
            (dom, tool, sig), v = byid[parts[0]]
            t = json.loads(json.dumps(have.get((dom, tool, sig), v["tmpl"])))
            sl = slots(t); vals = parts[2].split("|") if parts[2] != "" else []
            if len(vals) != len(sl):
                print(f"[{f.name}:{i}] 槽值数 {len(vals)}≠{len(sl)}: {line}"); bad += 1; continue
            for pth, x in zip(sl, vals):
                o = t
                for kk in pth[:-1]: o = o[kk]
                old = o[pth[-1]]
                if isinstance(old, bool): x = {"True": True, "False": False}[x]
                elif isinstance(old, int) and x.lstrip("-").isdigit(): x = int(x)
                setp(t, pth, x)
            if key(dom, tool, t) != (dom, tool, sig):
                print(f"[{f.name}:{i}] 结构变了: {line}"); bad += 1; continue
            rows[dom].append(f"{parts[1]}\t{tool}\t{json.dumps(t, ensure_ascii=False)}")
    for dom, l in rows.items():
        (d / "src" / f"{dom}_fill.tsv").write_text("\n".join(l) + "\n", encoding="utf-8")
    print({k: len(v) for k, v in rows.items()}, "bad", bad)

if __name__ == "__main__":
    d = ROOT / sys.argv[2]
    cmd_plan(d, int(sys.argv[3]) if len(sys.argv) > 3 else 5) if sys.argv[1] == "plan" else cmd_apply(d)
