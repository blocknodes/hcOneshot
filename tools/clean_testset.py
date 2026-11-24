#!/usr/bin/env python3
"""评测集清洗（第一步）：把 tool_param 里不合法的 JSON 修成合法 JSON。

输入：testset/raw/*_eval.csv（飞书 wiki「schema」下载的原件，只读不改）
输出：testset/clean/*_eval.csv + testset/clean/clean_report.json

规则（只修格式，不改语义；每一处修改都记进报告）：
  1. 「A 修改为 B」  标注员改判 → 取 B。
  2. 「A 或者 B」    多个可接受答案 → tool_param 取 A，B 放进新列 tool_param_alt（JSON 数组）。
  3. 其余结构错误   按 _PATCHES 逐条显式修（key=(文件, query)），不做模糊猜测。
  4. 空 tool_param  不是坏 JSON，原样保留（fan_knowledge_agent 为「只标工具」；
                    其余为漏标，列进报告待补标）。
  5. 去掉顶层 `retext`（tool_param 与 tool_param_alt 均去）：retext 是改写句，非检索槽位，
     不参与评测比对。retext 为唯一参数的条目（各 fuzzy/music 透传类工具）去掉后为 `{}`。
  7. 去掉整句透传的自然语言参数（audio_search.query、fan messages），口径同 retext。
  8. 去掉 edu 金标里默认补的 k12_category=[同步课,同步拓展课]（标注习惯，用户没说就不该加）；
     query 里明确说了「同步」的保留。去掉后 and 只剩一个子节点则解包为该子节点。
  6. 删脏数据段（_DROP_FROM，按原件文件行号，表头=第 1 行）：sports 自 L416 起为
     matchId 写死的赛事卡片/阵容/统计样本及多轮残留，整段删除。
最后合并为 all_eval.csv（首列 domain=文件名前缀）。
输出的 tool_param 统一为紧凑 JSON（ensure_ascii=False, separators=(",", ":")）。

用法：python3 tools/clean_testset.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "testset" / "raw"
OUT = ROOT / "testset" / "clean"

# 显式补丁：(文件名, query) → (修复后的 JSON 文本, 说明, 是否需人工复核)
_PATCHES: dict[tuple[str, str], tuple[str, str, bool]] = {
    ("educ_eval.csv", "适合儿童看的动画"): (
        '{"retext":"适合儿童看的动画","query":{"field":"content_type","value":"动画"}}',
        '多余引号："value": ""动画" → "动画"', False),
    ("educ_eval.csv", "有哪些绘本适合情绪管理？"): (
        '{"retext":"有哪些绘本适合情绪管理","query":{"and":['
        '{"field":"children_second_genre","value":"绘本"},'
        '{"or":[{"field":"children_third_genre","value":"情绪管理"},'
        '{"field":"training_objectives","value":"情绪管理"}]}]}}',
        'and 数组里裸写了 "or": [...]，补成对象 {"or": [...]}', False),
    ("music_eval.csv", '来一个歌曲，歌词是 "窗外的麻雀，在电线杆上多嘴"'): (
        json.dumps({"retext": '来一个歌曲，歌词是 "窗外的麻雀，在电线杆上多嘴"',
                    "lyrics": ["窗外的麻雀在电线杆上多嘴"]}, ensure_ascii=False),
        "retext 内的双引号未转义 → 转义（文本不变）", False),
    ("edu_eval.csv", "高中物理必修数学必修一"): (
        '{"query":{"and":[{"field":"figures","value":"高中"},'
        '{"or":[{"field":"first_subject","value":"物理"},{"field":"first_subject","value":"数学"}]},'
        '{"field":"k12_category","values":["同步课","同步拓展课"],"operator":"or"}]}}',
        "缺 `]}` 收尾。最小补法会把 k12_category 也放进 or（语义变成「物理 或 数学 或 同步课」），"
        "这里按语义放在 and 下：(物理 or 数学) and 同步课/同步拓展课。原标注的 figures=高中 未动",
        True),
}

# 规则 6：自该文件行号（含，表头=第 1 行）起整段删除
_DROP_FROM: dict[str, tuple[int, str]] = {
    "sports_eval.csv": (416, "matchId 写死的赛事卡片/阵容/统计脏数据及多轮残留（用户指定）"),
}
# 规则 7：按工具的整句透传参数（与 app/catalog.py 的 _TOOL_FILLED 一致）
_PASSTHROUGH = {"audio_search": ("query",), "fan_knowledge_agent": ("messages",)}
# 规则 8：默认补的同步课条件
_K12_DEFAULT = {"field": "k12_category", "values": ["同步课", "同步拓展课"], "operator": "or"}


def _is_k12_default(n) -> bool:
    return (isinstance(n, dict) and n.get("field") == "k12_category"
            and sorted(n.get("values") or []) == sorted(_K12_DEFAULT["values"])
            and set(n) <= {"field", "values", "operator"})


def _drop_k12(node):
    """递归删掉默认同步课叶子；返回 (新节点, 删除个数)。and/or 只剩一个子节点时解包。"""
    if isinstance(node, list):
        out, n = [], 0
        for x in node:
            if _is_k12_default(x):
                n += 1
                continue
            y, m = _drop_k12(x)
            out.append(y)
            n += m
        return out, n
    if not isinstance(node, dict):
        return node, 0
    out, n = {}, 0
    for k, v in node.items():
        out[k], m = _drop_k12(v)
        n += m
    for lg in ("and", "or"):
        if n and set(out) == {lg} and isinstance(out[lg], list) and len(out[lg]) == 1:
            return out[lg][0], n
    return out, n


def _strip_k12(params, query: str):
    """对一份参数应用规则 8；返回 (新参数, 删除个数)。"""
    if not isinstance(params, dict) or "query" not in params or "同步" in query:
        return params, 0
    q, n = _drop_k12(params["query"])
    if n:
        params = dict(params, query=q)
    return params, n


_DOMAIN_ORDER = ["vod", "educ", "edu", "music", "audio", "sports", "tv_control"]

_REVISE = re.compile(r"^(?P<a>\{.*\})\s*修改为\s*(?P<b>\{.*\})$", re.S)
_ALT = re.compile(r"\}\s*或者\s*\{")


def _compact(v) -> str:
    return json.dumps(v, ensure_ascii=False, separators=(",", ":"))


def clean_file(path: Path, report: dict) -> None:
    name = path.name
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    dropped = []
    if name in _DROP_FROM:
        start, why = _DROP_FROM[name]
        dropped = [{"line": i, "query": r["query"], "tool": r["tool"]}
                   for i, r in enumerate(rows[start - 2:], start=start)]
        rows = rows[:start - 2]
    fixes, missing, tool_only = [], [], 0
    has_alt = False
    for line, r in enumerate(rows, start=2):
        p = r["tool_param"]
        r["tool_param_alt"] = ""
        if not p.strip():
            if r["tool"] == "fan_knowledge_agent":
                tool_only += 1
            else:
                missing.append({"line": line, "query": r["query"], "tool": r["tool"]})
            continue
        try:
            r["tool_param"] = _compact(json.loads(p))
            continue
        except json.JSONDecodeError:
            pass
        key = (name, r["query"])
        if key in _PATCHES:
            fixed, why, review = _PATCHES[key]
            r["tool_param"] = _compact(json.loads(fixed))
            fixes.append({"line": line, "query": r["query"], "rule": "patch",
                          "why": why, "needs_review": review, "before": p,
                          "after": r["tool_param"]})
            continue
        m = _REVISE.match(p.strip())
        if m:
            r["tool_param"] = _compact(json.loads(m.group("b")))
            fixes.append({"line": line, "query": r["query"], "rule": "修改为",
                          "why": "标注员改判，取「修改为」之后的版本", "needs_review": False,
                          "before": p, "after": r["tool_param"]})
            continue
        if _ALT.search(p):
            parts = [s.strip() for s in re.split(r"(?<=\})\s*或者\s*(?=\{)", p.strip())]
            vals = [json.loads(s) for s in parts]
            r["tool_param"] = _compact(vals[0])
            r["tool_param_alt"] = _compact(vals[1:])
            has_alt = True
            fixes.append({"line": line, "query": r["query"], "rule": "或者",
                          "why": f"多个可接受答案：主答案 + {len(vals) - 1} 个备选写入 tool_param_alt",
                          "needs_review": False, "before": p, "after": r["tool_param"],
                          "alt": r["tool_param_alt"]})
            continue
        raise SystemExit(f"{name} L{line} 无法自动修复且无补丁：{p[:200]!r}")

    # 规则 5：去掉顶层 retext（JSON 修复之后做，修复记录里的 after 保留修复态便于核对）
    retext_dropped = retext_only = passthrough_dropped = 0
    k12_dropped, k12_kept = 0, []
    for r in rows:
        if r["tool_param"]:
            v = json.loads(r["tool_param"])
            if isinstance(v, dict) and "retext" in v:
                retext_only += set(v) == {"retext"}
                v.pop("retext")
                retext_dropped += 1
            # 规则 7：整句透传的自然语言参数（值=用户原话）同 retext，由系统补、不比对
            for k in _PASSTHROUGH.get(r["tool"], ()):
                if isinstance(v, dict) and k in v:
                    v.pop(k)
                    passthrough_dropped += 1
            # 规则 8：默认补的同步课条件
            if "k12_category" in r["tool_param"] and "同步" in r["query"]:
                k12_kept.append(r["query"])
            v, n = _strip_k12(v, r["query"])
            k12_dropped += n
            r["tool_param"] = _compact(v)
        if r.get("tool_param_alt"):
            alts = json.loads(r["tool_param_alt"])
            for i, a in enumerate(alts):
                if isinstance(a, dict):
                    a.pop("retext", None)
                    alts[i], n = _strip_k12(a, r["query"])
                    k12_dropped += n
            r["tool_param_alt"] = _compact(alts)

    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["query", "tool", "tool_param"] + (["tool_param_alt"] if has_alt else [])
    with open(OUT / name, "w", encoding="utf-8-sig", newline="") as fh:   # 与原件一致带 BOM（Excel 直接打开不乱码）
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    report[name] = {"rows": len(rows), "fixed": fixes, "missing_label": missing,
                    "tool_only_fan": tool_only, "retext_dropped": retext_dropped,
                    "retext_only_now_empty": retext_only, "dropped": dropped,
                    "passthrough_dropped": passthrough_dropped,
                    "k12_default_dropped": k12_dropped, "k12_kept_query_says_同步": k12_kept}


def main() -> int:
    files = sorted(Path(p) for p in glob.glob(str(RAW / "*_eval.csv")))
    if not files:
        raise SystemExit(f"未找到 {RAW}/*_eval.csv")
    report: dict = {}
    for f in files:
        clean_file(f, report)
    with open(OUT / "clean_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    # 合并：首列 domain（文件名前缀），统一 5 列
    merged = []
    for d in _DOMAIN_ORDER:
        f = OUT / f"{d}_eval.csv"
        if not f.exists():
            continue
        with open(f, encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                merged.append({"domain": d, "query": r["query"], "tool": r["tool"],
                               "tool_param": r["tool_param"],
                               "tool_param_alt": r.get("tool_param_alt", "")})
    with open(OUT / "all_eval.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["domain", "query", "tool", "tool_param", "tool_param_alt"])
        w.writeheader()
        w.writerows(merged)
    for name, r in report.items():
        print(f"{name:22s} rows={r['rows']:5d} fixed={len(r['fixed'])} "
              f"dropped={len(r['dropped'])} "
              f"tool_only_fan={r['tool_only_fan']} missing_label={len(r['missing_label'])} "
              f"retext_dropped={r['retext_dropped']}(→{{}}:{r['retext_only_now_empty']}) "
              f"k12_dropped={r['k12_default_dropped']} k12_kept={len(r['k12_kept_query_says_同步'])}")
    print(f"all_eval.csv           rows={len(merged):5d}")
    print(f"-> {os.path.relpath(OUT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
