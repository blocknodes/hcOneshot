#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线编译工具目录：直接读 hcSlime/hcTools 的 domains/*/schema.json，
产出 catalog.full.json —— hcOneshot **运行期唯一的工具目录来源**。

**这是构建期工具，不是运行时依赖**：编译一次产出 catalog.full.json 后，hcOneshot
运行期只读该文件，不起 hcTools、不拉任何端点。schema 变更时重跑本脚本即可。

用法：
  python3 tools/build_catalog.py [--hctools <hcTools 目录>] [--out <输出 json>]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WS = Path("/opt/ai/input/data/juice/prd_ai_center/train_project/dingbo4/workspace")
DEF_HCTOOLS = WS / "hcSlime/hcTools"
DEF_OUT = Path(__file__).resolve().parent.parent / "catalog.full.json"


def build(hctools: Path) -> list[dict]:
    sys.path.insert(0, str(hctools))
    from app.domain import DOMAINS_DIR, _load_schema_json, load_domains  # noqa: E402

    out: list[dict] = []
    # 工具清单以 load_domains() 为准 —— 各域 __init__.py 会在 schema.json 之外追加
    # fan_knowledge_agent（每域一个），只扫 schema.json 会漏。参数结构优先 schema.json
    # 的完整版，缺失时（如 fan）退回 __init__ 的裁剪版并按 required 合成 properties。
    for key, domain in load_domains().items():
        full = {t.name: t for t in _load_schema_json(DOMAINS_DIR / key)}
        for t in domain.tools:
            src = full.get(t.name)
            params = dict((src.parameters if src else t.parameters) or {})
            if not isinstance(params, dict):
                params = {}
            if not params.get("properties") and params.get("required"):
                params["properties"] = {n: {"type": "string"} for n in params["required"]}
            params.setdefault("type", "object")
            out.append({
                "id": f"{key}:{t.name}",
                "domain": key,
                "tool": t.name,
                # 与 /api/tools/full 对齐：原始 description 原样落盘，不做裁剪；
                # 目录块裁剪口径唯一收归 app/catalog.py 的 _brief_of。
                "description": (src.description if src else t.description) or "",
                "parameters": params,
                "required": list(params.get("required") or []),
            })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hctools", default=str(DEF_HCTOOLS))
    ap.add_argument("--out", default=str(DEF_OUT))
    args = ap.parse_args()

    tools = build(Path(args.hctools))
    out = Path(args.out)
    out.write_text(json.dumps({"tools": tools}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"工具数 : {len(tools)}")
    print(f"输出   : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
