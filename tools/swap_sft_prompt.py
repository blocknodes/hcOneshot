#!/usr/bin/env python3
"""把 SFT/DPO jsonl 里 system 提示词的工具目录块换成指定 compact 档位，生成新数据集。

只动 system 消息，user/assistant 及其他字段原样保留，逐行流式处理（支持 GB 级文件）。

两种模式：
  block（默认）：只把 system 中的「旧档位目录块」整段替换为新档位目录块，其余规则/锚例保持原文。
                 旧块按 --from-compact（默认 0）用当前 catalog.full.json 渲染后精确匹配；
                 匹配不到的行报错计数，不静默放过。
  full         ：整段 system 换成当前 app.prompts.build_oneshot_prompt(新目录块)
                 （会带上当前 prompt 的新增规则，如 after 顺序条目）。

用法：
  python3 tools/swap_sft_prompt.py --input train_n3.sft.jsonl [--compact 3] [--from-compact 0]
         [--mode block|full] [--output xxx.jsonl]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HC_CATALOG_FILE", str(ROOT / "catalog.full.json"))

from app import catalog  # noqa: E402
from app.prompts import build_oneshot_prompt  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="")
    ap.add_argument("--compact", type=int, default=3, help="新目录档位")
    ap.add_argument("--from-compact", type=int, default=0, help="原数据所用目录档位（block 模式）")
    ap.add_argument("--mode", choices=["block", "full"], default="block")
    args = ap.parse_args()

    src = Path(args.input)
    dst = Path(args.output) if args.output else src.with_name(
        src.name.replace(".sft.jsonl", "") + f".c{args.compact}.sft.jsonl"
        if src.name.endswith(".sft.jsonl") else src.stem + f".c{args.compact}.jsonl")
    if dst.resolve() == src.resolve():
        sys.exit("输出路径与输入相同，拒绝覆盖")

    new_blk = catalog.render_oneshot_block(compact=args.compact)
    old_blk = catalog.render_oneshot_block(compact=args.from_compact)
    full_sys = build_oneshot_prompt(new_blk, "")
    if not new_blk or not old_blk:
        sys.exit("目录为空，检查 HC_CATALOG_FILE")

    n = miss = 0
    cache: dict[str, str] = {}          # 原 system → 新 system（数据集通常只有一种 system）
    tmp = dst.with_name(dst.name + ".tmp")
    with src.open(encoding="utf-8") as fi, tmp.open("w", encoding="utf-8") as fo:
        for ln, line in enumerate(fi, 1):
            if not line.strip():
                continue
            d = json.loads(line)
            for m in d.get("messages", []):
                if m.get("role") != "system":
                    continue
                old = m["content"]
                if old not in cache:
                    if args.mode == "full":
                        cache[old] = full_sys
                    elif old.count(old_blk) == 1:
                        cache[old] = old.replace(old_blk, new_blk)
                    else:
                        cache[old] = ""
                if not cache[old]:
                    miss += 1
                    if miss <= 5:
                        print(f"[miss] 第 {ln} 行 system 中找不到 compact={args.from_compact} 目录块",
                              file=sys.stderr)
                    continue
                m["content"] = cache[old]
            fo.write(json.dumps(d, ensure_ascii=False) + "\n")
            n += 1

    if miss:
        tmp.unlink()
        sys.exit(f"{miss} 条 system 未匹配旧目录块，未生成输出（可试 --from-compact 或 --mode full）")
    tmp.replace(dst)
    sample_old = next(iter(cache))
    print(f"写出 {n} 行 → {dst}")
    print(f"distinct system: {len(cache)}；system 字符 {len(sample_old)} → {len(cache[sample_old])}"
          f"（目录块 {len(old_blk)} → {len(new_blk)}，mode={args.mode}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
