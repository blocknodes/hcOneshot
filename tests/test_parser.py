#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""parser / catalog 单测（无外部依赖，不起服务）。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import catalog as _cat  # noqa: E402
from app.catalog import (block_stats, render_oneshot_block,  # noqa: E402
                         valid_tools_for)
from app.parser import parse_oneshot  # noqa: E402


def _set_file_catalog() -> None:
    # 目录恒为本地文件（自包含，无 HC_CATALOG_SOURCE 开关）；仅清缓存确保重读。
    # 槽值检索默认指向线上服务：单测关掉，保持无网络（需要时由用例显式打桩）。
    import os
    os.environ["HC_SLOT_RETRIEVE_URL"] = ""
    # 单测与 .env 隔离：精简目录 / few-shot 都按默认值（关）测
    os.environ["HC_CATALOG_COMPACT"] = "0"
    os.environ["HC_FEWSHOT_ENABLED"] = "0"
    from app import config
    config.get_settings.cache_clear()


def case(name: str, ok: bool, extra: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + extra) if extra else ''}")
    return ok


def _compact_lossless() -> bool:
    """精简版只合并逐字重复的说明：完整版的每个非空行（去缩进）至少在精简版出现一次；
    引用目标须是更早出现的工具；且精简版确实更短。"""
    import re as _re
    full = render_oneshot_block(compact=False)
    comp = render_oneshot_block(compact=True)
    c_lines = {ln.strip() for ln in comp.splitlines()}
    lost = [ln.strip() for ln in full.splitlines() if ln.strip() and ln.strip() not in c_lines]
    if lost:
        print(f"    精简版丢失: {lost[0][:80]!r} 等 {len(lost)} 行")
        return False
    first_pos: dict[str, int] = {}
    cur = ""
    for i, ln in enumerate(comp.splitlines()):
        m = _re.match(r"- ([a-z_]+:[a-z_]+)：", ln)
        if m:
            cur = m.group(1)
            first_pos.setdefault(cur, i)
        for ref in _re.findall(r"同 ([a-z_]+:[a-z_]+)", ln):
            if ref not in first_pos or first_pos[ref] >= first_pos[cur]:
                print(f"    引用指向未出现/后文工具: {cur} → {ref}")
                return False
    return len(comp) < len(full)


def _minimal_complete(level: int = 2) -> bool:
    """极简版不截断工具/字段：每个工具 id、每个非代填参数名、每个检索 field 名都出现；且明显更短。"""
    blk = render_oneshot_block(compact=level)
    for t in _cat.tools():
        if f"- {t['id']}：" not in blk and f"- {t['id']} |" not in blk:
            print(f"    极简版缺工具: {t['id']}")
            return False
        props, defs = _cat._schema_parts(t.get("parameters") or {})
        drop = _cat.system_filled(t.get("tool", ""))
        for k in props:
            if k in drop or k == "required":
                continue
            if k not in blk:
                print(f"    极简版缺参数: {t['id']}.{k}")
                return False
        for c, node in defs.items():
            if c.endswith("FieldCondition"):
                for name, _d in _cat._field_options(node):
                    if name not in blk:
                        print(f"    极简版缺字段: {t['id']}.{name}")
                        return False
    return len(blk) < len(render_oneshot_block(compact=level - 1)) * 0.6


def _all_docs_present() -> bool:
    """不截断不变量：每个工具描述的最后一行、每个非代填参数说明的最后一行、每个检索字段名都出现在目录块。
    （旧实现把描述截到 90 字，字段清单/时间格式/工具边界全丢；注入截断即 FAIL。）"""
    blk = render_oneshot_block(compact=False)
    for t in _cat.tools():
        desc = _cat._desc_of(t)
        if desc and desc[-1] not in blk:
            print(f"    缺描述末句: {t['id']}")
            return False
        props, defs = _cat._schema_parts(t.get("parameters") or {})
        drop = _cat.system_filled(t.get("tool", ""))
        for k, v in props.items():
            if k in drop or k == "required" or not isinstance(v, dict):
                continue
            if k == "query" and _cat._query_doc(defs):
                for c, node in defs.items():
                    if c.endswith("FieldCondition"):
                        for name, _d in _cat._field_options(node):
                            if f"- {name}" not in blk:
                                print(f"    缺字段: {t['id']}.{name}")
                                return False
                continue
            d = _cat._clean(_cat._unwrap(v, defs).get("description") or v.get("description") or "")
            if d and d[-1] not in blk:
                print(f"    缺参数说明: {t['id']}.{k}")
                return False
    return True


def main() -> int:
    _set_file_catalog()
    print("catalog:")
    # 自包含不变量：目录加载不得触网 / 不得依赖其他项目（不起 hcTools、不拉端点）。
    # 打桩【记录】httpx.get 调用（而非抛错 —— 旧实现把拉取包在 except Exception 里，
    # 抛错会被吞掉，测试假绿）。只要有一次调用即 FAIL。
    import httpx as _httpx
    _real_get = _httpx.get
    _calls: list = []
    _httpx.get = lambda *a, **k: (_calls.append((a, k)), _real_get(*a, **k))[1]
    try:
        _cat._load(force=True)
    finally:
        _httpx.get = _real_get
    results = [case("目录加载不触网（自包含，无 hcTools 依赖）", not _calls,
                    f"httpx.get 调用={_calls}" if _calls else "")]
    st = block_stats()
    results += [
        case("工具数=56(49+schema外7个fan)", st["tools"] == 56, str(st)),
        case("目录块不截断：各工具描述末句、各参数说明都在块里",
             _all_docs_present(), f"{st['block_chars']}/{st['full_schema_chars']}"),
        case("默认（HC_CATALOG_COMPACT 未设）= 完整版",
             render_oneshot_block() == render_oneshot_block(compact=False)),
        case("精简版不丢信息：完整版每一行都在精简版里出现（重复处为「同 <工具id>」引用）",
             _compact_lossless()),
        case("极简版(2)：56 个工具、全部参数名与检索字段名都在，且比精简版短",
             _minimal_complete()),
        case("超简版(3)：56 个工具、全部参数名与检索字段名都在，且比极简版短",
             _minimal_complete(3)),
        case("索引版(4)：56 个工具、全部参数名与检索字段名都在，且比超简版短",
             _minimal_complete(4)),
        case("索引版(4)：catalog 每个工具都有人工用途短句（新增工具须补 _PURPOSE）",
             all(t["id"] in _cat._PURPOSE for t in _cat.tools()) and len(_cat._PURPOSE) == len(_cat.tools()),
             str(sorted({t["id"] for t in _cat.tools()} ^ set(_cat._PURPOSE)))),
        case("vod 工具集合非空", len(valid_tools_for("vod")) > 0,
             str(sorted(valid_tools_for("vod")))),
    ]
    print(f"  目录块字符 {st['block_chars']}，估算 token ~{st['est_block_tokens']}")
    assert render_oneshot_block()

    # 回归：目录块裁剪口径只认 description —— 编译产物不含 brief 预裁剪字段，
    # 裁剪只发生在 catalog._brief_of（单一口径）。
    # 这里钉住「产物不含 brief / 含 description」与「首句过短会续句」两条不变量。
    items = _cat.tools()
    results.append(case("目录字段=description 而非 brief",
                        bool(items) and all("description" in t and "brief" not in t for t in items),
                        f"keys={sorted(items[0].keys())}" if items else "empty"))
    _faq = next((t for t in items if t.get("tool") == "fan_knowledge_agent"), None)
    _fb = _cat._brief_of(_faq) if _faq else ""
    results.append(case("短首句续句（fan_knowledge_agent > 40 字）",
                        len(_fb) > 40 and _fb.startswith("有声内容知识问答"),
                        f"{len(_fb)}字 {_fb[:30]!r}"))
    # 回归：同构子键的内层形状不得被精简记号吃掉（sort 的 rate/hot/new/play 均为
    # {order: asc|desc}，旧记号只到 `sort{rate,hot,new,play}` → 模型换子键/取值靠猜）。
    _sort_tool = next(
        (t for t in items
         if isinstance((t.get("parameters") or {}).get("properties"), dict)
         and "sort" in t["parameters"]["properties"]), None)
    if _sort_tool:
        _schema = _sort_tool["parameters"]
        _defs = _schema.get("definitions") or _schema.get("$defs") or {}
        _mark = _cat._brief_param("sort", _schema["properties"]["sort"], _defs, False)
        results.append(case("同构子键补内层形状（sort 带 order 枚举）",
                            "各" in _mark and "order" in _mark and "asc" in _mark, _mark))
    else:
        print("  [SKIP] 未找到含 sort 的工具")

    # 回归：description 以「。」结尾时不得拼出「。。」
    # （split("。") 末段为空，旧实现照旧追加「。」→ device 的 power/network/common_control 三条中招）
    _tail = _cat._brief_of({"description": "负责设备开机、关机、重启控制。"})
    results.append(case("description 以。结尾不拼出。。",
                        "。。" not in _tail and _tail.endswith("。"), repr(_tail)))
    results.append(case("真目录块无任何。。",
                        "。。" not in render_oneshot_block()))

    # 回归：目录条目缺 domain 时不得让渲染 KeyError → 500。
    # _load_file 此前只校验 id；缺 domain 的行进 render/domain_of_tool 即炸。
    import json as _json2
    import tempfile as _tf2
    with _tf2.TemporaryDirectory() as _td2:
        _p = Path(_td2) / "c.json"
        _p.write_text(_json2.dumps([
            {"id": "vod:vod_search_all", "domain": "vod", "tool": "vod_search_all"},
            {"id": "nope:thing", "tool": "thing"},          # 缺 domain：须被丢弃
        ]), encoding="utf-8")
        _loaded = _cat._load_file(_p)
    results.append(case("_load_file 丢弃缺 domain 的行",
                        len(_loaded) == 1 and _loaded[0]["domain"] == "vod",
                        f"{[t.get('id') for t in _loaded]}"))

    # 回归：schema 畸形（properties 为 list）不得让渲染抛 AttributeError。
    # engine.run_oneshot 在 try 之外渲染目录块，任何渲染异常都会冒泡成 HTTP 500。
    try:
        _mark_bad = _cat._brief_param(
            "x", {"type": "object", "properties": [{"a": 1}]}, {}, False)
        _ok_bad = isinstance(_mark_bad, str)
    except Exception as _e:  # noqa: BLE001
        _ok_bad, _mark_bad = False, f"{type(_e).__name__}: {_e}"
    results.append(case("properties 非 dict 不炸（降级为无子键）", _ok_bad, str(_mark_bad)))

    # 不变量：块渲染对 description 的裁剪结果稳定（同一输入 → 同一输出，无逐次抖动）
    _dom = items[0].get("domain") if items else ""
    results.append(case("同 description 渲染一致（无逐次抖动）",
                        bool(items) and render_oneshot_block() == render_oneshot_block(),
                        f"样例域 {_dom!r}"))
    # 产物同步：catalog.full.json 必须等于 build_catalog.py 重跑结果，
    # 防止改了构建脚本/手改 json 后 file 模式静默漂移。hcTools 目录不可用时跳过。
    # 走子进程调真实入口 —— hcTools/hcOneshot 包名都叫 app，同进程 import 会串。
    import json as _json
    import subprocess as _sp
    import tempfile as _tf
    _hc = Path("/opt/ai/input/data/juice/prd_ai_center/train_project/dingbo4/"
               "workspace/hcSlime/hcTools")
    _bcpy = Path(__file__).resolve().parent.parent / "tools" / "build_catalog.py"
    if _hc.is_dir() and _bcpy.exists():
        with _tf.TemporaryDirectory() as _td:
            _out = Path(_td) / "c.json"
            _sp.run([sys.executable, str(_bcpy), "--hctools", str(_hc),
                     "--out", str(_out)], check=True, capture_output=True)
            _fresh = _json.loads(_out.read_text(encoding="utf-8"))["tools"]
        results.append(case("catalog.full.json == 构建脚本重跑", _fresh == items))
    else:
        print("  [SKIP] hcTools 目录不可用，跳过产物同步检查")

    print("parser:")
    # 1) 合法
    txt = ('[{"q":"搜索2020年之后上映的电影","d":"vod","tool":"vod_search_all",'
           '"params":{"action":"search"}}]')
    it, err = parse_oneshot(txt)
    results.append(case("合法数组", len(it) == 1 and it[0].vt == "vod:vod_search_all" and not err,
                        f"{len(it)} err={err!r}"))
    # 2) ```json 包裹
    it, err = parse_oneshot("```json\n" + txt + "\n```")
    results.append(case("代码块包裹", len(it) == 1 and not err))
    # 3) 前后有解释文字
    it, err = parse_oneshot("好的，计划如下：\n" + txt + "\n以上。")
    results.append(case("前后夹文字", len(it) == 1 and not err))
    # 4) 长键
    it, err = parse_oneshot('[{"query":"放首歌","domain":"music","toolName":"music_song_search"}]')
    results.append(case("长键兼容", len(it) == 1 and it[0].tool == "music_song_search"))
    # 5) 非法工具 → 降级清空 tool、保留 d
    it, err = parse_oneshot('[{"q":"放首歌","d":"music","tool":"not_a_tool","params":{}}]')
    results.append(case("非法工具降级", len(it) == 1 and it[0].d == "music" and it[0].tool == ""))
    # 6) 域非法 → d 空
    it, err = parse_oneshot('[{"q":"你好","d":"qa","tool":"x"}]')
    results.append(case("非法域清空", len(it) == 1 and it[0].d == "" and it[0].tool == ""))
    # 6b) tool 写成 domain:tool（目录渲染形态）→ 拆出裸名并补全 d
    it, err = parse_oneshot('[{"q":"刘德华演过哪些电影","d":"vod","tool":"vod:vod_person_search"}]')
    results.append(case("tool 带域前缀", len(it) == 1 and it[0].tool == "vod_person_search"
                        and it[0].d == "vod"))
    # 6c) 只给 tool=domain:tool、没给 d → 前缀补域
    it, err = parse_oneshot('[{"q":"放首歌","tool":"music:music_song_search"}]')
    results.append(case("仅 vt 补域", len(it) == 1 and it[0].d == "music"
                        and it[0].tool == "music_song_search"))
    # 6d) 只给唯一工具名、没给 d → 据工具反推域（避免白丢已选对的工具）
    it, err = parse_oneshot('[{"q":"放首歌","tool":"music_song_search"}]')
    results.append(case("唯一工具名反推域", len(it) == 1 and it[0].d == "music"
                        and it[0].tool == "music_song_search"))
    # 6e) 跨域重名工具（fan_knowledge_agent）无 d → 不可反推，清空 tool（交下游）
    it, err = parse_oneshot('[{"q":"你好吗","tool":"fan_knowledge_agent"}]')
    results.append(case("跨域重名工具不反推", len(it) == 1 and it[0].d == ""
                        and it[0].tool == ""))
    # 6f) 反推域必须与显式 d 一致：显式 d=music + 唯一属于 vod 的工具 vod_search_all
    #     → 工具不在 music 目录，清空 tool、保留 d=music
    it, err = parse_oneshot('[{"q":"放首歌","d":"music","tool":"vod_search_all"}]')
    results.append(case("显式 d 与工具冲突→清工具留域",
                        len(it) == 1 and it[0].d == "music" and it[0].tool == ""))
    # 7) 多步 + dep
    it, err = parse_oneshot('[{"q":"a","d":"vod","tool":"vod_search_all"},{"q":"b","d":"vod","tool":"fan_knowledge_agent","dep":1}]')
    results.append(case("多步带 dep", len(it) == 2 and it[1].dep == 1))
    # 7b) dep 越界（指向不存在的条）→ 清零
    it, err = parse_oneshot('[{"q":"a","d":"vod","tool":"vod_search_all"},'
                            '{"q":"b","d":"vod","tool":"vod_history","dep":5}]')
    results.append(case("dep 越界清零", len(it) == 2 and it[1].dep == 0))
    # 7c) dep 自指/前指 → 清零
    it, err = parse_oneshot('[{"q":"a","d":"vod","tool":"vod_search_all","dep":1},'
                            '{"q":"b","d":"vod","tool":"vod_history"}]')
    results.append(case("dep 自指清零", len(it) == 2 and it[0].dep == 0))
    # 7d) 序号重映射：第 1 条 q 空被丢，第 2 条 dep=3 原指模型第 3 条 → 映射到 out 第 2 条
    it, err = parse_oneshot('[{"q":"","d":"vod"},'
                            '{"q":"b","d":"vod","tool":"vod_search_all"},'
                            '{"q":"c","d":"vod","tool":"vod_history","dep":2}]')
    results.append(case("dep 序号重映射（丢条后仍指向正确前序）",
                        len(it) == 2 and it[1].dep == 1))
    # 7e) dep 指向被丢条目 → 清零（退化并行）
    it, err = parse_oneshot('[{"q":"a","d":"vod","tool":"vod_search_all"},'
                            '{"q":"","d":"vod"},'
                            '{"q":"c","d":"vod","tool":"vod_history","dep":2}]')
    results.append(case("dep 指向已丢条目→清零", len(it) == 2 and it[1].dep == 0))

    # 7f) dep=true（别名 depends_on_previous 语义）= 依赖紧邻前一条，非第 1 条。
    #     bool 是 int 子类，旧代码 isinstance(dep,int) 会把 true 当成序号 1 → 错指首条。
    it, err = parse_oneshot('[{"q":"a","d":"vod","tool":"vod_search_all"},'
                            '{"q":"b","d":"vod","tool":"vod_history"},'
                            '{"q":"c","d":"vod","tool":"vod_person_search","dep":true}]')
    results.append(case("dep=true→紧邻前一条（第3条指第2条）",
                        len(it) == 3 and it[2].dep == 2))
    # 7g) dep=false → 无依赖
    it, err = parse_oneshot('[{"q":"a","d":"vod","tool":"vod_search_all"},'
                            '{"q":"b","d":"vod","tool":"vod_history","dep":false}]')
    results.append(case("dep=false→0", len(it) == 2 and it[1].dep == 0))
    # 7h) dep 为数字串 "1" → 序号 1
    it, err = parse_oneshot('[{"q":"a","d":"vod","tool":"vod_search_all"},'
                            '{"q":"b","d":"vod","tool":"vod_history","dep":"1"}]')
    results.append(case("dep=\"1\"→1", len(it) == 2 and it[1].dep == 1))
    # 7i) dep 为整数型浮点 1.0 → 1；1.5 非整 → 0
    it, err = parse_oneshot('[{"q":"a","d":"vod","tool":"vod_search_all"},'
                            '{"q":"b","d":"vod","tool":"vod_history","dep":1.0}]')
    results.append(case("dep=1.0→1", len(it) == 2 and it[1].dep == 1))
    it, err = parse_oneshot('[{"q":"a","d":"vod","tool":"vod_search_all"},'
                            '{"q":"b","d":"vod","tool":"vod_history","dep":1.5}]')
    results.append(case("dep=1.5→0", len(it) == 2 and it[1].dep == 0))

    # 7j) 组合：dep=true 归一为模型序 → 再经丢条重映射。
    #     模型 4 条，第1条 q 空被丢；末条 true=依赖模型第3条 → out 第2条。
    it, err = parse_oneshot('[{"q":"","d":"vod"},'
                            '{"q":"b","d":"vod","tool":"vod_search_all"},'
                            '{"q":"c","d":"vod","tool":"vod_history"},'
                            '{"q":"d","d":"vod","tool":"vod_person_search","dep":true}]')
    results.append(case("dep=true 归一 + 丢条重映射叠加",
                        len(it) == 3 and [x.dep for x in it] == [0, 0, 2]))

    # 7k) _repair 不得改坏字符串内容：query 含 ",]" / ",}" 必须原样保留
    it, err = parse_oneshot('[{"q":"查一下,}", "d":"vod", "tool":"vod_search_all"}]')
    results.append(case("_repair 不吞字符串内 ,}",
                        len(it) == 1 and it[0].q == "查一下,}"))
    it, err = parse_oneshot('[{"q":"列表[1,2,]结尾", "d":"vod", "tool":"vod_search_all"}]')
    results.append(case("_repair 不吞字符串内 ,]",
                        len(it) == 1 and it[0].q == "列表[1,2,]结尾"))
    # 7l) 真尾逗号仍要能修（字符串外）
    it, err = parse_oneshot('[{"q":"a","d":"vod","tool":"vod_search_all"},]')
    results.append(case("真尾逗号仍修复", len(it) == 1))

    # 7m) lstrip("json") 字符集剥离坑：'jsonnull'.lstrip('json') == 'ull'（null 被啃）。
    #     现改为精确剥 "json" 前缀，且 _repair 对无尾逗号文本恒等（不引入字符集副作用）。
    from app.parser import parse_json_loose, _repair
    results.append(case("_repair 对 null 恒等（不被字符集剥离啃坏）",
                        _repair("null") == "null" and _repair('{"a":"b"}') == '{"a":"b"}'))
    results.append(case("parse_json_loose 解析裸 null 得 None",
                        parse_json_loose("null") is None
                        and parse_json_loose("jsonnull") is None))
    # ```json 紧贴内容（无换行）
    it, err = parse_oneshot('```json[{"q":"a","d":"vod","tool":"vod_search_all"}]```')
    results.append(case("```json 无换行紧贴", len(it) == 1))

    # 7n) 顶层外置参数折回 params（实测模型把 sort 提到顶层，旧代码静默丢弃）
    it, err = parse_oneshot('[{"q":"搜索评分最高的电影","d":"vod","tool":"vod_search_all",'
                            '"params":{"action":"search"},"sort":{"rate":{"order":"desc"}}}]')
    results.append(case("顶层 sort 折回 params",
                        len(it) == 1 and it[0].params.get("sort") == {"rate": {"order": "desc"}}))
    # 7o) params 内已有同名键 → 以 params 内为准，不被顶层覆盖
    it, err = parse_oneshot('[{"q":"a","d":"vod","tool":"vod_search_all",'
                            '"params":{"sort":{"hot":{"order":"asc"}}},'
                            '"sort":{"rate":{"order":"desc"}}}]')
    results.append(case("params 内优先，顶层不覆盖",
                        len(it) == 1 and it[0].params.get("sort") == {"hot": {"order": "asc"}}))
    # 7p) 信封键/非本工具参数名不得折回
    it, err = parse_oneshot('[{"q":"a","d":"vod","tool":"vod_search_all","params":{},"foo":"bar"}]')
    results.append(case("非本工具顶层键不折回",
                        len(it) == 1 and "foo" not in it[0].params))
    # 7q) 多参数同时外置
    it, err = parse_oneshot('[{"q":"a","d":"vod","tool":"vod_search_all","params":{},'
                            '"action":"search","sort":{"new":{"order":"desc"}}}]')
    results.append(case("多参数外置一并折回",
                        len(it) == 1 and it[0].params.get("action") == "search"
                        and "sort" in it[0].params))

    # 8) 空数组（纯闲聊）：模型本就给空计划 → 合法结果，error 须为空
    # （旧实现一律回 "解析成功但无有效条目"，与 OneResponse.error「空=正常」契约矛盾，
    #  下游会把正确的闲聊响应当失败）。
    it, err = parse_oneshot("[]")
    results.append(case("空数组（闲聊）→ error 空（合法结果）", it == [] and err == ""))
    it, err = parse_oneshot('{"plan":[]}')
    results.append(case("空 plan 包裹 → error 空", it == [] and err == ""))
    # 反例：有条目但全被降级丢弃 → 仍报错（真异常，应与上面区分）
    it, err = parse_oneshot('[{"q":""},{"q":"  "}]')
    results.append(case("条目全无效 → 保留 error", it == [] and err == "解析成功但无有效条目"))
    # 9) 尾逗号修复
    it, err = parse_oneshot('[{"q":"a","d":"vod","tool":"vod_search_all","params":{},}]')
    results.append(case("尾逗号修复", len(it) == 1))
    # 10) 完全无法解析
    it, err = parse_oneshot("这不是 JSON")
    results.append(case("非 JSON 报错", it == [] and bool(err)))
    # 11) dict 包裹
    it, err = parse_oneshot('{"plan":[{"q":"a","d":"vod","tool":"vod_search_all"}]}')
    results.append(case("dict 包裹", len(it) == 1))

    # --- LLM 响应形状：网关异常响应不得让 IndexError 冒泡成 HTTP 500 ---
    from app.llm import _extract_message, LLMError  # noqa: E402
    print("llm:")
    def _raises_value(fn):
        try:
            fn()
            return False
        except ValueError:
            return True
        except Exception:
            return False
    results.append(case("空 choices → ValueError（非 IndexError）",
                        _raises_value(lambda: _extract_message({"choices": []}))))
    results.append(case("缺 choices 键 → ValueError",
                        _raises_value(lambda: _extract_message({"usage": {}}))))
    results.append(case("非 dict 响应 → ValueError",
                        _raises_value(lambda: _extract_message("oops"))))
    results.append(case("choices[0] 非 dict → ValueError",
                        _raises_value(lambda: _extract_message({"choices": ["x"]}))))
    results.append(case("缺 message → ValueError",
                        _raises_value(lambda: _extract_message({"choices": [{}]}))))
    results.append(case("正常响应取到 message",
                        _extract_message({"choices": [{"message": {"content": "hi"}}]})
                        == {"content": "hi"}))
    # LLMError 是 Exception 子类，engine 的 except LLMError 依赖此
    results.append(case("LLMError 可被捕获为基类异常", issubclass(LLMError, Exception)))

    # --- 请求模型：空 query 两种形态须同语义（都进端点结构化 error，而非 400）---
    from app.models import OneRequest  # noqa: E402
    print("request:")
    # 扁平 {"query":""}：旧 validator raise → HTTP 400；嵌套不触发校验器 → 200。已统一。
    results.append(case("扁平空 query 不抛校验错",
                        OneRequest(query="").resolved_query() == ""))
    results.append(case("嵌套空 query 同语义",
                        OneRequest(data={"query": ""}).resolved_query() == ""))
    results.append(case("纯空白 query → 空（端点 error）",
                        OneRequest(query="   ").resolved_query() == ""))
    results.append(case("query 仍首尾归一",
                        OneRequest(query="  hi  ").resolved_query() == "hi"))
    results.append(case("data 兜底当扁平缺省",
                        OneRequest(data={"query": "真的"}).resolved_query() == "真的"))
    results.append(case("扁平优先于 data",
                        OneRequest(query="x", data={"query": "y"}).resolved_query() == "x"))

    # --- slowAgent：计划→批次→step 组装（hcAgent 契约）---
    from app.engine import (Intent, next_batch, parse_plan,  # noqa: E402
                            _executed_indexes, _steps_from_batch, _turnslow_steps,
                            _tool_result_text)
    from app.fanparams import rebuild_fan_params  # noqa: E402
    print("slowagent:")
    _plan = parse_plan(
        '[{"q":"搜索评分最高的胡歌演的电影","d":"vod","tool":"vod_search_all","params":{"action":"search"}},'
        '{"q":"这一部的导演是谁","d":"vod","tool":"fan_knowledge_agent",'
        '"params":{"messages":"这一部的导演是谁"},"dep":1}]')
    results.append(case("计划条目带 dep 解析", _plan.total() == 2 and _plan.intents[1].depends
                        and _plan.intents[1].dep_on == 1))
    results.append(case("首轮只发无依赖条（s1）",
                        [i.index for i in next_batch(_plan, set())] == [1]))
    results.append(case("s1 执行后发依赖条（s2）",
                        [i.index for i in next_batch(_plan, {1})] == [2]))
    results.append(case("toolHistory 序号识别",
                        _executed_indexes([{"id": "s1"}, {"index": 2}]) == {1, 2}))

    # step 组装：并行两条 → 各自 id/dependsOn/retext
    _b = parse_plan('[{"q":"放首歌","d":"music","tool":"music_song_search","params":{}},'
                    '{"q":"调音量到30","d":"device","tool":"numeric_adjust","params":{"value":"30"}}]')
    _s, _f = _steps_from_batch(_b.intents)
    results.append(case("step id/dependsOn 拼装",
                        [x["id"] for x in _s] == ["s1", "s2"] and all(not x["dependsOn"] for x in _s)
                        and _f is False))

    # fan 契约规范化：模型给 messages:string → 必须变数组 + riskRespStrategy + sceneType
    _fb = [Intent(query="这部片导演是谁", domain="vod", tool="fan_knowledge_agent",
                  index=1, src="这部片导演是谁", params={"messages": "这部片导演是谁"})]
    _fs, _ = _steps_from_batch(_fb)
    _fp = _fs[0]["parameters"]
    results.append(case("fan messages 字符串→数组契约",
                        isinstance(_fp.get("messages"), list)
                        and _fp["messages"][0]["content"] == "这部片导演是谁"
                        and _fp.get("riskRespStrategy") == "all_replace"
                        and _fp["sceneTypeIntentList"][0]["sceneType"] == "video"))
    results.append(case("rebuild 三种 messages 形态等价",
                        (rebuild_fan_params({"messages": "x"}) == rebuild_fan_params({"messages": [{"role": "user", "content": "x"}]})
                         == rebuild_fan_params({"messages": {"content": "x"}}))
                        and rebuild_fan_params({}, fallback_query="fb")["messages"][0]["content"] == "fb"))

    # 无工具（选工具失败/目录不可用）→ fan 兜底。域信号须保留：
    # d="" 落域失败 → knowledge_qa（通用 qa）；d="music" → agent_music（域内 qa）。
    _nb = [Intent(query="你好吗", domain="", tool="", index=1, src="你好吗")]
    _ns, _ = _steps_from_batch(_nb)
    results.append(case("无工具(d空)→fan 兜底 knowledge_qa",
                        _ns[0]["toolName"] == "fan_knowledge_agent"
                        and isinstance(_ns[0]["parameters"]["messages"], list)
                        and _ns[0]["parameters"]["sceneTypeIntentList"][0]["sceneType"] == "knowledge_qa"))
    _nb2 = [Intent(query="放首歌", domain="music", tool="", index=1, src="放首歌")]
    _ns2, _ = _steps_from_batch(_nb2)
    results.append(case("无工具(d=music)→域内 qa(agent_music)",
                        _ns2[0]["parameters"]["sceneTypeIntentList"][0]["sceneType"] == "agent_music"))

    # turnSlow 路由：同域模糊 vs 收口 fan 终点（forceStop）
    def _hist(tool, reason=None):
        d = {"turnSlow": True}
        if reason:
            d["reason"] = reason
        return [{"index": 1, "id": "s1", "toolName": tool, "result": {"data": d}}]
    _ts = _turnslow_steps(_hist("vod_search_all", "NO_SLOT_MAPPING_VALUE"))
    results.append(case("turnSlow 提取工具→reason",
                        _ts == {"vod_search_all": "NO_SLOT_MAPPING_VALUE"}))
    _mb = [Intent(query="q", domain="vod", tool="vod_search_all", index=1, src="q")]
    _ms, _mf = _steps_from_batch(_mb, history=_hist("vod_search_all", "NO_SLOT_MAPPING_VALUE"))
    results.append(case("turnSlow 同域→模糊兜底(非终点)",
                        _ms[0]["toolName"] == "vod_fuzzy_search" and _mf is False))
    # fuzzy 兜底的参数须是 retext（三个 fuzzy 工具 schema 均为 {retext}，
    # additionalProperties:false → 发 query 会被丢弃、兜底参数空转）。
    results.append(case("turnSlow 模糊兜底参数=retext（vod 亦非 query）",
                        _ms[0]["parameters"] == {"retext": "q"}, str(_ms[0]["parameters"])))
    # children 域（工具前缀 educ_*，与 education 的 edu_* 是不同域）：少儿检索 turnSlow
    # 后须换同域 educ_fuzzy_search，而非穿到 fan 终点。曾因 _TURNSLOW_FUZZY 缺 children 键而漏。
    _mb_ch = [Intent(query="白雪公主绘本", domain="children",
                     tool="educ_search_all", index=1, src="白雪公主绘本")]
    _ms_ch, _mf_ch = _steps_from_batch(
        _mb_ch, history=_hist("educ_search_all", "NO_SLOT_MAPPING_VALUE"))
    results.append(case("turnSlow children→educ_fuzzy_search（不穿到 fan）",
                        _ms_ch[0]["toolName"] == "educ_fuzzy_search" and _mf_ch is False
                        and _ms_ch[0]["parameters"] == {"retext": "白雪公主绘本"},
                        f"{_ms_ch[0]['toolName']} {_ms_ch[0]['parameters']}"))
    _ms2, _mf2 = _steps_from_batch(_mb, history=_hist("vod_search_all", "OTHER"))
    results.append(case("turnSlow 未知→fan 终点 forceStop",
                        _ms2[0]["toolName"] == "fan_knowledge_agent" and _mf2 is True))
    _ms3, _mf3 = _steps_from_batch(_mb, history=_hist("vod_search_all"))  # 裸 turnSlow 无 reason
    results.append(case("裸 turnSlow(无 reason)→fan 终点",
                        _ms3[0]["toolName"] == "fan_knowledge_agent" and _mf3 is True))

    # _tool_result_text：最新一步的候选优先，带「第N部」序号
    _tr = _tool_result_text([{"index": 1, "result": {"data": {"memoryData": [{"data": [
        {"mediaTitle": "功夫", "director": ["周星驰"], "doubanRate": "7.8"},
        {"mediaTitle": "美人鱼", "director": ["周星驰"]}]}]}}}])
    results.append(case("工具结果提候选带排位",
                        "第1部" in _tr and "功夫" in _tr and "周星驰" in _tr))
    results.append(case("纯文本工具结果直取",
                        _tool_result_text([{"result": "《无间道》的导演是刘伟强"}]) == "《无间道》的导演是刘伟强"))

    # --- 移植自 hcAgent 的三项：阵容 matchId 绑定 / 双域并行标记 / 赛事预测串行挂起 ---
    from app.engine import (_bind_sports_short_memory_match_id,  # noqa: E402
                            _arm_forecast_serial_plan, _short_memory_candidate_id,
                            TraceStateMachine, _with_source)
    print("slowagent-port:")
    # (a) 阵容 matchId：从 shortMemory 候选按「第N场」序号取 id。
    _sm_mem = [{"businessData": {"serviceData": [{"data": [
        {"id": "1001", "title": "甲队 vs 乙队"},
        {"id": "1002", "title": "丙队 vs 丁队"},
        {"id": "1003", "title": "戊队 vs 己队"}]}]}}]
    results.append(case("shortMemory 数字序号取候选 id",
                        _short_memory_candidate_id(_sm_mem, "看第2场的首发") == 1002))
    results.append(case("shortMemory 中文序号取候选 id",
                        _short_memory_candidate_id(_sm_mem, "第三场阵容") == 1003))
    results.append(case("shortMemory 缺序号默认第1场",
                        _short_memory_candidate_id(_sm_mem, "这场谁首发") == 1001))
    results.append(case("shortMemory 序号越界 → None",
                        _short_memory_candidate_id(_sm_mem, "看第9场") is None))
    _lineup = [{"id": "s1", "toolName": "sports_match_lineup_search",
                "parameters": {}, "retext": "第2场首发阵容", "hitQuery": "第2场首发阵容"}]
    _bind_sports_short_memory_match_id(_lineup, _sm_mem, "第2场首发阵容")
    results.append(case("阵容 step 回填 matchId",
                        _lineup[0]["parameters"].get("matchId") == 1002))
    _lineup_legacy = [{"id": "s2", "toolName": "sports_match_lineup_search",
                       "parameters": {"id": "7777"}, "retext": "阵容", "hitQuery": "阵容"}]
    _bind_sports_short_memory_match_id(_lineup_legacy, _sm_mem, "阵容")
    results.append(case("旧 id 参数升级为 matchId（不留 id）",
                        _lineup_legacy[0]["parameters"].get("matchId") == "7777"
                        and "id" not in _lineup_legacy[0]["parameters"]))
    _lineup_keep = [{"id": "s3", "toolName": "sports_match_lineup_search",
                     "parameters": {"matchId": "5555"}, "retext": "阵容", "hitQuery": "阵容"}]
    _bind_sports_short_memory_match_id(_lineup_keep, _sm_mem, "阵容")
    results.append(case("已有 matchId 不被覆盖",
                        _lineup_keep[0]["parameters"]["matchId"] == "5555"))
    _other = [{"id": "s4", "toolName": "music_song_search",
               "parameters": {}, "retext": "第2场", "hitQuery": "第2场"}]
    _bind_sports_short_memory_match_id(_other, _sm_mem, "第2场")
    results.append(case("非阵容工具不动参数",
                        "matchId" not in _other[0]["parameters"]))

    # (b) 赛事预测串行挂起：首步命中 sports_match_forecast → 存 2 意图计划、本轮仅 s1。
    _sm2 = TraceStateMachine()
    import app.engine as _eng  # noqa: E402
    _orig_sm = _eng._sm
    _eng._sm = lambda: _sm2
    try:
        _fb_batch = [Intent(query="预测甲队乙队胜负", domain="sports",
                            tool="sports_match_forecast", index=1, src="预测甲队乙队胜负")]
        _fb_steps, _ = _steps_from_batch(_fb_batch)
        armed = _arm_forecast_serial_plan("dev-x", _fb_batch, _fb_steps, "预测甲队乙队胜负")
        results.append(case("赛事预测首步 → 挂起串行计划",
                            armed is True and _sm2._load("dev-x") is not None))
        _nxt = next_batch(_sm2._state["dev-x"]["plan"], {1})
        results.append(case("挂起计划 s2 依赖 s1 且为 fan 终点",
                            [i.index for i in _nxt] == [2] and _nxt[0].depends
                            and _nxt[0].dep_on == 1 and _nxt[0].tool == "fan_knowledge_agent"
                            and _nxt[0].include_tool_result is True))
        # 非预测工具 / 非首步（有历史）不应挂起
        _nb_batch = [Intent(query="放首歌", domain="music",
                            tool="music_song_search", index=1, src="放首歌")]
        _nb_steps, _ = _steps_from_batch(_nb_batch)
        results.append(case("非预测工具不挂起",
                            _arm_forecast_serial_plan("dev-y", _nb_batch, _nb_steps, "放首歌") is False
                            and _sm2._load("dev-y") is None))
        results.append(case("空 state_key 不挂起",
                            _arm_forecast_serial_plan("", _fb_batch, _fb_steps, "q") is False))

        # (c) 双域并行标记：首轮无历史、恰好 children+vod 两意图 → data.parallel=true 且 stop。
        import asyncio  # noqa: E402
        from app.models import SlowRequest  # noqa: E402
        _dual = [Intent(query="放动画片", domain="children", tool="children_search", index=1, src="q"),
                 Intent(query="放电影", domain="vod", tool="vod_search_all", index=2, src="q")]

        class _FakeSM:
            async def tick(self, key, q, hist, weak_hint=""):
                return _dual, True

        _eng._sm = lambda: _FakeSM()
        _req = SlowRequest(traceId="t1", deviceId="d1",
                           data={"query": "放动画片和电影"})
        _body = asyncio.get_event_loop().run_until_complete(_eng.build_response(_req))
        results.append(case("双域并行 → data.parallel=true",
                            _body["data"].get("parallel") is True and _body["stop"] is True))
        # 有历史时不标 parallel
        _req2 = SlowRequest(traceId="t2", deviceId="d2",
                            data={"query": "放动画片和电影",
                                  "toolHistory": [{"index": 1, "id": "s1"}]})
        _body2 = asyncio.get_event_loop().run_until_complete(_eng.build_response(_req2))
        results.append(case("有历史 → 不标 parallel",
                            "parallel" not in _body2["data"]))
        # 单域两意图不标 parallel
        _single_dom = [Intent(query="a", domain="vod", tool="t1", index=1, src="q"),
                       Intent(query="b", domain="vod", tool="t2", index=2, src="q")]

        class _FakeSM2:
            async def tick(self, key, q, hist, weak_hint=""):
                return _single_dom, True

        _eng._sm = lambda: _FakeSM2()
        _req3 = SlowRequest(traceId="t3", deviceId="d3", data={"query": "q"})
        _body3 = asyncio.get_event_loop().run_until_complete(_eng.build_response(_req3))
        results.append(case("同域两意图 → 不标 parallel",
                            "parallel" not in _body3["data"]))

        # 端到端：走 build_response 才能真正验证调用点接线（非仅 helper 本身）。
        # matchId 回填须发生在 build_response 内（short_memory 透传链）。
        _lineup_batch = [Intent(query="第2场首发阵容", domain="sports",
                                tool="sports_match_lineup_search", index=1,
                                src="第2场首发阵容")]

        class _FakeSM3:
            async def tick(self, key, q, hist, weak_hint=""):
                return _lineup_batch, True

        _eng._sm = lambda: _FakeSM3()
        _req4 = SlowRequest(traceId="t4", deviceId="d4", data={
            "query": "第2场首发阵容",
            "memory": {"shortMemory": _sm_mem}})
        _body4 = asyncio.get_event_loop().run_until_complete(_eng.build_response(_req4))
        results.append(case("build_response 透传 shortMemory 回填 matchId",
                            _body4["data"]["steps"][0]["parameters"].get("matchId") == 1002))

        # 端到端：首步预测 → stop 被强制 false（本轮仅发 s1，s2 待续跑）。
        # 注意：_arm_forecast_serial_plan 内部调 _sm()._store，故 fake 须转发 _store。
        _sm3 = TraceStateMachine()

        class _FakeSM4:
            async def tick(self, key, q, hist, weak_hint=""):
                return _fb_batch, True

            def _store(self, key, plan):
                _sm3._store(key, plan)

        _fake4 = _FakeSM4()
        _eng._sm = lambda: _fake4
        _req5 = SlowRequest(traceId="t5", deviceId="d5", data={"query": "预测甲队乙队胜负"})
        _body5 = asyncio.get_event_loop().run_until_complete(_eng.build_response(_req5))
        results.append(case("build_response 预测首步 → stop=false",
                            _body5["stop"] is False
                            and _body5["data"]["steps"][0]["toolName"] == "sports_match_forecast"))

        # 续跑钉工具：计划 tool 权威。模型若擅自换工具（如把挂起的 fan 终点换成重发预测），
        # 必须钉回计划工具，否则链路收不了口。回归 bug：曾放任换工具 → s2 重发
        # sports_match_forecast 而非 fan_knowledge_agent。
        async def _fake_rewrite(self, q, tool_result, step=0, plan=None):
            return (q or "改写句"), "sports_match_forecast", {"query": "x"}

        _orig_rewrite = TraceStateMachine._rewrite_step
        TraceStateMachine._rewrite_step = _fake_rewrite
        try:
            _sm4 = TraceStateMachine()
            _pin = parse_plan(
                '[{"q":"先预测","d":"sports","tool":"sports_match_forecast","params":{"query":"甲队乙队"}},'
                '{"q":"介绍两队近况","d":"sports","tool":"fan_knowledge_agent","params":{},"dep":1}]')
            _with_source(_pin, "预测胜负并介绍近况")
            # 依赖步的工具只可能由系统挂起（如 _arm_forecast_serial_plan 的 fan s2），
            # parse_plan 会丢弃模型给依赖步的工具，故这里按挂起口径显式设定。
            _pin.intents[1].tool = "fan_knowledge_agent"
            _sm4._store("dev-pin", _pin)
            _sm4._state["dev-pin"]["plan"].intents[1].include_tool_result = True
            _batch, _stop = asyncio.get_event_loop().run_until_complete(
                _sm4.tick("dev-pin", "预测胜负并介绍近况",
                          [{"index": 1, "id": "s1", "toolName": "sports_match_forecast"}]))
            results.append(case("续跑钉回计划工具（拒绝模型擅换）",
                                len(_batch) == 1 and _batch[0].index == 2
                                and _batch[0].tool == "fan_knowledge_agent"))
        finally:
            TraceStateMachine._rewrite_step = _orig_rewrite

        # --- 多轮改写（MT）：纯函数 + build_response 门控 ---
        from app.engine import (_mt_add, _mt_get, _mt_merge, _mt_serialize,
                                _mt_append_answer, _has_mt_history, _short_turns,
                                _MT_CONTEXT, _MT_LOCK)
        with _MT_LOCK:
            _MT_CONTEXT.clear()
        print("mt:")
        # _short_turns：只收已完成轮（无应答无候选的 stub 丢弃），候选带排位
        _smem = [
            {"dialogData": {"query": "搜索周星驰电影"},
             "businessData": {"serviceData": [{"data": [
                 {"mediaTitle": "功夫", "director": ["周星驰"]},
                 {"mediaTitle": "美人鱼", "director": ["周星驰"]}]}]}},
            {"dialogData": {"query": "看第2部", "answer": "已为你播放《美人鱼》"}},
            {"dialogData": {"query": "空 stub"}, "businessData": {"serviceData": []}},
        ]
        _st = _short_turns(_smem)
        results.append(case("_short_turns 收已完成轮并丢空 stub",
                            len(_st) == 2 and _st[1]["answer"] == "已为你播放《美人鱼》"))
        results.append(case("_short_turns 候选带排位",
                            "第1部" in _st[0]["answer"] and "功夫" in _st[0]["answer"]))
        results.append(case("_short_turns 空输入 → 空", _short_turns(None) == []))
        # _mt_merge：按轮对齐，q 取真实链路、a 取 shortMemory
        _rq = [{"q": "搜索周星驰电影", "domain": "vod"}, {"q": "看第2部", "domain": "vod"}]
        _merged = _mt_merge(_rq, _short_turns(_smem))
        results.append(case("_mt_merge 按轮对齐 q/a",
                            [m["q"] for m in _merged] == ["搜索周星驰电影", "看第2部"]
                            and "美人鱼" in _merged[1]["answer"]))
        # real_q 更长：余轮保留为纯 query（a 空）
        _long = _mt_merge(_rq + [{"q": "第三个呢"}], _short_turns(_smem))
        results.append(case("_mt_merge 余轮保留纯 query 不丢",
                            len(_long) == 3 and _long[2]["q"] == "第三个呢"
                            and _long[2]["answer"] == ""))
        results.append(case("_mt_serialize 编号 + 空应答占位",
                            _mt_serialize([{"q": "a", "answer": ""}]) == "1. 用户：a → 助手：（应答为空）"))
        # _mt_add/_mt_get：同 q 续同轮不重复累积；answer 空不追加
        _mt_add("d1", {"q": "q1", "answer": "a1"})
        _mt_add("d1", {"q": "q1", "answer": "a1b"})   # 同 q → 覆盖末轮
        _mt_add("", {"q": "q0", "answer": "a0"})      # 无 device → 忽略
        _g = _mt_get("d1")
        results.append(case("_mt_add 同 q 续同轮不累积",
                            len(_g) == 1 and _g[0]["answer"] == "a1b"))
        results.append(case("_mt_add 无 device 忽略", _mt_get("") == []))
        _mt_append_answer("d1", "q2", "")             # answer 空 → 不追加
        results.append(case("_mt_append_answer 空 answer 不追加", len(_mt_get("d1")) == 1))
        _mt_append_answer("d1", "q2", "a2", domain="vod")
        _g2 = _mt_get("d1")
        results.append(case("_mt_append_answer 带域名写回",
                            len(_g2) == 2 and _g2[1].get("domain") == "vod"))
        # _has_mt_history：只有进程内上下文或 shortMemory 才算有历史
        results.append(case("_has_mt_history 有进程内历史为真", _has_mt_history("d1", None)))
        results.append(case("_has_mt_history 仅 shortMemory 亦真",
                            _has_mt_history("d-empty", [{"dialogData": {"answer": "x"}}])))
        results.append(case("_has_mt_history 全空为假",
                            _has_mt_history("d-empty", None) is False
                            and _has_mt_history("d-empty", []) is False))
        # MT 的两个旋钮须走 settings（HC_MT_TTL / HC_MT_MAX_TURNS 可调），
        # 而非模块常量——否则环境变量静默失效。回归 bug：曾读 _MT_TTL/_MT_MAX_TURNS 常量。
        import os as _os3  # noqa: E402
        from app.config import get_settings as _gs  # noqa: E402
        _gs.cache_clear()
        _os3.environ["HC_MT_MAX_TURNS"] = "2"
        _os3.environ["HC_MT_TTL"] = "1"
        try:
            _gs.cache_clear()
            with _MT_LOCK:
                _MT_CONTEXT.clear()
            for i in range(3):
                _mt_add("dmax", {"q": f"q{i}", "answer": f"a{i}"})
            results.append(case("HC_MT_MAX_TURNS 生效（保留最近 2 轮）",
                                [t["q"] for t in _mt_get("dmax")] == ["q1", "q2"]))
            with _MT_LOCK:
                _MT_CONTEXT["dmax"]["ts"] = time.time() - 5
            results.append(case("HC_MT_TTL 生效（1s 后过期）", _mt_get("dmax") == []))
        finally:
            _os3.environ.pop("HC_MT_MAX_TURNS", None)
            _os3.environ.pop("HC_MT_TTL", None)
            _gs.cache_clear()
            with _MT_LOCK:
                _MT_CONTEXT.clear()

        # build_response 门控：无任何上下文 → 不调 _mt_rewrite（原句直通）；
        # 有 shortMemory → 调 _mt_rewrite（改写历史）。
        _mt_calls = []

        async def _fake_mt(q, device_id, short_memory=None, trace_id=""):
            _mt_calls.append(("mt", q))
            return "改写后的自包含句"

        _orig_mt = _eng._mt_rewrite
        _eng._mt_rewrite = _fake_mt
        try:
            class _FakeSM5:
                async def tick(self, key, q, hist, weak_hint=""):
                    _mt_calls.append(("tick", q))
                    return [], True

            _eng._sm = lambda: _FakeSM5()
            # (i) 无 history 无 shortMemory 无 MT 上下文 → 原句直通（tick 收到原句）
            _mt_calls.clear()
            asyncio.get_event_loop().run_until_complete(_eng.build_response(
                SlowRequest(traceId="m1", deviceId="md1", data={"query": "你好"})))
            results.append(case("MT 首轮无上下文 → 不改写（原句直通 tick）",
                                ("tick", "你好") in _mt_calls
                                and not any(c == "你好" for c in _mt_calls if isinstance(c, str))))
            # (ii) 有 shortMemory → 改写，tick 收到改写句
            _mt_calls.clear()
            asyncio.get_event_loop().run_until_complete(_eng.build_response(
                SlowRequest(traceId="m2", deviceId="md2", data={
                    "query": "第三个呢",
                    "memory": {"shortMemory": [{"dialogData": {"query": "搜索", "answer": "已播放《A》"}}]}})))
            results.append(case("MT 有 shortMemory → 改写（tick 收到改写句）",
                                ("mt", "第三个呢") in _mt_calls
                                and ("tick", "改写后的自包含句") in _mt_calls))
            # (iii) 有 history（续跑）→ 不改写，原句交给状态机
            _mt_calls.clear()
            asyncio.get_event_loop().run_until_complete(_eng.build_response(
                SlowRequest(traceId="m3", deviceId="md3", data={
                    "query": "第三个呢",
                    "memory": {"shortMemory": [{"dialogData": {"answer": "x"}}]},
                    "toolHistory": [{"index": 1, "id": "s1"}]})))
            results.append(case("MT 续跑轮 → 不改写（原句交状态机）",
                                not any(c == "第三个呢" for c in _mt_calls if isinstance(c, str))
                                and ("tick", "第三个呢") in _mt_calls))
        finally:
            _eng._mt_rewrite = _orig_mt
            with _MT_LOCK:
                _MT_CONTEXT.clear()
    finally:
        _eng._sm = _orig_sm
        _sm2._state.clear()

    # --- badcase 精确覆盖层（debug 专用，默认关）---
    # 纯函数：_norm 归一 / _render 简写与完整两种写法 / lookup 开关门控。
    import app.badcase as _bc  # noqa: E402
    print("badcase:")
    results.append(case("_norm 折叠空白",
                        _bc._norm("  你好   世界 ") == "你好 世界"))
    # _render 简写：补全 code/message/planId 等 + 回填 traceId/deviceId
    _rb = _bc._render({"steps": [{"id": "s1", "toolName": "t"}], "stop": True,
                       "forceStop": False}, "tr1", "dv1")
    results.append(case("_render 简写补全信封",
                        _rb["code"] == 200 and _rb["message"] == "success"
                        and _rb["traceId"] == "tr1" and _rb["deviceId"] == "dv1"
                        and _rb["data"]["steps"] == [{"id": "s1", "toolName": "t"}]
                        and _rb["data"]["planId"].startswith("bc_")
                        and _rb["data"]["planConfidence"] == 1.0
                        and _rb["stop"] is True and _rb["forceStop"] is False))
    _rpf = _bc._render({"steps": [], "parallel": True, "final": True}, "t", "d")
    results.append(case("_render 简写透传 parallel/final",
                        _rpf["data"]["parallel"] is True
                        and _rpf["data"]["final"] is True
                        and _rpf["data"]["schemaVersion"] == "1.0"))
    # _render 完整 body 写法：不改动其 data 内容，只回填 traceId/deviceId
    _rb2 = _bc._render({"data": {"steps": [{"id": "x"}], "parallel": True},
                        "stop": False, "forceStop": True}, "tr2", "dv2")
    results.append(case("_render 完整 body 保真",
                        _rb2["data"]["steps"] == [{"id": "x"}]
                        and _rb2["data"]["parallel"] is True
                        and _rb2["stop"] is False and _rb2["forceStop"] is True
                        and _rb2["traceId"] == "tr2"))
    results.append(case("_render 非 dict → None", _bc._render("oops", "t", "d") is None))
    # 开关门控：默认关 → lookup 恒 None（即便缓存里有内容）
    _orig_cache = _bc._CACHE
    _bc._CACHE = {"你好": {"steps": [], "stop": True}}
    try:
        import os as _os
        _os.environ.pop("HC_BADCASE_ENABLED", None)
        results.append(case("开关关 → lookup None（不命中）", _bc.lookup("你好") is None))
        _os.environ["HC_BADCASE_ENABLED"] = "1"
        results.append(case("开关开 → lookup 命中（空白归一）",
                            _bc.lookup("  你好  ") == {"steps": [], "stop": True}))
        results.append(case("开关开但未收录 → None", _bc.lookup("不存在") is None))
        # build_response 门控：命中即直出（不触状态机），未命中走主干
        from app.models import SlowRequest  # noqa: E402
        _tick_calls = []

        class _FakeSM6:
            async def tick(self, key, q, hist, weak_hint=""):
                _tick_calls.append(q)
                return [], True

        _orig_sm2 = _eng._sm
        _eng._sm = lambda: _FakeSM6()
        try:
            _bc._CACHE = {_bc._norm("我想看悬疑剧"): {
                "steps": [{"id": "s1", "toolName": "vod_fuzzy_search",
                           "parameters": {"retext": "悬疑剧"}}], "stop": True}}
            _hit_body = asyncio.get_event_loop().run_until_complete(_eng.build_response(
                SlowRequest(traceId="bt1", deviceId="bd1", data={"query": "我想看悬疑剧"})))
            results.append(case("badcase 命中 → 直出且不触状态机",
                                _hit_body["data"]["steps"][0]["toolName"] == "vod_fuzzy_search"
                                and _hit_body["traceId"] == "bt1"
                                and _tick_calls == []))
            _tick_calls.clear()
            _miss_body = asyncio.get_event_loop().run_until_complete(_eng.build_response(
                SlowRequest(traceId="bt2", deviceId="bd2", data={"query": "没收录的句子"})))
            results.append(case("badcase 未命中 → 走主干（触状态机）",
                                _tick_calls == ["没收录的句子"]))
        finally:
            _eng._sm = _orig_sm2
            _os.environ.pop("HC_BADCASE_ENABLED", None)
    finally:
        _bc._CACHE = _orig_cache
        import os as _os2
        _os2.environ.pop("HC_BADCASE_ENABLED", None)

    # --- 超长 query 守卫（HC_MAX_QUERY_LEN，声明式旋钮须真生效）---
    # 一把出把整句塞进 prompt；超长应回结构化 error（HTTP 200），非 500/静默。
    print("query-limit:")
    from app.config import get_settings as _gsq  # noqa: E402
    import os as _osq  # noqa: E402
    _gsq.cache_clear()
    _osq.environ["HC_MAX_QUERY_LEN"] = "8"
    try:
        _gsq.cache_clear()

        class _FakeSMq:                      # 假状态机：离线、不触 LLM
            async def tick(self, key, q, hist, weak_hint=""):
                return [], True

        _eng._sm = lambda: _FakeSMq()
        _long = "很长很长很长很长很长很长的用户请求"
        _lb = asyncio.get_event_loop().run_until_complete(_eng.build_response(
            SlowRequest(traceId="ql1", deviceId="ql1", data={"query": _long})))
        results.append(case("超长 query → 结构化 error（非 500）",
                            _lb["code"] == 400 and "too long" in _lb["message"]
                            and _lb["stop"] is True and _lb["data"]["steps"] == []))
        # 边界：恰好等于上限 → 放行（不误伤）
        _sb = asyncio.get_event_loop().run_until_complete(_eng.build_response(
            SlowRequest(traceId="ql2", deviceId="ql2", data={"query": "8" * 8})))
        results.append(case("恰好等于上限 → 放行", _sb["code"] == 200))
    finally:
        _osq.environ.pop("HC_MAX_QUERY_LEN", None)
        _gsq.cache_clear()
        _eng._sm = _orig_sm

    # --- 核心不变量：【无上下文单轮 = 恰好 1 次 LLM】（一把出的立身之本）---
    # 数真实 llm.chat 调用（打桩计次），走真实 TraceStateMachine + _steps_from_batch。
    # 1 次 = 一把出（拆句+改写+定域+选工具+填参一次出）；≥2 = 退回两跳/多余改写，
    # 即本工程要否定的东西。任何新增「在 T0 前再调一次 LLM」的改动都会被这里拦下。
    from app.engine import TraceStateMachine as _TSM2  # noqa: E402
    import app.engine as _eng2  # noqa: E402
    print("llm-budget:")
    _chat_calls = []

    async def _counting_chat(messages, metadata=None):
        _chat_calls.append((messages[-1].get("content", "") if messages else "",
                            (metadata or {}).get("purpose", "")))
        return {"content": '[{"q":"播放战狼2","d":"vod","tool":"vod_search_all",'
                           '"params":{"action":"play"}}]'}

    _orig_chat = _eng2.llm.chat
    _orig_sm3 = _eng2._sm
    _eng2.llm.chat = _counting_chat
    try:
        from app.engine import _MT_CONTEXT, _MT_LOCK as _MTL2  # noqa: E402
        with _MTL2:
            _MT_CONTEXT.clear()
        _eng2._sm = lambda: _TSM2()
        # (i) 无 history + 无 shortMemory + 无 MT 上下文 → 恰好 1 次
        _chat_calls.clear()
        asyncio.get_event_loop().run_until_complete(_eng2.build_response(
            SlowRequest(traceId="k1", deviceId="kd1", data={"query": "播放战狼2"})))
        results.append(case("单轮无上下文 = 恰好 1 次 LLM",
                            len(_chat_calls) == 1))
        # (ii) 同一 device 再来一轮（进程内已有 MT 上下文）→ +1 次改写，共 2 次
        _chat_calls.clear()
        asyncio.get_event_loop().run_until_complete(_eng2.build_response(
            SlowRequest(traceId="k2", deviceId="kd1", data={"query": "再来一个"})))
        results.append(case("有跨轮上下文 = 2 次（改写 + 一把出）",
                            len(_chat_calls) == 2
                            and any(p == "mt_rewrite" for _, p in _chat_calls)))
        # (iii) 多意图/闲聊兜底仍 1 次（模型回什么不影响次数）
        _chat_calls.clear()
        with _MTL2:
            _MT_CONTEXT.clear()
        asyncio.get_event_loop().run_until_complete(_eng2.build_response(
            SlowRequest(traceId="k3", deviceId="kd9",
                        data={"query": "你好", "memory": {"shortMemory": []}})))
        results.append(case("空 shortMemory 列表 ≠ 有上下文 → 1 次",
                            len(_chat_calls) == 1))
    finally:
        _eng2.llm.chat = _orig_chat
        _eng2._sm = _orig_sm3
        with _MTL2:
            _MT_CONTEXT.clear()

    # --- 槽值检索补充块：渲染 / 降级 / KV cache 前缀不变 / 接入首轮 prompt ---
    print("slots:")
    from app import slots as _slots  # noqa: E402
    from app.prompts import build_oneshot_prompt as _bop  # noqa: E402
    _resp = {"ok": True, "hits": [
        {"domain": "music", "field": "tag", "value": "国语", "alias": "国语", "similarity": 1.0},
        {"domain": "educ", "field": "children_second_genre", "value": "早教", "alias": "早教",
         "similarity": 1.0},
        {"domain": "educ", "field": "language", "value": "普通话", "alias": "国语", "similarity": 1.0},
        {"domain": "vod", "field": "language", "value": "普通话", "alias": "国语", "similarity": 0.8},
        {"domain": "vod", "field": "language", "value": "普通话", "alias": "国语", "similarity": 0.8},
        {"domain": "xx", "field": "f", "value": "v"},
        {"domain": "vod", "field": "", "value": "v"},
    ]}
    _blk = _slots.render_slot_block(_resp)
    results.append(case("slot 块：educ→children、按域固定顺序、去重、丢坏行",
                        "- vod：国语 → language=普通话（相似度0.80）\n"
                        "- children：早教 → children_second_genre=早教；国语 → language=普通话\n"
                        "- music：国语 → tag=国语\n" in _blk
                        and _blk.count("language=普通话（") == 1 and "xx" not in _blk))
    results.append(case("slot 块：确定性（同输入同输出）",
                        _blk == _slots.render_slot_block(_resp)))
    results.append(case("slot 块：无命中/坏响应 → 空串",
                        _slots.render_slot_block({"ok": True, "hits": []}) == ""
                        and _slots.render_slot_block(None) == ""
                        and _slots.render_slot_block({"hits": "x"}) == ""))
    results.append(case("slot 块：max_hits 截断",
                        _slots.render_slot_block(_resp, max_hits=1).count(" → ") == 1))
    _base_p = _bop("CAT")
    _with_p = _bop("CAT", _blk)
    results.append(case("KV cache：带 slot 块的 prompt 以无块 prompt 为前缀、块在最末",
                        _with_p.startswith(_base_p) and _with_p.endswith(_blk)
                        and _bop("CAT", "") == _base_p))
    results.append(case("检索未配置 → None（不触网）",
                        asyncio.get_event_loop().run_until_complete(
                            _slots.retrieve("早教")) is None))
    _osl = __import__("os")
    _osl.environ["HC_SLOT_RETRIEVE_URL"] = "http://127.0.0.1:9/retrieve"
    _config_mod = __import__("app.config", fromlist=["get_settings"])
    _config_mod.get_settings.cache_clear()
    try:
        results.append(case("检索失败（连接拒绝）→ 降级 None，不抛",
                            asyncio.get_event_loop().run_until_complete(
                                _slots.retrieve("早教")) is None))
    finally:
        _osl.environ["HC_SLOT_RETRIEVE_URL"] = ""
        _config_mod.get_settings.cache_clear()

    # 接入：首轮 system prompt 末尾带检索块，且检索用的是本轮 query
    _seen_sys: list = []
    _seen_q: list = []

    async def _fake_block(q):
        _seen_q.append(q)
        return _blk

    async def _sys_chat(messages, metadata=None):
        _seen_sys.append(messages[0]["content"])
        return {"content": "[]"}

    _orig_sb, _orig_chat4 = _slots.slot_block_for, _eng2.llm.chat
    _slots.slot_block_for, _eng2.llm.chat = _fake_block, _sys_chat
    try:
        asyncio.get_event_loop().run_until_complete(
            _TSM2()._plan("早教学数字123国语"))
        results.append(case("首轮 _plan：system prompt 以 slot 块结尾、检索参数为本轮 query",
                            len(_seen_sys) == 1 and _seen_sys[0].endswith(_blk)
                            and _seen_q == ["早教学数字123国语"]))
    finally:
        _slots.slot_block_for, _eng2.llm.chat = _orig_sb, _orig_chat4

    # --- 多步架构：并行一把出；串行首轮出 s1 全量 + 后续子请求/依赖，续跑每步一次 LLM ---
    print("multistep:")
    _p = parse_plan('[{"q":"搜索刘德华的电影","d":"vod","tool":"vod_search_all",'
                    '"params":{"action":"search"}},'
                    '{"q":"第二部的导演是谁","d":"vod","tool":"fan_knowledge_agent",'
                    '"params":{"messages":"x"},"dep":1}]')
    results.append(case("依赖条目丢弃首轮 tool/params、保留 q/d/dep",
                        _p.intents[1].tool == "" and _p.intents[1].params == {}
                        and _p.intents[1].domain == "vod" and _p.intents[1].dep_on == 1
                        and _p.intents[0].tool == "vod_search_all"))
    _pd, _ = parse_oneshot('[{"q":"a","d":"vod","tool":"vod_search_all","params":{}},'
                           '{"q":"b","d":"vod","tool":"fan_knowledge_agent",'
                           '"params":{"messages":"x","dep":1}}]')
    results.append(case("dep 写进 params → 提到顶层且不留在 params",
                        len(_pd) == 2 and _pd[1].dep == 1 and "dep" not in _pd[1].params))
    _pp = parse_plan('[{"q":"播放战狼2","d":"vod","tool":"vod_search_all","params":{"action":"play"}},'
                     '{"q":"音量调到30","d":"device","tool":"numeric_adjust","params":{"value":"30"}}]')
    results.append(case("并行条目各自保留 tool/params、一批全发",
                        [i.tool for i in next_batch(_pp, set())]
                        == ["vod_search_all", "numeric_adjust"]))

    _t1_in: list = []
    _replies = [
        '[{"q":"搜索刘德华的电影","d":"vod","tool":"vod_search_all",'
        '"params":{"action":"search","retext":"搜索刘德华的电影"}},'
        '{"q":"第二部的导演是谁","d":"vod","dep":1}]',
        '{"q":"《无间道》的导演是谁","tool":"fan_knowledge_agent",'
        '"params":{"messages":"《无间道》的导演是谁"}}',
    ]

    async def _seq_chat(messages, metadata=None):
        _t1_in.append(messages[-1]["content"])
        return {"content": _replies[len(_t1_in) - 1]}

    _orig_chat5, _orig_sm5 = _eng2.llm.chat, _eng2._sm
    _sm5 = _TSM2()
    _eng2.llm.chat, _eng2._sm = _seq_chat, (lambda: _sm5)
    try:
        with _MTL2:
            _MT_CONTEXT.clear()
        _r1 = asyncio.get_event_loop().run_until_complete(_eng2.build_response(
            SlowRequest(traceId="m1", deviceId="dm1",
                        data={"query": "刘德华的电影，第二部导演是谁？"})))
        _hist = [{"index": 1, "id": "s1", "toolName": "vod_search_all",
                  "result": {"data": {"memoryData": [{"data": [{"mediaTitle": "天若有情"},
                                                               {"mediaTitle": "无间道"}]}]}}}]
        _n1 = len(_t1_in)
        _r2 = asyncio.get_event_loop().run_until_complete(_eng2.build_response(
            SlowRequest(traceId="m2", deviceId="dm1",
                        data={"query": "刘德华的电影，第二部导演是谁？", "toolHistory": _hist})))
        _s2 = _r2["data"]["steps"]
        results.append(case("串行 tick1：只发 s1、stop=false、1 次 LLM",
                            [s["id"] for s in _r1["data"]["steps"]] == ["s1"]
                            and _r1["stop"] is False and _n1 == 1))
        results.append(case("串行 tick2：1 次 LLM 出 s2 的 q/tool/params，dependsOn=s1，stop",
                            len(_t1_in) == 2 and len(_s2) == 1 and _s2[0]["id"] == "s2"
                            and _s2[0]["toolName"] == "fan_knowledge_agent"
                            and _s2[0]["dependsOn"] == ["s1"]
                            and _s2[0]["hitQuery"] == "《无间道》的导演是谁"
                            and _s2[0]["parameters"]["messages"][0]["content"] == "《无间道》的导演是谁"
                            and _r2["stop"] is True))
        results.append(case("续跑输入含用户原话/所属域/前序结果（漏 d 时补计划域，fan 不被清空）",
                            "用户原话:刘德华的电影，第二部导演是谁？" in _t1_in[1]
                            and "所属域:vod" in _t1_in[1] and "第2部" in _t1_in[1]))
    finally:
        _eng2.llm.chat, _eng2._sm = _orig_chat5, _orig_sm5
        with _MTL2:
            _MT_CONTEXT.clear()

    # --- 当前时间注入：相对时间换算基准（user message，不进 system 前缀）---
    print("now:")
    from app.engine import _now_line  # noqa: E402
    import time as _tm  # noqa: E402
    _ms = int(_tm.time() * 1000)
    results.append(case("毫秒 timestamp 与秒 timestamp 渲染一致",
                        _now_line(_ms) == _now_line(_ms / 1000)))
    results.append(case("timestamp 缺失/乱值/偏离 >2 天 → 用本机时间",
                        _now_line(None) == _now_line("abc") == _now_line(0)
                        == _now_line(_ms - 10 * 86400 * 1000)
                        and _now_line().startswith("当前时间：")
                        and "星期" in _now_line()))
    _cap: list = []

    async def _cap_chat(messages, metadata=None):
        _cap.append(messages)
        return {"content": "[]"}

    _orig_chat6 = _eng2.llm.chat
    _eng2.llm.chat = _cap_chat
    try:
        asyncio.get_event_loop().run_until_complete(_TSM2()._plan("明天皇马的比赛"))
        results.append(case("首轮：当前时间在 user message、不在 system",
                            _cap[0][1]["content"].startswith("当前时间：")
                            and "当前时间：2" not in _cap[0][0]["content"]))
    finally:
        _eng2.llm.chat = _orig_chat6

    # --- 输出格式 q/d/vt/params：vt=域:工具，params 不含 retext，retext 由引擎补=用户原句 ---
    print("vt-format:")
    _v, _ve = parse_oneshot('[{"q":"搜索2020年之后上映的电影","d":"vod","vt":"vod:vod_search_all",'
                            '"params":{"action":"search","retext":"模型乱填","query":{"field":"title","value":"x"}}}]')
    results.append(case("vt=域:工具 解析出 d/tool；模型填的 retext 被丢弃",
                        len(_v) == 1 and _v[0].d == "vod" and _v[0].tool == "vod_search_all"
                        and "retext" not in _v[0].params and "query" in _v[0].params))
    _v2, _ = parse_oneshot('[{"q":"a","d":"music","tool":"music_song_search","params":{"song":["x"]}}]')
    results.append(case("旧 tool 字段仍兼容", len(_v2) == 1 and _v2[0].tool == "music_song_search"))
    _vb = [Intent(query="搜索2020年之后上映的电影", domain="vod", tool="vod_search_all",
                  params={"action": "search"}, index=1, src="2020年之后拍的电影"),
           Intent(query="音量调到30", domain="device", tool="numeric_adjust",
                  params={"object": "音量"}, index=2, src="2020年之后拍的电影"),
           Intent(query="来点轻松的歌", domain="music", tool="music_song_recommend",
                  params={}, index=3, src="2020年之后拍的电影")]
    _vs, _ = _steps_from_batch(_vb)
    results.append(case("引擎补 retext=用户原句（含 retext 的工具），无 retext 的工具不补",
                        _vs[0]["parameters"]["retext"] == "2020年之后拍的电影"
                        and "retext" not in _vs[1]["parameters"]
                        and _vs[2]["parameters"] == {"retext": "2020年之后拍的电影"}))
    _blk2 = render_oneshot_block()
    _ap, _ = parse_oneshot('[{"q":"我想听郭德纲的相声","d":"audio","vt":"audio:audio_search",'
                           '"params":{"action":"play","query":"郭德纲相声"}}]')
    _as, _ = _steps_from_batch([Intent(query="我想听郭德纲的相声", domain="audio", tool="audio_search",
                                       params=_ap[0].params, index=1, src="我想听郭德纲的相声")])
    results.append(case("audio_search.query 整句透传：模型给的丢弃，引擎补用户原句",
                        _ap[0].params == {"action": "play"}
                        and _as[0]["parameters"] == {"action": "play", "query": "我想听郭德纲的相声"}))
    _fp2, _ = parse_oneshot('[{"q":"天龙八部的导演是谁","d":"vod","vt":"vod:fan_knowledge_agent",'
                            '"params":{"messages":"x"}}]')
    results.append(case("fan messages 不进目录、模型给的丢弃",
                        _fp2[0].params == {} and "messages" not in _blk2))
    results.append(case("目录块与 prompt 锚例均不含 retext",
                        "retext" not in _blk2
                        and '"retext"' not in __import__("app.prompts", fromlist=["x"])._ANCHORS))

    ok = all(results)
    print(f"\n{'ALL PASS' if ok else 'SOME FAILED'} ({sum(results)}/{len(results)})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
