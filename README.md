# hcOneshot

**hcAgent 的完整替代品**：暴露同一 `POST /slowAgent/{poc_id}` POC 契约（juagent
orchestrator 慢任务接口），自包含、可直接替换。区别只有一点 —— **单步只需一次 LLM 调用**：
hcAgent 走「hcAgent T0 选 vt → hcTools pinned 填参」两跳，hcOneshot 一次调用即同时产出
`拆句 / 改写 / 落域 / 选工具 / 填参数`（`{q, d, tool, params, dep?}`），不再依赖 hcTools 运行时。

多步仍走 hcAgent 式续跑状态机（按 `deviceId` 缓存 plan，跨 tick 推进依赖链）：
- **并行**：一次 LLM 直接出多条 `{q,d,tool,params}`，一批全发。
- **串行**：首轮一次 LLM 出第 1 步的 `{q,d,tool,params}` + 后续步的 `{q,d,dep}`（只有子请求与依赖，
  不出 tool/params）；后续每个 tick 一次 LLM，拿用户原话 + 所属域 + 前序工具结果做指代消解，
  生成当步的 `q/tool/params`。只有串行才涉及多次 LLM（N 步 = N 次）。

## 为什么

| | 两跳（hcAgent + hcTools） | 一把出（hcOneshot） |
|---|---|---|
| 单步调用次数 | 2（T0 计划 + hcTools 填参） | **1** |
| 运行时依赖 | hcTools 在线（否则参数空转） | **无**（自包含） |
| 错误传播 | T0 选错 vt（线上 22.6%）→ 填参再准也废 | 无跨跳传播，路由与填参联合决策 |
| 参数量 | 每次只谈一个工具（极聚焦） | 56 工具同屏 |
| 对外契约 | `POST /slowAgent/{poc_id}` | **同名同形**，可直接替换 |

单步计划占线上 98.7%（5354/5423），故"一把出"的多步负担本就很小。

## 目录瘦身（关键）

hcTools 56 个工具的完整 schema 共 **126k 字符 ≈ 36k token**，直接塞 prompt 不可行。
`app/catalog.py` 编译成「用途首句 + 必填/可选参数记号」：

```
- vod:vod_search_all：影视媒资组合式搜索与起播工具(vod_search_all)。
    必填: query action[search|play] retext(string)
    可选: sort{rate,hot,new,play}
```

**实测 8912 字符（原 schema 126k 的 7.1%），估算 ~3.6k token。**

> 工具清单以 `load_domains()` 为准：各域 `__init__.py` 会在 `schema.json` 之外追加
> `fan_knowledge_agent`（每域一个，共 7 个），只扫 `schema.json` 会漏成 49 个。
> 参数结构优先取 `schema.json` 完整版，缺失时（如 fan）退回裁剪版并按 `required` 合成。

## 快速开始

```bash
cd hcOneshot
pip install -r requirements.txt

# 工具目录：离线编译一次（构建期读 hcSlime/hcTools 的 schema.json）
# 运行期只读产物 catalog.full.json —— 不依赖任何其他项目（不起 hcTools、不拉端点）
python3 tools/build_catalog.py

# 起服务（默认 8088）
bash run.sh

curl -s -X POST http://127.0.0.1:8088/slowAgent/poc_1 \
  -H 'Content-Type: application/json' \
  -d '{"traceId":"t1","deviceId":"dev1","data":{"query":"搜索评分最高的胡歌演的电影，导演是谁"}}'
```

返回（首轮只发 s1、`stop=false`，plan 已按 deviceId 暂存）：
```json
{"code":200,"message":"success","traceId":"t1","deviceId":"dev1",
 "data":{"planId":"plan_1f46","schemaVersion":"1.0","planType":"execute","planConfidence":0.9,
   "steps":[{"id":"s1","toolName":"vod_search_all",
             "parameters":{"action":"search","retext":"搜索评分最高的胡歌演的电影"},
             "plan":null,"dependsOn":[],"retext":"...","hitQuery":"..."}]},
 "stop":false,"forceStop":false}
```

runtime 回传 `toolHistory` 后，同一 `deviceId` 触发续跑（一次 LLM 改写依赖步并重填参）：
```json
{"code":200,...,"data":{"steps":[{"id":"s2","toolName":"fan_knowledge_agent",
   "parameters":{"messages":[{"role":"user","content":"《西游降魔篇》的导演是谁"}],
                 "riskRespStrategy":"all_replace",
                 "sceneTypeIntentList":[{"sceneType":"video"}]},
   "dependsOn":["s1"],...}]},"stop":true,"forceStop":false}
```

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/slowAgent/{poc_id}` | **唯一对外接口**，hcAgent 同款 POC 契约（可直接替换） |
| GET | `/api/catalog?block=1` | 工具目录（block=1 附渲染块与 token 统计；调试/训练对齐用） |
| GET | `/health` | 健康检查 + 目录工具数 |

## 配置（`HC_` 前缀，见 `.env.example`）

| 变量 | 默认 | 说明 |
|---|---|---|
| `HC_API_BASE` | `http://127.0.0.1:7000/v1` | LLM 网关（OpenAI 兼容） |
| `HC_MODEL` | `baseline` | 模型名（切 LoRA 时设 `lora1` 等） |
| `HC_CATALOG_FILE` | `catalog.full.json` | 工具目录产物路径（**唯一来源**，无 api/file 开关） |
| `HC_MAX_QUERY_LEN` | `2000` | query 长度上限（超限回结构化 error） |
| `HC_MAX_PLAN_STEPS` | `6` | 单计划最多条数（`next_batch` 截断） |
| `HC_PLAN_TRACE_TTL` | `300` | 串行状态（按 `deviceId` 存 plan）存活秒数 |
| `HC_SLOT_RETRIEVE_URL` | `http://aibi-model.hisense.com:8091/retrieve` | 槽值检索接口；置空即关闭 |
| `HC_SLOT_RETRIEVE_TIMEOUT` | `1.5` | 槽值检索超时秒数（超时/失败降级为无补充块） |
| `HC_SLOT_MAX_HITS` | `30` | 补充块最多收录的命中条数 |
| `HC_PORT` | `8088` | 服务端口 |

## 结构

```
hcOneshot/
  app/
    config.py     环境变量配置（含续跑 TTL / max_plan_steps）
    models.py     SlowRequest/SlowData（hcAgent 契约）+ OneIntent/OneResponse（一把出原生）
    llm.py        OpenAI 兼容异步客户端（+ _extract_message 形状校验）
    catalog.py    ★ 目录加载（只读 catalog.full.json → 精简参数块，运行期零外部依赖）
    prompts.py    首轮 prompt（T0 同源 + 目录 + 锚例）+ T1 续跑 prompt（改写+填参合一）
    parser.py     解析 [{q,d,tool,params}]，容错 + 逐字段降级
    fanparams.py  ★ fan_knowledge_agent 参数契约（逐字节移植自 hcTools）
    engine.py     ★ 编排：首轮计划 / 续跑状态机 / step 组装 / turnSlow 路由
    main.py       FastAPI —— POST /slowAgent/{poc_id}
  tools/build_catalog.py   离线编译目录（构建期读 hcTools schema.json；非运行时依赖）
  tests/test_parser.py     解析/目录/引擎单测（130 例，无网络）
```

**引擎一 tick 恰一次 LLM**（对齐 hcAgent 的「编排层零语言规则」）：
首轮由 `parse_plan` 复用 `parser.parse_oneshot` 的全部容错；续跑 `next_batch` 由 dep 序号
派发当前可执行批（无依赖全发=并行，依赖项需已执行=串行）；`_steps_from_batch` 做
turnSlow 路由 + fan 契约规范化 + step 拼装（与 hcAgent `_build_steps` 同构）。

## 零运行时依赖（自包含）

hcOneshot **运行期不依赖任何其他项目**：工具目录只读本地产物 `catalog.full.json`
（`HC_CATALOG_FILE`，默认仓根），不起 hcTools、不拉任何端点。产物由
`tools/build_catalog.py` 在**构建期**编译 —— 该脚本直读 hcSlime/hcTools 的
`domains/*/schema.json`（含完整 `parameters` 与原始 `description`；各域 `__init__.py`
会裁剪 `Tool.parameters`，故直读 schema.json），schema 变更时重跑一次即可。

`description` 在构建期**不截断**：目录块裁剪口径唯一收归 `app/catalog.py` 的 `_brief_of`，
产物与渲染逐字节可复现（8912 字符 / 56 工具）。单测有「目录加载不触网」不变量钉住此约束。

**例外：槽值检索**（`app/slots.py`，可关）。首轮计划前按本轮 query 调
`POST HC_SLOT_RETRIEVE_URL {"query":...}`，把命中渲染成「槽位检索参考」块，**追加在首轮
system prompt 最末尾**（锚例之后）。前面的 prompt+目录+锚例对所有请求逐字节相同，KV cache
前缀可复用，只有末尾这段随 query 变。块按域固定顺序、去重，确定性渲染；检索服务的
`educ`→`children`、`edu`→`education`。超时/失败降级为无块，不影响主干；不是 LLM 调用，
不改变「单轮 1 次 LLM」。只接入首轮（`_plan` / `run_oneshot`），续跑 T1 与 MT 改写未接入。
实测「早教学数字123国语」：无检索时出 `title=数字123`、`language=国语`；有检索时出
`children_third_genre=数字`、`language=普通话`。

## 已验证

- 服务起停正常，目录 56 工具加载，`GET /health` 返回 `catalog_tools: 56`。
- **slowAgent 两 tick 实跑**（假网关，验证续跑状态机）：
  - 第 1 tick（`deviceId=dev1`，无 toolHistory）→ 只发 `s1`（`vod_search_all`），`stop=false`，plan 暂存。
  - 第 2 tick（同 `deviceId`，回传 `toolHistory`[s1 候选媒资]）→ 发 `s2`
    （`fan_knowledge_agent`，`dependsOn:["s1"]`），T1 改写收到「候选媒资(按搜索排位): 第1部…第3部」，
    产出《西游降魔篇》自包含句；`messages` 已规范化为契约数组；`stop=true`。
- **turnSlow 兜底路由**（单测 + 实跑）：上一步返回 `turnSlow` 的工具禁止重发 ——
  `NO_SLOT_MAPPING_VALUE`+vod/children/education → 同域 fuzzy（`vod_fuzzy_search` /
  `educ_fuzzy_search` / `edu_fuzzy_search`，非终点，`forceStop=false`；参数恒为 `retext`）；
  未知 reason / 裸 `turnSlow` → `fan_knowledge_agent` 终点（`forceStop=true`）。
- **fan 契约规范化**：模型按目录把 `messages` 填成字符串 → 统一重建为
  `[{role,content}]` + `riskRespStrategy` + `sceneTypeIntentList`（逐字节移植 hcTools 词表，
  13/13 用例与 hcTools 输出一致）。
- 一把出原生端到端（`baseline` 网关 @127.0.0.1:7000，`run_oneshot`）：

| query | 输出 |
|---|---|
| 2020年之后拍的电影 | `vod:vod_search_all` + release_time/category 布尔树 ✅（与 iter1 金标逐字一致） |
| 播放战狼2，顺便把音量调到30 | 拆两条：`vod:vod_search_all` + `device:numeric_adjust` ✅ |
| 小猪佩奇的弟弟是谁 | `children:fan_knowledge_agent` + `{"messages": ...}` ✅ |
| 刘德华演过哪些电影 | `vod:vod_search_all` + actor/category ✅ |
| 你好 | 空数组（纯闲聊）✅ |

- 解析/目录/引擎单测 **130/130**（无网络）：代码块、夹文字、长键、`domain:tool` 前缀补域、唯一工具名反推域、
  尾逗号（字符串外才算，不吞字符串内 `,]`）、`dep` 归一/重映射、顶层参数折回、
  同构子键内层形状、网关响应形状（空 `choices`/缺 `message` → ValueError 而非 IndexError）、
  描述不以「。」结尾拼出「。。」、目录坏行（缺 `domain`）被丢弃而非渲染期 KeyError、
  `properties` 畸形不炸、纯闲聊空计划 error 为空而条目全无效仍报错、
  非法域/工具降级、目录块无逐次抖动、`catalog.full.json` 与构建脚本同步、
  空 query 两形态同语义；**引擎侧**：计划 dep 解析、批次派发（无依赖/依赖）、
  step 拼装、fan 契约规范化（三种 messages 形态等价）、turnSlow 路由三态、
  工具体结果提候选带排位；**移植项**：shortMemory 序号取候选 id、
  matchId 回填（旧 id 升级/已有值不覆盖/非阵容不动）、赛事预测串行挂起（s2 依赖 s1 且为 fan 终点）、
  双域并行标记（同域两条/有历史时不标）；**续跑钉工具**（拒绝模型擅换、钉回计划工具）；
  **多轮改写 MT**（`_short_turns` 收已完成轮/候选带排位、`_mt_merge` 按轮对齐、`_mt_add` 同 q 续轮、
  `_has_mt_history` 三态、`build_response` 门控：无上下文不改写/有 shortMemory 改写/续跑轮不改写）；
  **badcase 覆盖层**（`_norm` 归一、`_render` 简写/完整两写法、开关门控、命中直出不触状态机/未命中走主干）。
  移植三项走 `build_response` 端到端、钉工具项走状态机端到端、MT/badcase 门控走 `build_response`；
  **核心不变量「无上下文单轮 = 恰好 1 次 LLM」**（打桩 `llm.chat` 计次走真实状态机；
  注入「T0 前多一次 LLM」即 FAIL，钉死一把出的立身之本）。
  以上均经「注入旧行为即 FAIL」验证。
- 实测延迟 **0.3–1.1s**（单次调用，含 3.5k token 目录块）。

### 调试中修掉的十八处

1. **工具被静默清空**：目录按 `域:工具` 渲染，模型会把整串照抄进 `"tool"`
   （`"tool":"vod:vod_person_search"`），旧 parser 只认裸名 → 清空。
   现 parser 兼容前缀并顺带补 `d`，prompt 也加了明确说明。
2. **漏 7 个工具**：`fan_knowledge_agent` 只在各域 `pipeline.py` 的 `_TOOL_SPECS`，
   不在 `schema.json`，只扫 schema 得 49。现以 `load_domains()` 清单为准，得 56。
3. **目录描述两处裁剪不一致 → 目录块漂移**：早期有 `api`（拉 `/api/tools/full`）与 `file`
   两个来源，前者用 hcTools 的 `_brief()`（仅首句，无续句），后者用 `build_catalog.py` 的
   `_brief`（首句过短续句），同一工具两处裁出不同文本 → 目录块 7152 vs 8855。
   现**收敛为单一 file 来源**：构建期原样落盘 `description` 不截断，运行期裁剪口径唯一收归
   `app/catalog.py` 的 `_brief_of`（8912 字符 / 56 工具，逐字节可复现）。
   回归测试钉住「加载不触网」不变量，注入 api 拉取即 FAIL。
4. **解析静默篡改/依赖错配**（两处隐蔽 bug，都是"成功但错"）：
   - **`_repair` 穿透字符串**：`re.sub(r",\s*([\}\]])")` 会把 query/retext 里的 `,]`/`,}`
     当尾逗号删掉（`{"a":"x,}"}` → `{"a":"x}"}`）——用户原话被静默改字。现改**引号/括号感知**扫描。
   - **`lstrip("json")` 是字符集剥离**：`'jsonnull'.lstrip('json') == 'ull'`（`null` 被啃）；
     现精确剥 `json` 前缀。
   - **`dep` 处理**：`isinstance(True, int)` 使 `"dep": true` 被当成序号 `1` 错指首条
     （其别名 `depends_on_previous` 意为"紧邻前一条"）；`"1"`/`1.0` 被丢弃。现 `_coerce_dep`
     归一为模型原始序号，再经**丢条重映射**到输出序号，越界/自指/指向已丢条一律清零。
   以上均有回归用例（含"注入旧实现即 FAIL"验证）。
5. **空 query 两形态语义不一**：扁平 `{"query":""}` 因 field_validator `raise` 走 **HTTP 400**，
   而嵌套 `{"data":{"query":""}}` 不触发该校验器、走 **HTTP 200 + `error`**。同一逻辑请求两种
   HTTP 语义。现 validator 只做 strip 不抛错，空判定统一收敛到端点 `error:"query 不能为空"`
   （与 `OneResponse.error` 设计一致）；类型错误（如 `query:123`）仍正确回 400。
6. **顶层参数被静默丢弃**（真准确率 bug）：模型常把参数**外置到顶层**（与 `q/d/tool` 同级），
   如 `{...,"params":{"action":"search"},"sort":{"rate":{"order":"desc"}}}` —— 锚例把 `sort`
   写在 `params` 内，实测模型却提到顶层，旧 parser 只读 `params` → **静默丢参**
   （「评分最高的电影」连测 4 次 `sort` 全空）。现把「属于本工具参数名」的顶层键折回 `params`
   （内层同名以 `params` 为准；信封键与非本工具键不折回）。修复后 `sort` 稳定入参。
7. **同构子键内层形状被吃掉**：`sort` 的 `rate/hot/new/play` 子键均为 `{order: asc|desc}`，
   旧记号只到 `sort{rate,hot,new,play}` → 换子键/取值只能靠猜。现补公共形状
   `sort{rate,hot,new,play，各{order，各[asc|desc]}}`（块 8852→8912 字符，+0.7%）。
8. **网关形状异常 → HTTP 500**：网关返回 `{"choices": []}` 时 `data["choices"][0]` 抛 **IndexError**，
   不在 `chat()` 的重试捕获列表内 → 逃出 `chat()`、逃出 engine 的 `except LLMError` → 端点 **HTTP 500**
   （已用假网关复现）。现抽出纯函数 `_extract_message(data)`：`choices` 缺失/非列表/为空、
   `choices[0]` 非 dict、缺 `message` 一律抛 **ValueError**（视为可重试），并把 `ValueError` 纳入重试捕获。
   修复后同响应**重试 3 次后回 HTTP 200 + 结构化 `error`**，不再 500。已在无网络单测钉住形状校验。
9. **描述以「。」结尾拼出「。。」**：`_brief_of` 按 `split("。")` 逐段拼回时，末段恒为空串却被照旧
   追加「。」→ `power_control`/`network_control`/`common_control` 三条（描述短、首段即达阈值未提前 break）
   渲染成「…重启控制。。」。现跳过空段，块 8915→8912 字符（-3 个句号，纯美观/解析一致性）。
10. **目录条目缺 `domain` → 渲染 KeyError → HTTP 500**：加载口只校验 `id`（早期还校验 `tool`），
   缺 `domain` 的行进入 `render_oneshot_block`（按域分组 `t["domain"]`）或 `domain_of_tool`
   即 **KeyError**，穿出 `run_oneshot` → 500（schema 变动即触发）。现加载口要求
   `id`+`tool`+`domain` 齐全，与裁剪口径一致地丢弃坏行。
11. **schema 畸形（`properties` 为 list/str）→ 渲染 AttributeError → HTTP 500**：`_brief_param`
   对子键表直接 `props.values()`/`list(props)`，非 dict 时抛 `AttributeError`。因 `engine.run_oneshot`
   在 **try 之外**渲染目录块，任何渲染异常都冒泡成 500（当前 56 工具的真实 schema 未触发，
   属防漂移的健壮性加固）。现 `_brief_param`/`_param_line` 均要求 `properties` 为 dict，
   否则降级为「无子键」/「无参数」，目录块不生变。
12. **纯闲聊（空计划）被误报为错误**：模型对「你好」等按 prompt 规则 3 回 `[]`（合法），
   但 parser 一律返回 `error="解析成功但无有效条目"` —— 与 `OneResponse.error` 自身契约
   （「空=正常」）矛盾，下游会把**正确的**闲聊当失败。现区分：模型**本就给空计划**（`[]` /
   `{"plan":[]}`）→ `error=""`；**给了条目但全被降级丢弃** → 保留 error（真异常）。
13. **fan 的 `messages` 形状不合契约**（转 slowAgent 后暴露）：目录块为省 token 把 fan 的
   `messages` 压成 `string`，模型照目录填就是裸字符串；而真实契约是
   `messages:[{role,content}]` + `riskRespStrategy` + `sceneTypeIntentList`（见 hcTools
   `fanparams.py`）。现 `_steps_from_batch` 对 fan 步骤统一经 `rebuild_fan_params` 规范化
   （三种 messages 形态等价重建，取不到内容回退用户原话）。词表**逐字节移植**自 hcTools，
   13/13 用例输出一致（避免手抄词表静默漂移）。
14. **域内 qa 被塌成通用 qa**（转 slowAgent 后暴露）：无工具兜底（选工具失败/目录不可用）时
   曾无条件清空 `it.domain` → 泛知识问答一律落到 `sceneType=knowledge_qa`。但 hcAgent 区分两级：
   **落域失败**（d 空）→ 通用 qa；**域有效但无工具** → 域内 qa（带该域 sceneType）。
   现保留域信号：`rebuild_fan_params` 的 `scene_hint` 取 `QA_SCENE_BY_DOMAIN.get(it.domain)`，
   `d="music"` → `agent_music`，`d=""` → `knowledge_qa`。
15. **续跑擅自换工具 → 串行链收不了口**（e2e 实测暴露）：首轮 `sports_match_forecast` 挂起
   「整句→`fan_knowledge_agent`」的 s2 后，下一 tick 的改写 LLM 把 s2 的工具**自作主张**换成
   `sports_match_forecast` → 重发那个本该收口的工具，链路无限重发、永不落到 fan 终点。
   根因：`_continue.resolve` 直接把 `_rewrite_step` 返回的 tool 当结果用。**hcAgent 的 T1
   只改写 query**（工具由 T0 计划固定、`_rewrite_step` 返回 `str`），计划工具才是权威。
   hcOneshot 因无 hcTools 需在续跑补参，故多出 params，但**工具仍须钉住**：模型换工具时
   钉回计划工具并丢弃其（不匹配的）params。修复后 s2 正确发 `fan_knowledge_agent`
   （含完整契约、`dep=[s1]`、`stop=true`）。回归用例经状态机端到端钉住，注入旧行为即 FAIL。
16. **两处声明式旋钮形同虚设**（配置声明了却没人读）：
   - **MT 旋钮**：`HC_MT_TTL`/`HC_MT_MAX_TURNS` 加进了 `config.py`，但 `engine._mt_add/_mt_get`
     仍读模块常量 `_MT_TTL`/`_MT_MAX_TURNS` → 环境变量静默失效（改配置不生效）。现改为
     调用期 `get_settings().mt_ttl/mt_max_turns`（与 `plan_trace_ttl`/`candidate_context_limit` 同款）。
   - **`max_query_len`**（`HC_MAX_QUERY_LEN`）：`config.py` 里声明、全仓无人读。一把出把整句塞进
     LLM prompt，超长会撑爆上下文/拖垮延迟。现于 `build_response` 入口收敛为结构化 error
     （`code:400 "query too long"`；端点按 `code` 镜像 HTTP 状态 → **HTTP 400**，与
     hcAgent 的 `_error` 同款，非 500）。两处均有回归用例，注入旧行为即 FAIL。
17. **turnSlow 表缺 `children` 域 → 少儿检索兜穿到 fan 终点**（benchmark 实测暴露）：
    `_TURNSLOW_FUZZY` 只登记了 `vod`/`education`，但域命名有坑 —— **children 域的工具前缀是
    `educ_*`、education 域是 `edu_*`，两者是不同域**。`educ_search_all` 属 children，按域查表
    无 `children` 键 → 兜底穿到 `fan_knowledge_agent`（链路终点 + `forceStop=true`），
    而正确行为是换同域模糊检索 `educ_fuzzy_search`（与 vod→`vod_fuzzy_search` 同理）。
    现补 `children` 键。同一 bug 在 `hcSlime/hcAgent/app/engine.py` 同样存在，已一并修。
18. **fuzzy 兜底参数发错键**：三个 fuzzy 工具（`vod_fuzzy_search`/`educ_fuzzy_search`/
    `edu_fuzzy_search`）schema 均为 `{retext}` 且 `additionalProperties:false`，但代码对
    **vod 特判**发 `{"query": fb_q}` → `query` 不在 schema 内、被丢弃，兜底参数**空转**。
    hcTools 自己的 vod 兜底（`domains/vod/pipeline.py:269`）发的正是 `{"retext": query}`。
    现一律发 `{"retext": fb_q}`。同 bug 亦存在于 `hcSlime/hcAgent`，已一并修。
    两条均有回归用例（children 路由 + vod 参数键），注入旧行为即 FAIL。

## 与 hcAgent 的差异

hcOneshot 已与 hcAgent 的 slowAgent 契约**逐项对齐**（含串行状态机、turnSlow 兜底、
fan 契约规范化、域信号保留）。以下 hcAgent 专有逻辑**已移植**：

- **胜负预测挂起串行**（`_arm_forecast_serial_plan`）：首步命中 `sports_match_forecast`
  时，首轮只发 s1 并 `stop=false`，把「整句→fan_knowledge_agent」作为依赖 s1 的 s2 挂起，
  下一 tick 经状态机改写执行。
- **双域并行标记**（`data.parallel`）：首轮无历史 + 本批恰为 `children`+`vod` 两条且 `stop`
  时打 `parallel=true`，供客户端并发渲染两个 tab。
- **阵容 matchId 绑定**（`_bind_sports_short_memory_match_id`）：`sports_match_lineup_search`
  的 step 按 retext 里的「第N场/第N场」序号，从最近一轮 shortMemory 候选取 id 回填 `matchId`
  （旧 `id` 参数升级为 `matchId`，已有 matchId 不覆盖）。
- **多轮上下文改写**（`_mt_rewrite` 家族，按 deviceId 合并对话历史）：门控——`history` 非空（续跑）
  或**无任何跨轮上下文**（进程内 `_MT_CONTEXT` 与 `shortMemory` 皆空）→ 原句直通；否则把
  「真实执行 query 链 + shortMemory 应答/候选」按轮对齐成 q1-a1/q2-a2，交 LLM 改写成自包含 query。
  仅改写 query、不改工具（域判定/选工具仍归 T0）。`MT_REWRITE_PROMPT` **逐字节移植**自 hcAgent
  （1.4k 字符，勿手改措辞，改则整体同步）。含「歌曲名锚点」（“播放它”→保留“这首歌”字样使下游落 music 域）、
  「人物属性优先还原人名」、「串行依赖句保留指代」、「无关则原句」等线上已验规则。
- **badcase 精确覆盖层**（`app/badcase.py`，debug 专用、**默认关**）：`HC_BADCASE_ENABLED=1` 生效，
  query 归一化后精确匹配 `badcases_oneshot.json`（路径可由 `HC_BADCASE_PATH` 覆盖），命中即直出
  配置的完整响应体、**跳过 LLM 主干**（不触状态机/改写）。与 hcAgent 同构（简写/完整 body 两种写法、
  补全信封、回填 traceId/deviceId）。

至此 hcAgent 的 slowAgent 行为已**全量移植**，无剩余差距。

## 注意

- **`retext` 类噪音字段**：一把出里模型自填 `retext`，与金标措辞常有差异；
  微调时若沿用 iter1 口径（`retext` 不参与比对），需在数据侧对齐。
- **参数准确率是本工程要验证的核心假设**：pinned 单工具 prompt 只有 695 字符、极聚焦，
  一把出要在 13k prompt 里同时路由+填参，边界槽位（时间格式/布尔树/单位）准确率可能下降。
  须用 iter1 的 4934 条金标做 A/B，比 vt 准确率与 params 逐字段准确率。

## 后续（本阶段未做）

- 微调数据生成（从 iter1 金标/日志造一把出 SFT/DPO + 审计 CSV）
- 训练脚本（ms-swift Megatron，可参考 `hcSlime/../hcTrain/script/train_{sft,dpo}_megatron.sh`）
- 评测对比（一把出 vs 两跳，须用 iter1 的 4934 条金标 A/B）
