"""运行时配置：LLM 网关 / 服务参数，环境变量优先（HC_ 前缀）。

载入路径固定到本文件所在包的上级目录，避免受 cwd 影响。

**自包含**：工具目录只从本地编译产物 `catalog.full.json` 读，**运行时不依赖任何其他
项目**（不起 hcTools、不拉 hcTools 端点）。产物由 `tools/build_catalog.py` 离线编译。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_BASE = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HC_",
        env_file=str(_BASE / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- LLM 网关（OpenAI 兼容）----
    api_base: str = "http://127.0.0.1:7000/v1"
    api_key: str = ""
    model: str = "baseline"                     # baseline / lora1 …（对应 vllm --served-model-name）

    # ---- 工具目录：只读本地编译产物（运行时不依赖 hcTools）----
    # 由 tools/build_catalog.py 离线编译自 hcSlime/hcTools 的 schema.json；
    # 运行期只读本文件，无网络、无跨项目依赖。
    catalog_file: str = str(_BASE / "catalog.full.json")

    # 目录块档位：0=完整版（~38k token）；1=精简版，逐字重复说明改「同 <工具id>」（~30.7k）；
    # 2=极简版，每工具只留用途/边界句 + 签名 + 字段名及短标签/枚举/格式（按整句取舍，不截断）；
    # 3=超简版（~6.9k）；4=工具索引版，id|签名|用途短句 + 字段名（候选值交给动态层）。
    catalog_compact: int = 0

    # ---- 动态 few-shot context（实验移植，见 app/fewshot.py）：默认关 ----
    # 开启后首轮 system 末尾追加「规则提示 + BM25 相似示例」，并修正静态 prompt 两处冲突。
    fewshot_enabled: bool = False
    fewshot_lib: str = "experiments/retrieval/lib/para_all.csv"   # 相对路径按项目根解析
    fewshot_topk: int = 4
    fewshot_min_score: float = 4.0
    # 检索时排除与本轮 query 完全相同的库条目（评测时防止直接命中本题金标）
    fewshot_exclude_self: bool = True
    # 多意图 few-shot（需 fewshot_enabled）：query 按连接词切出 ≥2 个子句时，改为注入
    #   多意图拆单提示 + 多意图库整句 top-k 示例 + 每个子句从单意图库检 top-k 示例。置空关闭。
    fewshot_multi_lib: str = ""        # 如 experiments/retrieval/lib/multi_bench10k.csv（列 query,exec_mode,plan）
    fewshot_multi_topk: int = 5
    fewshot_clause_topk: int = 3
    # 多意图示例的触发方式：rule = 只在连接词/标点切出 ≥2 子句时注入（原逻辑）；
    #   score = 规则没切出时，再看「多意图库最高分 / 单意图库最高分」≥ fewshot_multi_ratio 也注入（拆不拆仍由模型决定）
    # 规则库（CSV，见 experiments/retrieval/lib/rules.csv）：按正则/实体类型检索口径提示注入首轮 system；置空用内置少儿规则
    fewshot_rules_lib: str = ""
    fewshot_rules_max: int = 3
    fewshot_multi_trigger: str = "rule"
    fewshot_multi_ratio: float = 0.32
    # 实体链接（需 fewshot_enabled）：按词典最长匹配 query 里的名字（歌曲/歌手/影视/少儿/人名）。
    #   · 检索：原句 + 名字换成类型占位符的句子各检一次，合并去重（库句同样离线换占位符）；
    #   · fewshot_entity_hint：首轮 system 末尾追加「实体参考」块（名字 → 类型）。
    # 词典 CSV 列 name,types（| 分隔，首个为主类型）；置空关闭。纯字符串匹配，不调模型。
    fewshot_entity_dict: str = ""     # 如 experiments/retrieval/lib/entity_dict.csv
    fewshot_entity_hint: bool = True

    # ---- 实体库（app/entity.py）：默认关 ----
    # 开启后按 query 词典最长匹配名字（片名/歌名/歌手/球队…），在首轮 system 最末尾追加「实体库参考」块，
    # 说明名字在库里属于哪个域的哪类槽位。纯字符串匹配，不增加 LLM 调用。库由 tools/build_entity_lib.py 生成。
    entity_enabled: bool = False
    entity_lib: str = "experiments/retrieval/lib/entity_lib.csv"   # 相对路径按项目根解析
    entity_max_hits: int = 5

    # ---- 设备域虚拟 schema（app/device_schema.py）：默认关 ----
    # 开启后 LLM 只看到虚拟参数（action/target/value/time：关机、暂停、全屏…），目录块、锚例、
    # few-shot 示例都用虚拟写法；引擎在组装 step 前（后处理）查表还原成真实 operation/object。
    # 关闭时行为与此前完全一致。改动此项需重启（few-shot 库在启动期预建）。
    device_virtual: bool = False

    # 空目录重试 TTL（秒）：文件暂缺（如编译中）时不至于永久空目录。
    catalog_empty_ttl: float = 60.0

    # ---- 槽值检索（slot retrieval）：结果追加到首轮 system prompt 末尾 ----
    # 置空即关闭；失败/超时降级为无补充块，不影响主干。
    slot_retrieve_url: str = "http://aibi-model.hisense.com:8091/retrieve"
    slot_retrieve_timeout: float = 1.5
    # 补充块最多收录的命中条数（防 prompt 膨胀）
    slot_max_hits: int = 30

    # ---- 服务 ----
    max_query_len: int = 2000

    # ---- slowAgent 续跑（对齐 hcAgent config 常量）----
    # 单计划最多条数（next_batch 截断）
    max_plan_steps: int = 6
    # 串行状态（按 deviceId 存 plan）存活秒数
    plan_trace_ttl: float = 300.0
    # 续跑改写时携带的上一步候选媒资上限（避免超长）
    candidate_context_limit: int = 30

    # ---- 多轮改写（对齐 hcAgent 常量）----
    # 进程内对话上下文（按 deviceId 存 q/a 轮）存活秒数
    mt_ttl: float = 120.0
    # 每设备最多保留的对话轮数（防超长 prompt）
    mt_max_turns: int = 20

    @property
    def chat_url(self) -> str:
        return f"{self.api_base.rstrip('/')}/chat/completions"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
