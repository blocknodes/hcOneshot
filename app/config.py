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
