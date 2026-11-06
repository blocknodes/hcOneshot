#!/usr/bin/env bash
# hcOneshot 启动脚本：juagent poc 慢任务接口（POST /slowAgent/{poc_id}）—— 一把出，单步一次 LLM
set -euo pipefail
cd "$(dirname "$0")"

# HC_PORT / HC_HOST：shell 环境变量优先，其次读 .env（只取这两项，不 source 整个文件），默认 8088 / 0.0.0.0
_env_get() { { [ -f .env ] && grep -E "^[[:space:]]*$1=" .env | tail -1 | cut -d= -f2- | tr -d '[:space:]"'"'"; } || true; }
PORT="${HC_PORT:-$(_env_get HC_PORT)}"; PORT="${PORT:-8088}"
HOST="${HC_HOST:-$(_env_get HC_HOST)}"; HOST="${HOST:-0.0.0.0}"

# 先杀掉占用同一端口的旧进程（往往是上次没退干净的 uvicorn）。
# 注意：uvicorn --reload 的 worker 是 multiprocessing fork 出来的，其 cmdline
# 不含 "uvicorn"，pgrep -x uvicorn 查不到、且常扛住 SIGTERM —— 故按【端口持有者】
# 清理，SIGTERM 后升级 SIGKILL，并轮询到端口真正释放为止。
# 无 lsof 时兜底：按命令行里的「--port <PORT>」找本项目的 uvicorn（含 --reload 的父进程）并结束
if ! command -v lsof >/dev/null 2>&1; then
  pids="$(pgrep -f "uvicorn app.main:app .*--port ${PORT}( |$)" || true)"
  if [ -n "$pids" ]; then
    echo "[run.sh] 端口 $PORT 上的旧 uvicorn 进程，先结束：$pids"
    kill $pids 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      [ -z "$(pgrep -f "uvicorn app.main:app .*--port ${PORT}( |$)" || true)" ] && break
      sleep 0.5
    done
    leftover="$(pgrep -f "uvicorn app.main:app .*--port ${PORT}( |$)" || true)"
    [ -n "$leftover" ] && { echo "[run.sh] SIGKILL：$leftover"; kill -9 $leftover 2>/dev/null || true; sleep 1; }
  fi
fi
if command -v lsof >/dev/null 2>&1; then
  pids="$(lsof -t -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
  if [ -n "$pids" ]; then
    echo "[run.sh] 端口 $PORT 被以下进程占用，先结束：$pids"
    kill $pids 2>/dev/null || true
    for _ in 1 2 3 4 5 6; do
      [ -z "$(lsof -t -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)" ] && break
      sleep 0.5
    done
    leftover="$(lsof -t -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
    if [ -n "$leftover" ]; then
      echo "[run.sh] 端口 $PORT 仍被占用，SIGKILL：$leftover"
      kill -9 $leftover 2>/dev/null || true
      sleep 1
    fi
  fi
  if [ -n "$(lsof -t -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)" ]; then
    echo "[run.sh] 错误：端口 $PORT 无法释放，请手动处理。" >&2
    exit 1
  fi
fi

# 日志按次落盘：logs/hcOneshot_<启动时间戳>.log，logs/latest.log 软链指向最新。
LOG_DIR="logs"
if [ -n "${HC_LOG_FILE:-}" ]; then
  LOG_FILE="$HC_LOG_FILE"; mkdir -p "$(dirname "$LOG_FILE")"
else
  mkdir -p "$LOG_DIR"
  TS="$(date +%Y%m%d_%H%M%S)"
  LOG_FILE="${LOG_DIR}/hcOneshot_${TS}.log"
  n=1; while [ -e "$LOG_FILE" ]; do LOG_FILE="${LOG_DIR}/hcOneshot_${TS}_${n}.log"; n=$((n+1)); done
fi
touch "$LOG_FILE"
ln -sfn "$(cd "$(dirname "$LOG_FILE")" && pwd)/$(basename "$LOG_FILE")" "${LOG_DIR}/latest.log"
echo "[run.sh] 日志 -> ${LOG_FILE}"

# 打印生效配置（app.config.Settings：环境变量 > .env > 默认值；key/token/secret 类字段打码），同时写入日志
{
  echo "[run.sh] 生效配置（HOST=${HOST} PORT=${PORT}）："
  python3 - <<'PYEOF' || echo "[run.sh] 警告：配置打印失败（不影响启动）"
import re
from app.config import get_settings
s = get_settings()
_lv = {0: "完整版", 1: "精简版", 2: "极简版", 3: "超简版", 4: "工具索引版"}
print(f"  >>> 目录档位 compact={s.catalog_compact}（{_lv.get(int(s.catalog_compact), '?')}）  "
      f"few-shot={'开' if s.fewshot_enabled else '关'}  模型={s.model}  网关={s.api_base}")
for k, v in s.model_dump().items():
    if re.search(r"key|token|secret|password", k, re.I) and v:
        v = str(v)[:3] + "***"
    print(f"  HC_{k.upper():<28} = {v}")
PYEOF
} 2>&1 | tee -a "$LOG_FILE"

exec uvicorn app.main:app --host "$HOST" --port "$PORT" --reload \
     --timeout-graceful-shutdown 5 2>&1 | tee -a "$LOG_FILE"
