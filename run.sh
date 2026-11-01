#!/usr/bin/env bash
# hcOneshot 启动脚本：juagent poc 慢任务接口（POST /slowAgent/{poc_id}）—— 一把出，单步一次 LLM
set -euo pipefail
cd "$(dirname "$0")"

PORT="${HC_PORT:-8088}"
HOST="${HC_HOST:-0.0.0.0}"

# 先杀掉占用同一端口的旧进程（往往是上次没退干净的 uvicorn）。
# 注意：uvicorn --reload 的 worker 是 multiprocessing fork 出来的，其 cmdline
# 不含 "uvicorn"，pgrep -x uvicorn 查不到、且常扛住 SIGTERM —— 故按【端口持有者】
# 清理，SIGTERM 后升级 SIGKILL，并轮询到端口真正释放为止。
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

exec uvicorn app.main:app --host "$HOST" --port "$PORT" --reload \
     --timeout-graceful-shutdown 5 2>&1 | tee -a "$LOG_FILE"
