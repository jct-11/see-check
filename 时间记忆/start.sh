#!/bin/bash
set -e

cd "$(dirname "$0")"

mkdir -p data/capture data/frames data/memory

SETUP_DIR="$(pwd)/.setup"
SETUP_STATE_FILE="${SETUP_DIR}/install-state.env"

read_state_value() {
  local key="$1"
  if [ ! -f "$SETUP_STATE_FILE" ]; then
    return
  fi
  awk -F= -v k="$key" '$1==k {sub($1"=",""); print; exit}' "$SETUP_STATE_FILE"
}

ensure_ffmpeg() {
  if command -v ffmpeg >/dev/null 2>&1; then
    return
  fi

  if ! command -v brew >/dev/null 2>&1; then
    echo "❌ 缺少 ffmpeg，且未检测到 Homebrew，无法自动安装。"
    echo "请先安装 Homebrew 后执行: brew install ffmpeg"
    exit 1
  fi

  echo "📦 未检测到 ffmpeg，正在自动安装..."
  brew install ffmpeg
}

ensure_node_runtime() {
  if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
    if ! command -v brew >/dev/null 2>&1; then
      echo "❌ 缺少 Node.js/npm，且未检测到 Homebrew，无法自动安装。"
      echo "请先安装 Homebrew 后执行: brew install node"
      exit 1
    fi
    echo "📦 未检测到 Node.js/npm，正在自动安装 Node.js..."
    brew install node
  fi

  # 版本检查：better-sqlite3 / canvas / @lancedb/lancedb 均需要 Node.js v18+
  # 强烈建议 v22（当前 LTS），老版本（v14 以下）会有 spread 语法报错
  local node_major
  node_major="$(node -e 'process.stdout.write(process.versions.node.split(".")[0])')"
  if [ "$node_major" -lt 18 ] 2>/dev/null; then
    echo "⚠️  检测到 Node.js v$(node -v)（主版本号 < 18）"
    echo "   better-sqlite3 和 @lancedb/lancedb 需要 Node.js v18+，建议使用 v22 LTS"
    echo "   使用 nvm: nvm install 22 && nvm use 22"
    echo "   或 Homebrew: brew install node@22 && brew link --force node@22"
    exit 1
  fi
}

ensure_node_dependencies() {
  if ! command -v npm >/dev/null 2>&1; then
    echo "❌ npm 仍不可用，请检查 Node.js 安装是否成功。"
    exit 1
  fi
  local npm_major
  npm_major="$(npm -v | awk -F. '{print $1}')"
  if [ "$npm_major" -lt 7 ] 2>/dev/null; then
    echo "⚠️ 当前 npm 版本较旧（$(npm -v)），可能出现 lockfileVersion 警告，建议升级到 npm 9+。"
  fi

  local current_lock_hash=""
  local last_lock_hash=""
  if [ -f package-lock.json ]; then
    current_lock_hash="$(shasum -a 256 package-lock.json | awk '{print $1}')"
  fi
  last_lock_hash="$(read_state_value lock_hash)"

  if [ ! -d node_modules ] || [ "$current_lock_hash" != "$last_lock_hash" ]; then
    echo "📦 正在安装/更新 Node.js 依赖..."
    if ! npm install --no-audit; then
      echo "⚠️ npm install 失败，尝试降级为 --ignore-scripts 模式（将跳过 tfjs-node 二进制下载）..."
      if ! npm install --ignore-scripts --no-audit; then
        echo "❌ Node.js 依赖安装失败，请检查网络或 npm 环境后重试。"
        exit 1
      fi
    fi
  fi
}

save_install_state() {
  mkdir -p "$SETUP_DIR"

  local lock_hash=""
  local node_version="missing"
  local npm_version="missing"
  local ffmpeg_path="missing"
  local ffmpeg_version="missing"
  local js_dependencies=""

  if [ -f package-lock.json ]; then
    lock_hash="$(shasum -a 256 package-lock.json | awk '{print $1}')"
  fi
  if command -v node >/dev/null 2>&1; then
    node_version="$(node -v)"
  fi
  if command -v npm >/dev/null 2>&1; then
    npm_version="$(npm -v)"
  fi
  if command -v ffmpeg >/dev/null 2>&1; then
    ffmpeg_path="$(command -v ffmpeg)"
    ffmpeg_version="$(ffmpeg -version 2>&1 | head -n 1 | tr ' ' '_')"
  fi
  if command -v node >/dev/null 2>&1 && [ -f package.json ]; then
    js_dependencies="$(node -e 'const p=require("./package.json");const d={...(p.dependencies||{}),...(p.devDependencies||{})};process.stdout.write(Object.entries(d).map(([k,v])=>`${k}@${v}`).join(","));')"
  fi

  {
    echo "updated_at=$(date '+%Y-%m-%dT%H:%M:%S%z')"
    echo "node_version=${node_version}"
    echo "npm_version=${npm_version}"
    echo "lock_hash=${lock_hash}"
    echo "ffmpeg_path=${ffmpeg_path}"
    echo "ffmpeg_version=${ffmpeg_version}"
    echo "js_dependencies=${js_dependencies}"
  } > "$SETUP_STATE_FILE"
}

terminate_pids() {
  local pids="$1"
  if [ -z "$pids" ]; then
    return
  fi

  kill $pids 2>/dev/null || true
  sleep 1

  for pid in $pids; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
  done
}

cleanup_existing_processes() {
  local port_pids
  local node_pids
  local ffmpeg_pids
  local all_pids

  port_pids="$(lsof -ti :8080 -ti :8081 2>/dev/null | sort -u)"
  node_pids="$(pgrep -f 'node .*src/dashboard.js' 2>/dev/null | sort -u)"
  # ffmpeg snapshot 子进程（每次 -vframes 1 捕获）在正常情况下会自行退出，此处清理残留
  ffmpeg_pids="$(pgrep -f 'ffmpeg .*avfoundation' 2>/dev/null | sort -u)"
  all_pids="$(printf "%s\n%s\n%s\n" "$port_pids" "$node_pids" "$ffmpeg_pids" | awk 'NF' | sort -u)"

  if [ -n "$all_pids" ]; then
    echo "🧹 清理旧进程: $all_pids"
    terminate_pids "$all_pids"
  fi
}

cleanup_existing_processes

ensure_node_runtime
ensure_node_dependencies
ensure_ffmpeg
save_install_state

echo ""
echo "🚀 启动 Node.js 服务（ffmpeg 取帧由服务端 CaptureController 接管）..."
node src/dashboard.js --port 8080 &
NODE_PID=$!
echo "   Node.js PID: $NODE_PID"
echo ""

cleanup() {
  set +e
  trap - INT TERM
  echo ""
  echo "⏹️  停止所有服务..."
  if [ -n "${NODE_PID:-}" ]; then
    kill "$NODE_PID" 2>/dev/null || true
    wait "$NODE_PID" 2>/dev/null || true
  fi
  pkill -f 'ffmpeg .*avfoundation' 2>/dev/null || true
  echo "✅ 已退出"
  exit 0
}
trap cleanup INT TERM

wait
