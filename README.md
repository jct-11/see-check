# 识境 ShiJing — 端侧视觉时空记忆系统

> 让摄像头拥有「记忆」：持续快照 → 场景变化检测 → 多模态理解 → 向量索引 → 语义查询

---

## 项目简介

识境（ShiJing / stmem）是运行在边缘设备（Mac 等）上的**视觉短期记忆系统**。  
它以 **ffmpeg 快照模式**定期从摄像头获取当前画面，经像素差筛选（pixelDiff，默认阈值 50）判断是否有意义的场景变化，通过 Gemma4 E2B 多模态大模型以**批量推理**方式生成文本描述，结合 nomic-embed-text 向量嵌入与可选人脸识别，构建可语义检索的时空记忆库。所有记忆持久化至 SQLite + LanceDB，支持语义查询与记忆生命周期自动管理。

---

## 系统架构

```
摄像头 (avfoundation)
    │
    │  每 ~2-3 秒一次独立 ffmpeg 快照进程 (-vframes 1)
    ▼
┌─────────────────────────────────────────────────────┐
│  CaptureController                                  │
│    ├─ QuickFilter (pixelDiff)                        │
│    │    • 128×72 缩略图 MAD 对比                      │
│    │    • 全量字节比对（byteSame 快速路径）            │
│    │    • diff ≥ threshold → 立即落地                 │
│    │    • forceIntervalMs 到期 → 无论如何落地一帧      │
│    └─ 落地: data/frames/frame_NNN.jpg               │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  FramePipeline                                      │
│    ├─ KeyframeSelector (passthrough / windowRep)    │
│    ├─ Batcher                                       │
│    │    • 帧落地后入队                               │
│    │    • 1 帧 → 单次推理                            │
│    │    • 2-6 帧 → 批量推理（一次 API 调用）          │
│    │    • >6 帧 → 按 6 帧一组串行批量                 │
│    ├─ Inference: Gemma4 E2B (Ollama / OpenAI 兼容)  │
│    ├─ Embedding: nomic-embed-text (768 维)          │
│    ├─ Face: @vladmandic/face-api (可选)             │
│    ├─ MemoryStore: SQLite (better-sqlite3)          │
│    └─ VectorIndex: LanceDB (持久化)                 │
└─────────────────────────────────────────────────────┘
    │
    ▼
Web 仪表板 / POST /api/query 语义查询
```

---

## 项目结构

```
stmem/
├── src/
│   ├── dashboard.js                # 瘦入口：组装容器 → 启动 HTTP(S)
│   ├── config/
│   │   ├── index.js                # 配置加载/保存/规范化
│   │   └── defaults.js             # 默认值
│   ├── capture/
│   │   ├── controller.js           # CaptureController
│   │   ├── sources/
│   │   │   ├── ffmpegSnapshot.js   # 主采帧源：每次独立 ffmpeg -vframes 1
│   │   │   └── ffmpegAvfoundation.js  # 流式源（macOS 存在首帧后冻结问题，已不默认使用）
│   │   └── quickFilter/
│   │       ├── index.js            # 工厂
│   │       ├── passthrough.js      # 全放行
│   │       └── pixelDiff.js        # 像素差筛选（当前默认）
│   ├── pipeline/
│   │   ├── framePipeline.js        # 落地帧 → VLM → 嵌入 → 存储
│   │   ├── keyframe/
│   │   │   ├── passthrough.js      # 默认：每帧都送
│   │   │   └── windowRepresentative.js  # 时间窗口代表帧选择
│   │   └── batcher.js
│   ├── inference/ embeddings/ faces/
│   ├── memory/
│   │   └── adapters/sqlite.js      # SQLite 持久化
│   ├── vector/
│   │   └── adapters/lancedb.js     # LanceDB 向量索引
│   ├── query/ lifecycle/ server/ utils/
│   └── web/
│       ├── index.html              # 前端 SPA
│       └── assets/                 # style.css / temporal.js / spatial.js …
├── data/                           # 运行期生成（首次启动自动创建）
│   ├── capture/latest.jpg          # 摄像头最新预览
│   ├── frames/frame_NNN.jpg        # 已落地帧
│   ├── memory.sqlite               # 记忆数据库
│   └── vectors/                    # LanceDB 向量目录
├── config.json                     # 用户配置
├── start.sh                        # 一键启动脚本
└── package.json
```

---

## 快速开始

### 1. 系统依赖

| 工具 | 最低版本 | 安装方式 |
|------|----------|----------|
| Node.js | **v22 LTS**（严格要求，见[已知坑](#已知坑)） | `brew install node` 或 `nvm install 22` |
| ffmpeg | 任意近期版本 | `brew install ffmpeg` |
| Xcode CLI Tools | — | `xcode-select --install`（better-sqlite3 编译需要）|

### 2. 安装 Node 依赖

```bash
# 推荐先确认 Node 版本
node -v   # 应输出 v22.x.x

npm install
```

> **如果 npm install 失败**（常见于网络限制下下载 @tensorflow/tfjs-node）：
> ```bash
> npm install --ignore-scripts   # 跳过原生二进制下载；人脸识别功能不可用，其余正常
> ```

### 3. 推理服务

系统需要一个运行 **Ollama** 的推理机（本机或局域网）：

```bash
# 拉取视觉模型（本机 Ollama）
ollama pull gemma4:e2b
ollama pull nomic-embed-text
```

在 `config.json` 中配置推理机地址与模型：

```jsonc
{
  "inferenceServers": [
    { "id": "local", "host": "localhost", "port": 11434, "apiStyle": "ollama" }
  ],
  "selectedInferenceServerId": "local",
  "model": "gemma4:e2b",
  "embedBase": "http://localhost:11434"
}
```

### 4. 启动

```bash
# 一键启动（推荐）
./start.sh

# 或手动
node src/dashboard.js --port 8080
```

访问 `http://localhost:8080`，或局域网内其他设备通过 HTTPS `https://<IP>:8081`（需 `certs/` 目录下有自签名证书）。

### 5. 摄像头权限

首次使用时，macOS 会弹出摄像头授权对话框。若未弹出或拒绝后需重新授权：

> **系统设置 → 隐私与安全性 → 摄像头** → 找到运行本服务的终端程序（Terminal / iTerm2 / Cursor 内置终端等）→ 开启

---

## 配置说明（`config.json`）

```jsonc
{
  "capture": {
    "source": "ffmpeg",
    "ffmpeg": {
      "mode": "snapshot",         // "snapshot"（默认）| "stream"（macOS 有 bug，不推荐）
      "deviceIndex": 0,           // 摄像头设备索引
      "width": 1280, "height": 720,
      "quality": 5,               // ffmpeg -q:v，1=最高质量，31=最低
      "snapshotIntervalMs": 1000, // 每次快照之间最小等待时间（ms）；实际周期 ≈ 2-3s
      "snapshotTimeoutMs": 8000   // 单次快照超时（ms）
    }
  },
    "pipeline": {
    "quickFilter": "pixelDiff",   // "passthrough" | "pixelDiff"
    "pixelDiff": {
      "diffThreshold": 50,        // MAD 阈值（0-255）；越大越不灵敏；默认 50
      "forceIntervalMs": 10000    // 最长 10s 强制落一帧（静止场景保证时间线完整）
    },
    "keyframe": "passthrough",    // "passthrough" | "windowRepresentative"
    "batch": { "enabled": false } // true = 按 batchSize 聚合后批推理；false = 动态 1/2-6/>6 分支
  },
  "memory": { "store": "sqlite" },
  "vector": { "store": "lancedb", "dir": "data/vectors" }
}
```

---

## 语义查询流程

用户在前端输入自然语言问题，或直接 `POST /api/query`，触发以下完整流水线：

```
用户问题（自然语言）
    │
    ▼  Step 1 · 时间范围解析
    │  parseTimeRange(question)
    │  支持：最近N分钟 / N小时 / 半小时 / 今天 / 昨天 / 上午 / 下午 / 晚上
    │  → start / end 时间戳（解析失败则 timeRange = null）
    │
    ▼  Step 2 · 候选帧召回
    │  vectorIndex.filterByTime({})  → 取全量索引（含向量 + captureTime）
    │  若 timeRange 有效：过滤 captureTime ∈ [start, end]
    │  若过滤后为空：回退到全量，并在 Prompt 中注明该时间段无记忆
    │
    ▼  Step 3 · 向量相似度排序（Top-20）
    │  embeddingService.embed(question) → queryEmb (768 维)
    │  候选数 ≤ 20            → 全部保留
    │  候选数 > 20 且有 queryEmb → cosine(queryEmb, 帧向量) 降序，取前 20
    │  候选数 > 20 且无 queryEmb → 按 captureTime 降序，取前 20
    │
    ▼  Step 4 · LLM 上下文构建
    │  topMemories 按时间升序排列，每条格式：
    │    [时间] 场景描述文本 [识别到人物: personId(姓名, 性别, 年龄)] (如有人脸)
    │  附加提示词：
    │    • 若时间过滤为空：告知 LLM 该段无记忆，以其他时段作参考
    │    • 若有人脸：提示同一 personId = 同一人，勿重复计数
    │
    ▼  Step 5 · LLM 推理
    │  inferenceService.textInfer(fullPrompt)
    │  → answer（自然语言回答）+ inferenceTime
    │
    ▼  Step 6 · 相关帧提取（三级回退）
    │  ① 从 LLM 回答中正则提取引用时间戳（格式：YYYY-MM-DD HH:mm:ss）
    │    若时间戳与某帧 captureTime 误差 < 5 秒 → 命中，score = 100%
    │  ② 若 ① 无命中：cosine(queryEmb, 帧向量) 重新打分，
    │    保留 score ≥ 最高分 × 75% 的帧，最多 8 帧
    │  ③ 若仍为空：取 topMemories 中最新 5 帧
    │
    ▼  Step 7 · 前端展示过滤
       matchedFrames 仅渲染 score ≥ 80% 的帧
       若无帧达标：提示"最高 N%，请尝试更具体描述"
       标注格式：📅 时间 · N%（绿色）
```

### 时间表达式支持列表

| 表达式 | 解析结果 |
|--------|---------|
| `最近N分钟` / `N分钟内` / `N分钟前` | now − N 分钟 → now |
| `最近N小时` / `N小时内` / `N小时前` | now − N 小时 → now |
| `最近半小时` / `半小时内` | now − 30 分钟 → now |
| `今天` | 今天 00:00 → now |
| `昨天` | 昨天 00:00 → 今天 00:00 |
| `上午` / `早上` | 当天 00:00 → 12:00 |
| `下午` | 当天 12:00 → 18:00 |
| `晚上` | 当天 18:00 → 23:59 |
| 无时间词 | 不过滤，全量检索 |

### 相关帧打分说明

| 来源 | score 含义 |
|------|-----------|
| LLM 回答中引用的时间戳（误差 < 5s） | 100%（精确引用） |
| cosine 相似度 | `Math.round(cosine × 100)`，0–100 |
| 无嵌入时间回退 | null（不在前端展示） |

> 前端仅展示 **score ≥ 80%** 的帧；若无帧达标，显示提示信息（含最高分）；LLM 回答本身始终完整返回，不受此过滤影响。

---

## API 路由

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | Web 仪表板 |
| GET | `/api/status` | 运行状态（isCapturing、faceApiReady 等） |
| GET/POST | `/api/config` | 读取/更新配置 |
| POST | `/api/start` | 开始采帧记忆 |
| POST | `/api/stop` | 停止采帧 |
| GET | `/api/frames` | 帧列表与记忆摘要 |
| GET | `/api/latest-frame` | 摄像头最新预览图 |
| GET | `/frames/:filename` | 读取落地帧图片（含 no-cache 头）|
| POST | `/api/capture` | 浏览器上传帧（走完整管道）|
| POST | `/api/capture-native` | 手动触发单帧（调试用，绕过筛选直接送 VLM）|
| POST | `/api/query` | 语义查询（body: `{ question }` ）|
| GET | `/api/logs` | 最新日志片段 |

---

## 已知坑

以下是重构过程中遇到的问题，供后续维护参考。

### 1. Node.js 版本必须 ≥ v18（强烈建议 v22）

- `better-sqlite3` 需要原生编译，v14 及以下会报编译失败
- `@lancedb/lancedb` 要求 v18+
- 旧版本（v7-v14）会报 `SyntaxError: Unexpected token ...`（spread 语法）和 `Unexpected token )` （trailing comma）
- **解决**：`nvm install 22 && nvm use 22`，然后重新 `npm install`

### 2. macOS AVFoundation ffmpeg 流式模式"首帧后冻结"

- 使用 `-f avfoundation ... -f image2pipe -` 流式输出时，**只有第一帧是真实画面**，后续所有帧是第一帧的 bit-identical 副本——即使用户走开摄像头也不更新
- 诊断方法：观察日志中 `摄像头原始帧内容变化` 是否在 session 开始后只触发一次
- **解决**：改为每次独立调用 `ffmpeg -vframes 1`（snapshot 模式，`mode: "snapshot"`）；每次独立 AVFoundation session，保证拿到当前实时画面；代价是每帧约 2-3 秒采集延迟

### 3. 浏览器缓存导致帧缩略图显示旧内容

- 早期 `/frames/` 路由没有设置 `Cache-Control` 头，浏览器会缓存帧图片
- 当某个 `frame_NNN.jpg` 因重新开始录制被覆盖，浏览器仍显示缓存里的旧图
- **解决**：`/frames/` 路由返回头加上 `Cache-Control: no-cache, no-store, must-revalidate`

### 4. Buffer 共享内存导致帧内容被覆盖

- MJPEG splitter 中 `buf.slice(start, end+2)` 返回的是 view（共享底层 ArrayBuffer）
- 在 `quickFilter.shouldKeep()` 的 async 执行期间（约 100ms canvas decode），后续 pipe 数据可能覆盖同一段内存，导致比较和落地的是错误帧
- **解决**：在 `handleRawFrame` 入口立即 `Buffer.from(rawFrame.jpeg)` 强制拷贝，切断共享引用

### 5. better-sqlite3 安装失败

- 需要 Xcode Command Line Tools：`xcode-select --install`
- 在沙盒/受限 shell 中 `npm install` 会报 `EPERM`：用 `required_permissions: ["all"]` 或在系统终端里运行
- 编译需匹配当前 Node.js ABI：切换 Node 版本后务必重新 `npm install`

### 6. @tensorflow/tfjs-node 下载失败

- 包体积约 200 MB，国内网络下容易超时
- `start.sh` 的 fallback：`npm install --ignore-scripts`；此时人脸识别不可用，日志会有 `faceApiReady=false`，其余功能正常

### 7. LanceDB 向量索引初始化失败

- 平台预编译包可能不支持某些系统配置
- 系统会自动降级到内存向量索引（重启后向量数据丢失），日志打 WARN
- **完整持久化**需要 `@lancedb/lancedb` 正常安装（v22 Node.js 下正常）

### 8. macOS 摄像头权限

- 终端程序需在「系统设置 → 隐私与安全性 → 摄像头」中被授权
- 未授权时 ffmpeg 5 秒内无帧输出，日志打 `ffmpeg 已运行 5s 但未收到任何帧`
- 换了终端程序（如从 Terminal 换到 iTerm2）需重新授权

---

## 技术栈

| 层次 | 技术 |
|------|------|
| 运行时 | Node.js v22 LTS（CommonJS）|
| 服务 | 内置 `http` / `https` |
| 采帧 | ffmpeg（avfoundation snapshot 模式）|
| 场景变化检测 | 像素差（128×72 缩略图 MAD，阈值 50；全量字节快速对比）|
| 视觉/文本推理 | Gemma4 E2B（Ollama `/api/generate` 或 OpenAI 兼容 `/chat/completions`；支持 1~6 帧批量推理）|
| 文本嵌入 | nomic-embed-text（Ollama `/api/embed`，768 维）|
| 人脸识别 | @vladmandic/face-api + @tensorflow/tfjs-node + canvas（可选）|
| 记忆存储 | better-sqlite3（SQLite）|
| 向量索引 | @lancedb/lancedb（LanceDB）|
| 上传解析 | formidable |
| 前端 | 纯原生 HTML/CSS/JS（无构建工具）|
| 协议 | HTTP 8080；有证书时 HTTPS 8081 |

---

## 许可证

ISC（见 `package.json`）
