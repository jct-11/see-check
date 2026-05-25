#!/bin/bash
# 胶水脚本：batch_service 推理结果 → dgsg 格式 → 建图 → viewer
# 用法: bash run_dgsg_pipeline.sh [batch_id] [scene_name]
#   不传参数则自动使用最新 batch
set -e

STMEN_DIR="/home/sscy/lingbot-map/stmem-main"
DGSG_DIR="/home/liangjiahua/dgsg-orin"

# 自动检测 batch
if [ -z "$1" ]; then
    if [ -L "${STMEN_DIR}/data/latest" ] && [ -d "${STMEN_DIR}/data/latest" ]; then
        BATCH_DIR="${STMEN_DIR}/data/latest"
        BATCH_ID=$(basename "$(readlink -f "$BATCH_DIR")")
        SCENE_NAME="${BATCH_ID}"
        echo "🔍 使用 latest: ${BATCH_ID}"
    else
        LATEST=$(ls -dt "${STMEN_DIR}/data/"batch_* 2>/dev/null | head -1)
        if [ -z "$LATEST" ]; then
            echo "❌ 没有找到 batch 数据"
            exit 1
        fi
        BATCH_DIR="$LATEST"
        BATCH_ID=$(basename "$BATCH_DIR")
        SCENE_NAME="${BATCH_ID}"
        echo "🔍 自动检测最新 batch: ${BATCH_ID}"
    fi
else
    BATCH_ID="$1"
    BATCH_DIR="${STMEN_DIR}/data/${BATCH_ID}"
    SCENE_NAME="${2:-${BATCH_ID}}"
fi

BATCH_DIR="${STMEN_DIR}/data/${BATCH_ID}"
OUTPUT_DIR="${DGSG_DIR}/data/mydata/${SCENE_NAME}"

# ── 清空 lingbot 旧数据，确保每次建图从干净状态开始 ──
if [ "${SCENE_NAME}" = "lingbot" ]; then
    echo "🧹 清空 lingbot 旧数据..."
    EXP_DIR="${DGSG_DIR}/experiments/mydata/${SCENE_NAME}"
    rm -rf "${OUTPUT_DIR}" 2>/dev/null || true
    rm -rf "${EXP_DIR}" 2>/dev/null || true
    echo "✓ lingbot 旧数据已清空"
fi

# ── 检查 batch 数据是否存在 ──
if [ ! -d "${BATCH_DIR}/frames" ]; then
    echo "❌ batch 数据不存在: ${BATCH_DIR}/frames"
    exit 1
fi

# ── 初始化 conda（用 liangjiahua 的 miniconda3，dgsg 环境装在那里）──
export PATH="/home/liangjiahua/miniconda3/bin:${PATH}"
eval "$(/home/liangjiahua/miniconda3/bin/conda shell.bash hook 2>/dev/null)" || { echo "❌ conda 初始化失败"; exit 1; }
conda activate dgsg

# 验证环境
DGSG_PYTHON=$(which python)
echo "✓ Python 环境: ${DGSG_PYTHON}"

# ── 1. 转格式：拷贝数据 + 生成 intrinsics.yaml ──
echo "📦 转换数据格式: ${BATCH_ID} → ${SCENE_NAME}"

mkdir -p "${OUTPUT_DIR}/rgb" "${OUTPUT_DIR}/depth" "${OUTPUT_DIR}/poses" "${OUTPUT_DIR}/point"

# 拷贝 frames → rgb
cp "${BATCH_DIR}/frames"/* "${OUTPUT_DIR}/rgb/" 2>/dev/null || true

# 拷贝 depth
if [ -d "${BATCH_DIR}/depth" ]; then
    cp "${BATCH_DIR}/depth"/* "${OUTPUT_DIR}/depth/" 2>/dev/null || true
else
    echo "❌ 没有 depth 数据，跳过"
    exit 1
fi

# 拷贝 poses
if [ -d "${BATCH_DIR}/poses" ]; then
    cp "${BATCH_DIR}/poses"/* "${OUTPUT_DIR}/poses/" 2>/dev/null || true
else
    echo "❌ 没有 poses 数据，跳过"
    exit 1
fi

# 拷贝 point
if [ -d "${BATCH_DIR}/point" ]; then
    cp "${BATCH_DIR}/point"/* "${OUTPUT_DIR}/point/" 2>/dev/null || true
else
    echo "⚠️  没有 point 数据，跳过"
fi

# ── 验证 rgb/depth/poses 数量一致 ──
RGB_COUNT=$(ls "${OUTPUT_DIR}/rgb/" 2>/dev/null | wc -l)
DEPTH_COUNT=$(ls "${OUTPUT_DIR}/depth/" 2>/dev/null | wc -l)
POSES_COUNT=$(ls "${OUTPUT_DIR}/poses/" 2>/dev/null | wc -l)
POINT_COUNT=$(ls "${OUTPUT_DIR}/point/" 2>/dev/null | wc -l)
echo "📊 数据统计: rgb=${RGB_COUNT}, depth=${DEPTH_COUNT}, poses=${POSES_COUNT}, point=${POINT_COUNT}"

# ── 限制 rgb 数量与 depth/poses 一致（前端可能多采几帧）──
TARGET_COUNT="${DEPTH_COUNT}"
if [ "${POSES_COUNT}" -lt "${TARGET_COUNT}" ]; then
    TARGET_COUNT="${POSES_COUNT}"
fi

if [ "${RGB_COUNT}" -gt "${TARGET_COUNT}" ]; then
    echo "⚠️  rgb 图片数(${RGB_COUNT})多于 depth(${DEPTH_COUNT})/poses(${POSES_COUNT})，裁剪为 ${TARGET_COUNT} 帧"
    EXTRA=$((RGB_COUNT - TARGET_COUNT))
    # 删除多余的 rgb 文件（按文件名排序，删除最后的）
    ls "${OUTPUT_DIR}/rgb/" | sort | tail -n "${EXTRA}" | while read f; do
        rm -f "${OUTPUT_DIR}/rgb/${f}"
    done
    RGB_COUNT=$(ls "${OUTPUT_DIR}/rgb/" 2>/dev/null | wc -l)
fi
echo "✓ 数据就绪: rgb=${RGB_COUNT}, depth=${DEPTH_COUNT}, poses=${POSES_COUNT}"

# 读取 batch 的 intrinsics.json 生成 intrinsics.yaml
python3 - "${BATCH_ID}" "${OUTPUT_DIR}" <<'PYEOF'
import sys, json, os, yaml

batch_id = sys.argv[1]
output_dir = sys.argv[2]
intr_path = f"/home/sscy/lingbot-map/stmem-main/data/{batch_id}/intrinsics.json"

defaults = {"fx": 644.6, "fy": 643.7, "cx": 641.4, "cy": 368.7, "w": 1280, "h": 720}
if os.path.exists(intr_path):
    with open(intr_path) as f:
        intr = json.load(f)
    print(f"✓ 读取真实相机参数: {intr}")
else:
    intr = defaults
    print("⚠️  没有 intrinsics.json，使用默认值")

config = {
    "dataset_name": "mydata",
    "camera_params": {
        "image_height": intr["h"],
        "image_width": intr["w"],
        "fx": intr["fx"],
        "fy": intr["fy"],
        "cx": intr["cx"],
        "cy": intr["cy"],
        "png_depth_scale": 1000.0,
        "crop_edge": 0,
    },
}
yaml_path = os.path.join(output_dir, "intrinsics.yaml")
with open(yaml_path, "w") as f:
    yaml.dump(config, f, default_flow_style=False)
print(f"✓ intrinsics.yaml 已生成")
PYEOF

# 确保所有文件其他用户可读写
chmod -R 777 "${OUTPUT_DIR}" 2>/dev/null || true

FRAME_COUNT=$(ls "${OUTPUT_DIR}/rgb/" | wc -l)
echo "✓ 数据就绪: ${FRAME_COUNT} 帧"

# ── 4. 跑建图管线（调用 dgsg-orin 根目录下的 lingbot.sh）──
echo ""
echo "🚀 运行 dgsg 建图管线 (lingbot.sh)..."
cd "${DGSG_DIR}"

# 动态生成配置（指向当前 scene）
cat > "${DGSG_DIR}/configs/mydata/pipeline_temp.py" <<PYEOF
import sys
sys.path.insert(0, "${DGSG_DIR}")
import configs.mydata.test as cfg_module
cfg_module.scene_name = "${SCENE_NAME}"
cfg_module.run_name = "${SCENE_NAME}"
config = cfg_module.config
config["data"]["sequence"] = "${SCENE_NAME}"
config["run_name"] = "${SCENE_NAME}"
config["viz"]["variables_path"] = f"./experiments/mydata/${SCENE_NAME}/variables.npz"
config["viz"]["keyframe_list_path"] = f"./experiments/mydata/${SCENE_NAME}/keyframelist.pkl.gz"
PYEOF

bash lingbot.sh configs/mydata/pipeline_temp.py
LINGBOT_RC=$?
if [ $LINGBOT_RC -ne 0 ]; then
    echo "❌ lingbot.sh 失败，退出码: $LINGBOT_RC"
    exit $LINGBOT_RC
fi

echo ""
echo "✅ 建图完成！结果在 ${DGSG_DIR}/experiments/mydata/${SCENE_NAME}/"

# 确保实验输出文件其他用户可读写
chmod -R 777 "${DGSG_DIR}/experiments/mydata/${SCENE_NAME}" 2>/dev/null || true

# ── 写完成标记（防止 viewer 加载不完整的实验数据）──
touch "${DGSG_DIR}/experiments/mydata/${SCENE_NAME}/.pipeline_done"

