#!/usr/bin/env bash
# =============================================================
# 快速验证脚本：在正式训练前用小数据快速跑通全流程
#
# 功能：
#   1. 自动生成 10 条极小数据集（优先复用 data/train.jsonl 前 10 行，
#      不存在时调用 prepare_countdown_data.py 生成）
#   2. 通过临时 recipe 注入 max_steps=5 / logging_steps=1 / save_steps=5，
#      实现每步输出 loss 与 reward 指标
#   3. 执行 accelerate launch（DeepSpeed ZeRO-3），stdout+stderr 同时
#      tee 到 output/quick_test/train_log.txt
#   4. 训练结束后校验输出目录与模型产物是否生成，并提取每步 loss/reward 摘要
#
# 用法：
#   bash scripts/quick_test.sh                      # 默认 QLoRA 模式（省显存、最快）
#   bash scripts/quick_test.sh --mode full          # 全参数模式快速验证
#   bash scripts/quick_test.sh --recipe <path.yaml> # 用其他 recipe 验证
#   bash scripts/quick_test.sh --num-gpus 4         # 手动指定 GPU 数
#
# 注意：
#   - 默认使用 QLoRA recipe（output/quick_test，可 --recipe 覆盖）
#   - 生成的临时数据目录 data_quick/ 与日志可手动清理
# =============================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# 默认配置
# ---------------------------------------------------------------------------
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECIPE_QLORA="${PROJECT_DIR}/recipes/grpo-qwen-2.5-3b-countdown-qlora.yaml"
RECIPE_FULL="${PROJECT_DIR}/recipes/grpo-qwen-2.5-3b-countdown.yaml"
RECIPE="$RECIPE_QLORA"       # 默认 QLoRA（显存占用小，验证快）
MODE="auto"
ACCELERATE_CONFIG="${PROJECT_DIR}/configs/accelerate_configs/deepspeed_zero3.yaml"
NUM_GPUS="auto"
OUTPUT_DIR="${PROJECT_DIR}/output/quick_test"
QUICK_DATA_DIR="${PROJECT_DIR}/data_quick"
MAX_STEPS=5
QUICK_SAMPLES=10

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
log_info() { echo -e "${CYAN}[INFO]${NC} $*"; }
log_ok()   { echo -e "${GREEN}[ OK ]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
die()      { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

usage() {
    sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//' | sed 's/^#//'
    exit 0
}

# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --recipe)            RECIPE="$2"; shift 2 ;;
        --mode)              MODE="$2"; shift 2 ;;
        --accelerate-config) ACCELERATE_CONFIG="$2"; shift 2 ;;
        --num-gpus)          NUM_GPUS="$2"; shift 2 ;;
        --output-dir)        OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help)           usage ;;
        *)                   EXTRA_ARGS+=("$1"); shift ;;
    esac
done

case "$MODE" in
    full|qlora|auto) : ;;
    *) die "--mode 只支持 full / qlora / auto，收到: ${MODE}" ;;
esac

# recipe 别名解析（与 launch_train.sh 保持一致）
if [[ "$RECIPE" == "full" ]]; then
    RECIPE="$RECIPE_FULL"
elif [[ "$RECIPE" == "qlora" ]]; then
    RECIPE="$RECIPE_QLORA"
fi
[[ -f "$RECIPE" ]] || die "recipe 配置文件不存在: ${RECIPE}"

# ---------------------------------------------------------------------------
# GPU 检测（复用 launch_train.sh 逻辑）
# ---------------------------------------------------------------------------
if [[ "$NUM_GPUS" == "auto" ]]; then
    command -v nvidia-smi >/dev/null 2>&1 \
        || die "未找到 nvidia-smi。请确认 NVIDIA 驱动已安装，或使用 --num-gpus N 手动指定。"
    GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
    GPU_COUNT="$(echo -n "$GPU_COUNT" | tr -d '[:space:]')"
else
    GPU_COUNT="$NUM_GPUS"
fi
[[ "$GPU_COUNT" =~ ^[0-9]+$ ]] && [[ "$GPU_COUNT" -gt 0 ]] || die "GPU 数量非法: ${GPU_COUNT}"
if [[ "$GPU_COUNT" -lt 2 ]]; then
    die "检测到 ${GPU_COUNT} 张 GPU，不足 2 张（1 卡训练 + 1 卡 vLLM）。请在多卡机器上运行。"
fi
NUM_PROCESSES=$((GPU_COUNT - 1))

# ---------------------------------------------------------------------------
# Step 1: 准备 10 条极小数据集
# ---------------------------------------------------------------------------
mkdir -p "$QUICK_DATA_DIR" "$OUTPUT_DIR"
QUICK_TRAIN="${QUICK_DATA_DIR}/train.jsonl"

log_info "================ 快速验证（10 条样本 × ${MAX_STEPS} 步）================"
log_info "[1/4] 准备 ${QUICK_SAMPLES} 条样本的小数据集 ..."

if [[ -f "${PROJECT_DIR}/data/train.jsonl" ]]; then
    # 复用已有正式数据的前 N 行（无需重新下载，最快）
    head -n "$QUICK_SAMPLES" "${PROJECT_DIR}/data/train.jsonl" > "$QUICK_TRAIN"
    log_ok "已从 data/train.jsonl 截取前 ${QUICK_SAMPLES} 行 -> ${QUICK_TRAIN}"
else
    # 无正式数据时调用预处理脚本生成（需网络下载 HF 数据集）
    log_warn "未找到 data/train.jsonl，调用预处理脚本生成小数据集（需联网下载）..."
    python "${PROJECT_DIR}/scripts/prepare_countdown_data.py" \
        --max_samples "$QUICK_SAMPLES" --output_dir "$QUICK_DATA_DIR" --seed 42
    log_ok "小数据集已生成 -> ${QUICK_TRAIN}"
fi

# ---------------------------------------------------------------------------
# Step 2: 生成临时 recipe（注入 max_steps=5 / logging_steps=1 / save_steps=5）
# ---------------------------------------------------------------------------
log_info "[2/4] 生成临时 recipe（max_steps=${MAX_STEPS}, logging_steps=1, save_steps=${MAX_STEPS}）..."
TMP_RECIPE="${OUTPUT_DIR}/quick_recipe.yaml"
cp "$RECIPE" "$TMP_RECIPE"
sed -i \
    -e "s/^max_steps:.*/max_steps: ${MAX_STEPS}/" \
    -e "s/^logging_steps:.*/logging_steps: 1/" \
    -e "s/^save_steps:.*/save_steps: ${MAX_STEPS}/" \
    "$TMP_RECIPE"
# 防呆：若 recipe 中缺少上述字段，sed 未命中时追加到文件末尾
for kv in "max_steps: ${MAX_STEPS}" "logging_steps: 1" "save_steps: ${MAX_STEPS}"; do
    key="${kv%%:*}"
    grep -q "^${key}:" "$TMP_RECIPE" || echo "$kv" >> "$TMP_RECIPE"
done
log_ok "临时 recipe: ${TMP_RECIPE}"

# ---------------------------------------------------------------------------
# Step 3: 执行加速训练（tee 日志）
# ---------------------------------------------------------------------------
LOG_FILE="${OUTPUT_DIR}/train_log.txt"
log_info "[3/4] 开始训练（GPU=${GPU_COUNT}, num_processes=${NUM_PROCESSES}, 日志: ${LOG_FILE}）..."

set +e    # 此处需要捕获训练退出码，避免 set -e 提前退出
accelerate launch --config_file "$ACCELERATE_CONFIG" --num_processes "$NUM_PROCESSES" \
    python "${PROJECT_DIR}/scripts/run_grpo.py" \
        --config "$TMP_RECIPE" \
        --train_file "$QUICK_TRAIN" \
        --output_dir "$OUTPUT_DIR" \
        --report_to tensorboard \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    | tee "$LOG_FILE"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e

if [[ $TRAIN_EXIT -ne 0 ]]; then
    die "训练失败（退出码 ${TRAIN_EXIT}）。请查看日志: ${LOG_FILE}"
fi
log_ok "训练流程跑通（退出码 0）"

# ---------------------------------------------------------------------------
# Step 4: 校验产物 + 提取 loss/reward
# ---------------------------------------------------------------------------
log_info "[4/4] 校验输出与提取每步 loss/reward ..."

# 4.1 校验模型产物：QLoRA -> adapter，全参数 -> 完整权重
# 模式判定优先级：--mode 显式值 > recipe 中 load_in_4bit: true 推断
IS_QLORA="no"
if [[ "$MODE" == "qlora" ]]; then
    IS_QLORA="yes"
elif [[ "$MODE" == "auto" ]] && grep -q "load_in_4bit: true" "$TMP_RECIPE"; then
    IS_QLORA="yes"
fi
if [[ "$IS_QLORA" == "yes" ]]; then
    PROD_OK=0
    for f in adapter_config.json adapter_model.safetensors; do
        if [[ -f "$OUTPUT_DIR/$f" ]]; then PROD_OK=1; fi
    done
else
    PROD_OK=0
    for f in config.json model.safetensors.index.json; do
        if [[ -f "$OUTPUT_DIR/$f" ]]; then PROD_OK=1; fi
    done
fi
if [[ $PROD_OK -eq 1 ]]; then
    log_ok "模型产物已生成:"
    ls -lh "$OUTPUT_DIR" | grep -E "config.json|adapter|model|tokenizer" || true
else
    log_warn "未在 ${OUTPUT_DIR} 中找到模型产物（请人工检查 trainer.save_model 是否执行）"
fi

# 4.2 提取每步 loss / reward 指标（transformers 日志为每步一行 JSON dict）
echo ""
echo "------------------------------------------------------------"
echo "  每步 loss / reward 摘要（来自 ${LOG_FILE}）"
echo "------------------------------------------------------------"
if grep -qE "'loss'|'rewards'" "$LOG_FILE"; then
    # 打印含 loss / rewards 的日志行（每步一行），取最后 MAX_STEPS*2 行避免刷屏
    grep -E "'loss'|'rewards'" "$LOG_FILE" | tail -n $((MAX_STEPS * 2))
else
    log_warn "日志中未找到 loss/reward 指标。请确认 recipe 的 logging_steps=1 生效，并检查:"
    log_warn "    grep -E 'loss|reward' ${LOG_FILE}"
fi
echo "------------------------------------------------------------"

log_ok "快速验证完成！"
echo ""
echo "  后续操作："
echo "    - 查看完整日志: tail -n 50 ${LOG_FILE}"
echo "    - 查看 TensorBoard: tensorboard --logdir ${OUTPUT_DIR}/runs"
echo "    - 正式训练: bash scripts/launch_train.sh"
echo "    - 清理临时数据（可选）: rm -rf ${QUICK_DATA_DIR}"
