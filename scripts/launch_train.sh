#!/usr/bin/env bash
# =============================================================
# 训练启动脚本：GRPO（DeepSpeed ZeRO-3 + vLLM）一键启动
#
# 功能：
#   1. 自动检测可用 GPU 数量（支持 --num-gpus 手动覆盖）
#   2. 自动计算 num_processes = GPU 数 - 1（最后 1 卡留给 vLLM 生成后端）
#   3. 通过参数选择 recipe 配置文件与训练模式（full / qlora）
#   4. 组装并执行完整的 accelerate launch 命令，其他参数原样透传给 run_grpo.py
#   5. GPU 数量不足（<2）时给出明确报错与修复建议
#
# 用法：
#   bash scripts/launch_train.sh                                          # 全参数微调（默认）
#   bash scripts/launch_train.sh --mode qlora                             # QLoRA 模式
#   bash scripts/launch_train.sh --recipe qlora --learning_rate 1e-6      # 选 recipe + 透传超参
#   bash scripts/launch_train.sh --recipe grpo-qwen-2.5-3b-countdown.yaml --max_steps 100
#   bash scripts/launch_train.sh --num-gpus 4                             # 手动指定 GPU 数
#
# 参数说明：
#   --recipe NAME          recipe 配置文件（默认全参数 recipes/grpo-qwen-2.5-3b-countdown.yaml）
#                          NAME 支持：full / qlora / 短名（自动补全为 recipes/NAME.yaml）/ 完整路径
#   --mode MODE            full=全参数微调，qlora=4bit量化+LoRA，auto=不传 --mode（由脚本推断）
#                          （默认 auto；指定 recipe 时请保持与 recipe 一致）
#   --accelerate-config    accelerate 配置文件（默认 configs/accelerate_configs/deepspeed_zero3.yaml）
#   --num-gpus N           手动指定 GPU 数量（默认自动检测）
#   其余参数               原样透传给 scripts/run_grpo.py（如 --learning_rate / --max_steps
#                          / --output_dir / --train_file / --report_to 等）
#
# 说明：
#   - num_processes = GPU 数 - 1：训练进程占用前 N-1 卡，vLLM（vllm_device="auto"）
#     自动占用最后一块空闲卡，避免训练/推理争抢显存
#   - 4bit 量化（QLoRA）与 ZeRO-3 offload 存在兼容性限制，若报 offload 相关错误
#     请修改 configs/accelerate_configs/deepspeed_zero3.yaml 中 offload_optimizer_device: none
# =============================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# 默认配置
# ---------------------------------------------------------------------------
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_FILE="${PROJECT_DIR}/scripts/run_grpo.py"

# recipe 默认值（--recipe 未指定时按 --mode 选择）
RECIPE_FULL="${PROJECT_DIR}/recipes/grpo-qwen-2.5-3b-countdown.yaml"
RECIPE_QLORA="${PROJECT_DIR}/recipes/grpo-qwen-2.5-3b-countdown-qlora.yaml"
RECIPE=""
MODE="auto"
ACCELERATE_CONFIG="${PROJECT_DIR}/configs/accelerate_configs/deepspeed_zero3.yaml"
NUM_GPUS="auto"

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
log_info() { echo -e "${CYAN}[INFO]${NC} $*"; }
log_ok()   { echo -e "${GREEN}[ OK ]${NC} $*"; }
die()      { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

usage() {
    sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//' | sed 's/^#//'
    exit 0
}

# ---------------------------------------------------------------------------
# 解析命令行参数：--xx 选项 + 其余参数透传
# ---------------------------------------------------------------------------
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --recipe)            RECIPE="$2"; shift 2 ;;
        --mode)              MODE="$2"; shift 2 ;;
        --accelerate-config) ACCELERATE_CONFIG="$2"; shift 2 ;;
        --num-gpus)          NUM_GPUS="$2"; shift 2 ;;
        -h|--help)           usage ;;
        *)                   EXTRA_ARGS+=("$1"); shift ;;   # 透传给 run_grpo.py
    esac
done

# ---------------------------------------------------------------------------
# 校验 mode 参数
# ---------------------------------------------------------------------------
case "$MODE" in
    full|qlora|auto) : ;;
    *) die "--mode 只支持 full / qlora / auto，收到: ${MODE}" ;;
esac

# ---------------------------------------------------------------------------
# 解析 recipe：默认值 + 别名（full/qlora）+ 短名补全
# ---------------------------------------------------------------------------
if [[ -z "$RECIPE" ]]; then
    if [[ "$MODE" == "qlora" ]]; then
        RECIPE="$RECIPE_QLORA"
    else
        RECIPE="$RECIPE_FULL"    # auto 模式默认全参数配置
    fi
else
    case "$RECIPE" in
        full)  RECIPE="$RECIPE_FULL" ;;
        qlora) RECIPE="$RECIPE_QLORA" ;;
        *)     # 不含路径分隔符时自动补全为 recipes/<name>.yaml
            if [[ "$RECIPE" != */* && "$RECIPE" != *.yaml ]]; then
                RECIPE="${PROJECT_DIR}/recipes/${RECIPE}.yaml"
            elif [[ "$RECIPE" != /* ]]; then
                RECIPE="${PROJECT_DIR}/${RECIPE}"
            fi
            ;;
    esac
fi

[[ -f "$RECIPE" ]] || die "recipe 配置文件不存在: ${RECIPE}"
[[ -f "$SCRIPT_FILE" ]] || die "训练脚本不存在: ${SCRIPT_FILE}"

# ---------------------------------------------------------------------------
# 检测 GPU 数量（--num-gpus 可手动覆盖）
# ---------------------------------------------------------------------------
if [[ "$NUM_GPUS" == "auto" ]]; then
    command -v nvidia-smi >/dev/null 2>&1 \
        || die "未找到 nvidia-smi。请确认已安装 NVIDIA 驱动，或使用 --num-gpus N 手动指定。"
    GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
    GPU_COUNT="$(echo -n "$GPU_COUNT" | tr -d '[:space:]')"
else
    GPU_COUNT="$NUM_GPUS"
fi

# 数字校验
[[ "$GPU_COUNT" =~ ^[0-9]+$ ]] && [[ "$GPU_COUNT" -gt 0 ]] \
    || die "GPU 数量非法: ${GPU_COUNT}"

# ---------------------------------------------------------------------------
# GPU 数量不足检查：至少 1 卡训练 + 1 卡 vLLM
# ---------------------------------------------------------------------------
if [[ "$GPU_COUNT" -lt 2 ]]; then
    die "检测到 ${GPU_COUNT} 张 GPU，不足 2 张。本脚本训练侧使用 GPU 数 - 1 个进程，
    最后 1 卡需留给 vLLM 生成后端（vllm_device=auto）。
    处理建议：
      1) 在有多卡的机器/节点上运行本脚本；
      2) 单卡调试可临时修改 recipe 中 vllm: false，改用 transformers 生成后端（显存要求更高）；
      3) 确保集群上 nvidia-smi 可见所有 GPU（排除权限/容器未挂载 GPU 的情况）。"
fi

NUM_PROCESSES=$((GPU_COUNT - 1))

# ---------------------------------------------------------------------------
# 校验 accelerate 配置与训练数据
# ---------------------------------------------------------------------------
[[ -f "$ACCELERATE_CONFIG" ]] \
    || die "accelerate 配置文件不存在: ${ACCELERATE_CONFIG}"

TRAIN_FILE="${PROJECT_DIR}/data/train.jsonl"
if [[ ! -f "$TRAIN_FILE" ]]; then
    die "训练数据不存在: ${TRAIN_FILE}
    请先运行数据预处理脚本生成数据：
        conda activate <你的环境>
        python scripts/prepare_countdown_data.py
    （如需使用其他数据，可透传 --train_file <路径> 覆盖）"
fi

# ---------------------------------------------------------------------------
# 组装 accelerate launch 命令
# ---------------------------------------------------------------------------
# accelerate launch --config_file <加速配置> --num_processes <GPU-1> \
#     python scripts/run_grpo.py --config <recipe> [--mode <full|qlora>] [透传参数...]
ACCELERATE_CMD=(accelerate launch --config_file "$ACCELERATE_CONFIG" --num_processes "$NUM_PROCESSES")
PY_CMD=(python "$SCRIPT_FILE" --config "$RECIPE")
if [[ "$MODE" != "auto" ]]; then
    PY_CMD+=(--mode "$MODE")
fi
PY_CMD+=("${EXTRA_ARGS[@]}")

# ---------------------------------------------------------------------------
# 打印启动摘要 + 完整命令（便于复现/排查），然后执行
# ---------------------------------------------------------------------------
log_info "================ 训练启动信息 ================"
log_info "GPU 数量        : ${GPU_COUNT}"
log_info "num_processes   : ${NUM_PROCESSES}（GPU 数 - 1，最后 1 卡留给 vLLM）"
log_info "训练模式        : ${MODE}"
log_info "recipe 配置     : ${RECIPE}"
log_info "accelerate 配置 : ${ACCELERATE_CONFIG}"
log_info "训练数据        : ${TRAIN_FILE}"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    log_info "透传参数        : ${EXTRA_ARGS[*]}"
fi
log_info "=============================================="
echo ""
echo ">>> 即将执行:"
echo "    ${ACCELERATE_CMD[*]} \\"
echo "        ${PY_CMD[*]}"
echo ""

# 注意：不要在 --num_processes 后跟 \\ 续行导致 shell 拼接错误，这里直接执行数组
"${ACCELERATE_CMD[@]}" "${PY_CMD[@]}"
EXIT_CODE=$?

if [[ $EXIT_CODE -eq 0 ]]; then
    log_ok "训练正常结束。模型已保存到 recipe 中 output_dir 指定目录（可用 --output_dir 覆盖）。"
else
    echo -e "${RED}[ERROR]${NC} accelerate launch 退出码: ${EXIT_CODE}"
    echo "请参考 docs/CHECKLIST.md 中的常见问题排查表定位原因。"
fi
exit $EXIT_CODE
