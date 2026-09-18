#!/usr/bin/env bash
# =============================================================
# vLLM 服务容器入口脚本（统一入口，支持两种多机架构）
#
# 功能：
#   1. 读取环境变量并校验（模型路径、并行度、端口、显存占用等）
#   2. 校验物理 GPU 数是否满足 TP x PP（避免启动到一半才失败）
#   3. 按 DISTRIBUTED_BACKEND 选择后端并组装 vllm serve 命令
#      - mp （默认）：单机多卡张量并行，用于「多实例 + 负载均衡」模式
#      - ray         ：跨节点张量/流水并行，用于「单实例跨节点 TP/PP」模式
#   4. exec 启动服务（保证信号可透传，容器停止时优雅退出）
#
# 环境变量（均可由 docker run -e 或 compose environment 覆盖）：
#   MODEL_PATH              模型路径（HF repo id 或已挂载的本地目录，必填）
#   SERVED_MODEL_NAME       对外暴露的模型名（默认 qwen2.5-3b）
#   TP_SIZE / PP_SIZE       张量并行 / 流水并行度（默认 1 / 1）
#   DISTRIBUTED_BACKEND     mp（单机多卡）或 ray（跨节点，默认 mp）
#   HOST / PORT             监听地址与端口（默认 0.0.0.0 / 8000）
#   MAX_MODEL_LEN           最大上下文长度（默认 4096）
#   GPU_MEMORY_UTILIZATION  单卡显存占用比例（默认 0.9）
#   DTYPE                   auto / bfloat16 / float16（默认 auto）
#   MAX_NUM_SEQS            单实例最大并发序列数（默认 256）
#   API_KEY                 鉴权令牌，留空表示不鉴权
#   EXTRA_ARGS              追加参数，如 "--enable-prefix-caching"
#
# 说明：
#   - 模式 A（多实例）：每个节点各跑一个本容器，TP_SIZE 设为该节点 GPU 数
#   - 模式 B（Ray）   ：仅在 head 节点跑本容器，Ray worker 已由 deploy_multinode.sh 拉起
# =============================================================
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen2.5-3b}"
TP_SIZE="${TP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-1}"
DISTRIBUTED_BACKEND="${DISTRIBUTED_BACKEND:-mp}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
DTYPE="${DTYPE:-auto}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
API_KEY="${API_KEY:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

CYAN='\033[0;36m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log_info() { echo -e "${CYAN}[INFO]${NC} $*"; }
log_ok()   { echo -e "${GREEN}[ OK ]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
die()      { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 参数校验
# ---------------------------------------------------------------------------
[[ -n "$MODEL_PATH" ]] || die "MODEL_PATH 未设置（HF repo id 或容器内模型目录）"

for kv in "TP_SIZE=$TP_SIZE" "PP_SIZE=$PP_SIZE" "PORT=$PORT" "MAX_MODEL_LEN=$MAX_MODEL_LEN" "MAX_NUM_SEQS=$MAX_NUM_SEQS"; do
    key="${kv%%=*}"; val="${kv#*=}"
    [[ "$val" =~ ^[0-9]+$ ]] && [[ "$val" -ge 1 ]] || die "${key} 必须是 >=1 的整数，当前值: ${val}"
done

case "$DISTRIBUTED_BACKEND" in
    mp|ray) : ;;
    *) die "DISTRIBUTED_BACKEND 只支持 mp（单机多卡）或 ray（跨节点），收到: ${DISTRIBUTED_BACKEND}" ;;
esac

case "$DTYPE" in
    auto|bfloat16|float16|float32) : ;;
    *) die "DTYPE 只支持 auto / bfloat16 / float16 / float32，收到: ${DTYPE}" ;;
esac

# ---------------------------------------------------------------------------
# GPU 校验：mp 模式下要求本机 GPU 数 >= TP x PP
# （ray 模式由整个 Ray 集群提供 GPU，本机数量不构成约束）
# ---------------------------------------------------------------------------
GPU_COUNT=0
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d '[:space:]')"
fi
TOTAL_PARALLEL=$(( TP_SIZE * PP_SIZE ))

if [[ "$DISTRIBUTED_BACKEND" == "mp" ]]; then
    [[ "$GPU_COUNT" =~ ^[0-9]+$ ]] || GPU_COUNT=0
    if (( GPU_COUNT < TOTAL_PARALLEL )); then
        die "本机 GPU 数(${GPU_COUNT}) 少于 TP x PP = ${TOTAL_PARALLEL}。
  - 单机多卡(mp)模式请调小 TP_SIZE/PP_SIZE；
  - 若确实要跨节点并行，请设置 DISTRIBUTED_BACKEND=ray 并先启动 Ray 集群。"
    fi
    log_ok "GPU 校验通过：本机 ${GPU_COUNT} 卡 >= TP(${TP_SIZE}) x PP(${PP_SIZE}) = ${TOTAL_PARALLEL}"
else
    log_info "Ray 模式：跳过本机 GPU 校验（由 Ray 集群统一调度，TP x PP = ${TOTAL_PARALLEL}）"
    # Ray 模式需要能连上 head 节点
    if [[ -n "${RAY_ADDRESS:-}" ]]; then
        log_info "RAY_ADDRESS=${RAY_ADDRESS}"
    else
        log_warn "未设置 RAY_ADDRESS：若本容器不在 head 节点上启动，将无法发现集群资源"
    fi
fi

# ---------------------------------------------------------------------------
# 组装 vllm serve 命令
# ---------------------------------------------------------------------------
CMD=(
    vllm serve "$MODEL_PATH"
    --served-model-name "$SERVED_MODEL_NAME"
    --tensor-parallel-size "$TP_SIZE"
    --pipeline-parallel-size "$PP_SIZE"
    --host "$HOST"
    --port "$PORT"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --dtype "$DTYPE"
    --max-num-seqs "$MAX_NUM_SEQS"
)

# 跨节点必须显式指定 ray 后端，否则只会用本机 GPU
[[ "$DISTRIBUTED_BACKEND" == "ray" ]] && CMD+=( --distributed-executor-backend ray )
[[ -n "$API_KEY" ]] && CMD+=( --api-key "$API_KEY" )
if [[ -n "$EXTRA_ARGS" ]]; then
    read -r -a EXTRA_ARR <<< "$EXTRA_ARGS"
    CMD+=( "${EXTRA_ARR[@]}" )
fi

log_info "==================== vLLM 服务启动 ===================="
log_info "模型: ${MODEL_PATH}（对外名称: ${SERVED_MODEL_NAME}）"
log_info "并行: TP=${TP_SIZE} PP=${PP_SIZE} 后端=${DISTRIBUTED_BACKEND} | 显存占用=${GPU_MEMORY_UTILIZATION} dtype=${DTYPE}"
log_info "服务: http://${HOST}:${PORT}  | OpenAI 兼容端点 /v1/chat/completions"
log_info "======================================================="

# exec 保证 vllm 进程接管 PID 1，容器可正确接收 SIGTERM
exec "${CMD[@]}"
