#!/usr/bin/env bash
# =============================================================
# Megatron-LM SFT 监督微调启动脚本（Qwen2.5-3B）
#
# 功能：
#   1. 加载 configs/megatron/sft_qwen2.5-3b.env（低学习率 / 短调度参数）
#   2. 复用 pretrain_gpt.py 入口，配合 SFT 数据集做有监督微调
#   3. 同样提供 GPU 检测、并行度推荐与全套约束校验
#
# 与预训练的差异：
#   - 数据必须是 input/output 分离格式：prepare_sft_data.py --mode sft 产出
#   - 学习率显著降低（默认 1e-5），warmup 占比更高
#   - 建议 --load 指向 convert_checkpoint.py 转换出的 Megatron 检查点，
#     或从预训练产出的检查点继续，避免从随机初始化开始
#
# 用法：
#   bash scripts/megatron/sft_qwen.sh                       # 默认 sft env + 自动并行
#   bash scripts/megatron/sft_qwen.sh --load output/megatron/pretrain   # 从预训练检查点继续
#   bash scripts/megatron/sft_qwen.sh --tp 2 --pp 1 --lr 2e-5
#   bash scripts/megatron/sft_qwen.sh --dry-run
#
# 参数说明：
#   --conf FILE          参数配置文件（默认 configs/megatron/sft_qwen2.5-3b.env）
#   --megatron-path DIR  Megatron-LM 源码根目录（默认取环境变量 MEGATRON_PATH）
#   --data-path PREFIX   SFT 数据前缀（input/output 分离格式）
#   --load DIR           起始检查点（强烈建议指定）
#   --tp/--pp/--cp/--ep N  张量/流水/上下文/专家并行度
#   --gpus N / --seqlen N / --mbs N / --gbs N / --lr F / --iters N
#   --save DIR
#   --num-nodes N / --node-rank N / --master-addr IP / --master-port P
#   --launcher torchrun|deepspeed
#   --dry-run            仅打印命令
#   -h / --help          显示本帮助
#
# 说明：
#   - SFT 完成后如需转回 HF 格式给 vLLM/transformers 部署，运行：
#       torchrun --nproc_per_node=8 scripts/megatron/convert_checkpoint.py --mode export ...
#   - 本链路产出的权重可直接作为 scripts/megatron/grpo_verl_megatron.sh 的 GRPO 起点
# =============================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CONF="${PROJECT_DIR}/configs/megatron/sft_qwen2.5-3b.env"
MEGATRON_PATH="${MEGATRON_PATH:-}"
DRY_RUN="no"
LAUNCHER="torchrun"
NUM_NODES=1
NODE_RANK=0
MASTER_ADDR="127.0.0.1"
MASTER_PORT="29500"
GPU_COUNT="auto"

OVR_DATA_PATH=""; OVR_TOKENIZER=""; OVR_TP=""; OVR_PP=""; OVR_CP=""; OVR_EP=""
OVR_SEQLEN=""; OVR_MBS=""; OVR_GBS=""; OVR_LR=""; OVR_ITERS=""
OVR_SAVE=""; OVR_LOAD=""; OVR_GPUS=""

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
log_info()  { echo -e "${CYAN}[INFO]${NC} $*"; }
log_ok()    { echo -e "${GREEN}[ OK ]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
die()       { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

usage() { sed -n '2,/^# =\{10,\}/p' "$0" | sed 's/^# \{0,1\}//' | sed 's/^#//'; exit 0; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --conf)           CONF="$2"; shift 2 ;;
        --megatron-path)  MEGATRON_PATH="$2"; shift 2 ;;
        --data-path)      OVR_DATA_PATH="$2"; shift 2 ;;
        --tokenizer-path) OVR_TOKENIZER="$2"; shift 2 ;;
        --tp)             OVR_TP="$2"; shift 2 ;;
        --pp)             OVR_PP="$2"; shift 2 ;;
        --cp)             OVR_CP="$2"; shift 2 ;;
        --ep)             OVR_EP="$2"; shift 2 ;;
        --gpus)           OVR_GPUS="$2"; shift 2 ;;
        --seqlen)         OVR_SEQLEN="$2"; shift 2 ;;
        --mbs)            OVR_MBS="$2"; shift 2 ;;
        --gbs)            OVR_GBS="$2"; shift 2 ;;
        --lr)             OVR_LR="$2"; shift 2 ;;
        --iters)          OVR_ITERS="$2"; shift 2 ;;
        --save)           OVR_SAVE="$2"; shift 2 ;;
        --load)           OVR_LOAD="$2"; shift 2 ;;
        --num-nodes)      NUM_NODES="$2"; shift 2 ;;
        --node-rank)      NODE_RANK="$2"; shift 2 ;;
        --master-addr)    MASTER_ADDR="$2"; shift 2 ;;
        --master-port)    MASTER_PORT="$2"; shift 2 ;;
        --launcher)       LAUNCHER="$2"; shift 2 ;;
        --dry-run)        DRY_RUN="yes"; shift ;;
        -h|--help)        usage ;;
        *) die "未知参数: $1（使用 --help 查看用法）" ;;
    esac
done

[[ -f "$CONF" ]] || die "参数配置文件不存在: ${CONF}"
# shellcheck disable=SC1090
source "$CONF"
log_info "已加载配置: ${CONF}"

[[ -n "$OVR_DATA_PATH" ]] && DATA_PATH="$OVR_DATA_PATH"
[[ -n "$OVR_TOKENIZER" ]] && TOKENIZER_PATH="$OVR_TOKENIZER"
[[ -n "$OVR_TP" ]] && TP="$OVR_TP"
[[ -n "$OVR_PP" ]] && PP="$OVR_PP"
[[ -n "$OVR_CP" ]] && CP="$OVR_CP"
[[ -n "$OVR_EP" ]] && EP="$OVR_EP"
[[ -n "$OVR_SEQLEN" ]] && SEQLEN="$OVR_SEQLEN"
[[ -n "$OVR_MBS" ]] && MBS="$OVR_MBS"
[[ -n "$OVR_GBS" ]] && GBS="$OVR_GBS"
[[ -n "$OVR_LR" ]] && LR="$OVR_LR"
[[ -n "$OVR_ITERS" ]] && TRAIN_ITERS="$OVR_ITERS"
[[ -n "$OVR_SAVE" ]] && SAVE_DIR="$OVR_SAVE"
[[ -n "$OVR_LOAD" ]] && LOAD_DIR="$OVR_LOAD"

if [[ -n "$OVR_GPUS" ]]; then
    GPU_COUNT="$OVR_GPUS"
elif [[ "$GPU_COUNT" == "auto" ]]; then
    command -v nvidia-smi >/dev/null 2>&1 || die "未找到 nvidia-smi，请用 --gpus N 手动指定 GPU 数。"
    GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
    GPU_COUNT="$(echo -n "$GPU_COUNT" | tr -d '[:space:]')"
fi
[[ "$GPU_COUNT" =~ ^[0-9]+$ ]] && [[ "$GPU_COUNT" -gt 0 ]] || die "GPU 数量非法: ${GPU_COUNT}"

WORLD_SIZE=$(( NUM_NODES * GPU_COUNT ))
log_info "节点数 ${NUM_NODES} x 单节点 GPU ${GPU_COUNT} => world_size=${WORLD_SIZE}"

recommend_parallel() {
    case "$1" in
        1)  echo "1 1 1" ;;
        2)  echo "1 1 1" ;;
        4)  echo "2 1 1" ;;
        8)  echo "2 1 1" ;;
        16) echo "2 2 1" ;;
        32) echo "2 4 1" ;;
        *)  echo "2 1 1" ;;
    esac
}
if [[ -z "${TP:-}" || -z "${PP:-}" || -z "${CP:-}" ]]; then
    read -r R_TP R_PP R_CP <<< "$(recommend_parallel "$GPU_COUNT")"
    TP="${TP:-$R_TP}"; PP="${PP:-$R_PP}"; CP="${CP:-$R_CP}"
    log_info "未指定并行度，按 ${GPU_COUNT} 卡推荐：TP=${TP} PP=${PP} CP=${CP}"
fi

PARALLEL_PRODUCT=$(( TP * PP * CP ))
if (( WORLD_SIZE % PARALLEL_PRODUCT != 0 )); then
    die "并行度乘积非法：world_size(${WORLD_SIZE}) 不能被 TP(${TP}) x PP(${PP}) x CP(${CP})=${PARALLEL_PRODUCT} 整除"
fi
DP=$(( WORLD_SIZE / PARALLEL_PRODUCT ))
if (( NUM_LAYERS % PP != 0 )); then
    die "层数 ${NUM_LAYERS} 不能被 PP(${PP}) 整除"
fi
if (( NUM_ATTN_HEADS % TP != 0 )); then
    die "注意力头数 ${NUM_ATTN_HEADS} 不能被 TP(${TP}) 整除"
fi
if (( NUM_QUERY_GROUPS % TP != 0 )); then
    die "GQA 分组数 ${NUM_QUERY_GROUPS} 不能被 TP(${TP}) 整除 —— Qwen2.5-3B 的 GQA=2，TP 最大只能为 2"
fi
if (( GBS % (MBS * DP) != 0 )); then
    die "GBS(${GBS}) 不能被 MBS(${MBS}) x DP(${DP})=$((MBS * DP)) 整除"
fi
log_ok "并行校验通过：TP=${TP} PP=${PP} CP=${CP} EP=${EP} -> DP=${DP}"

if [[ -z "$LOAD_DIR" ]]; then
    log_warn "未指定 --load，将从随机初始化开始训练（通常仅用于跑通流程）"
    log_warn "正式 SFT 建议：先用 convert_checkpoint.py 转换 HF 权重，或从上一步预训练检查点继续"
fi

if [[ -z "$MEGATRON_PATH" ]]; then
    die "未指定 Megatron-LM 源码路径。请加 --megatron-path <repo>，或导出环境变量 MEGATRON_PATH。"
fi
PRETRAIN_SCRIPT="$MEGATRON_PATH/pretrain_gpt.py"
[[ -f "$PRETRAIN_SCRIPT" ]] || die "未找到 ${PRETRAIN_SCRIPT}（--megatron-path 需指向含 pretrain_gpt.py 的仓库根目录）"
if ! ls "${DATA_PATH}"*.bin >/dev/null 2>&1; then
    log_warn "未在 ${DATA_PATH} 旁发现 .bin 分词产物，请先运行 scripts/megatron/prepare_sft_data.py --mode sft"
fi

mkdir -p "$SAVE_DIR"

ARGS=(
    --num-layers "$NUM_LAYERS"
    --hidden-size "$HIDDEN_SIZE"
    --ffn-hidden-size "$FFN_HIDDEN_SIZE"
    --num-attention-heads "$NUM_ATTN_HEADS"
    --group-query-attention
    --num-query-groups "$NUM_QUERY_GROUPS"
    --max-position-embeddings "$MAX_POSITION_EMBEDDINGS"
    --rotary-base "$ROTARY_BASE"
    --norm-epsilon "$NORM_EPSILON"
    --seq-length "$SEQLEN"
    --micro-batch-size "$MBS"
    --global-batch-size "$GBS"
    --lr "$LR"
    --min-lr "$MIN_LR"
    --lr-decay-style "$LR_DECAY_STYLE"
    --lr-warmup-fraction "$LR_WARMUP_FRAC"
    --weight-decay "$WEIGHT_DECAY"
    --adam-beta1 "$ADAM_BETA1"
    --adam-beta2 "$ADAM_BETA2"
    --clip-grad "$CLIP_GRAD"
    --train-iters "$TRAIN_ITERS"
    --eval-interval "$EVAL_INTERVAL"
    --save-interval "$SAVE_INTERVAL"
    --log-interval "$LOG_INTERVAL"
    --tensor-model-parallel-size "$TP"
    --pipeline-model-parallel-size "$PP"
    --context-parallel-size "$CP"
    --expert-model-parallel-size "$EP"
    --tokenizer-type PretrainedFromHF
    --tokenizer-name-or-path "$TOKENIZER_PATH"
    --data-path "$DATA_PATH"
    --save "$SAVE_DIR"
    --position-embedding-type rope
    --normalization RMSNorm
    --swiglu
    --distributed-backend nccl
    --transformer-impl "$TRANSFORMER_IMPL"
)

if [[ "$SEQ_PARALLEL" == "1" && "$TP" -gt 1 ]]; then ARGS+=( --sequence-parallel ); fi
if [[ "$BF16" == "1" ]]; then ARGS+=( --bf16 ); fi
if [[ "$USE_FLASH_ATTN" == "1" ]]; then ARGS+=( --use-flash-attn ); fi
[[ -n "$LOAD_DIR" ]] && ARGS+=( --load "$LOAD_DIR" )
if [[ -n "${EXTRA_ARGS:-}" ]]; then
    read -r -a EXTRA_ARR <<< "$EXTRA_ARGS"
    ARGS+=( "${EXTRA_ARR[@]}" )
fi

case "$LAUNCHER" in
    torchrun)
        if [[ "$NUM_NODES" -gt 1 ]]; then
            LAUNCH=(torchrun --nnodes "$NUM_NODES" --node-rank "$NODE_RANK" \
                --rdzv-backend c10d --rdzv-endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
                --nproc-per-node "$GPU_COUNT")
        else
            LAUNCH=(torchrun --standalone --nproc-per-node "$GPU_COUNT")
        fi ;;
    deepspeed)
        if [[ "$NUM_NODES" -gt 1 ]]; then
            LAUNCH=(deepspeed --num_nodes "$NUM_NODES" --num_gpus "$GPU_COUNT" \
                --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT")
        else
            LAUNCH=(deepspeed --num_gpus "$GPU_COUNT")
        fi ;;
    *) die "--launcher 只支持 torchrun / deepspeed，收到: ${LAUNCHER}" ;;
esac

log_info "==================== SFT 启动参数 ===================="
log_info "数据: ${DATA_PATH}   保存: ${SAVE_DIR}${LOAD_DIR:+   加载: $LOAD_DIR}"
log_info "TP=${TP} PP=${PP} CP=${CP} DP=${DP} | MBS=${MBS} GBS=${GBS} | LR=${LR} ITERS=${TRAIN_ITERS}"
log_info "====================================================="

if [[ "$DRY_RUN" == "yes" ]]; then
    echo ""
    echo "最终命令（--dry-run，未执行）："
    echo ""
    echo "  ${LAUNCH[@]} ${PRETRAIN_SCRIPT} \\"
    printf '    %s \\\n' "${ARGS[@]}" | sed '$ s/ \\$//'
    echo ""
    exit 0
fi

log_info "启动 Megatron-LM SFT 微调（launcher=${LAUNCHER}）..."
set +e
"${LAUNCH[@]}" "$PRETRAIN_SCRIPT" "${ARGS[@]}"
EXIT_CODE=$?
set -e
[[ $EXIT_CODE -eq 0 ]] || die "SFT 训练失败（退出码 ${EXIT_CODE}），请参考 docs/CHECKLIST.md 的 Megatron 排查章节。"
log_ok "SFT 完成！输出目录: ${SAVE_DIR}"
echo ""
echo "  后续建议："
echo "    1) 导出 HF 格式部署：torchrun --nproc_per_node=${GPU_COUNT} scripts/megatron/convert_checkpoint.py --mode export ..."
echo "    2) 继续做 GRPO：bash scripts/megatron/grpo_verl_megatron.sh"
