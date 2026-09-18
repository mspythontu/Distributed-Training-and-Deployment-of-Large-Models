#!/usr/bin/env bash
# =============================================================
# Megatron-LM 预训练 / 继续预训练启动脚本（Qwen2.5-3B）
#
# 功能：
#   1. 自动检测 GPU 数量，并按卡数推荐合理的 TP/PP/CP 并行组合
#   2. 前置校验并行度约束（避免启动后几分钟才崩，浪费排队时间）
#   3. 加载 configs/megatron/*.env 的模型架构与训练超参
#   4. 用 torchrun（或 deepspeed）启动 Megatron-LM 的 pretrain_gpt.py
#
# 用法：
#   bash scripts/megatron/pretrain_qwen.sh                     # 默认 pretrain env + 自动并行
#   bash scripts/megatron/pretrain_qwen.sh --tp 2 --pp 1       # 显式指定并行度
#   bash scripts/megatron/pretrain_qwen.sh --data-path data/megatron/countdown_pt
#   bash scripts/megatron/pretrain_qwen.sh --launcher deepspeed
#   bash scripts/megatron/pretrain_qwen.sh --dry-run           # 只打印命令不执行
#
# 参数说明：
#   --conf FILE          参数配置文件（默认 configs/megatron/pretrain_qwen2.5-3b.env）
#   --megatron-path DIR  Megatron-LM 源码根目录（默认取环境变量 MEGATRON_PATH）
#   --data-path PREFIX   bin/idx 数据前缀（不含后缀）
#   --tokenizer-path S   HF 分词器路径
#   --tp/--pp/--cp/--ep N  张量/流水/上下文/专家并行度
#   --gpus N             GPU 数（默认 nvidia-smi 自动检测）
#   --seqlen/--mbs/--gbs N  序列长度 / micro batch / global batch
#   --lr F / --iters N   学习率 / 训练迭代数
#   --save DIR / --load DIR   检查点保存与加载目录
#   --num-nodes N / --node-rank N / --master-addr IP / --master-port P   多机参数
#   --launcher torchrun|deepspeed
#   --dry-run            仅打印组装出的最终命令
#   -h / --help          显示本帮助
#
# 说明：
#   - 依赖安装：bash scripts/setup_env.sh --with-megatron
#   - Qwen2.5-3B 的 GQA 分组数为 2，因此 TP 最大值为 2（脚本会校验，超限直接报错）
#   - Megatron 参数名随版本演进，若报未识别参数请对照所用版本的官方文档
# =============================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ---------------------------------------------------------------------------
# 默认配置（vars loaded later from conf; CLI ovr applied after source）
# ---------------------------------------------------------------------------
CONF="${PROJECT_DIR}/configs/megatron/pretrain_qwen2.5-3b.env"
MEGATRON_PATH="${MEGATRON_PATH:-}"
DRY_RUN="no"
LAUNCHER="torchrun"
NUM_NODES=1
NODE_RANK=0
MASTER_ADDR="127.0.0.1"
MASTER_PORT="29500"
GPU_COUNT="auto"

# CLI 覆盖变量：解析阶段先存放，source conf 之后再统一生效
OVR_DATA_PATH=""; OVR_TOKENIZER=""; OVR_TP=""; OVR_PP=""; OVR_CP=""; OVR_EP=""
OVR_SEQLEN=""; OVR_MBS=""; OVR_GBS=""; OVR_LR=""; OVR_ITERS=""
OVR_SAVE=""; OVR_LOAD=""; OVR_GPUS=""

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
log_info()  { echo -e "${CYAN}[INFO]${NC} $*"; }
log_ok()    { echo -e "${GREEN}[ OK ]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
die()       { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

usage() { sed -n '2,/^# =\{10,\}/p' "$0" | sed 's/^# \{0,1\}//' | sed 's/^#//'; exit 0; }

# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 加载参数配置文件，再应用命令行覆盖
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# GPU 检测
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 并行度：未指定时按卡数推荐（Qwen2.5-3B 的 GQA=2，TP 上限为 2）
# ---------------------------------------------------------------------------
recommend_parallel() {
    case "$1" in
        1)  echo "1 1 1" ;;
        2)  echo "1 1 1" ;;   # 2 卡：纯 DP，避免 TP 跨 PCIe
        4)  echo "2 1 1" ;;
        8)  echo "2 1 1" ;;   # 单节点 8 卡最常用组合
        16) echo "2 2 1" ;;   # 两节点：PP 跨机分摊显存
        32) echo "2 4 1" ;;
        *)  echo "2 1 1" ;;
    esac
}
if [[ -z "${TP:-}" || -z "${PP:-}" || -z "${CP:-}" ]]; then
    read -r R_TP R_PP R_CP <<< "$(recommend_parallel "$GPU_COUNT")"
    TP="${TP:-$R_TP}"; PP="${PP:-$R_PP}"; CP="${CP:-$R_CP}"
    log_info "未指定并行度，按 ${GPU_COUNT} 卡推荐：TP=${TP} PP=${PP} CP=${CP}"
fi

# ---------------------------------------------------------------------------
# 并行度约束校验（提前失败，避免排到集群才发现配置非法）
# ---------------------------------------------------------------------------
PARALLEL_PRODUCT=$(( TP * PP * CP ))
if (( WORLD_SIZE % PARALLEL_PRODUCT != 0 )); then
    die "并行度乘积非法：world_size(${WORLD_SIZE}) 不能被 TP(${TP}) x PP(${PP}) x CP(${CP})=${PARALLEL_PRODUCT} 整除"
fi
DP=$(( WORLD_SIZE / PARALLEL_PRODUCT ))

if (( NUM_LAYERS % PP != 0 )); then
    die "层数 ${NUM_LAYERS} 不能被 PP(${PP}) 整除（可选 PP：1/2/3/4/6/9/12/18/36）"
fi
if (( NUM_ATTN_HEADS % TP != 0 )); then
    die "注意力头数 ${NUM_ATTN_HEADS} 不能被 TP(${TP}) 整除"
fi
if (( NUM_QUERY_GROUPS % TP != 0 )); then
    die "GQA 分组数 ${NUM_QUERY_GROUPS} 不能被 TP(${TP}) 整除 —— Qwen2.5-3B 的 GQA=2，TP 最大只能为 2"
fi
if (( GBS % (MBS * DP) != 0 )); then
    die "GBS(${GBS}) 不能被 MBS(${MBS}) x DP(${DP})=${DP*MBS} 整除，请调整 --gbs 或 --mbs"
fi
log_ok "并行校验通过：TP=${TP} PP=${PP} CP=${CP} EP=${EP} -> DP=${DP}"

# ---------------------------------------------------------------------------
# 定位 Megatron-LM 的 pretrain_gpt.py
# ---------------------------------------------------------------------------
if [[ -z "$MEGATRON_PATH" ]]; then
    die "未指定 Megatron-LM 源码路径。请加 --megatron-path <repo>，或导出环境变量 MEGATRON_PATH。
     获取源码：git clone https://github.com/NVIDIA/Megatron-LM.git"
fi
PRETRAIN_SCRIPT="$MEGATRON_PATH/pretrain_gpt.py"
[[ -f "$PRETRAIN_SCRIPT" ]] || die "未找到 ${PRETRAIN_SCRIPT}（--megatron-path 需指向含 pretrain_gpt.py 的仓库根目录）"
# Megatron 产物形如 <prefix>_<key>_document.bin / .idx，用通配判断是否存在
if ! ls "${DATA_PATH}"*.bin >/dev/null 2>&1; then
    log_warn "未在 ${DATA_PATH} 旁发现 .bin 分词产物，请先运行 scripts/megatron/prepare_sft_data.py"
fi

mkdir -p "$SAVE_DIR"

# ---------------------------------------------------------------------------
# 组装训练命令
# ---------------------------------------------------------------------------
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

# launcher 组装：单机用 standalone / 纯 deepspeed，多机走 rdzv 端点
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

log_info "==================== 预训练启动参数 ===================="
log_info "数据: ${DATA_PATH}   保存: ${SAVE_DIR}${LOAD_DIR:+   加载: $LOAD_DIR}"
log_info "TP=${TP} PP=${PP} CP=${CP} DP=${DP} | MBS=${MBS} GBS=${GBS} | LR=${LR} ITERS=${TRAIN_ITERS}"
log_info "========================================================"

if [[ "$DRY_RUN" == "yes" ]]; then
    echo ""
    echo "最终命令（--dry-run，未执行）："
    echo ""
    echo "  ${LAUNCH[@]} ${PRETRAIN_SCRIPT} \\"
    printf '    %s \\\n' "${ARGS[@]}" | sed '$ s/ \\$//'
    echo ""
    exit 0
fi

log_info "启动 Megatron-LM 预训练（launcher=${LAUNCHER}）..."
set +e
"${LAUNCH[@]}" "$PRETRAIN_SCRIPT" "${ARGS[@]}"
EXIT_CODE=$?
set -e
[[ $EXIT_CODE -eq 0 ]] || die "预训练失败（退出码 ${EXIT_CODE}），请参考 docs/CHECKLIST.md 的 Megatron 排查章节。"
log_ok "预训练完成！输出目录: ${SAVE_DIR}"
