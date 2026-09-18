#!/usr/bin/env bash
# =============================================================
# GRPO 强化学习训练脚本：veRL + Megatron-LM 后端
#
# 为什么用 veRL 而不是 TRL？
#   本项目主链路用 TRL(GRPOTrainer) + DeepSpeed(ZeRO-3)，适合中小规模；
#   veRL 的 Megatron 后端额外提供 5D 并行（TP/EP/CP/DP/PP）+ 序列并行，
#   并通过 3D HybridEngine 在 actor(Megatron) 与 rollout(vLLM/SGLang) 之间
#   做高效权重重分片，适合更大规模 / 更长序列的 RL 训练。
#
# 功能：
#   1. 自动把项目 jsonl 数据转成 veRL 需要的 parquet（含 extra_info 透传 target）
#   2. GPU 检测 + Megatron 并行度校验（world_size 必须整除 TP x PP x CP）
#   3. 组装 veRL main_ppo 命令，启用 Megatron 后端的 GRPO 算法
#
# 用法：
#   bash scripts/megatron/grpo_verl_megatron.sh
#   bash scripts/megatron/grpo_verl_megatron.sh --tp 2 --pp 1 --rollout-tp 1
#   bash scripts/megatron/grpo_verl_megatron.sh --model-path output/megatron/sft   # 从 SFT 模型起步
#   bash scripts/megatron/grpo_verl_megatron.sh --offload                          # 显存紧张时开启卸载
#   bash scripts/megatron/grpo_verl_megatron.sh --dry-run                          # 只打印命令
#
# 参数说明：
#   --model-path S      策略起始权重（HF 格式路径或模型 id）
#   --train-file F      jsonl 训练数据（默认 data/train.jsonl，自动转 parquet）
#   --val-file F        jsonl 验证数据（默认 data/eval.jsonl）
#   --tp/--pp/--cp N    actor 的张量/流水/上下文并行度
#   --rollout-tp N      rollout(vLLM) 的张量并行度（默认 1，与训练 TP 解耦）
#   --gpus N            GPU 数（默认自动检测）
#   --train-batch N     全局 prompt 批大小
#   --num-generations N GRPO 每个 prompt 的采样数（等于 group size）
#   --lr F              学习率
#   --epochs N          训练轮数
#   --micro-batch N     单卡 micro batch（影响显存）
#   --max-prompt-len N / --max-response-len N
#   --gpu-mem-util F    vLLM 显存占用比例
#   --offload           开启 actor/ref 的参数、梯度、优化器卸载
#   --custom-reward-module F / --custom-reward-name F   自定义奖励函数（见下）
#   --project S / --exp-name S   wandb 项目 / 实验名
#   --dry-run           仅打印命令
#   -h / --help         显示本帮助
#
# 关于自定义奖励函数：
#   本项目的 scripts/reward_functions.py 面向 TRL 接口（接收 completions/prompts），
#   而 veRL 的 custom reward 接受解包后的参数。需要按 veRL 签名再包一层，例如：
#
#     # my_reward.py
#     def reward_fn(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
#         # ground_truth 来自数据的 ground_truth 字段
#         # extra_info 可透传 index / target 等
#         return 1.0 if ok else 0.0
#
#   然后：--custom-reward-module /abs/path/my_reward.py --custom-reward-name reward_fn
#   提示：直接复用 reward_functions.py 的解析/校验逻辑，只替换入参形式即可。
#
# 说明：
#   - 依赖安装：bash scripts/setup_env.sh --with-megatron（含 veRL）
#   - veRL 的配置键随版本演进较快，若报 "Could not resolve config xxx"，
#     请在虚拟环境中查看对应版本的 verl/trainer/config/ppo_trainer.yaml 后调整本脚本参数
# =============================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ---------------------------------------------------------------------------
# 默认配置
# ---------------------------------------------------------------------------
MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"
TRAIN_FILE="${PROJECT_DIR}/data/train.jsonl"
VAL_FILE="${PROJECT_DIR}/data/eval.jsonl"
WORK_DIR="${PROJECT_DIR}/data/megatron/verl"

GPU_COUNT="auto"
NUM_NODES=1
TP=1
PP=1
CP=1
ROLLOUT_TP=1

TRAIN_BATCH=64
MICRO_BATCH=2
NUM_GENERATIONS=8
LR=1e-6
TOTAL_EPOCHS=1
MAX_PROMPT_LEN=512
MAX_RESPONSE_LEN=1024
GPU_MEM_UTIL=0.4

OFFLOAD="no"
PROJECT_NAME="grpo-megatron"
EXP_NAME="qwen2.5-3b-countdown"
LOG_DIR="${PROJECT_DIR}/output/megatron/verl_grpo"

CUSTOM_REWARD_MODULE=""
CUSTOM_REWARD_NAME=""
DRY_RUN="no"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
log_info()  { echo -e "${CYAN}[INFO]${NC} $*"; }
log_ok()    { echo -e "${GREEN}[ OK ]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
die()       { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

usage() { sed -n '2,/^# =\{10,\}/p' "$0" | sed 's/^# \{0,1\}//' | sed 's/^#//'; exit 0; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path)    MODEL_PATH="$2"; shift 2 ;;
        --train-file)    TRAIN_FILE="$2"; shift 2 ;;
        --val-file)      VAL_FILE="$2"; shift 2 ;;
        --tp)            TP="$2"; shift 2 ;;
        --pp)            PP="$2"; shift 2 ;;
        --cp)            CP="$2"; shift 2 ;;
        --rollout-tp)    ROLLOUT_TP="$2"; shift 2 ;;
        --gpus)          GPU_COUNT="$2"; shift 2 ;;
        --num-nodes)     NUM_NODES="$2"; shift 2 ;;
        --train-batch)   TRAIN_BATCH="$2"; shift 2 ;;
        --micro-batch)   MICRO_BATCH="$2"; shift 2 ;;
        --num-generations) NUM_GENERATIONS="$2"; shift 2 ;;
        --lr)            LR="$2"; shift 2 ;;
        --epochs)        TOTAL_EPOCHS="$2"; shift 2 ;;
        --max-prompt-len)  MAX_PROMPT_LEN="$2"; shift 2 ;;
        --max-response-len) MAX_RESPONSE_LEN="$2"; shift 2 ;;
        --gpu-mem-util)  GPU_MEM_UTIL="$2"; shift 2 ;;
        --offload)       OFFLOAD="yes"; shift ;;
        --custom-reward-module) CUSTOM_REWARD_MODULE="$2"; shift 2 ;;
        --custom-reward-name)   CUSTOM_REWARD_NAME="$2"; shift 2 ;;
        --project)       PROJECT_NAME="$2"; shift 2 ;;
        --exp-name)      EXP_NAME="$2"; shift 2 ;;
        --dry-run)       DRY_RUN="yes"; shift ;;
        -h|--help)       usage ;;
        *) die "未知参数: $1（使用 --help 查看用法）" ;;
    esac
done

# ---------------------------------------------------------------------------
# GPU 检测与并行度校验
# ---------------------------------------------------------------------------
if [[ "$GPU_COUNT" == "auto" ]]; then
    command -v nvidia-smi >/dev/null 2>&1 || die "未找到 nvidia-smi，请用 --gpus N 手动指定 GPU 数。"
    GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
    GPU_COUNT="$(echo -n "$GPU_COUNT" | tr -d '[:space:]')"
fi
[[ "$GPU_COUNT" =~ ^[0-9]+$ ]] && [[ "$GPU_COUNT" -gt 0 ]] || die "GPU 数量非法: ${GPU_COUNT}"

WORLD_SIZE=$(( NUM_NODES * GPU_COUNT ))
PARALLEL_PRODUCT=$(( TP * PP * CP ))
if (( WORLD_SIZE % PARALLEL_PRODUCT != 0 )); then
    die "并行度乘积非法：world_size(${WORLD_SIZE}) 不能被 TP(${TP}) x PP(${PP}) x CP(${CP})=${PARALLEL_PRODUCT} 整除"
fi
DP=$(( WORLD_SIZE / PARALLEL_PRODUCT ))
if (( TRAIN_BATCH % (MICRO_BATCH * DP * NUM_GENERATIONS) != 0 )); then
    log_warn "train_batch(${TRAIN_BATCH}) 不是 micro_batch x DP x num_generations=$((MICRO_BATCH * DP * NUM_GENERATIONS)) 的整数倍，veRL 可能自动调整或报错"
fi
log_ok "并行校验通过：world_size=${WORLD_SIZE} -> TP=${TP} PP=${PP} CP=${CP} DP=${DP}（rollout TP=${ROLLOUT_TP}）"

# veRL 依赖 Ray 与 Hydra，依赖缺失时提前给出明确提示
python3 -c "import verl" 2>/dev/null || die "未找到 veRL（import verl 失败）。请先安装：bash scripts/setup_env.sh --with-megatron"

# ---------------------------------------------------------------------------
# 数据准备：jsonl -> parquet（veRL 默认读取 parquet）
#   字段约定：
#     prompt        : conversation 格式（list[{"role","content"}]），veRL 会自行套 chat template
#     ground_truth  : 本项目的 target 值
#     extra_info    : 透传给 reward function 的附加信息
# ---------------------------------------------------------------------------
mkdir -p "$WORK_DIR" "$LOG_DIR"
TRAIN_PARQUET="${WORK_DIR}/train.parquet"
VAL_PARQUET="${WORK_DIR}/val.parquet"

convert_jsonl_to_parquet() {
    SRC="$1"; DST="$2"
    [[ -f "$SRC" ]] || { log_warn "未找到数据文件 ${SRC}，跳过转换"; return 1; }
    SRC_BASENAME="$(basename "$SRC")"
    log_info "转换 ${SRC_BASENAME} -> $(basename "$DST") ..."
    SRC_PATH="$SRC" DST_PATH="$DST" python3 - <<'PY'
import json, os

src, dst = os.environ["SRC_PATH"], os.environ["DST_PATH"]
try:
    import pandas as pd
except ImportError:
    raise SystemExit("缺少 pandas，请先安装：pip install pandas pyarrow")

rows = []
with open(src, encoding="utf-8") as fh:
    for i, line in enumerate(fh):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        prompt = rec.get("prompt") or ""
        rows.append({
            "data_source": "countdown",
            "prompt": [{"role": "user", "content": prompt}],
            "ground_truth": str(rec.get("target", "")),
            "extra_info": {"index": i, "solution": rec.get("solution", "")},
        })
if not rows:
    raise SystemExit(f"{src} 中没有可用样本")
pd.DataFrame(rows).to_parquet(dst, index=False)
print(f"  -> {dst}（{len(rows)} 条）")
PY
}

convert_jsonl_to_parquet "$TRAIN_FILE" "$TRAIN_PARQUET" || die "训练数据转换失败，请检查 ${TRAIN_FILE}"
convert_jsonl_to_parquet "$VAL_FILE" "$VAL_PARQUET" || true

# ---------------------------------------------------------------------------
# 卸载开关（显存紧张时把参数/梯度/优化器临时移到 CPU）
# ---------------------------------------------------------------------------
PARAM_OFFLOAD=False; GRAD_OFFLOAD=False; OPT_OFFLOAD=False; REF_OFFLOAD=False
if [[ "$OFFLOAD" == "yes" ]]; then
    PARAM_OFFLOAD=True; GRAD_OFFLOAD=True; OPT_OFFLOAD=True; REF_OFFLOAD=True
    log_info "已开启 offload：param/grad/optimizer + ref.param"
fi

# ---------------------------------------------------------------------------
# 组装 veRL 命令（注意：配置键以安装的 veRL 版本为准）
# ---------------------------------------------------------------------------
CMD=(
    python3 -m verl.trainer.main_ppo
    algorithm.adv_estimator=grpo
    data.train_files="$TRAIN_PARQUET"
    data.val_files="$VAL_PARQUET"
    data.train_batch_size=$TRAIN_BATCH
    data.max_prompt_length=$MAX_PROMPT_LEN
    data.max_response_length=$MAX_RESPONSE_LEN
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.strategy=megatron
    actor_rollout_ref.actor.optim.lr=$LR
    actor_rollout_ref.actor.ppo_mini_batch_size=$MICRO_BATCH
    actor_rollout_ref.actor.micro_batch_size_per_device_for_update=$MICRO_BATCH
    actor_rollout_ref.actor.micro_batch_size_per_device_for_experience=$MICRO_BATCH
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP
    actor_rollout_ref.actor.megatron.context_parallel_size=$CP
    actor_rollout_ref.actor.megatron.param_offload=$PARAM_OFFLOAD
    actor_rollout_ref.actor.megatron.grad_offload=$GRAD_OFFLOAD
    actor_rollout_ref.actor.megatron.optimizer_offload=$OPT_OFFLOAD
    actor_rollout_ref.ref.strategy=megatron
    actor_rollout_ref.ref.megatron.param_offload=$REF_OFFLOAD
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM_UTIL
    actor_rollout_ref.rollout.n=$NUM_GENERATIONS
    trainer.n_gpus_per_node=$GPU_COUNT
    trainer.nnodes=$NUM_NODES
    trainer.total_epochs=$TOTAL_EPOCHS
    trainer.project_name="$PROJECT_NAME"
    trainer.experiment_name="$EXP_NAME"
    trainer.default_local_dir="$LOG_DIR"
)

if [[ -n "$CUSTOM_REWARD_MODULE" && -n "$CUSTOM_REWARD_NAME" ]]; then
    CMD+=(
        custom_reward_function.path="$CUSTOM_REWARD_MODULE"
        custom_reward_function.name="$CUSTOM_REWARD_NAME"
    )
    log_info "使用自定义奖励函数: ${CUSTOM_REWARD_MODULE}:${CUSTOM_REWARD_NAME}"
else
    log_warn "未指定 --custom-reward-module/--custom-reward-name，将使用 veRL 内置奖励（Countdown 场景务必自定义，否则奖励无意义）"
fi

log_info "==================== GRPO（veRL + Megatron） ===================="
log_info "模型: ${MODEL_PATH}"
log_info "数据: ${TRAIN_PARQUET}"
log_info "TP=${TP} PP=${PP} CP=${CP} DP=${DP} | batch=${TRAIN_BATCH} micro=${MICRO_BATCH} generations=${NUM_GENERATIONS}"
log_info "================================================================"

if [[ "$DRY_RUN" == "yes" ]]; then
    echo ""
    echo "最终命令（--dry-run，未执行）："
    echo ""
    printf '  %s \\\n' "${CMD[@]}" | sed '$ s/ \\$//'
    echo ""
    exit 0
fi

log_info "启动 veRL GRPO 训练（Megatron 后端）..."
set +e
"${CMD[@]}"
EXIT_CODE=$?
set -e
[[ $EXIT_CODE -eq 0 ]] || die "GRPO 训练失败（退出码 ${EXIT_CODE}），请参考 docs/CHECKLIST.md 的 Megatron 排查章节。"
log_ok "GRPO 训练完成！输出目录: ${LOG_DIR}"
