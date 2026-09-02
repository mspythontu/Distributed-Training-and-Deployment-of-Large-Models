#!/usr/bin/env bash
# =============================================================
# 一键环境安装脚本：GRPO（DeepSpeed + TRL + vLLM）训练环境
#
# 功能：
#   1. 创建 conda 环境（默认 Python 3.10）
#   2. 按集群 CUDA 版本安装 torch（官方 wheel），再安装 requirements.txt 其余依赖
#   3. 可选：从源码安装 trl，启用 Liger GRPO Loss（LigerGRPOConfig）
#   4. 配置 HuggingFace 镜像源（国内加速，默认 https://hf-mirror.com）
#   5. 安装完成后打印各库版本，验证环境可用性
#
# 用法：
#   bash scripts/setup_env.sh                              # 全默认安装
#   bash scripts/setup_env.sh --cuda cu124                 # 显式指定 CUDA 12.4
#   bash scripts/setup_env.sh --source-trl                 # 从源码安装 trl（Liger）
#   bash scripts/setup_env.sh --env-name grpo --python 3.10
#   bash scripts/setup_env.sh --skip-verify                # 跳过末尾版本验证
#   bash scripts/setup_env.sh --with-wandb                 # 额外安装 wandb（训练可视化）
#
# 参数说明：
#   --env-name NAME         conda 环境名（默认 grpo）
#   --python X.Y            Python 版本（默认 3.10）
#   --cuda CUXXX            cu121 / cu124 / cu128；缺省根据 nvidia-smi 自动推断
#   --source-trl            从源码安装 trl（支持 Liger GRPO Loss）
#   --with-wandb            安装 wandb（训练可视化，配合 --report_to wandb 使用，默认不装）
#   --hf-endpoint URL       HuggingFace 镜像地址（默认 https://hf-mirror.com）
#   --force                 环境已存在时删除重建（默认遇到已存在环境直接退出）
#   --skip-verify           跳过安装后的版本验证
#   -h / --help             显示本帮助
#
# 说明：
#   - 脚本面向 Linux GPU 集群（bash + conda + nvidia-smi）
#   - torch 安装顺序遵循 requirements.txt 头部说明：先按 CUDA 版本装 torch，再装其余依赖
#   - CUDA 12.8 时 vLLM 需要额外索引重装（vllm 0.8.x 对 CUDA 版本强依赖）
# =============================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# 默认配置
# ---------------------------------------------------------------------------
ENV_NAME="grpo"
PYTHON_VERSION="3.10"
CUDA_FLAG="auto"        # auto | cu121 | cu124 | cu128
SOURCE_TRL="no"         # yes/no：是否从源码安装 trl（Liger GRPO Loss）
WITH_WANDB="no"         # yes/no：是否额外安装 wandb（训练可视化）
HF_ENDPOINT="https://hf-mirror.com"
FORCE="no"
SKIP_VERIFY="no"

# 项目根目录 = 脚本所在目录的上一级
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQ_FILE="${PROJECT_DIR}/requirements.txt"

# ---------------------------------------------------------------------------
# 工具函数：彩色进度提示 + 错误退出
# ---------------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

log_info()  { echo -e "${CYAN}[INFO]${NC} $*"; }
log_ok()    { echo -e "${GREEN}[ OK ]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
die()       { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

usage() {
    sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//' | sed 's/^#//'
    exit 0
}

# ---------------------------------------------------------------------------
# 解析命令行参数
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-name)      ENV_NAME="$2"; shift 2 ;;
        --python)        PYTHON_VERSION="$2"; shift 2 ;;
        --cuda)          CUDA_FLAG="$2"; shift 2 ;;
        --source-trl)    SOURCE_TRL="yes"; shift ;;
        --with-wandb)    WITH_WANDB="yes"; shift ;;
        --hf-endpoint)   HF_ENDPOINT="$2"; shift 2 ;;
        --force)         FORCE="yes"; shift ;;
        --skip-verify)   SKIP_VERIFY="yes"; shift ;;
        -h|--help)       usage ;;
        *) die "未知参数: $1（使用 --help 查看用法）" ;;
    esac
done

# ---------------------------------------------------------------------------
# 前置检查：conda / git
# ---------------------------------------------------------------------------
log_info "项目目录: ${PROJECT_DIR}"
[[ -f "$REQ_FILE" ]] || die "未找到 requirements.txt: ${REQ_FILE}"
command -v conda >/dev/null 2>&1 || die "未找到 conda 命令。请先安装 Miniconda/Anaconda 并加入 PATH（参见 https://docs.conda.io/en/latest/miniconda.html）"
command -v git  >/dev/null 2>&1 || die "未找到 git 命令。请先安装 git（源码安装 trl 需要）"

# ---------------------------------------------------------------------------
# CUDA 版本推断 / 校验
# ---------------------------------------------------------------------------
if [[ "$CUDA_FLAG" == "auto" ]]; then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        die "未检测到 nvidia-smi，无法自动推断 CUDA 版本。请用 --cuda cu121/cu124/cu128 显式指定。"
    fi
    # 从 nvidia-smi 输出尾部提取 "CUDA Version: X.Y"
    DRIVER_CUDA="$(nvidia-smi | grep -oP 'CUDA Version:\s*\K[0-9.]+' | head -1)"
    [[ -n "$DRIVER_CUDA" ]] || die "nvidia-smi 无法解析 CUDA 版本，请用 --cuda 显式指定。"
    case "$DRIVER_CUDA" in
        12.8*)  CUDA_FLAG="cu128" ;;
        12.[4-7]*) CUDA_FLAG="cu124" ;;
        12.[0-3]*) CUDA_FLAG="cu121" ;;
        *) die "不支持的 CUDA 版本 ${DRIVER_CUDA}（当前支持 12.0~12.8+，请用 --cuda 显式指定）" ;;
    esac
    log_info "驱动 CUDA ${DRIVER_CUDA} -> 使用 PyTorch 索引 ${CUDA_FLAG}"
else
    case "$CUDA_FLAG" in
        cu121|cu124|cu128) : ;;
        *) die "--cuda 只支持 cu121 / cu124 / cu128，收到: ${CUDA_FLAG}" ;;
    esac
    log_info "使用用户指定的 PyTorch 索引 ${CUDA_FLAG}"
fi

# ---------------------------------------------------------------------------
# 创建 conda 环境
# ---------------------------------------------------------------------------
# 在非交互 shell 中必须先 source conda.sh 才能使用 conda activate
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    if [[ "$FORCE" == "yes" ]]; then
        log_warn "环境 ${ENV_NAME} 已存在，--force 已指定，删除重建 ..."
        conda env remove -y -n "$ENV_NAME"
    else
        die "conda 环境 ${ENV_NAME} 已存在。如需重建请先执行: conda env remove -n ${ENV_NAME}，或加 --force 参数。"
    fi
fi

log_info "[1/5] 创建 conda 环境 ${ENV_NAME} (Python ${PYTHON_VERSION}) ..."
conda create -y -n "$ENV_NAME" "python=${PYTHON_VERSION}"
conda activate "$ENV_NAME"
python --version

# ---------------------------------------------------------------------------
# 安装 torch（按 CUDA 版本，官方 wheel）
# ---------------------------------------------------------------------------
log_info "[2/5] 安装 torch/torchvision（CUDA ${CUDA_FLAG}）..."
# 与 requirements.txt 中 torch>=2.5.0 / torchvision>=0.20.0 保持一致
pip install "torch>=2.5.0" "torchvision>=0.20.0" \
    --index-url "https://download.pytorch.org/whl/${CUDA_FLAG}"

# ---------------------------------------------------------------------------
# 安装其余依赖
# ---------------------------------------------------------------------------
log_info "[3/5] 安装其余依赖（requirements.txt）..."
pip install -r "$REQ_FILE"

# CUDA 12.8 时，vllm 0.8.x 需要从官方额外索引安装对应 CUDA 版本 wheel
if [[ "$CUDA_FLAG" == "cu128" ]]; then
    log_warn "CUDA 12.8 检测到：从官方额外索引重装 vllm（默认 wheel 对应 CUDA 12.4）..."
    pip install "vllm>=0.8.0" --extra-index-url "https://download.pytorch.org/whl/cu128"
fi

# ---------------------------------------------------------------------------
# 可选：从源码安装 trl（Liger GRPO Loss）
# ---------------------------------------------------------------------------
if [[ "$SOURCE_TRL" == "yes" ]]; then
    log_info "[4/5] 从源码安装 trl（启用 Liger GRPO Loss）..."
    TRL_SRC="$(mktemp -d)/trl"
    git clone --depth 1 https://github.com/huggingface/trl.git "$TRL_SRC"
    # 从源码以可编辑模式安装（含 liger 扩展依赖：liger-kernel 等）
    (cd "$TRL_SRC" && pip install -e ".[liger]")
    rm -rf "$(dirname "$TRL_SRC")"
    log_ok "trl 源码安装完成，可通过 LigerGRPOConfig 使用 Liger GRPO Loss"
else
    log_info "[4/5] 使用 PyPI 版 trl（跳过源码安装，如需 Liger GRPO Loss 请加 --source-trl）"
fi

# ---------------------------------------------------------------------------
# 可选：安装 wandb（训练可视化）
# ---------------------------------------------------------------------------
if [[ "$WITH_WANDB" == "yes" ]]; then
    log_info "安装 wandb（可选，训练可视化）..."
    pip install "wandb>=0.17.0"
    log_ok "wandb 已安装，可配合 --report_to wandb 使用"
else
    log_info "跳过 wandb 安装（如需训练可视化，请加 --with-wandb）"
fi

# ---------------------------------------------------------------------------
# 配置 HuggingFace 镜像源（国内加速，幂等追加到 ~/.bashrc）
# ---------------------------------------------------------------------------
log_info "配置 HuggingFace 镜像源: ${HF_ENDPOINT}"
for line in "export HF_ENDPOINT=${HF_ENDPOINT}" "export HF_HUB_ENABLE_HF_TRANSFER=1"; do
    if ! grep -qF "$line" ~/.bashrc 2>/dev/null; then
        echo "$line" >> ~/.bashrc
        log_ok "已写入 ~/.bashrc: $line"
    else
        log_info "~/.bashrc 已存在: $line（跳过）"
    fi
done
export HF_ENDPOINT="$HF_ENDPOINT"

# ---------------------------------------------------------------------------
# 验证安装
# ---------------------------------------------------------------------------
if [[ "$SKIP_VERIFY" == "yes" ]]; then
    log_ok "安装完成（已跳过验证）。请执行: conda activate ${ENV_NAME}"
    exit 0
fi

log_info "[5/5] 验证安装结果 ..."
pip check || log_warn "pip check 报告依赖冲突，请仔细阅读上方冲突信息"

cat <<'EOF'
------------------------------------------------------------
  库版本核对（PyTorch / HF / 分布式 / 推理加速）
------------------------------------------------------------
EOF
python - <<'PY'
import sys
try:
    import torch
    print(f"  torch          : {torch.__version__}  (CUDA: {torch.version.cuda}, 可用: {torch.cuda.is_available()})")
except Exception as e:
    print(f"  torch          : 导入失败 -> {e}")
for name in ("transformers", "trl", "datasets", "tokenizers", "huggingface_hub",
             "accelerate", "deepspeed", "peft", "bitsandbytes", "vllm",
             "sentencepiece", "einops", "rich", "psutil", "yaml", "tensorboard"):
    try:
        mod = __import__(name)
        print(f"  {name:<16}: {getattr(mod, '__version__', 'N/A')}")
    except Exception as e:
        print(f"  {name:<16}: 导入失败 -> {e}")
try:
    from trl import GRPOTrainer
    print("  GRPOTrainer    : 可用")
except Exception as e:
    print(f"  GRPOTrainer    : 导入失败 -> {e}")
try:
    from trl import LigerGRPOConfig  # noqa: F401
    print("  LigerGRPOConfig: 可用（Liger GRPO Loss 已启用）")
except Exception:
    print("  LigerGRPOConfig: 不可用（如需请加 --source-trl 重新安装）")
PY
if [[ "$WITH_WANDB" == "yes" ]]; then
    python -c "import wandb; print(f'  wandb           : {wandb.__version__}')"
fi

echo "------------------------------------------------------------"
deepspeed --version || echo "  deepspeed --version 执行失败"
log_ok "环境安装完成！"
echo ""
echo "  使用方式："
echo "    conda activate ${ENV_NAME}"
echo "    echo \$HF_ENDPOINT        # 应输出 ${HF_ENDPOINT}"
echo "    bash scripts/launch_train.sh   # 启动正式训练"
echo "    bash scripts/quick_test.sh     # 快速冒烟验证"
if [[ "$WITH_WANDB" == "yes" ]]; then
    echo "    bash scripts/launch_train.sh --report_to wandb   # 使用 wandb 记录训练"
fi
