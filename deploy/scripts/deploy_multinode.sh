#!/usr/bin/env bash
# =============================================================
# 多机 vLLM 部署编排脚本（支持两种架构）
#
# 模式一 multi-instance（默认）：多实例 + 负载均衡
#   每个节点起一个独立 vLLM 实例（TP = 节点内 GPU 数），
#   再在网关节点起 Nginx 反向代理分发请求。并发吞吐最高，适合显存装得下的模型（如 3B）。
#
# 模式二 ray：单实例跨节点 TP/PP
#   先由各节点组成 Ray 集群，再启动**一个** vLLM 实例跨节点占用所有 GPU。
#   适合单节点装不下的大模型（70B+），请求无需分发（单一入口）。
#
# 用法：
#   bash deploy/scripts/deploy_multinode.sh --mode multi-instance          # 模式一（默认）
#   bash deploy/scripts/deploy_multinode.sh --mode ray                     # 模式二
#   bash deploy/scripts/deploy_multinode.sh --nodes worker1,worker2        # 覆盖 .env 的节点列表
#   bash deploy/scripts/deploy_multinode.sh --remote-dir /workspace/grpo   # 指定远程项目目录
#   bash deploy/scripts/deploy_multinode.sh --build                        # 部署前先构建镜像
#   bash deploy/scripts/deploy_multinode.sh --dry-run                      # 只打印将执行的命令
#
# 参数说明：
#   --mode NAME           multi-instance（默认）/ ray
#   --nodes LIST          逗号分隔节点列表（默认取 .env 的 NODES）
#   --head NODE           ray 模式的 head 节点（默认取 .env 的 RAY_HEAD_NODE 或首个节点）
#   --gateway NODE        nginx 网关节点（默认本机 localhost）
#   --remote-dir DIR      远程节点上的项目目录（默认与本地项目目录同路径，适用于 NFS）
#   --env-file FILE       环境文件（默认 deploy/.env）
#   --build               部署前先 docker build 构建镜像
#   --dry-run             只打印命令，不真正执行
#   -h / --help           显示本帮助
#
# 前置条件：
#   1. 各节点已安装 Docker 与 nvidia-container-toolkit，且能免密 SSH 登录
#   2. 各节点存在项目目录（共享存储 / 或先同步代码）
#   3. 已 cp deploy/.env.example deploy/.env 并填写 MODEL_PATH / NODES 等
# =============================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEPLOY_DIR="${PROJECT_DIR}/deploy"

MODE="multi-instance"
NODES=""
HEAD_NODE=""
GATEWAY="localhost"
REMOTE_DIR="${PROJECT_DIR}"
ENV_FILE="${DEPLOY_DIR}/.env"
DO_BUILD="no"
DRY_RUN="no"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
log_info() { echo -e "${CYAN}[INFO]${NC} $*"; }
log_ok()   { echo -e "${GREEN}[ OK ]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
die()      { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

usage() { sed -n '2,/^# =\{10,\}/p' "$0" | sed 's/^# \{0,1\}//' | sed 's/^#//'; exit 0; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)        MODE="$2"; shift 2 ;;
        --nodes)       NODES="$2"; shift 2 ;;
        --head)        HEAD_NODE="$2"; shift 2 ;;
        --gateway)     GATEWAY="$2"; shift 2 ;;
        --remote-dir)  REMOTE_DIR="$2"; shift 2 ;;
        --env-file)    ENV_FILE="$2"; shift 2 ;;
        --build)       DO_BUILD="yes"; shift ;;
        --dry-run)     DRY_RUN="yes"; shift ;;
        -h|--help)     usage ;;
        *) die "未知参数: $1（使用 --help 查看用法）" ;;
    esac
done

case "$MODE" in
    multi-instance|ray) : ;;
    *) die "--mode 只支持 multi-instance / ray，收到: ${MODE}" ;;
esac

[[ -f "$ENV_FILE" ]] || die "未找到环境文件 ${ENV_FILE}。请先执行：cp deploy/.env.example deploy/.env"
# shellcheck disable=SC1090
source "$ENV_FILE"
[[ -n "$NODES" ]] || NODES="${NODES:-}"
[[ -n "$HEAD_NODE" ]] || HEAD_NODE="${RAY_HEAD_NODE:-}"

IFS=',' read -ra NODE_ARR <<< "$NODES"
NODE_ARR=("${NODE_ARR[@]//[[:space:]]/}")
[[ ${#NODE_ARR[@]} -gt 0 && -n "${NODE_ARR[0]}" ]] || die "节点列表为空，请在 .env 设置 NODES 或用 --nodes 指定"
[[ -n "$HEAD_NODE" ]] || HEAD_NODE="${NODE_ARR[0]}"

log_info "部署模式: ${MODE}"
log_info "节点列表: ${NODES}"
log_info "远程项目目录: ${REMOTE_DIR}"

# ---------------------------------------------------------------------------
# 工具：远程执行（本机用 bash -lc 直接执行，避免无谓 SSH）
# ---------------------------------------------------------------------------
run_on() {
    local host="$1"; shift
    local cmd="$*"
    if [[ "$DRY_RUN" == "yes" ]]; then
        echo "  [dry-run] ${host}: ${cmd}"
        return 0
    fi
    if [[ "$host" == "localhost" || "$host" == "127.0.0.1" ]]; then
        bash -lc "$cmd" || die "在 ${host} 执行失败: ${cmd}"
    else
        ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$host" "$cmd" \
            || die "SSH 到 ${host} 执行失败: ${cmd}"
    fi
}

# ---------------------------------------------------------------------------
# 前置检查：SSH 连通性 + Docker 可用性
# ---------------------------------------------------------------------------
log_info "前置检查：SSH 连通性与 Docker 可用性 ..."
if [[ "$DRY_RUN" == "no" ]]; then
    for node in "${NODE_ARR[@]}"; do
        if [[ "$node" == "localhost" || "$node" == "127.0.0.1" ]]; then
            log_ok "${node}（本机）跳过 SSH 检查"
            continue
        fi
        ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$node" "echo ok" >/dev/null 2>&1 \
            || die "无法 SSH 到 ${node}。请配置免密登录：ssh-copy-id ${node}"
        ssh -o StrictHostKeyChecking=no "$node" "command -v docker >/dev/null" 2>/dev/null \
            || die "${node} 上未找到 docker 命令"
        log_ok "${node} SSH + Docker 正常"
    done
else
    log_warn "--dry-run：跳过 SSH / Docker 检查"
fi

# ---------------------------------------------------------------------------
# 可选：构建镜像（各节点本地构建；生产环境建议推到镜像仓库后拉取）
# ---------------------------------------------------------------------------
if [[ "$DO_BUILD" == "yes" ]]; then
    log_info "构建镜像（各节点本地构建）..."
    for node in "${NODE_ARR[@]}"; do
        run_on "$node" "cd ${REMOTE_DIR} && docker build -t ${VLLM_IMAGE:-grpo-vllm:v0.8.5} -f deploy/Dockerfile ."
    done
    log_ok "镜像构建完成"
fi

# ---------------------------------------------------------------------------
# 模式一：多实例 + 负载均衡
# ---------------------------------------------------------------------------
if [[ "$MODE" == "multi-instance" ]]; then
    log_info "[模式 A] 在各节点启动独立 vLLM 实例 ..."
    for node in "${NODE_ARR[@]}"; do
        run_on "$node" "cd ${REMOTE_DIR} && docker compose -f deploy/docker-compose.yml up -d"
        log_ok "${node} vLLM 实例已启动（端口 ${NODE_PORT:-8000}）"
    done

    log_info "生成 Nginx 负载均衡配置 ..."
    if [[ "$DRY_RUN" == "yes" ]]; then
        echo "  [dry-run] bash ${DEPLOY_DIR}/scripts/gen_nginx_conf.sh"
    else
        bash "${DEPLOY_DIR}/scripts/gen_nginx_conf.sh"
    fi

    log_info "在网关节点(${GATEWAY}) 启动 Nginx ..."
    run_on "$GATEWAY" "cd ${REMOTE_DIR} && docker compose -f deploy/docker-compose.nginx.yml up -d"

    log_ok "模式 A 部署完成"
    echo ""
    echo "  验证："
    echo "    curl http://${GATEWAY}:${NGINX_PORT:-8080}/v1/models"
    echo "    bash deploy/scripts/health_check.sh"
    echo "    python deploy/scripts/client_example.py --base-url http://${GATEWAY}:${NGINX_PORT:-8080}/v1"
    exit 0
fi

# ---------------------------------------------------------------------------
# 模式二：Ray 集群 + 单实例跨节点 TP/PP
# ---------------------------------------------------------------------------
log_info "[模式 B] 启动 Ray head（${HEAD_NODE}）..."
run_on "$HEAD_NODE" "cd ${REMOTE_DIR} && docker compose -f deploy/docker-compose.ray.yml --profile head up -d"

for node in "${NODE_ARR[@]}"; do
    [[ "$node" == "$HEAD_NODE" ]] && continue
    log_info "启动 Ray worker（${node}）..."
    run_on "$node" "cd ${REMOTE_DIR} && docker compose -f deploy/docker-compose.ray.yml --profile worker up -d"
done

TOTAL_PARALLEL=$(( ${TP_SIZE:-1} * ${PP_SIZE:-1} ))
log_warn "请确认 TP(${TP_SIZE:-1}) x PP(${PP_SIZE:-1}) = ${TOTAL_PARALLEL} 等于集群总 GPU 数，否则 vLLM 会启动失败"

log_ok "模式 B 部署完成"
echo ""
echo "  验证："
echo "    Ray 面板: http://${HEAD_NODE}:${RAY_DASHBOARD_PORT:-8265}"
echo "    curl http://${HEAD_NODE}:${PORT:-8000}/v1/models"
echo "    bash deploy/scripts/health_check.sh --mode ray"
