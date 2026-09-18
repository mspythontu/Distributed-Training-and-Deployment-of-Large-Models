#!/usr/bin/env bash
# =============================================================
# 一键停止并清理多机 vLLM 部署
#
# 用法：
#   bash deploy/scripts/stop_all.sh                        # 停止模式 A（多实例 + Nginx）
#   bash deploy/scripts/stop_all.sh --mode ray             # 停止模式 B（Ray 集群）
#   bash deploy/scripts/stop_all.sh --nodes worker1,worker2
#   bash deploy/scripts/stop_all.sh --volumes              # 同时删除匿名卷
#   bash deploy/scripts/stop_all.sh --dry-run              # 只打印命令
#
# 参数说明：
#   --mode NAME      multi-instance（默认）/ ray
#   --nodes LIST     逗号分隔节点列表（默认取 .env 的 NODES）
#   --gateway NODE   Nginx 网关节点（默认 localhost）
#   --remote-dir DIR 远程项目目录（默认与本地同路径，适用于 NFS）
#   --env-file FILE  环境文件（默认 deploy/.env）
#   --volumes        down 时附带 -v 删除卷
#   --dry-run        只打印命令
#   -h / --help      显示本帮助
#
# 说明：本脚本只停止容器，不会删除已下载的镜像与模型缓存（HF_HOME），
#       便于下次快速重启。
# =============================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEPLOY_DIR="${PROJECT_DIR}/deploy"

MODE="multi-instance"
NODES=""
GATEWAY="localhost"
REMOTE_DIR="${PROJECT_DIR}"
ENV_FILE="${DEPLOY_DIR}/.env"
REMOVE_VOLUMES="no"
DRY_RUN="no"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
log_info() { echo -e "${CYAN}[INFO]${NC} $*"; }
log_ok()   { echo -e "${GREEN}[ OK ]${NC} $*"; }
die()      { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

usage() { sed -n '2,/^# =\{10,\}/p' "$0" | sed 's/^# \{0,1\}//' | sed 's/^#//'; exit 0; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)       MODE="$2"; shift 2 ;;
        --nodes)      NODES="$2"; shift 2 ;;
        --gateway)    GATEWAY="$2"; shift 2 ;;
        --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
        --env-file)   ENV_FILE="$2"; shift 2 ;;
        --volumes)    REMOVE_VOLUMES="yes"; shift ;;
        --dry-run)    DRY_RUN="yes"; shift ;;
        -h|--help)    usage ;;
        *) die "未知参数: $1（使用 --help 查看用法）" ;;
    esac
done

case "$MODE" in
    multi-instance|ray) : ;;
    *) die "--mode 只支持 multi-instance / ray，收到: ${MODE}" ;;
esac

[[ -f "$ENV_FILE" ]] || die "未找到环境文件 ${ENV_FILE}"
# shellcheck disable=SC1090
source "$ENV_FILE"
[[ -n "$NODES" ]] || NODES="${NODES:-}"

IFS=',' read -ra NODE_ARR <<< "$NODES"
NODE_ARR=("${NODE_ARR[@]//[[:space:]]/}")
[[ ${#NODE_ARR[@]} -gt 0 && -n "${NODE_ARR[0]}" ]] || die "节点列表为空，请在 .env 设置 NODES 或用 --nodes 指定"

DOWN_FLAG=""
[[ "$REMOVE_VOLUMES" == "yes" ]] && DOWN_FLAG="-v"

run_on() {
    local host="$1"; shift
    local cmd="$*"
    if [[ "$DRY_RUN" == "yes" ]]; then
        echo "  [dry-run] ${host}: ${cmd}"
        return 0
    fi
    if [[ "$host" == "localhost" || "$host" == "127.0.0.1" ]]; then
        bash -lc "$cmd" 2>/dev/null || log_warn "在 ${host} 执行失败（可能容器本就不存在）: ${cmd}"
    else
        ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$host" "$cmd" 2>/dev/null \
            || log_warn "SSH 到 ${host} 执行失败（可能容器本就不存在）: ${cmd}"
    fi
}

log_info "停止模式 ${MODE}，节点: ${NODES}"

if [[ "$MODE" == "ray" ]]; then
    COMPOSE_FILE="deploy/docker-compose.ray.yml"
    for node in "${NODE_ARR[@]}"; do
        log_info "停止 ${node} 上的 Ray 服务 ..."
        run_on "$node" "cd ${REMOTE_DIR} && docker compose -f ${COMPOSE_FILE} --profile head down ${DOWN_FLAG} 2>/dev/null; docker compose -f ${COMPOSE_FILE} --profile worker down ${DOWN_FLAG}"
    done
else
    log_info "停止网关 Nginx（${GATEWAY}）..."
    run_on "$GATEWAY" "cd ${REMOTE_DIR} && docker compose -f deploy/docker-compose.nginx.yml down ${DOWN_FLAG}"
    for node in "${NODE_ARR[@]}"; do
        log_info "停止 ${node} 上的 vLLM 实例 ..."
        run_on "$node" "cd ${REMOTE_DIR} && docker compose -f deploy/docker-compose.yml down ${DOWN_FLAG}"
    done
fi

log_ok "所有节点已停止"
echo ""
echo "  提示：镜像与模型缓存（HF_HOME）保留，如需彻底清理请手动执行："
echo "    docker image rm ${VLLM_IMAGE:-grpo-vllm:v0.8.5}"
