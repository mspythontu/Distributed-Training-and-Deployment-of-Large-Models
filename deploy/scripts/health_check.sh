#!/usr/bin/env bash
# =============================================================
# 多机 vLLM 部署健康检查
#
# 检查项：
#   1. 服务存活：/health 端点
#   2. 模型加载：/v1/models 返回的模型列表与 SERVED_MODEL_NAME 是否一致
#   3. GPU 状态：各节点 nvidia-smi（显存占用 / 利用率 / 温度）
#   4. 容器状态：docker ps 中 vLLM / Nginx / Ray 容器是否 Up
#   5. Ray 模式额外：ray status 的集群 GPU 资源统计
#
# 用法：
#   bash deploy/scripts/health_check.sh                      # 模式 A（默认）
#   bash deploy/scripts/health_check.sh --mode ray           # 模式 B
#   bash deploy/scripts/health_check.sh --endpoint http://1.2.3.4:8080   # 指定统一入口
#   bash deploy/scripts/health_check.sh --nodes worker1,worker2
#   bash deploy/scripts/health_check.sh --verbose            # 额外打印 GPU / 容器详情
#
# 参数说明：
#   --mode NAME       multi-instance（默认）/ ray
#   --endpoint URL    统一入口地址（模式 A 默认为网关，模式 B 默认为 head 节点）
#   --nodes LIST      逗号分隔节点列表（默认取 .env 的 NODES）
#   --env-file FILE   环境文件（默认 deploy/.env）
#   --timeout SEC     单次 HTTP 探测超时（默认 5 秒）
#   --verbose         打印 GPU 与容器明细
#   -h / --help       显示本帮助
#
# 退出码：全部检查通过为 0，任一检查失败为 1（便于接入监控告警）
# =============================================================
set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEPLOY_DIR="${PROJECT_DIR}/deploy"

MODE="multi-instance"
ENDPOINT=""
NODES=""
ENV_FILE="${DEPLOY_DIR}/.env"
TIMEOUT=5
VERBOSE="no"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
log_info() { echo -e "${CYAN}[INFO]${NC} $*"; }
log_ok()   { echo -e "${GREEN}[ OK ]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_fail() { echo -e "${RED}[FAIL]${NC} $*"; }

usage() { sed -n '2,/^# =\{10,\}/p' "$0" | sed 's/^# \{0,1\}//' | sed 's/^#//'; exit 0; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)      MODE="$2"; shift 2 ;;
        --endpoint)  ENDPOINT="$2"; shift 2 ;;
        --nodes)     NODES="$2"; shift 2 ;;
        --env-file)  ENV_FILE="$2"; shift 2 ;;
        --timeout)   TIMEOUT="$2"; shift 2 ;;
        --verbose)   VERBOSE="yes"; shift ;;
        -h|--help)   usage ;;
        *) echo -e "${RED}[ERROR]${NC} 未知参数: $1" >&2; exit 1 ;;
    esac
done

[[ -f "$ENV_FILE" ]] || { log_fail "未找到环境文件 ${ENV_FILE}"; exit 1; }
# shellcheck disable=SC1090
source "$ENV_FILE"
[[ -n "$NODES" ]] || NODES="${NODES:-}"

IFS=',' read -ra NODE_ARR <<< "$NODES"
NODE_ARR=("${NODE_ARR[@]//[[:space:]]/}")

FAILED=0

echo "============================================================"
echo "  多机 vLLM 部署健康检查（模式: ${MODE}）"
echo "============================================================"

# ---------------------------------------------------------------------------
# 1. 统一入口健康检查
# ---------------------------------------------------------------------------
if [[ -z "$ENDPOINT" ]]; then
    if [[ "$MODE" == "ray" ]]; then
        HEAD="${RAY_HEAD_NODE:-${NODE_ARR[0]:-localhost}}"
        ENDPOINT="http://${HEAD}:${PORT:-8000}"
    else
        ENDPOINT="http://localhost:${NGINX_PORT:-8080}"
    fi
fi

log_info "探测统一入口: ${ENDPOINT}"
if curl -fsS --max-time "$TIMEOUT" "${ENDPOINT}/health" >/dev/null 2>&1; then
    log_ok "健康检查通过: ${ENDPOINT}/health"
else
    log_fail "健康检查失败: ${ENDPOINT}/health（服务未就绪或地址不可达）"
    FAILED=1
fi

# ---------------------------------------------------------------------------
# 2. 模型列表校验
# ---------------------------------------------------------------------------
MODELS_JSON="$(curl -fsS --max-time "$TIMEOUT" "${ENDPOINT}/v1/models" 2>/dev/null || echo "")"
if [[ -n "$MODELS_JSON" ]]; then
    EXPECTED="${SERVED_MODEL_NAME:-}"
    if [[ -n "$EXPECTED" ]] && echo "$MODELS_JSON" | grep -q "\"${EXPECTED}\""; then
        log_ok "模型已加载: ${EXPECTED}"
    elif [[ -n "$EXPECTED" ]]; then
        log_warn "模型 ${EXPECTED} 未在 /v1/models 中出现，实际返回: ${MODELS_JSON:0:200}"
        FAILED=1
    else
        log_ok "模型列表: ${MODELS_JSON:0:200}"
    fi
else
    log_fail "无法获取模型列表: ${ENDPOINT}/v1/models"
    FAILED=1
fi

# ---------------------------------------------------------------------------
# 3. 各节点检查（服务 / 容器 / GPU）
# ---------------------------------------------------------------------------
remote() {
    local host="$1"; shift
    if [[ "$host" == "localhost" || "$host" == "127.0.0.1" ]]; then
        bash -lc "$*" 2>/dev/null
    else
        ssh -o StrictHostKeyChecking=no -o ConnectTimeout=8 "$host" "$*" 2>/dev/null
    fi
}

for node in "${NODE_ARR[@]}"; do
    [[ -n "$node" ]] || continue
    echo ""
    log_info "---- 节点 ${node} ----"

    # 3.1 节点本地服务健康（仅模式 A 有意义；模式 B 只有 head 有服务）
    if [[ "$MODE" == "multi-instance" ]]; then
        if curl -fsS --max-time "$TIMEOUT" "http://${node}:${NODE_PORT:-8000}/health" >/dev/null 2>&1; then
            log_ok "${node}:${NODE_PORT:-8000} 服务健康"
        else
            log_fail "${node}:${NODE_PORT:-8000} 服务不可达"
            FAILED=1
        fi
    fi

    # 3.2 容器状态
    CONTAINERS="$(remote "$node" "docker ps --filter name=vllm --filter name=ray --format '{{.Names}}\t{{.Status}}'")"
    if [[ -n "$CONTAINERS" ]]; then
        log_ok "容器运行中:"
        echo "$CONTAINERS" | sed 's/^/         /'
    else
        log_warn "未发现运行中的 vllm / ray 容器"
        FAILED=1
    fi

    # 3.3 GPU 状态
    GPU_INFO="$(remote "$node" "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader")"
    if [[ -n "$GPU_INFO" ]]; then
        log_ok "GPU 状态:"
        echo "$GPU_INFO" | sed 's/^/         /'
    else
        log_warn "未能获取 GPU 信息（nvidia-smi 不可用或 SSH 失败）"
        FAILED=1
    fi
done

# ---------------------------------------------------------------------------
# 4. Ray 模式：集群资源统计
# ---------------------------------------------------------------------------
if [[ "$MODE" == "ray" && ${#NODE_ARR[@]} -gt 0 ]]; then
    HEAD="${RAY_HEAD_NODE:-${NODE_ARR[0]}}"
    echo ""
    log_info "---- Ray 集群状态（${HEAD}）----"
    RAY_STATUS="$(remote "$HEAD" "docker exec ray-head ray status 2>/dev/null | head -30" || echo "")"
    if [[ -n "$RAY_STATUS" ]]; then
        echo "$RAY_STATUS" | sed 's/^/     /'
    else
        log_warn "未能获取 ray status（head 容器名是否为 ray-head？）"
    fi
fi

# ---------------------------------------------------------------------------
# 5. verbose 明细
# ---------------------------------------------------------------------------
if [[ "$VERBOSE" == "yes" ]]; then
    for node in "${NODE_ARR[@]}"; do
        [[ -n "$node" ]] || continue
        echo ""
        log_info "---- ${node} 明细 ----"
        remote "$node" "nvidia-smi" | sed 's/^/     /'
        remote "$node" "docker ps -a --filter name=vllm --filter name=ray --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'" | sed 's/^/     /'
    done
fi

echo ""
echo "============================================================"
if [[ $FAILED -eq 0 ]]; then
    log_ok "全部检查通过"
    exit 0
fi
log_fail "存在检查项未通过（详见上方 FAIL/WARN）"
exit 1
