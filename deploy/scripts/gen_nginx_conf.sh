#!/usr/bin/env bash
# =============================================================
# 生成 Nginx 负载均衡配置（模式 A：多实例 + 负载均衡）
#
# 功能：
#   1. 读取 deploy/.env 中的 NODES / NODE_PORT / NGINX_PORT / NGINX_LB_METHOD
#   2. 把节点列表渲染成 nginx upstream 的 server 条目
#   3. 基于 deploy/nginx/nginx.conf.template 生成 deploy/nginx/nginx.conf
#
# 用法：
#   bash deploy/scripts/gen_nginx_conf.sh                 # 用 deploy/.env 生成
#   bash deploy/scripts/gen_nginx_conf.sh --print         # 只打印不落盘
#
# 说明：
#   - 分发策略：least_conn（最小连接，推荐）/ ip_hash（会话保持）/ round_robin（轮询）
#   - 每个 upstream server 带 max_fails + fail_timeout，节点故障时自动摘除
#   - 生成后需重启 Nginx 容器使配置生效：
#     docker compose -f deploy/docker-compose.nginx.yml restart
# =============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${DEPLOY_DIR}/.env"
TEMPLATE="${DEPLOY_DIR}/nginx/nginx.conf.template"
CONF_OUT="${DEPLOY_DIR}/nginx/nginx.conf"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
log_info() { echo -e "${CYAN}[INFO]${NC} $*"; }
log_ok()   { echo -e "${GREEN}[ OK ]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
die()      { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

PRINT_ONLY="no"
[[ "${1:-}" == "--print" ]] && PRINT_ONLY="yes"

# ---------------------------------------------------------------------------
# 读取配置
# ---------------------------------------------------------------------------
[[ -f "$ENV_FILE" ]] || die "未找到 ${ENV_FILE}。请先执行：cp deploy/.env.example deploy/.env 并填写 NODES"
[[ -f "$TEMPLATE" ]] || die "未找到模板文件: ${TEMPLATE}"
# shellcheck disable=SC1090
source "$ENV_FILE"

NODES="${NODES:-}"
NODE_PORT="${NODE_PORT:-8000}"
NGINX_PORT="${NGINX_PORT:-8080}"
LB_METHOD="${NGINX_LB_METHOD:-least_conn}"

[[ -n "$NODES" ]] || die "deploy/.env 中 NODES 为空。请填写节点列表，例如：NODES=worker1,worker2"

# ---------------------------------------------------------------------------
# 分发策略 -> nginx 指令
# ---------------------------------------------------------------------------
case "$LB_METHOD" in
    least_conn)  LB_DIRECTIVE="        least_conn;" ;;
    ip_hash)     LB_DIRECTIVE="        ip_hash;" ;;
    round_robin) LB_DIRECTIVE="" ;;   # nginx 默认即为轮询，无需指令
    *) die "NGINX_LB_METHOD 只支持 least_conn / ip_hash / round_robin，收到: ${LB_METHOD}" ;;
esac

# ---------------------------------------------------------------------------
# 渲染 upstream 条目
# ---------------------------------------------------------------------------
SERVERS=""
IFS=',' read -ra NODE_ARR <<< "$NODES"
NODE_COUNT=0
for node in "${NODE_ARR[@]}"; do
    node="$(echo -n "$node" | tr -d '[:space:]')"
    [[ -n "$node" ]] || continue
    SERVERS+="        server ${node}:${NODE_PORT} max_fails=3 fail_timeout=30s;"$'\n'
    NODE_COUNT=$(( NODE_COUNT + 1 ))
done
(( NODE_COUNT > 0 )) || die "NODES 解析后没有有效节点（收到: ${NODES}）"

log_info "节点数 ${NODE_COUNT}，端口 ${NODE_PORT}，分发策略 ${LB_METHOD}，监听 ${NGINX_PORT}"

# ---------------------------------------------------------------------------
# 渲染模板（占位符替换用 Python 处理，避免 sed 转义问题）
# ---------------------------------------------------------------------------
export LB_DIRECTIVE SERVERS NGINX_PORT

if [[ "$PRINT_ONLY" == "yes" ]]; then
python3 - "$TEMPLATE" <<'PY'
import os, sys
with open(sys.argv[1], encoding="utf-8") as fh:
    content = fh.read()
content = (content
    .replace("__LB_METHOD__", os.environ["LB_DIRECTIVE"])
    .replace("__UPSTREAM_SERVERS__", os.environ["SERVERS"].rstrip("\n"))
    .replace("__NGINX_PORT__", os.environ["NGINX_PORT"]))
print(content)
PY
    exit 0
fi

mkdir -p "$(dirname "$CONF_OUT")"
python3 - "$TEMPLATE" "$CONF_OUT" <<'PY'
import os, sys
src, dst = sys.argv[1], sys.argv[2]
with open(src, encoding="utf-8") as fh:
    content = fh.read()
content = (content
    .replace("__LB_METHOD__", os.environ["LB_DIRECTIVE"])
    .replace("__UPSTREAM_SERVERS__", os.environ["SERVERS"].rstrip("\n"))
    .replace("__NGINX_PORT__", os.environ["NGINX_PORT"]))
with open(dst, "w", encoding="utf-8") as fh:
    fh.write(content)
print(f"  已生成: {dst}")
PY

log_ok "Nginx 配置生成完成（${NODE_COUNT} 个后端节点）"
echo ""
echo "  下一步："
echo "    docker compose -f deploy/docker-compose.nginx.yml up -d"
echo "    curl http://<网关IP>:${NGINX_PORT}/v1/models"
