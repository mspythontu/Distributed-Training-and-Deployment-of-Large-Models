# 多机 vLLM 推理部署

用 Docker 把训练好的模型（或任意 HuggingFace 模型）部署成**多机多卡推理服务**，
对外提供 OpenAI 兼容 API。支持两种架构，按模型规模与并发需求选择。

> 注意：项目训练时也会用到 vLLM（GRPOTrainer 内嵌 rollout 引擎），
> 但那是**训练进程内部的生成加速**；本目录是**独立的在线推理服务**，两者互不干扰。

---

## 目录结构

```
deploy/
├── Dockerfile                       # vLLM OpenAI 服务镜像（基于官方 vllm-openai）
├── .env.example                     # 环境变量模板（先 cp 成 .env 再改）
├── docker-compose.yml               # 模式 A 单节点单元：一个独立 vLLM 实例
├── docker-compose.nginx.yml         # 模式 A 网关：Nginx 负载均衡
├── docker-compose.ray.yml           # 模式 B：Ray 集群 + 单实例跨节点 TP/PP
├── nginx/
│   ├── nginx.conf.template          # 负载均衡配置模板（含长超时、流式不缓冲）
│   └── nginx.conf                   # 由 gen_nginx_conf.sh 自动生成（勿手改）
└── scripts/
    ├── start_vllm.sh                # 容器入口：参数校验 + 组装 vllm serve
    ├── deploy_multinode.sh          # 多机编排：SSH 批量拉起两种模式
    ├── stop_all.sh                  # 一键停止与清理
    ├── gen_nginx_conf.sh            # 按节点列表生成 nginx.conf
    ├── health_check.sh              # 健康检查（服务/模型/GPU/容器/Ray）
    ├── client_example.py            # OpenAI 兼容调用示例（含流式、并发）
    └── benchmark.py                 # 并发压测（吞吐、TTFT、P50/P95/P99）
```

---

## 两种架构怎么选

| 维度 | 模式 A：多实例 + 负载均衡 | 模式 B：单实例跨节点 TP/PP |
|---|---|---|
| 适用场景 | 单节点能装下模型（如 3B），追求**高并发吞吐** | 单节点装不下（70B+），必须**跨节点切分** |
| 并行方式 | 每节点独立 TP（节点内），实例间无通信 | TP/PP 铺满集群所有 GPU |
| 入口 | Nginx 网关统一分发 | 单入口（head 节点） |
| 横向扩容 | 加节点即可线性提升吞吐 | 需重算 TP × PP |
| 网络要求 | 常规以太网即可 | 跨节点 TP 需 InfiniBand/RoCE |
| 启动参数 | `--mode multi-instance` | `--mode ray` |

> **本项目（Qwen2.5-3B）推荐模式 A**：3B 模型单卡即可容纳，
> 多实例能把并发吞吐做到接近线性提升。

---

## 快速开始

### 0. 前置条件

- 各节点已安装 Docker 与 [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- 部署机到各节点 **免密 SSH**（`ssh-copy-id <node>`）
- 各节点存在项目目录（共享存储，或先同步代码）
- 已准备 `deploy/.env`

```bash
cp deploy/.env.example deploy/.env
# 至少修改：MODEL_PATH、NODES（模式 A）、TP_SIZE/PP_SIZE（模式 B）
```

### 1. 构建镜像

```bash
docker build -t grpo-vllm:v0.8.5 --build-arg VLLM_VERSION=v0.8.5 -f deploy/Dockerfile .
# 多机时让每个节点都构建（或推到镜像仓库后拉取）：
bash deploy/scripts/deploy_multinode.sh --build --dry-run   # 先看会做什么
```

### 2. 部署

```bash
# 模式 A（多实例 + 负载均衡，推荐）
bash deploy/scripts/deploy_multinode.sh --mode multi-instance

# 模式 B（Ray 集群，单实例跨节点）
bash deploy/scripts/deploy_multinode.sh --mode ray
```

脚本会：校验 SSH/Docker →（可选）构建镜像 → 各节点拉起容器 → 生成 Nginx 配置 → 启动网关。

### 3. 验证

```bash
bash deploy/scripts/health_check.sh                     # 健康检查
bash deploy/scripts/health_check.sh --verbose           # 附加 GPU / 容器明细
python deploy/scripts/client_example.py --list-models   # 查看已加载模型
python deploy/scripts/client_example.py --stream        # 流式对话
```

---

## 模式 A 详解（多实例 + 负载均衡）

**架构**：每个 GPU 节点跑一个 vLLM 容器（`TP_SIZE` = 该节点 GPU 数），
Nginx 网关按策略把请求分发到各节点。

```bash
# 1) 配置 .env
#    NODES=worker1,worker2,worker3
#    TP_SIZE=8              # 每节点 8 卡，节点内张量并行
#    NGINX_LB_METHOD=least_conn

# 2) 一键部署
bash deploy/scripts/deploy_multinode.sh --mode multi-instance

# 3) 请求打网关（不要直连单节点）
curl http://<网关IP>:8080/v1/models
python deploy/scripts/client_example.py --base-url http://<网关IP>:8080/v1
```

分发策略（`.env` 的 `NGINX_LB_METHOD`）：

| 策略 | 说明 |
|---|---|
| `least_conn` | 最小连接，长请求场景最均衡（**推荐**） |
| `ip_hash` | 按客户端 IP 会话保持，适合需要命中同一实例缓存的场景 |
| `round_robin` | 轮询，请求耗时接近时可用 |

Nginx 配置已预设：
- `proxy_read_timeout 3600s`：长文本生成不中断
- `proxy_buffering off`：流式输出（SSE）必须关闭缓冲，否则客户端收不到增量 token
- `max_fails=3 fail_timeout=30s`：节点故障自动摘除

---

## 模式 B 详解（Ray 集群 + 单实例跨节点）

**架构**：各节点组成 Ray 集群，vLLM 在 head 节点起**一个**实例，
通过 `--distributed-executor-backend ray` 跨节点占用所有 GPU。

```bash
# 1) 配置 .env
#    RAY_HEAD_NODE=worker1
#    TP_SIZE=8  PP_SIZE=2      # 满足 TP x PP = 集群总 GPU 数（例如 2 节点 x 8 卡）
#    DISTRIBUTED_BACKEND=ray

# 2) 一键部署（head 起 ray-head+vllm，其余起 ray-worker）
bash deploy/scripts/deploy_multinode.sh --mode ray

# 3) 验证
#    Ray 面板: http://worker1:8265
#    curl http://worker1:8000/v1/models
```

**关键约束**：

- `TP × PP` 必须等于集群 GPU 总数，否则 vLLM 启动失败（脚本会提示核对）
- Ray 容器内 `--num-gpus` 要如实填写（`HEAD_GPUS` / `WORKER_GPUS`），否则资源统计错误
- 仅有以太网时，**优先加大 PP、减小跨节点 TP**（TP 的 all-reduce 通信量远高于 PP 的点对点）

---

## 客户端调用

```bash
# 单次对话
python deploy/scripts/client_example.py --base-url http://<网关IP>:8080/v1

# 流式（逐 token 打印）
python deploy/scripts/client_example.py --stream

# 并发 8 路
python deploy/scripts/client_example.py --concurrent 8

# 带鉴权（与 .env 的 API_KEY 对应）
python deploy/scripts/client_example.py --api-key sk-xxx
```

脚本使用标准库 `urllib`，无需安装 `openai` SDK。

---

## 压测

```bash
python deploy/scripts/benchmark.py --base-url http://<网关IP>:8080/v1 \
    --concurrency 16 --requests 100 --max-tokens 256

# 流式压测（可测首 token 延迟 TTFT）
python deploy/scripts/benchmark.py --concurrency 16 --requests 100 --stream

# 导出 JSON 结果，便于对比不同配置
python deploy/scripts/benchmark.py --concurrency 32 --requests 200 \
    --output deploy/bench_result.json
```

输出指标：请求吞吐（req/s）、输出 token 吞吐（tok/s）、端到端延迟 avg/P50/P95/P99、
TTFT（流式）、成功率。

**调优顺序**：先确认单并发正常 → 逐步加大并发 → 观察 P99 是否陡增 → 找到拐点即为合理并发上限。

---

## 常用参数调优

| 参数 | 影响 | 建议 |
|---|---|---|
| `GPU_MEMORY_UTILIZATION` | 单实例显存占用比例 | 独占卡 0.9；多实例同卡部署需下调（如 0.45） |
| `MAX_MODEL_LEN` | 最大上下文长度 | 超过会报错；越长 KV cache 占显存越多 |
| `MAX_NUM_SEQS` | 单实例最大并发序列 | 并发上不去时调大，显存不足时调小 |
| `TP_SIZE` | 张量并行度 | 模式 A 设为单节点卡数；跨节点 TP 需高速网络 |
| `EXTRA_ARGS` | 追加参数 | 可加 `--enable-prefix-caching`（前缀缓存，多轮对话提速） |

---

## 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| 容器启动报 `bus error` 或卡在加载 | `/dev/shm` 太小。已设置 `ipc: host` + `shm_size: 16g`，勿删 |
| `unknown runtime nvidia` | 未装 nvidia-container-toolkit，或 compose 版本过旧（需 v2 语法支持 `devices`） |
| 网关返回 502 | 后端未就绪（模型加载 1~3 分钟）；用 `health_check.sh` 看节点状态 |
| 流式输出卡顿/一次性返回 | Nginx 缓冲未关。确认 `proxy_buffering off` 生效 |
| 跨节点 TP 极慢 | 以太网跑张量并行。改用模式 A，或加大 PP 减少跨节点 TP |
| 模型下载慢/超时 | 设置 `HF_ENDPOINT=https://hf-mirror.com`，或提前下载到挂载的 `HF_HOME` |
| vLLM 报 TP × PP 与 GPU 数不匹配 | 核对 `.env` 的 `TP_SIZE`/`PP_SIZE` 与集群实际卡数 |
