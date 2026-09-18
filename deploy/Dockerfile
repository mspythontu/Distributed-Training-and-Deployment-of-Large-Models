# =============================================================
# vLLM OpenAI 兼容服务镜像（多机推理部署用）
#
# 基础镜像：vLLM 官方 vllm-openai（已内置 CUDA / vLLM / 依赖，开箱即用）
# 版本可通过 --build-arg 指定，应与项目 requirements.txt 的 vllm>=0.8.0 对齐
#
# 构建：
#   docker build -t grpo-vllm:v0.8.5 --build-arg VLLM_VERSION=v0.8.5 -f deploy/Dockerfile .
#
# 运行（单机多卡，TP=2）：
#   docker run --gpus all --ipc=host --shm-size=16g -p 8000:8000 \
#     -e MODEL_PATH=Qwen/Qwen2.5-3B-Instruct -e TP_SIZE=2 \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     grpo-vllm:v0.8.5
#
# 说明：
#   - --ipc=host 与 --shm-size 必须设置：vLLM 多进程（TP）依赖共享内存传递张量，
#     默认的 64MB /dev/shm 会导致启动时报 "bus error" 或卡死
#   - 挂载 HF 缓存目录可避免每个节点重复下载模型（几十 GB）
#   - 国内网络建议设置 -e HF_ENDPOINT=https://hf-mirror.com
# =============================================================

ARG VLLM_VERSION=v0.8.5
FROM vllm/vllm-openai:${VLLM_VERSION}

# 运行时默认参数（全部可被 docker run -e / compose environment 覆盖）
ENV MODEL_PATH="Qwen/Qwen2.5-3B-Instruct" \
    SERVED_MODEL_NAME="qwen2.5-3b" \
    TP_SIZE=1 \
    PP_SIZE=1 \
    DISTRIBUTED_BACKEND="mp" \
    HOST="0.0.0.0" \
    PORT=8000 \
    MAX_MODEL_LEN=4096 \
    GPU_MEMORY_UTILIZATION=0.9 \
    DTYPE="auto" \
    MAX_NUM_SEQS=256 \
    API_KEY="" \
    EXTRA_ARGS="" \
    HF_ENDPOINT="https://hf-mirror.com"

# 跨节点并行（模式 B）依赖 Ray 作为分布式执行后端。
# 官方镜像通常已内置 ray；若基础镜像未包含，此步会补装（已安装时 pip 会快速跳过）。
RUN pip install --no-cache-dir "ray>=2.40"

# 容器入口：统一的参数校验 + vllm serve 组装（支持 mp / ray 两种后端）
COPY deploy/scripts/start_vllm.sh /opt/start_vllm.sh
RUN chmod +x /opt/start_vllm.sh

EXPOSE 8000

# 健康检查：vLLM 的 OpenAI 服务提供 /health 端点
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${PORT}/health || exit 1

ENTRYPOINT ["/opt/start_vllm.sh"]
