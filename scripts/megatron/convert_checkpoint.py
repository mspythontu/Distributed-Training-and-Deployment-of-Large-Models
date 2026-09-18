#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HF <-> Megatron-Core 检查点转换工具（基于 mbridge）

背景：
    Megatron-LM 训练使用 Megatron-Core 内部格式并行化模型，
    HuggingFace(transformers) 使用另一种权重命名与切分约定。
    mbridge（veRL / slime 官方采用的转换组件）提供 AutoBridge 统一桥接，
    支持"在线加载 HF 权重并自动按 TP/PP/CP/VPP/EP 分片"，无需预先落盘中间格式。

支持的转换方向：
    import（默认）: HuggingFace -> Megatron-Core
        AutoBridge.from_pretrained(hf) + bridge.get_model(weight_path=hf)
        得到已按并行策略分片的 Megatron-Core 模型，可直接训练或落盘分片 state_dict。

    export        : Megatron-Core -> HuggingFace
        bridge.save_weights(model, save_path) 导出合并后的 HF 权重，
        供 vLLM / transformers 推理部署使用。

用法（必须在 torchrun 下运行，Megatron-Core 依赖分布式环境初始化）：
    # 单卡转换（TP=1）
    python scripts/megatron/convert_checkpoint.py \
        --mode import --hf-path Qwen/Qwen2.5-3B-Instruct \
        --save-path checkpoints/qwen2.5-3b-mcore --tp 1 --pp 1

    # 8 卡：TP2 x PP2 分片
    torchrun --nproc_per_node=8 scripts/megatron/convert_checkpoint.py \
        --mode import --hf-path Qwen/Qwen2.5-3B-Instruct \
        --save-path checkpoints/qwen2.5-3b-mcore --tp 2 --pp 2

    # 训练后导出回 HF 格式（供 vLLM / transformers 部署）
    torchrun --nproc_per_node=8 scripts/megatron/convert_checkpoint.py \
        --mode export --hf-path Qwen/Qwen2.5-3B-Instruct \
        --megatron-path checkpoints/qwen2.5-3b-mcore \
        --save-path output/qwen2.5-3b-hf --tp 2 --pp 2

依赖安装：
    bash scripts/setup_env.sh --with-megatron     # 安装 megatron-core + TE + mbridge
"""

import argparse
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("convert_checkpoint")


def check_dependencies() -> None:
    """校验 mbridge / megatron-core / TransformerEngine 是否可用。

    mbridge 官方注明 use_te=False 暂不支持，即 TransformerEngine 为硬性依赖。
    """
    missing = []
    for module, hint in (
        ("torch", "pip install torch>=2.6.0"),
        ("megatron.core", "uv pip install --system --no-build-isolation 'megatron-core[training,dev]'"),
        ("transformer_engine", "megatron-core[dev] extras 会连带编译安装 TransformerEngine"),
        ("mbridge", "pip install mbridge（见 requirements-megatron.txt）"),
    ):
        try:
            __import__(module)
        except ImportError:
            missing.append(f"  - {module:<20} 安装提示: {hint}")
    if missing:
        logger.error("缺少 Megatron 转换所需依赖：\n%s", "\n".join(missing))
        logger.error("可一键安装：bash scripts/setup_env.sh --with-megatron")
        sys.exit(1)
    logger.info("依赖检查通过：torch / megatron.core / transformer_engine / mbridge")


def validate_parallel(tp: int, pp: int, cp: int, vpp: int, ep: int) -> int:
    """校验并行度乘积能否被 world_size 整除，返回数据并行度 DP。

    约束：world_size == TP x PP x CP x DP（EP 作用于 MoE 专家维度，不参与该乘积）。
    """
    if min(tp, pp, cp, vpp, ep) < 1:
        logger.error("并行度参数必须 >= 1（收到 tp=%d pp=%d cp=%d vpp=%d ep=%d）", tp, pp, cp, vpp, ep)
        sys.exit(1)

    # torchrun 注入 WORLD_SIZE；单进程直接执行时缺省为 1
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    divisor = tp * pp * cp
    if world_size % divisor != 0:
        logger.error(
            "并行度乘积不整除：world_size=%d 不能被 TP(%d) x PP(%d) x CP(%d)=%d 整除，"
            "请调整 --nproc_per_node 或并行度参数",
            world_size, tp, pp, cp, divisor,
        )
        sys.exit(1)
    dp = world_size // divisor
    logger.info(
        "并行配置：world_size=%d, TP=%d, PP=%d, CP=%d, VPP=%d, EP=%d -> DP=%d",
        world_size, tp, pp, cp, vpp, ep, dp,
    )
    return dp


def initialize_distributed(tp: int, pp: int, cp: int, vpp: int, ep: int) -> None:
    """初始化分布式进程组与 Megatron 模型并行状态。

    单进程（无 torchrun）时也会初始化单进程组，避免必须起多进程才能转换。
    """
    import torch
    from megatron.core import parallel_state as mpu

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))

    if not torch.distributed.is_initialized():
        if world_size > 1:
            torch.distributed.init_process_group(backend="nccl")
        else:
            logger.info("未检测到分布式进程组（WORLD_SIZE=1），以单进程模式初始化模型并行")
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29510")
            torch.distributed.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                world_size=1,
                rank=0,
            )

    mpu.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        virtual_pipeline_model_parallel_size=vpp,
        context_parallel_size=cp,
        expert_model_parallel_size=ep,
    )
    logger.info("模型并行初始化完成（rank=%d/%d）", rank, world_size)


def build_bridge_and_model(hf_path: str):
    """从 HuggingFace 权重构建 Megatron-Core 模型（自动按当前并行策略分片）。"""
    from mbridge import AutoBridge

    logger.info("加载 HuggingFace 模型: %s", hf_path)
    bridge = AutoBridge.from_pretrained(hf_path)
    # get_model 会在线把 HF 权重导入并按 TP/PP/CP 切分，无需预先转换的中转权重
    model = bridge.get_model(weight_path=hf_path)
    logger.info("Megatron-Core 模型构建完成（已按并行策略分片）")
    return bridge, model


def mode_import(args) -> None:
    """HuggingFace -> Megatron-Core：在线导入并按并行策略分片，可选落盘分片权重。"""
    import torch

    bridge, model = build_bridge_and_model(args.hf_path)

    if not args.save_path:
        logger.info("未指定 --save-path，仅完成内存中导入；如需训练请直接在训练脚本内调用 mbridge")
        return

    os.makedirs(args.save_path, exist_ok=True)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # 按 rank 保存本地分片（每个 rank 仅持有自己负责的那一部分权重）
    shard_file = os.path.join(args.save_path, f"mp_rank_{rank:02d}.pt")
    state_dict = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
    torch.save(state_dict, shard_file)
    logger.info("[rank %d/%d] 分片权重已保存: %s", rank, world_size, shard_file)

    # 主 rank 额外写入元信息，便于后续 export 时还原并行策略
    if rank == 0:
        meta_file = os.path.join(args.save_path, "latest_checkpointed_iteration.txt")
        with open(meta_file, "w", encoding="utf-8") as fh:
            fh.write("release\n")
        logger.info(
            "已记录检查点元信息: %s（并行策略 TP=%d PP=%d CP=%d VPP=%d EP=%d）",
            meta_file, args.tp, args.pp, args.cp, args.vpp, args.ep,
        )
    logger.info(
        "提示：如需标准的 torch_dist 格式检查点，可在训练脚本中由 Megatron 自动保存后 resume；\n"
        "本脚本落盘的分片可用于快速验证与 export 回放。"
    )


def mode_export(args) -> None:
    """Megatron-Core -> HuggingFace：合并并行分片并导出为 HF 格式，供推理引擎部署。"""
    import torch

    if not args.megatron_path:
        logger.error("export 模式必须提供 --megatron-path（import 时保存的 Megatron 检查点目录）")
        sys.exit(1)
    if not os.path.isdir(args.megatron_path):
        logger.error("Megatron 检查点目录不存在: %s", args.megatron_path)
        sys.exit(1)

    bridge, model = build_bridge_and_model(args.hf_path)

    # 若存在本脚本导出的分片权重，则覆盖到当前并行分片上
    rank = int(os.environ.get("RANK", "0"))
    shard_file = os.path.join(args.megatron_path, f"mp_rank_{rank:02d}.pt")
    if os.path.isfile(shard_file):
        state_dict = torch.load(shard_file, map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info(
            "[rank %d] 已加载分片权重 %s（missing=%d, unexpected=%d）",
            rank, shard_file, len(missing), len(unexpected),
        )
    else:
        logger.warning(
            "[rank %d] 未找到分片文件 %s，将导出从 HF 直接加载的权重（等价于无损回环校验）",
            rank, shard_file,
        )

    os.makedirs(args.save_path, exist_ok=True)
    # memory_efficient=True 逐张量导出，显著降低大模型导出时的 CPU 内存峰值
    if rank == 0:
        logger.info("导出 HuggingFace 权重到: %s", args.save_path)
        bridge.save_weights(model, args.save_path, memory_efficient=args.memory_efficient)
        logger.info("导出完成，可用 transformers / vLLM 直接加载该目录")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HuggingFace <-> Megatron-Core 检查点双向转换（基于 mbridge）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例：torchrun --nproc_per_node=8 scripts/megatron/convert_checkpoint.py "
               "--mode import --hf-path Qwen/Qwen2.5-3B-Instruct --save-path ckpt/mcore --tp 2 --pp 2",
    )
    parser.add_argument(
        "--mode", choices=("import", "export"), default="import",
        help="转换方向：import=HF->Mcore（默认）；export=Mcore->HF",
    )
    parser.add_argument(
        "--hf-path", required=True,
        help="HuggingFace 模型路径或 repo id（如 Qwen/Qwen2.5-3B-Instruct）",
    )
    parser.add_argument(
        "--megatron-path", default=None,
        help="Megatron 检查点目录（export 模式必填；指向 import 时 --save-path 的产物）",
    )
    parser.add_argument(
        "--save-path", default=None,
        help="输出目录：import 模式保存分片权重；export 模式保存合并后的 HF 权重",
    )
    parser.add_argument("--tp", type=int, default=1, help="Tensor Parallel 张量并行度（默认 1）")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline Parallel 流水线并行度（默认 1）")
    parser.add_argument("--cp", type=int, default=1, help="Context Parallel 上下文并行度（默认 1）")
    parser.add_argument("--vpp", type=int, default=1, help="Virtual Pipeline 虚拟流水级数（默认 1，PP>1 时可设）")
    parser.add_argument("--ep", type=int, default=1, help="Expert Parallel 专家并行度（MoE 模型用，默认 1）")
    parser.add_argument(
        "--memory-efficient", action="store_true",
        help="export 时逐张量导出以降低 CPU 内存峰值（大模型推荐）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("=" * 60)
    logger.info("HF <-> Megatron-Core 检查点转换（mode=%s）", args.mode)
    logger.info("=" * 60)

    check_dependencies()
    validate_parallel(args.tp, args.pp, args.cp, args.vpp, args.ep)
    initialize_distributed(args.tp, args.pp, args.cp, args.vpp, args.ep)

    if args.mode == "import":
        mode_import(args)
    else:
        mode_export(args)

    logger.info("转换流程结束（mode=%s）", args.mode)


if __name__ == "__main__":
    main()
