#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GRPO 训练主脚本：在 Countdown-Tasks 上用 DeepSpeed ZeRO-3 + vLLM 训练 Qwen2.5-3B-Instruct。

功能：
1. 用 transformers 加载基础模型（bf16 + flash_attention_2）
2. 通过命令行参数选择全参数微调或 QLoRA（4bit 量化 + LoRA）模式
3. 加载 data/train.jsonl（{"prompt", "target", "solution"} 格式，由 prepare_countdown_data.py 生成）
4. 导入 scripts/reward_functions.py 的 combined_reward 作为奖励函数
5. 使用 TRL GRPOTrainer 训练，关键超参全部由 recipes/*.yaml 配置文件驱动
6. 训练结束自动保存最终模型到 output/ 目录

配置文件驱动规则：
- recipes YAML 中 GRPOConfig 字段直接透传给 GRPOConfig；扩展字段（model_id、lora_*、bnb_* 等）用于模式判定与模型初始化
- 命令行参数优先级高于配置文件，便于快速调参

用法示例：
    # 1) 先准备数据（未运行过时）
    python scripts/prepare_countdown_data.py --max_samples 5000

    # 2) 全参数微调（num_processes = GPU 数量 - 1，最后一块留给 vLLM 推理）
    accelerate launch --config_file configs/accelerate_configs/deepspeed_zero3.yaml \
        --num_processes $((nproc - 1)) \
        scripts/run_grpo.py --config recipes/grpo-qwen-2.5-3b-countdown.yaml

    # 3) QLoRA（4bit + LoRA）
    accelerate launch --config_file configs/accelerate_configs/deepspeed_zero3.yaml \
        --num_processes $((nproc - 1)) \
        scripts/run_grpo.py --config recipes/grpo-qwen-2.5-3b-countdown-qlora.yaml --mode qlora

    # 4) 快速调参（CLI 覆盖配置）
    accelerate launch --config_file configs/accelerate_configs/deepspeed_zero3.yaml \
        --num_processes $((nproc - 1)) \
        scripts/run_grpo.py --config recipes/grpo-qwen-2.5-3b-countdown.yaml \
        --learning_rate 1e-6 --max_steps 100 --output_dir output/test --report_to wandb

说明：
- 训练侧使用 nproc - 1 张 GPU，vLLM 生成后端（vllm_device="auto"）自动占用最后一块空闲 GPU
- QLoRA 模式下 vLLM 从 model_id 加载未量化的 bf16 基础模型用于生成，训练侧为 4bit + LoRA
- 4bit 量化与 DeepSpeed ZeRO-3 的参数/优化器 offload 存在兼容性限制，若报错可关闭
  offload 或改用全参数微调模式
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields as dataclass_fields
from pathlib import Path

import yaml
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# 使 scripts/ 目录下的模块（reward_functions）可被直接导入
sys.path.insert(0, str(Path(__file__).resolve().parent))

from reward_functions import combined_reward  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_TRAIN_FILE = PROJECT_ROOT / "data" / "train.jsonl"
DEFAULT_EVAL_FILE = PROJECT_ROOT / "data" / "eval.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "grpo-countdown"

# Qwen2.5 全模块 LoRA target：注意力四投影 + 前馈门控/上/下投影
DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="GRPO training for Qwen2.5-3B-Instruct on Countdown-Tasks (DeepSpeed ZeRO-3 + vLLM)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="训练配置 recipes/*.yaml 路径（必填）。",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "qlora"],
        default=None,
        help="训练模式：full=全参数微调，qlora=4bit量化+LoRA；缺省按配置中 load_in_4bit 自动推断。",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help=f"基础模型名或本地路径（缺省取配置文件 model_id，再缺省 {DEFAULT_MODEL_ID}）。",
    )
    parser.add_argument(
        "--train_file",
        type=Path,
        default=DEFAULT_TRAIN_FILE,
        help="训练数据 jsonl 路径（默认 data/train.jsonl）。",
    )
    parser.add_argument(
        "--eval_file",
        type=Path,
        default=None,
        help="评估数据 jsonl 路径（可选，默认 data/eval.jsonl 存在时自动使用）。",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="覆盖配置文件 output_dir。")
    parser.add_argument("--learning_rate", type=float, default=None, help="覆盖配置文件 learning_rate。")
    parser.add_argument("--max_steps", type=int, default=None, help="覆盖配置文件 max_steps。")
    parser.add_argument("--seed", type=int, default=None, help="覆盖配置文件 seed。")
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        help="覆盖配置文件 report_to：tensorboard 或 wandb。",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    """读取 YAML 配置，返回 dict。"""
    if not path.is_file():
        raise FileNotFoundError(
            f"配置文件不存在: {path}\n请检查路径，可用配置见 recipes/ 目录。"
        )
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def resolve_dtype(value) -> torch.dtype:
    """将字符串/对象解析为 torch.dtype（默认 bfloat16）。"""
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        key = value.strip().lower().replace("torch.", "")
        mapping = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "auto": torch.bfloat16,
        }
        if key in mapping:
            return mapping[key]
    return torch.bfloat16


def build_model_init_kwargs(mode: str, extra: dict) -> dict:
    """构建 AutoModelForCausalLM 加载参数（bf16 + flash_attention_2，QLoRA 追加 4bit 量化）。"""
    model_init_kwargs: dict = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_2",
    }
    if mode == "qlora" and extra.get("load_in_4bit", True):
        model_init_kwargs.update(
            {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": extra.get("bnb_4bit_quant_type", "nf4"),
                "bnb_4bit_compute_dtype": resolve_dtype(
                    extra.get("bnb_4bit_compute_dtype", "bfloat16")
                ),
                "bnb_4bit_use_double_quant": extra.get("bnb_4bit_use_double_quant", True),
            }
        )
    return model_init_kwargs


def build_peft_config(extra: dict) -> LoraConfig:
    """构建 QLoRA 所需的 LoraConfig。"""
    return LoraConfig(
        r=int(extra.get("lora_r", 16)),
        lora_alpha=int(extra.get("lora_alpha", 32)),
        lora_dropout=float(extra.get("lora_dropout", 0.05)),
        target_modules=extra.get("lora_target_modules", DEFAULT_LORA_TARGET_MODULES),
        task_type="CAUSAL_LM",
    )


def load_jsonl_dataset(path: Path, split: str = "train"):
    """加载 jsonl 数据集，文件缺失时给出清晰的修复提示。"""
    if not path.is_file():
        raise FileNotFoundError(
            f"数据文件不存在: {path}\n请先运行数据预处理脚本生成训练数据：\n"
            "    python scripts/prepare_countdown_data.py"
        )
    return load_dataset("json", data_files=str(path), split=split)


def main() -> None:
    args = parse_args()

    # 1. 加载配置文件，拆分 GRPOConfig 字段与扩展字段（model_id / lora_* / bnb_* 等）
    cfg = load_yaml(args.config)
    grpo_field_names = {f.name for f in dataclass_fields(GRPOConfig)}
    grpo_kwargs = {k: v for k, v in cfg.items() if k in grpo_field_names}
    extra = {k: v for k, v in cfg.items() if k not in grpo_field_names}

    # 2. CLI 参数覆盖配置（优先级：CLI > YAML > 默认值）
    if args.output_dir is not None:
        grpo_kwargs["output_dir"] = args.output_dir
    if args.learning_rate is not None:
        grpo_kwargs["learning_rate"] = args.learning_rate
    if args.max_steps is not None:
        grpo_kwargs["max_steps"] = args.max_steps
    if args.seed is not None:
        grpo_kwargs["seed"] = args.seed
    if args.report_to is not None:
        grpo_kwargs["report_to"] = args.report_to
    grpo_kwargs.setdefault("output_dir", str(DEFAULT_OUTPUT_DIR))

    model_id = args.model_id or extra.get("model_id") or DEFAULT_MODEL_ID

    # 3. 训练模式判定：--mode 优先，缺省按 load_in_4bit 推断
    mode = args.mode or ("qlora" if extra.get("load_in_4bit") else "full")
    if mode == "full" and extra.get("load_in_4bit"):
        print("[提示] 配置含 load_in_4bit 但当前模式为 full，量化参数将被忽略。")

    # 4. 模型加载参数 + PEFT 配置
    model_init_kwargs = build_model_init_kwargs(mode, extra)
    peft_config = build_peft_config(extra) if mode == "qlora" else None

    # 5. 数据集与分词器
    train_dataset = load_jsonl_dataset(args.train_file)
    eval_dataset = None
    eval_file = args.eval_file or (DEFAULT_EVAL_FILE if DEFAULT_EVAL_FILE.is_file() else None)
    if eval_file is not None:
        if Path(eval_file).is_file():
            eval_dataset = load_jsonl_dataset(Path(eval_file))
        else:
            print(f"[提示] 评估文件不存在，跳过 eval: {eval_file}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # 6. GRPOTrainer 训练
    training_args = GRPOConfig(**grpo_kwargs)
    print("\n" + "=" * 70)
    print(f"模型         : {model_id}")
    print(f"训练模式     : {mode.upper()} ({'全参数微调' if mode == 'full' else 'QLoRA (4bit + LoRA)'})")
    print(f"输出目录     : {training_args.output_dir}")
    print(f"训练数据     : {args.train_file}（{len(train_dataset)} 条）")
    print(f"评估数据     : {len(eval_dataset) if eval_dataset is not None else '无'}")
    print(f"奖励函数     : [combined_reward]（格式 + 正确性，范围 0~2）")
    print(f"日志记录     : {training_args.report_to}")
    print("=" * 70)

    trainer = GRPOTrainer(
        model=model_id,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,  # TRL 0.18+ 使用 processing_class（替代 tokenizer 参数）
        reward_funcs=[combined_reward],
        peft_config=peft_config,
        model_init_kwargs=model_init_kwargs,
    )

    trainer.train()

    # 7. 保存最终模型（QLoRA 模式保存 LoRA adapter + 分词器）
    trainer.save_model(training_args.output_dir)
    print(f"\n[完成] 模型已保存到: {training_args.output_dir}")


if __name__ == "__main__":
    main()
