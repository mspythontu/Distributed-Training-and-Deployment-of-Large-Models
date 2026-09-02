#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Countdown-Tasks 数据预处理脚本：为 GRPO 训练生成 train.jsonl / eval.jsonl。

功能：
1. 从 HuggingFace Hub 下载 Countdown-Tasks 数据集（默认 Jiayi-Pan/Countdown-Tasks）
2. 将原始字段标准化为 numbers / target / solution，并构建带 <thinking> CoT 引导的 prompt
3. 按 train_ratio（默认 9:1）划分训练集 / 评估集，输出为 UTF-8 编码的 jsonl
4. 通过命令行参数控制数据集名称、输出路径、划分比例、随机种子等

输出格式（每行一条 JSON 记录）：
    {"prompt": "...", "target": "15", "solution": "..."}

用法示例：
    python scripts/prepare_countdown_data.py
    python scripts/prepare_countdown_data.py --dataset_name Jiayi-Pan/Countdown-Tasks \
        --output_dir data --train_ratio 0.9 --seed 42
    python scripts/prepare_countdown_data.py --max_samples 100 --output_dir data_test  # 小样本调试
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Optional

from datasets import load_dataset

DEFAULT_DATASET = "Jiayi-Pan/Countdown-Tasks"

PROMPT_TEMPLATE = """\
You are given a set of numbers and a target value. Use basic arithmetic operations (+, -, *, /) to reach the target value. You must use each number at most once.

Numbers: {numbers}
Target: {target}

First, reason step by step inside <thinking> tags. Then provide your final answer in the format: <answer>expression = target"""


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Prepare Countdown-Tasks dataset for GRPO training (output: train.jsonl / eval.jsonl)."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=DEFAULT_DATASET,
        help=f"HuggingFace dataset name (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data"),
        help="Output directory for train.jsonl / eval.jsonl (default: data).",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Ratio of samples used for training split (default: 0.9).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible shuffling (default: 42).",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=Path("data/cache"),
        help="Cache directory for the downloaded dataset (default: data/cache).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Only process the first N samples (useful for debugging, default: all).",
    )
    return parser.parse_args()


def _first_value(example: dict, keys: list[str], default: Optional[str] = None):
    """按顺序尝试多个字段名，返回第一个非空值。"""
    for key in keys:
        value = example.get(key)
        if value is not None and str(value).strip() != "":
            return value
    return default


def normalize_example(example: dict) -> Optional[dict]:
    """字段容错标准化：将原始样本映射为 numbers / target / solution。

    支持字段名：
        numbers:  input_numbers / numbers / nums / input
        target:   target / target_value
        solution: solution / answer

    返回 None 表示核心字段缺失，该样本应被跳过。
    """
    raw_numbers = _first_value(example, ["input_numbers", "numbers", "nums", "input"])
    raw_target = _first_value(example, ["target", "target_value"])
    if raw_numbers is None or raw_target is None:
        return None

    numbers = [str(x) for x in raw_numbers]
    target = str(raw_target)
    solution = _first_value(example, ["solution", "answer"], default="") or ""
    return {"numbers": numbers, "target": target, "solution": str(solution).strip()}


def build_prompt(numbers: list[str], target: str) -> str:
    """用模板构建 GRPO prompt，Numbers 以列表形式展示。"""
    numbers_str = "[" + ", ".join(numbers) + "]"
    return PROMPT_TEMPLATE.format(numbers=numbers_str, target=target)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """将记录列表写出为 UTF-8 编码的 jsonl 文件。"""
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError(f"--train_ratio must be in (0, 1), got {args.train_ratio}")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError(f"--max_samples must be positive, got {args.max_samples}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = str(args.cache_dir) if args.cache_dir else None
    if cache_dir:
        args.cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. 下载 / 加载数据集
    print(f"[1/4] Loading dataset '{args.dataset_name}' ...")
    dataset_dict = load_dataset(args.dataset_name, cache_dir=cache_dir)
    split = "train" if "train" in dataset_dict else list(dataset_dict.keys())[0]
    raw_data = dataset_dict[split]
    if args.max_samples is not None:
        raw_data = raw_data.select(range(min(args.max_samples, len(raw_data))))
    print(f"      Dataset split '{split}' size: {len(raw_data)}")

    # 2. 字段标准化 + 构建 prompt
    print("[2/4] Normalizing fields and building prompts ...")
    records: list[dict] = []
    skipped = 0
    for ex in raw_data:
        norm = normalize_example(ex)
        if norm is None:
            skipped += 1
            continue
        prompt = build_prompt(norm["numbers"], norm["target"])
        records.append(
            {
                "prompt": prompt,
                "target": norm["target"],
                "solution": norm["solution"],
            }
        )
    if not records:
        raise RuntimeError(
            "No valid samples after normalization. Check the dataset fields with "
            f"'print(ds[{split!r}].features)' (skipped {skipped} samples)."
        )

    # 3. 固定 seed 划分 train / eval
    print("[3/4] Splitting into train / eval ...")
    rng = random.Random(args.seed)
    rng.shuffle(records)
    n_train = int(len(records) * args.train_ratio)
    train_rows, eval_rows = records[:n_train], records[n_train:]

    # 4. 输出 jsonl
    print("[4/4] Writing jsonl files ...")
    train_path = args.output_dir / "train.jsonl"
    eval_path = args.output_dir / "eval.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)

    # 统计信息
    print("\n===== Done =====")
    print(f"Dataset          : {args.dataset_name} (split '{split}')")
    print(f"Processed        : {len(records)} samples (skipped {skipped} invalid)")
    print(f"Train / Eval     : {len(train_rows)} / {len(eval_rows)} "
          f"(ratio {args.train_ratio:.2f}, seed {args.seed})")
    print(f"Train file       : {train_path.resolve()}")
    print(f"Eval file        : {eval_path.resolve()}")
    print("Sample prompt:")
    print("-" * 60)
    print(records[0]["prompt"])
    print("-" * 60)


if __name__ == "__main__":
    main()
