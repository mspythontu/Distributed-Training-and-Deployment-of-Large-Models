#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把本项目 jsonl 数据转换为 Megatron-LM 训练所需的二进制索引格式（.bin / .idx）。

背景：
    Megatron-LM 不直接读取原始 jsonl，而是通过官方 tools/preprocess_data.py
    把文本语料分词后落盘为两套文件：
        <prefix>_<key>_document.idx   # 样本索引（每条样本的起止位置）
        <prefix>_<key>_document.bin   # 扁平化的 token ids
    训练时用 --data-path <prefix> 指向该 prefix（不带后缀）。

    本脚本负责前半段：把项目 data/train.jsonl（字段 prompt/target/solution）
    重整为 preprocess_data.py 可直接消费的 jsonl；
    然后后半段自动调用官方脚本产出 bin/idx。

两种模式：
    pretrain : 拼接 prompt + solution 组成连续文本 -> {"text": "..."}
               用于预训练 / 继续预训练（自回归学习）
    sft      : 保留 prompt 与回答分离 -> {"input": "...", "output": "..."}
               用于监督微调（对 output 部分计算 loss）

用法：
    # SFT 数据（推荐先跑这个，与下游 GRPO 衔接）
    python scripts/megatron/prepare_sft_data.py \
        --input data/train.jsonl --output-dir data/megatron \
        --prefix countdown_sft --mode sft \
        --tokenizer-path Qwen/Qwen2.5-3B-Instruct \
        --megatron-path /path/to/Megatron-LM

    # 预训练语料
    python scripts/megatron/prepare_sft_data.py \
        --input data/train.jsonl --prefix countdown_pt --mode pretrain \
        --megatron-path /path/to/Megatron-LM

    # 只生成中间 jsonl，不调用官方脚本（离线环境后续手动处理）
    python scripts/megatron/prepare_sft_data.py \
        --input data/train.jsonl --prefix countdown_sft --mode sft --skip-preprocess

依赖：
    Megatron-LM 源码中的 tools/preprocess_data.py（--megatron-path 指定仓库根目录）
"""

import argparse
import json
import logging
import os
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("prepare_megatron_data")

DEFAULT_SYSTEM_HINT = (
    "You are given a set of numbers and a target value. "
    "Use each number exactly once with basic arithmetic operations (+ - * /) to reach the target."
)


def read_jsonl(path: str):
    """逐行读取 jsonl，跳过空行并做 JSON 合法性校验。"""
    if not os.path.isfile(path):
        logger.error("输入数据文件不存在: %s", path)
        logger.error("可先运行: python scripts/prepare_countdown_data.py 生成数据")
        sys.exit(1)

    records, bad_lines = [], 0
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                bad_lines += 1
                logger.warning("第 %d 行不是合法 JSON，已跳过: %s", lineno, exc)
    logger.info("读取 %s：有效样本 %d 条，跳过损坏行 %d 条", path, len(records), bad_lines)
    if not records:
        logger.error("没有可用的样本，终止处理")
        sys.exit(1)
    return records


def build_prompt(rec: dict, system_hint: str) -> str:
    """从 Countdown 样本构造指令式 prompt。"""
    prompt = rec.get("prompt")
    if prompt:
        return f"{system_hint}\n\n{prompt}"
    numbers = rec.get("numbers")
    target = rec.get("target")
    if numbers and target is not None:
        return f"{DEFAULT_SYSTEM_HINT}\n\nNumbers: {numbers}\nTarget: {target}"
    logger.warning("样本既无 prompt 也无 numbers/target 字段，将仅使用系统提示")
    return system_hint


def build_answer(rec: dict) -> str:
    """构造回答文本：优先 solution，其次按 think/answer 模板包裹。"""
    solution = rec.get("solution")
    if solution:
        return solution
    target = rec.get("target")
    if target is not None:
        return (
            "<thinking>We need to combine the given numbers with basic arithmetic "
            f"operations to obtain {target}.<answer>{target}</answer>"
        )
    logger.warning("样本缺少 solution / target，回答为空")
    return ""


def write_loose_jsonl(records, output_dir: str, prefix: str, mode: str, system_hint: str) -> str:
    """把项目格式重整为 Megatron preprocess_data.py 可消费的 jsonl。

    pretrain 模式 -> {"text": ...}
    sft      模式 -> {"input": ..., "output": ...}
    """
    os.makedirs(output_dir, exist_ok=True)
    loose_path = os.path.join(output_dir, f"{prefix}.jsonl")

    written, skipped = 0, 0
    with open(loose_path, "w", encoding="utf-8") as out:
        for rec in records:
            prompt = build_prompt(rec, system_hint)
            answer = build_answer(rec)
            if not answer:
                skipped += 1
                continue
            if mode == "pretrain":
                # 拼接成单一文本流，EOS 由 preprocess_data.py 的 --append-eod 负责
                out.write(json.dumps({"text": f"{prompt}\n{answer}"}, ensure_ascii=False) + "\n")
            else:
                out.write(json.dumps({"input": prompt, "output": answer}, ensure_ascii=False) + "\n")
            written += 1

    logger.info("中间 jsonl 生成完成: %s（写入 %d 条，跳过 %d 条无回答样本）", loose_path, written, skipped)
    if written == 0:
        logger.error("没有任何样本被写入，请检查数据字段是否为 prompt / numbers / target / solution")
        sys.exit(1)
    return loose_path


def locate_preprocess_script(megatron_path: str) -> str:
    """定位 Megatron-LM 官方的 tools/preprocess_data.py。"""
    candidates = [
        os.path.join(megatron_path, "tools", "preprocess_data.py"),
        os.path.join(megatron_path, "preprocess_data.py"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            logger.info("找到官方预处理脚本: %s", candidate)
            return candidate
    logger.error(
        "未在 %s 下找到 Megatron-LM 的 tools/preprocess_data.py。\n"
        "请先克隆源码: git clone https://github.com/NVIDIA/Megatron-LM.git\n"
        "然后用 --megatron-path 指向仓库根目录。",
        megatron_path,
    )
    sys.exit(1)


def run_preprocess(script: str, input_jsonl: str, prefix: str, tokenizer_path: str,
                   mode: str, workers: int, seq_length: int) -> None:
    """调用 Megatron 官方 preprocess_data.py 生成 .bin / .idx 索引。"""
    json_keys = "text" if mode == "pretrain" else "input output"
    output_prefix = os.path.join(os.path.dirname(input_jsonl), prefix)

    cmd = [
        sys.executable, script,
        "--input", input_jsonl,
        "--output-prefix", output_prefix,
        "--tokenizer-type", "PretrainedFromHF",
        "--tokenizer-name-or-path", tokenizer_path,
        "--json-keys", json_keys,
        "--workers", str(workers),
        "--append-eod",
        "--log-interval", "1000",
    ]
    if mode == "sft":
        # SFT 需要按样本切分，避免多条样本被拼接进同一个序列
        cmd.append("--split-sentences")
        cmd.extend(["--seq-length", str(seq_length)])

    logger.info("=" * 60)
    logger.info("开始分词并生成 bin/idx（%s）", "预训练" if mode == "pretrain" else "SFT")
    logger.info("命令: %s", " ".join(cmd))
    logger.info("=" * 60)

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.error("preprocess_data.py 执行失败（退出码 %d）", result.returncode)
        logger.error(
            "常见原因：tokenizer 路径不可达（国内网络可 export HF_ENDPOINT=https://hf-mirror.com）；"
            "内存不足需调小 --workers；Megatron 源码版本与参数不兼容"
        )
        sys.exit(result.returncode)

    logger.info("=" * 60)
    logger.info("数据转换完成！产物前缀: %s", output_prefix)
    logger.info("训练脚本 --data-path 请填写: %s", output_prefix)
    logger.info("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="项目 jsonl -> Megatron-LM bin/idx 二进制索引",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", default="data/train.jsonl", help="输入 jsonl（默认 data/train.jsonl）")
    parser.add_argument("--output-dir", default="data/megatron", help="输出目录（默认 data/megatron）")
    parser.add_argument("--prefix", default="countdown_sft", help="输出文件名前缀（默认 countdown_sft）")
    parser.add_argument(
        "--mode", choices=("sft", "pretrain"), default="sft",
        help="sft=保留 input/output 分离（默认）；pretrain=拼接为单一 text 文本流",
    )
    parser.add_argument(
        "--tokenizer-path", default="Qwen/Qwen2.5-3B-Instruct",
        help="分词器 HF 路径或本地目录（默认 Qwen/Qwen2.5-3B-Instruct）",
    )
    parser.add_argument(
        "--megatron-path", default=os.environ.get("MEGATRON_PATH", ""),
        help="Megatron-LM 源码根目录（也可用环境变量 MEGATRON_PATH）",
    )
    parser.add_argument("--workers", type=int, default=8, help="分词并行进程数（默认 8）")
    parser.add_argument("--seq-length", type=int, default=1024, help="序列长度，SFT 切分用（默认 1024）")
    parser.add_argument("--system-hint", default=DEFAULT_SYSTEM_HINT, help="附加在 prompt 前的系统提示语")
    parser.add_argument("--max-samples", type=int, default=0, help="仅处理前 N 条（调试用，0 表示全量）")
    parser.add_argument("--skip-preprocess", action="store_true", help="只生成中间 jsonl，不调用官方脚本")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Megatron 数据预处理：mode=%s, input=%s", args.mode, args.input)

    records = read_jsonl(args.input)
    if args.max_samples > 0:
        records = records[: args.max_samples]
        logger.info("已按 --max-samples 截取前 %d 条", len(records))

    loose_jsonl = write_loose_jsonl(records, args.output_dir, args.prefix, args.mode, args.system_hint)

    if args.skip_preprocess:
        logger.info("已跳过官方分词步骤（--skip-preprocess）。中间文件: %s", loose_jsonl)
        return

    if not args.megatron_path:
        logger.error(
            "缺少 Megatron-LM 源码路径，无法生成 bin/idx。\n"
            "  方式一: git clone https://github.com/NVIDIA/Megatron-LM.git 后加 --megatron-path <repo>\n"
            "  方式二: export MEGATRON_PATH=<repo>\n"
            "  方式三: 加 --skip-preprocess 只产出中间 jsonl，稍后离线处理"
        )
        sys.exit(1)

    script = locate_preprocess_script(args.megatron_path)
    run_preprocess(
        script, loose_jsonl, args.prefix, args.tokenizer_path,
        args.mode, args.workers, args.seq_length,
    )


if __name__ == "__main__":
    main()
