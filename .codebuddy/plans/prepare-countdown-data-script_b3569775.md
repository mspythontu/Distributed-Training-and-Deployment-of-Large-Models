---
name: prepare-countdown-data-script
overview: 编写 scripts/prepare_countdown_data.py：从 HuggingFace Hub 下载 Countdown-Tasks 数据集，转换为 GRPO 训练格式（含 <thinking> CoT 引导的 prompt 模板 + 保留 solution 字段），按 9:1 划分输出为 data/train.jsonl 与 data/eval.jsonl，支持命令行参数配置数据集名称、输出路径、划分比例。
todos:
  - id: write-prepare-script
    content: 编写 scripts/prepare_countdown_data.py，实现 CLI 参数、字段容错标准化、prompt 模板构建与 9:1 划分 jsonl 输出
    status: completed
  - id: smoke-test
    content: 用 --max_samples 100 运行脚本冒烟测试，验证 train/eval jsonl 输出格式与统计信息后清理临时产物
    status: completed
    dependencies:
      - write-prepare-script
---

## 产品概述

为 DeepSpeed + TRL 的 GRPO 强化学习训练项目编写数据预处理脚本 `scripts/prepare_countdown_data.py`，将 HuggingFace Hub 上的 Countdown-Tasks（倒计时游戏）数据集转换为 GRPO 训练可直接消费的 jsonl 格式。

## 核心功能

- 从 HuggingFace Hub 下载 Countdown-Tasks 数据集（默认 `Jiayi-Pan/Countdown-Tasks`）
- 字段标准化：将原始字段映射为 `input_numbers`（可用数字列表）、`target`（目标值）、`solution`（参考解法），并对字段命名差异做容错（`input_numbers`/`numbers`/`nums` 兼容）
- 构建 prompt：任务描述 + 可用数字 + 目标值，引导模型先输出 `<thinking>...</thinking>` 进行 CoT 推理，再以 `<answer>expression = target` 格式给出最终答案；`solution` 字段原样保留供奖励函数使用
- 按 9:1 比例划分训练集/评估集，固定随机种子保证可复现
- 输出 `data/train.jsonl` 与 `data/eval.jsonl`（UTF-8 编码，每行一条 JSON 记录）
- 命令行参数：`--dataset_name`、`--output_dir`、`--train_ratio`、`--seed`、`--cache_dir`、`--max_samples`（调试用）
- 运行后打印数据集规模、划分数量等统计信息

## 技术栈

- Python >= 3.10（与项目 requirements.txt 一致）
- `datasets` >= 3.0.0（已在 requirements.txt 中声明，用于下载与加载数据集）
- 标准库：`argparse`、`json`、`pathlib`、`random`

## 实现方案

### 总体策略

编写一个独立、可复用的 CLI 脚本，职责单一：下载 → 标准化 → 构建 prompt → 划分 → 输出。脚本不依赖项目其他模块，可直接在 Linux 训练集群或本地执行。

### 关键设计

1. **字段容错标准化**：HF 上 Countdown-Tasks 变体的字段命名不完全一致（用户描述为 `input_numbers`/`target`/`solution`，部分版本为 `prompt`/`target`）。脚本实现 `_normalize_example()` 函数：

- 数字列表：依次尝试 `input_numbers` → `numbers` → `nums` → `input`，将值统一转为 `[str(x) for x in nums]` 并保持顺序
- 目标值：尝试 `target` → `target_value`，统一转字符串
- 参考解法：尝试 `solution` → `answer`，缺失时记为 `""`（jsonl 中保留空字符串键，奖励函数可自行判断）
- 若三种核心字段均缺失，跳过该样本并计数警告

2. **Prompt 模板**：基于用户提供模板，补充 `<thinking>` 引导句（需求明确要求 CoT 推理），最终模板为：

```
You are given a set of numbers and a target value. Use basic arithmetic operations (+, -, *, /) to reach the target value. You must use each number at most once.

Numbers: [1, 2, 3, 4]
Target: 10

First, reason step by step inside <thinking> tags. Then provide your final answer in the format: <answer>expression = target
```

模板常量定义在脚本顶部，便于后续调整。

3. **划分逻辑**：使用固定 seed 的 `random.Random(seed)` 对样本 `shuffle` 后按 `train_ratio` 切分；若数据集本身带 `eval`/`test` split 则忽略（仅用 `train` split，保证可控），保持逻辑简单可预测。
4. **输出格式**：jsonl 每行 `{"prompt": ..., "solution": ...}`，`ensure_ascii=False` + `utf-8` 写文件，保证符号与中文正确编码；`--max_samples` 用于小样本调试（取数据集前 N 条）。
5. **健壮性**：`output_dir` 不存在时自动创建；下载时支持 `cache_dir` 指定缓存位置（默认 `data/cache`）；打印下载/处理/划分统计信息便于核对。

### 复杂度与性能

- 时间：O(n) 单遍处理，5 万条样本秒级完成
- 内存：逐样本流式转换后写入（不整体驻留大列表），`shuffle` 仅对索引列表操作
- 数据集下载是主要耗时点，与网络有关；脚本不做额外缓存逻辑（依赖 datasets 自身缓存机制）

## 实现注意

- 脚本顶部 `if __name__ == "__main__":` 入口 + `main()` 函数组织，便于单测与复用
- 使用 `pathlib.Path` 而非字符串拼接路径，兼容 Windows/Linux
- 不修改 README.md、requirements.txt 等其他文件，保持本次改动范围最小（仅新增一个脚本文件）
- 冒烟测试：以 `--max_samples 100 --output_dir data_test/` 运行验证输出格式后删除临时产物，再以完整数据正式生成

## 架构设计

单文件 CLI 脚本，无复杂组件关系，结构如下：

```
scripts/prepare_countdown_data.py
├── PROMPT_TEMPLATE 常量（模板字符串）
├── parse_args()          # argparse 参数解析
├── normalize_example()   # 字段标准化 + 容错
├── build_prompt()        # 模板填充
├── split_and_save()      # 划分 + jsonl 写出
└── main()                # 流程编排 + 统计打印
```

## 目录结构

仅新增 1 个文件（其余为已存在的骨架文件）：

```
deepspeed+trl分布式/
├── scripts/
│   ├── prepare_countdown_data.py   # [NEW] 数据预处理脚本。实现 CLI 参数解析、
│   │                               #       数据集下载（load_dataset）、字段容错标准化、
│   │                               #       prompt 模板构建（<thinking> CoT 引导）、
│   │                               #       9:1 划分（固定 seed）与 jsonl 输出，
│   │                               #       运行后打印统计信息
│   └── .gitkeep                    # 已有占位
└── data/                           # 脚本默认输出目录（train.jsonl / eval.jsonl）