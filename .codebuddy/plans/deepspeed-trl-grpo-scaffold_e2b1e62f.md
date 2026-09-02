---
name: deepspeed-trl-grpo-scaffold
overview: 在空工作区中搭建基于 DeepSpeed + TRL 的 GRPO 强化学习训练项目脚手架：创建清晰的项目目录结构（configs/、scripts/、recipes/、data/、output/ 等）、锁定版本号的 requirements.txt，以及包含环境安装/数据准备/多机多卡启动命令说明的 README.md。
todos:
  - id: create-skeleton
    content: 创建项目目录骨架：configs/deepspeed、scripts、recipes、data、output 及各目录 .gitkeep 占位，编写根目录 .gitignore
    status: completed
  - id: write-requirements
    content: 编写 requirements.txt，按功能分组列出全部依赖并标注最低版本号，含 vllm/CUDA 兼容性注释
    status: completed
    dependencies:
      - create-skeleton
  - id: write-readme
    content: 编写 README.md，覆盖环境安装、数据准备、多机多卡启动命令（hostfile/SSH/accelerate launch）及 QLoRA/ZeRO-3/vLLM 注意事项
    status: completed
    dependencies:
      - create-skeleton
---

## 用户需求

搭建一个基于 DeepSpeed + TRL 的 GRPO 强化学习训练项目（倒计时游戏 Countdown-Tasks，基础模型 Qwen/Qwen2.5-3B-Instruct），本次交付范围为：

1. 完整的项目目录结构（遵循 HuggingFace TRL 官方 GRPO 示例风格，包含 configs/、scripts/、recipes/、data/、output/ 等）
2. requirements.txt（每个包指定最低版本号，含 vllm 与 CUDA 版本兼容性提示）
3. 项目 README.md（环境安装、数据准备、多机多卡启动命令说明）

## 澄清结果

- 生成范围：仅目录骨架 + requirements.txt + README.md，不生成训练脚本/配置文件实体（目录内用 .gitkeep 占位，脚本由用户后续自行编写）
- 分布式规模：多机多卡，需 deepspeed hostfile + 无密码 SSH 配置，README 启动命令面向 Linux 集群
- 微调方式：QLoRA（bitsandbytes 4bit 量化 + LoRA）
- 硬件：A100/A800 80G 或更高，README 默认参数按较大 batch 编写

## 核心交付物

- 目录骨架（含 .gitignore、各目录 .gitkeep 占位）
- requirements.txt（torch/transformers/trl/accelerate/deepspeed/vllm/peft/bitsandbytes/datasets 等，全部含最低版本号）
- README.md（环境安装、数据准备、多机多卡启动命令、QLoRA/ZeRO-3/vLLM 注意事项）

## 技术栈

- Python >= 3.10（vllm 对 Python 版本有要求，README 中注明）
- torch >= 2.5.0、transformers >= 4.56.1、trl >= 0.18.1（GRPOTrainer）
- accelerate >= 1.0.0（分布式启动）
- deepspeed >= 0.15.4（ZeRO Stage 3）
- vllm >= 0.8.0（GRPOTrainer 自动调用 vLLM 做推理生成，注意按 CUDA 版本安装对应 wheel）
- peft >= 0.17.1 + bitsandbytes >= 0.45.0（QLoRA 4bit 量化）
- datasets >= 3.0.0（加载 Countdown-Tasks）
- 辅助：sentencepiece、einops、numpy、huggingface_hub、rich、tensorboard（可选 wandb）

## 实现方案

- 目录结构参照 TRL 官方示例组织方式：scripts/ 放训练脚本、recipes/ 放训练参数配置（yaml）、configs/deepspeed/ 放 ZeRO-3 配置、data/ 放数据集缓存与预处理产物、output/ 放 checkpoint 与日志。本次仅创建目录骨架，用 .gitkeep 占位保留结构。
- requirements.txt 按功能分组（核心框架 / 分布式训练 / QLoRA / 推理 / 工具）逐行标注最低版本号，顶部注释说明 vllm 与 CUDA 版本（12.4/12.8）的对应关系及安装顺序。
- README.md 覆盖：项目简介与结构说明、环境安装（conda 创建 + pip 安装 + 验证命令）、数据准备（Countdown-Tasks 字段说明与加载方式）、多机多卡启动（passwordless SSH 配置、hostfile 编写示例、`accelerate launch` 与 `deepspeed --hostfile` 两种启动命令示例）、QLoRA/ZeRO-3/vLLM 注意事项（4bit 量化与 ZeRO-3 共存要点、vLLM 张量并行、端口隔离、checkpoint 保存）。

## 版本兼容性要点（写入 requirements/README）

- vllm 0.8.x 对应 CUDA 12.4/12.8，需按集群 CUDA 版本选择安装源，README 提供 `pip index versions vllm` 等验证命令
- trl 0.18.x 要求 transformers >= 4.56、accelerate >= 1.0、torch >= 2.5，requirements 中已体现
- bitsandbytes 4bit 量化需 CUDA 12.x 且 Linux 环境，Windows 仅用于编写文件，训练在 Linux 集群执行
- 最终以 pip 依赖解析结果为准，README 中提供 `pip check` 验证步骤