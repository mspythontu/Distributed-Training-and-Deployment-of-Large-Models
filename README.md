# DeepSpeed + TRL：GRPO 强化学习训练项目

基于 **DeepSpeed (ZeRO-3) + TRL (GRPOTrainer)** 的分布式强化学习（RL）训练项目，在 **Countdown-Tasks**（倒计时游戏）数据集上微调 **Qwen2.5-3B-Instruct**，支持**全参数微调**或 **QLoRA**（4bit 量化 + LoRA）两种模式，推理生成由 **vLLM** 加速，支持**多机多卡**训练。

## 技术栈

| 组件 | 版本要求 | 说明 |
|---|---|---|
| Python | >= 3.10 | vLLM 对 Python 版本有要求 |
| PyTorch | >= 2.5.0 | 按集群 CUDA 版本安装对应 wheel |
| transformers | >= 4.56.1 | 基础模型加载 |
| trl | >= 0.18.1 | `GRPOTrainer`（含 vLLM 推理集成） |
| accelerate | >= 1.0.0 | 分布式启动与调度 |
| deepspeed | >= 0.15.4 | ZeRO Stage 3 显存优化 |
| vllm | >= 0.8.0 | GRPOTrainer 自动调用做推理生成 |
| peft | >= 0.17.1 | LoRA 适配器 |
| bitsandbytes | >= 0.45.0 | 4bit 量化（QLoRA） |
| datasets | >= 3.0.0 | 数据集加载 |

- 基础模型：`Qwen/Qwen2.5-3B-Instruct`
- 数据集：`Jiayi-Pan/Countdown-Tasks`（约 5 万条训练样本）
- 训练框架：DeepSpeed ZeRO-3 + vLLM，支持全参数微调 / QLoRA（4bit 量化）

## 项目结构

```
deepspeed+trl分布式/
├── configs/                       # 训练配置文件
│   ├── accelerate_configs/        # accelerate launch 启动配置
│   │   └── deepspeed_zero3.yaml   # DeepSpeed ZeRO-3（bf16 + CPU offload）
│   └── deepspeed/                 # DeepSpeed 原始配置（可选）
├── scripts/                       # 训练/工具脚本
│   ├── prepare_countdown_data.py  # 数据预处理（生成 train.jsonl / eval.jsonl）
│   ├── reward_functions.py        # GRPO 奖励函数（correctness / format / combined）
│   ├── run_grpo.py                # GRPO 训练主脚本（TRL + DeepSpeed 主链路）
│   ├── launch_train.sh            # 一键启动训练（GPU 自动检测，预留 vLLM 卡）
│   ├── quick_test.sh              # 快速冒烟验证（10 条样本 × 5 步）
│   ├── setup_env.sh               # 一键环境安装（可选 --with-megatron / --with-wandb）
│   └── megatron/                  # Megatron-LM 平行方案（预训练 / SFT / GRPO）
│       ├── convert_checkpoint.py  # HF↔Megatron-Core 权重双向转换（mbridge）
│       ├── prepare_sft_data.py    # jsonl → Megatron bin/idx 分词索引
│       ├── pretrain_qwen.sh       # 预训练 / 继续预训练
│       ├── sft_qwen.sh            # SFT 监督微调
│       └── grpo_verl_megatron.sh  # GRPO（veRL + Megatron 后端）
├── recipes/                       # 训练参数配方（yaml）
│   ├── grpo-qwen-2.5-3b-countdown.yaml       # 全参数微调
│   └── grpo-qwen-2.5-3b-countdown-qlora.yaml # QLoRA（4bit + LoRA）
├── configs/                       # 训练配置文件
│   ├── accelerate_configs/        # accelerate launch 启动配置
│   ├── deepspeed/                 # DeepSpeed 原始配置（可选）
│   └── megatron/                  # Megatron 参数配置（pretrain/sft env + 并行速查）
├── data/                          # 数据集缓存与预处理产物（train.jsonl / eval.jsonl / megatron/）
├── docs/                          # 文档（CHECKLIST.md 运行检查清单）
├── output/                        # checkpoint、日志、模型输出
├── requirements.txt               # 依赖清单（主链路）
├── requirements-megatron.txt      # Megatron 专项依赖（3D/5D 并行 + veRL）
├── project.md                     # 工程循环档案（Loop Engineering）
└── README.md
```

## 环境安装

> 训练环境为 Linux 集群，下述命令均在集群上执行。

### 1. 创建虚拟环境

```bash
conda create -n grpo python=3.10 -y
conda activate grpo
```

### 2. 安装 PyTorch（按集群 CUDA 版本）

先确认 CUDA 版本：`nvidia-smi`，然后选择对应安装源：

```bash
# CUDA 12.1
pip install torch>=2.5.0 --index-url https://download.pytorch.org/whl/cu121
# CUDA 12.4
pip install torch>=2.5.0 --index-url https://download.pytorch.org/whl/cu124
# CUDA 12.8
pip install torch>=2.5.0 --index-url https://download.pytorch.org/whl/cu128
```

### 3. 安装项目依赖

```bash
cd deepspeed+trl分布式
pip install -r requirements.txt
```

> **vLLM 与 CUDA 版本对应**（vllm 0.8.x）：
> - 默认 wheel 对应 CUDA 12.4
> - CUDA 12.8 需额外索引：`pip install "vllm>=0.8.0" --extra-index-url https://download.pytorch.org/whl/cu128`
> - 查看可用版本：`pip index versions vllm`

### 4. 验证安装

```bash
pip check                       # 依赖无冲突
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import trl, deepspeed, vllm, peft; print('OK')"
deepspeed --version
```

## 数据准备

`Countdown-Tasks` 是倒计时游戏任务：给定一组数字与一个目标值，要求用四个数字各一次，通过四则运算（+ - * /）得到目标值。

数据集字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `prompt` | str | 游戏指令，如 `Countdown game start with the number list [3, 5, 2, 9], target is 15, use the numbers 3, 5, 2, 9 each exactly once, and use only basic arithmetic operations (+ - * /) to reach 15.` |
| `target` | str | 目标值，如 `15` |

加载方式：

```python
from datasets import load_dataset

ds = load_dataset("Jiayi-Pan/Countdown-Tasks")
print(ds)          # DatasetDict: train(约 50k 条)
print(ds["train"][0])
```

推荐将数据集预处理后缓存在 `data/` 目录：

```python
ds = load_dataset("Jiayi-Pan/Countdown-Tasks", cache_dir="./data/cache")
```

### 一键预处理（推荐）

运行 `scripts/prepare_countdown_data.py` 可自动完成下载、字段标准化、prompt 构建（含 `<thinking>` CoT 引导）与 9:1 划分：

```bash
# 全量处理（约 5 万条 -> data/train.jsonl 与 data/eval.jsonl）
python scripts/prepare_countdown_data.py

# 小样本调试（只取前 100 条，输出到 data_test/）
python scripts/prepare_countdown_data.py --max_samples 100 --output_dir data_test

# 自定义数据集 / 划分比例 / 随机种子
python scripts/prepare_countdown_data.py --dataset_name Jiayi-Pan/Countdown-Tasks \
    --output_dir data --train_ratio 0.9 --seed 42
```

输出格式（每行一条 JSON，`target` 同时写入样本供奖励函数直接使用）：

```json
{"prompt": "You are given a set of numbers and a target value. ... Numbers: [3, 5, 2, 9] Target: 15 ...", "target": "15", "solution": "..."}
```

**奖励函数设计要点**（见 `scripts/reward_functions.py`）：
- 正确性奖励：模型输出中的最终表达式计算结果是否等于 `target`（AST 白名单安全求值）
- 格式奖励：输出是否遵循 `<thinking>...</thinking><answer>...</answer>` 格式
- 组合奖励 `combined_reward`：格式(0/1) + 正确性(0/1)，范围 0~2，本项目默认奖励函数

## GRPO 训练

### 1. 准备数据

```bash
python scripts/prepare_countdown_data.py
```

### 2. 启动训练（accelerate launch）

> **关键**：`--num_processes` 设为 **GPU 数量 - 1**，最后一块 GPU 留给 vLLM 推理引擎
> （`vllm_device="auto"` 会自动占用空闲卡）。以 8 卡机器为例：

```bash
# 全参数微调
accelerate launch --config_file configs/accelerate_configs/deepspeed_zero3.yaml \
    --num_processes 7 \
    scripts/run_grpo.py --config recipes/grpo-qwen-2.5-3b-countdown.yaml

# QLoRA（4bit + LoRA）
accelerate launch --config_file configs/accelerate_configs/deepspeed_zero3.yaml \
    --num_processes 7 \
    scripts/run_grpo.py --config recipes/grpo-qwen-2.5-3b-countdown-qlora.yaml

# 通用写法（自动计算 GPU 数量 - 1）
NPROC=$((nproc - 1))
accelerate launch --config_file configs/accelerate_configs/deepspeed_zero3.yaml \
    --num_processes $NPROC \
    scripts/run_grpo.py --config recipes/grpo-qwen-2.5-3b-countdown.yaml
```

### 3. 快速调参（CLI 覆盖配置文件）

```bash
accelerate launch --config_file configs/accelerate_configs/deepspeed_zero3.yaml \
    --num_processes 7 \
    scripts/run_grpo.py --config recipes/grpo-qwen-2.5-3b-countdown.yaml \
    --learning_rate 1e-6 --max_steps 100 --output_dir output/test --report_to wandb
```

支持覆盖的参数：`--mode`（full/qlora）、`--model_id`、`--train_file`、`--eval_file`、
`--output_dir`、`--learning_rate`、`--max_steps`、`--seed`、`--report_to`（tensorboard/wandb）

### 4. 查看训练日志

```bash
tensorboard --logdir output/grpo-countdown
# 浏览器打开 http://localhost:6006
```

### 5. 训练产物

- 断点保存在 `output/grpo-countdown/checkpoint-*`（由 `save_steps` 控制）
- 训练结束自动保存最终模型到 `output/grpo-countdown`（QLoRA 模式保存 LoRA adapter + 分词器）
- 断点续训：修改 recipes 或启动参数支持 `resume_from_checkpoint`

## 多机多卡训练

### 1. 配置免密 SSH（多机必需）

所有节点间需能免密 SSH 互联：

```bash
# 在每台机器上
ssh-keygen -t rsa
ssh-copy-id user@worker1
ssh-copy-id user@worker2
# 验证
ssh worker1 hostname
```

### 2. 编写 DeepSpeed hostfile

在项目根目录创建 `hostfile`（无扩展名）：

```
worker1 slots=8        # 主节点，8 卡
worker2 slots=8        # 从节点，8 卡
```

### 3. 启动训练（方式一：deepspeed --hostfile）

```bash
# 在主节点执行即可，deepspeed 会自动分发到各节点
deepspeed --hostfile=hostfile \
    scripts/run_grpo.py --config recipes/grpo-qwen-2.5-3b-countdown.yaml
```

### 4. 启动训练（方式二：accelerate launch 多机）

先生成多机加速配置：

```bash
accelerate config    # 按提示配置：分布式训练 -> 多机 -> num_machines / num_processes / machine_rank
```

保存后（如 `accelerate_config.yaml`），在**每台机器**上分别启动（`--num_processes` 同样为 GPU 数量 - 1）：

```bash
# worker1（rank 0）
accelerate launch --config_file accelerate_config.yaml --num_machines 2 --machine_rank 0 \
    --main_process_ip <worker1_ip> --main_process_port 29500 --num_processes 7 \
    scripts/run_grpo.py --config recipes/grpo-qwen-2.5-3b-countdown.yaml

# worker2（rank 1）
accelerate launch --config_file accelerate_config.yaml --num_machines 2 --machine_rank 1 \
    --main_process_ip <worker1_ip> --main_process_port 29500 --num_processes 7 \
    scripts/run_grpo.py --config recipes/grpo-qwen-2.5-3b-countdown.yaml
```

## Megatron-LM 多卡分布式训练（预训练 / SFT / GRPO）

> 项目主链路是 **TRL + DeepSpeed(ZeRO-3)**；本章提供 **Megatron-LM 平行方案**，
> 用 3D/5D 并行（TP / PP / CP / EP / DP + 序列并行）覆盖预训练、SFT 与 GRPO 三条链路。
> 两套方案共享同一份 Countdown-Tasks 数据与 Qwen2.5-3B 模型，便于对照实验。

### 0. 方案怎么选

| 维度 | TRL + DeepSpeed（主链路） | Megatron-LM（本章） |
|---|---|---|
| 并行方式 | ZeRO-3 参数分片 + 数据并行 | TP / PP / CP / EP / DP + 序列并行（5D） |
| 上手成本 | 低（HF 生态，配置即用） | 中高（需转换权重与数据格式） |
| 适用场景 | 单机 / 小规模多机，快速实验 | 大规模集群、超长序列、追求极致吞吐 |
| RL rollout | GRPOTrainer 内建 vLLM | veRL 3D HybridEngine（actor↔rollout 权重重分片） |
| 性能特征 | 简单稳定 | 依靠 TransformerEngine 融合算子获得更高 MFU |

### 1. 环境安装

```bash
# 在原有环境上追加 Megatron 生态（megatron-core + TransformerEngine + mbridge + veRL）
bash scripts/setup_env.sh --with-megatron

# 资源受限 / 不需要 FP8：跳过 TransformerEngine 编译
bash scripts/setup_env.sh --megatron-lite
```

安装要点：

- **torch 版本冲突**：Megatron Core 要求 `torch>=2.6.0`，而主链路仅需 `>=2.5.0`；
  脚本会先校验，不满足时按集群 CUDA 索引自动升级
- **编译耗时**：`megatron-core[training,dev]` 会就地编译 TransformerEngine，约 **20+ 分钟**；
  默认 `MAX_JOBS=4` 限制并发防止多核机器 OOM，可用 `--max-jobs N` 调整
- **推荐路径**：生产环境优先使用 [NVIDIA NGC PyTorch 容器](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)（依赖预编译）
- 安装完成后验证段会打印 `megatron.core` / `transformer_engine` / `mbridge` / `verl` 的版本

### 2. 并行策略速查表

3D/5D 并行必须满足 `world_size == TP × PP × CP × DP`：

| GPU 数 | TP | PP | CP | DP | 说明 |
|---|---|---|---|---|---|
| 1 | 1 | 1 | 1 | 1 | 调试跑通 |
| 4 | 2 | 1 | 1 | 2 | 单节点，TP 覆盖 NVLink |
| 8 | 2 | 1 | 1 | 4 | 单节点 8 卡最常用 |
| 16 | 2 | 2 | 1 | 4 | 两节点，PP 跨机分摊显存 |
| 32 | 2 | 4 | 1 | 4 | 大规模扩展 |

> ⚠ **Qwen2.5-3B 特别注意**：模型使用 GQA（`num_heads=16`、`num_query_groups=2`），
> 两个值都必须能被 TP 整除，因此 **TP 最大只能取 2**。
> 需要更大并行度请用 PP / DP 扩展。详见 `configs/megatron/README.md`。

脚本会自动按卡数推荐组合，并在启动前做全套约束校验（不满足直接报错，避免排进集群才失败）。

### 3. 数据准备：jsonl → bin/idx

Megatron 读取分词后的二进制索引格式，需先用官方 `preprocess_data.py` 转换：

```bash
# SFT 数据（input / output 分离，loss 只算回答部分）
python scripts/megatron/prepare_sft_data.py \
    --input data/train.jsonl --output-dir data/megatron \
    --prefix countdown_sft --mode sft \
    --tokenizer-path Qwen/Qwen2.5-3B-Instruct \
    --megatron-path /path/to/Megatron-LM

# 预训练语料（拼接为连续文本流）
python scripts/megatron/prepare_sft_data.py \
    --input data/train.jsonl --prefix countdown_pt --mode pretrain \
    --megatron-path /path/to/Megatron-LM

# 只生成中间 jsonl（离线环境稍后再处理）
python scripts/megatron/prepare_sft_data.py --mode sft --skip-preprocess
```

- 产物：`data/megatron/<prefix>_<key>_document.bin/.idx`，
  训练时用 `--data-path data/megatron/<prefix>` 引用（**不带后缀**）
- `--megatron-path` 指向 Megatron-LM 源码根目录，也可 `export MEGATRON_PATH=/path/to/Megatron-LM`
- 若未克隆源码：`git clone https://github.com/NVIDIA/Megatron-LM.git`

### 4. 权重转换：HF ↔ Megatron-Core

基于 [mbridge](https://github.com/ISEEKYAN/mbridge)（veRL 官方采用的 Megatron-Core 转换组件）：

```bash
# HF -> Megatron-Core（按 TP/PP 在线分片并保存）
torchrun --nproc_per_node=8 scripts/megatron/convert_checkpoint.py \
    --mode import --hf-path Qwen/Qwen2.5-3B-Instruct \
    --save-path checkpoints/qwen2.5-3b-mcore --tp 2 --pp 1

# Megatron-Core -> HF（训练后导出，供 vLLM / transformers 部署）
torchrun --nproc_per_node=8 scripts/megatron/convert_checkpoint.py \
    --mode export --hf-path Qwen/Qwen2.5-3B-Instruct \
    --megatron-path checkpoints/qwen2.5-3b-mcore \
    --save-path output/qwen2.5-3b-hf --tp 2 --pp 1 --memory-efficient
```

- 转换依赖 TransformerEngine（mbridge 官方注明 `use_te=False` 暂不支持）
- 支持架构：Qwen2 / Qwen2.5-VL / Qwen3 / Qwen3-MoE / LLaMA / DeepseekV3 / Mixtral 等

### 5. 预训练 / 继续预训练

```bash
# 自动检测 GPU 并按卡数推荐并行组合
bash scripts/megatron/pretrain_qwen.sh \
    --megatron-path /path/to/Megatron-LM \
    --data-path data/megatron/countdown_pt

# 显式指定并行度 + 从转换好的检查点继续预训练
bash scripts/megatron/pretrain_qwen.sh \
    --tp 2 --pp 1 --gbs 64 --mbs 4 --lr 3e-4 \
    --load checkpoints/qwen2.5-3b-mcore
```

参数集中在 `configs/megatron/pretrain_qwen2.5-3b.env`（模型架构 / 并行 / 学习率 / 保存间隔），
命令行参数优先级更高；加 `--dry-run` 可只打印组装后的完整命令而不执行。

### 6. SFT 监督微调

```bash
bash scripts/megatron/sft_qwen.sh \
    --megatron-path /path/to/Megatron-LM \
    --data-path data/megatron/countdown_sft \
    --load output/megatron/pretrain
```

与预训练的差异集中在 `configs/megatron/sft_qwen2.5-3b.env`：
学习率降至 `1e-5`、warmup 占比 3%、关闭权重衰减、迭代步数更少。
**务必用 `--load` 指定起始检查点**，否则会从随机初始化开始。

### 7. GRPO 强化学习（veRL + Megatron 后端）

```bash
bash scripts/megatron/grpo_verl_megatron.sh \
    --model-path output/megatron/sft \
    --tp 2 --pp 1 --rollout-tp 1 \
    --train-batch 64 --num-generations 8
```

要点：

- 脚本会自动把 `data/*.jsonl` 转成 veRL 需要的 **parquet**（输出到 `data/megatron/verl/`），
  其中 `prompt` 为 conversation 格式、`ground_truth` 为 target、`extra_info` 透传原始字段
- Megatron 后端启用 **5D 并行 + 序列并行**，并通过 3D HybridEngine 在 actor(Megatron) 与
  rollout(vLLM/SGLang) 之间做高效权重重分片
- 显存紧张时加 `--offload`（参数 / 梯度 / 优化器卸载到 CPU）
- **奖励函数**：项目 `scripts/reward_functions.py` 面向 TRL 接口，需按 veRL 签名再包一层，
  然后用 `--custom-reward-module` / `--custom-reward-name` 指定（脚本头部注释含适配示例）
- veRL 配置键随版本演进较快，报 `Could not resolve config` 时，
  对照安装版本的 `verl/trainer/config/ppo_trainer.yaml` 调整本脚本参数即可

### 8. 多机多卡（Megatron）

```bash
# 每个节点分别执行，仅 --node-rank 不同
bash scripts/megatron/pretrain_qwen.sh \
    --num-nodes 2 --node-rank 0 --master-addr <主节点IP> --master-port 29500 \
    --gpus 8 --tp 2 --pp 2
```

- 可用 `--launcher deepspeed` 切换为 deepspeed 启动器（走其多机分发）
- **跨机原则**：TP 限制在单节点内（NVLink 带宽高），跨机扩展优先用 PP / DP

---

## QLoRA / ZeRO-3 / vLLM 注意事项

### DeepSpeed ZeRO-3
- ZeRO-3 将模型参数/梯度/优化器状态分片到所有 GPU，是 3B 模型 + 较大 batch 的首选
- 本项目使用 accelerate 配置 `configs/accelerate_configs/deepspeed_zero3.yaml`：
  - `zero_stage: 3`、`zero3_init_flag: true`、`zero3_save_16bit_model: true`
  - `mixed_precision: bf16`（与 GRPOConfig 的 `bf16: true` 一致）
  - `offload_optimizer_device: cpu`：优化器状态卸载到 CPU，进一步省显存（可改 `none`）
- gradient_checkpointing 由 `recipes/*.yaml` 的 `gradient_checkpointing: true` 控制
  （不在 accelerate 的 ds 配置中）

### QLoRA（4bit 量化）
- 4bit 量化权重由 bitsandbytes 管理，可显著降低显存占用
- 参数见 `recipes/grpo-qwen-2.5-3b-countdown-qlora.yaml`：`lora_r` / `lora_alpha` /
  `lora_dropout` / `load_in_4bit` / `bnb_4bit_quant_type` / `bnb_4bit_compute_dtype`
- 训练结束保存的是 LoRA adapter（`trainer.save_model`），可后续合并回全量权重
- 4bit 量化要求 CUDA 12.x + Linux，Windows 无法使用 bnb 量化训练
- **兼容性**：bnb 4bit 模型与 ZeRO-3 的 offload 存在已知限制，若加载报错请将
  `deepspeed_zero3.yaml` 中的 offload 改为 `none`，或改用全参数模式（`--mode full`）

### vLLM 推理（GRPOTrainer 集成）
- TRL 0.18+ 通过 `GRPOConfig(vllm=True)` 启用 vLLM 生成 rollouts（recipes 中 `vllm: true`）
- **重要**：vLLM 推理引擎需独占显存，本项目约定训练用 n-1 张 GPU，
  `vllm_device="auto"` 自动占用最后一块空闲卡
- 若显存紧张可调低 `vllm_gpu_memory_utilization`（默认 0.9）
- 多机场景下每台机器都会启动 vLLM 实例，注意 `vllm_port` 端口隔离（默认 8000）
- 关键 vLLM 参数：`vllm_device` / `vllm_gpu_memory_utilization` / `vllm_max_model_len` / `vllm_port`
- QLoRA 模式下 vLLM 加载的是未量化的 bf16 基础模型（仅用于生成，不与训练权重共享）

### 其他
- 训练日志输出到 `output/`，`report_to: tensorboard`（默认）或 `wandb`（`--report_to wandb`）
- 单卡调试建议：先用 `--max_samples` 生成小数据，再以 `--max_steps 20` 快速跑通

## 常见问题排查

| 问题 | 排查方向 |
|---|---|
| `GLIBC_2.29 not found` | vLLM 版本与系统 glibc 不兼容，升级系统或降级 vllm |
| vLLM 报 CUDA 版本错误 | 按上文安装对应 CUDA 版本的 vllm wheel |
| 多机连不上 | 检查 SSH 免密、防火墙、`--main_process_ip` |
| OOM 显存不足 | 降低 `per_device_train_batch_size`、`num_generations` 或 vLLM 显存占用 |
| ZeRO-3 保存的 checkpoint 无法加载 | 确认 `stage3_gather_16bit_weights_on_model_save` 为 `true` |
