---
name: grpo-training-script-configs
overview: 编写 GRPO 训练主脚本 scripts/run_grpo.py 及全部配套配置：在 reward_functions.py 中新增 combined_reward、新增 accelerate DeepSpeed ZeRO-3 启动配置、全参数与 QLoRA 两套 recipes YAML，并在 README 补充训练启动命令（num_processes = GPU 数 - 1 留给 vLLM）。
todos:
  - id: add-combined-reward
    content: 在 reward_functions.py 新增 combined_reward 组合奖励函数并补充自测断言
    status: completed
  - id: write-run-grpo-script
    content: 编写 run_grpo.py：CLI 参数、模式切换、GRPOTrainer 训练与模型保存
    status: completed
    dependencies:
      - add-combined-reward
  - id: write-deepspeed-config
    content: 编写 configs/accelerate_configs/deepspeed_zero3.yaml 启动配置
    status: completed
  - id: write-recipes
    content: 编写全参数与 QLoRA 两套 recipes YAML 配置文件
    status: completed
  - id: update-readme
    content: 更新 README.md 补充 GRPO 训练启动命令与注意事项
    status: completed
    dependencies:
      - write-run-grpo-script
      - write-deepspeed-config
      - write-recipes
  - id: verify-scripts
    content: 验证：py_compile 语法检查、reward 自测、yaml 配置解析检查
    status: completed
    dependencies:
      - write-run-grpo-script
      - write-deepspeed-config
      - write-recipes
---

## 用户需求

编写 GRPO 强化学习训练主脚本及全部配套配置文件，用于在 Countdown-Tasks 数据集上训练 Qwen2.5-3B-Instruct（DeepSpeed ZeRO-3 + vLLM 推理 + TRL GRPOTrainer）。

## 核心功能

1. **训练主脚本 `scripts/run_grpo.py`**：

- 用 transformers 加载 `Qwen/Qwen2.5-3B-Instruct`，启用 bf16 与 `flash_attention_2`
- 命令行参数选择**全参数微调**或 **QLoRA**（4bit 量化 + LoRA）模式
- 加载 `data/train.jsonl` 数据集（`{"prompt", "target", "solution"}` 格式）
- 导入 `scripts/reward_functions.py` 中的 `combined_reward` 作为奖励函数
- 使用 `GRPOTrainer` 训练，关键超参全部通过 recipes YAML 传入
- 训练结束自动保存模型到 `output/` 目录

2. **配置 `configs/accelerate_configs/deepspeed_zero3.yaml`**：ZeRO Stage 3、bf16 混合精度、optimizer 卸载到 CPU、梯度检查点说明
3. **配置 `recipes/grpo-qwen-2.5-3b-countdown.yaml`**：输出目录、学习率 5e-7、cosine 调度、bf16、max_prompt_length=256、max_completion_length=1024、num_generations=2、beta=0.001、gradient_checkpointing、batch_size=1、梯度累积 4、logging_steps=10、save_steps=100、max_steps=450、warmup_ratio=0.1、report_to=tensorboard
4. **配置 `recipes/grpo-qwen-2.5-3b-countdown-qlora.yaml`**：在基础配置上增加 lora_r=16、lora_alpha=32、lora_dropout=0.05、load_in_4bit=true、bnb_4bit_compute_dtype=bfloat16

## 补充约束

- 启动时 `num_processes` 应为 GPU 数量 - 1（最后一块留给 vLLM 推理）
- 所有超参支持配置文件驱动并可被命令行覆盖
- 支持 tensorboard/wandb 日志记录
- `reward_functions.py` 目前缺少 `combined_reward`，需新增

## 技术栈

- Python >= 3.10、`trl>=0.18.1`（GRPOTrainer）、`transformers>=4.56.1`、`peft>=0.17.1`、`bitsandbytes>=0.45.0`
- `accelerate>=1.0.0` + `deepspeed>=0.15.4`（ZeRO-3）、`vllm>=0.8.0`（生成后端）、`datasets>=3.0.0`、`pyyaml>=6.0.0`

## 实现方案

### 总体策略

延续现有脚本风格（模块 docstring、`from __future__ import annotations`、类型标注、`if __name__ == "__main__"` 入口），新增一个训练主脚本 + 三个 YAML 配置 + 对 `reward_functions.py` 做最小增量修改。全部超参以 YAML 为单一事实来源，脚本用 `dataclasses.fields(GRPOConfig)` 差分识别 GRPOConfig 字段与扩展字段（`model_id`/`lora_*`/`bnb_*`），实现"配置文件驱动 + CLI 覆盖"。

### 关键设计

1. **`combined_reward` 新增**（`reward_functions.py`）：`combined_reward = format_reward(0/1) + correctness_reward(0/1)`，范围 0~2，鼓励"格式正确 + 答案正确"；复用现有两个函数，`correctness_reward` 已内置"优先 kwargs['target']、回退 prompt 解析"的健壮逻辑，与 TRL 0.18.x 奖励函数签名 `(completions, **kwargs) -> list[float]` 完全兼容；自测补充断言。

2. **`run_grpo.py` 结构**：

- argparse：`--config`（必填）、`--mode {full,qlora}`、`--model_id`、`--train_file`、`--output_dir`、`--learning_rate`、`--max_steps`、`--seed`、`--report_to`、`--eval_file`（可选）
- 加载 YAML → 用 `dataclass_fields(GRPOConfig)` 拆分：GRPOConfig 参数直接 `GRPOConfig(**kwargs)`；扩展字段用于模式判定与模型初始化
- 模式判定：`--mode` 优先，缺省按 `load_in_4bit` 自动推断
- `model_init_kwargs`：`torch_dtype=torch.bfloat16`、`attn_implementation="flash_attention_2"`；QLoRA 模式追加 `load_in_4bit`、`bnb_4bit_quant_type="nf4"`、`bnb_4bit_compute_dtype=torch.bfloat16`、`bnb_4bit_use_double_quant`
- QLoRA 模式构建 `LoraConfig(r, lora_alpha, lora_dropout, target_modules=Qwen2.5 全模块, task_type="CAUSAL_LM")`；全参数模式 `peft_config=None`
- 数据集：`load_dataset("json", data_files=...)`；tokenizer：`AutoTokenizer.from_pretrained`
- `GRPOTrainer(model=model_id, args=GRPOConfig(...), train_dataset=ds, processing_class=tokenizer, reward_funcs=[combined_reward], peft_config=..., model_init_kwargs=...)`（TRL 0.18 参数命名）
- 训练后 `trainer.save_model(output_dir)` 保存最终模型（QLoRA 保存 adapter）；`sys.path` 注入脚本所在目录实现 `from reward_functions import combined_reward`
- 日志：`report_to` 从配置读取（tensorboard 默认），支持 `--report_to wandb` 覆盖

3. **`deepspeed_zero3.yaml`**（accelerate launch 配置格式）：`distributed_type: DEEPSPEED`、`zero_stage: 3`、`zero3_init_flag: true`、`zero3_save_16bit_model: true`、`offload_optimizer_device: cpu`、`offload_param_device: none`、`mixed_precision: bf16`、`bf16.enabled: true`；注释说明：gradient_checkpointing 由 GRPOConfig 控制（accelerate ds 配置无此字段）、`num_processes` 启动时用 `--num_processes $(nproc-1)` 覆盖（留 1 卡给 vLLM）。

4. **两个 recipes YAML**：完整列出用户要求的 GRPOConfig 字段，补充 `model_id: Qwen/Qwen2.5-3B-Instruct`、`vllm: true`（TRL 0.18 字段）、`seed: 42`；QLoRA 版本追加 `lora_r/lora_alpha/lora_dropout/load_in_4bit/bnb_4bit_quant_type/bnb_4bit_use_double_quant/bnb_4bit_compute_dtype/lora_target_modules`。

5. **README 更新**：新增"GRPO 训练"章节——数据准备命令、full/qlora 两种启动命令（`accelerate launch --config_file ... --num_processes $(nproc-1)`）、tensorboard 查看、QLoRA+ZeRO-3 兼容性注意事项（4bit 与 offload 冲突时可关闭 offload 或改 full 模式）。

### 性能与健壮性

- 单样本奖励评估为 O(表达式长度)，毫秒级，不构成训练瓶颈
- 训练/推理职责分离：训练用 n-1 卡，vLLM 用最后 1 卡（`vllm_device="auto"` 自动占用），避免显存竞争
- 所有外部调用均有异常兜底；数据集列缺失时奖励函数回退 prompt 解析，不中断训练
- 本地 Windows 无 torch/trl/peft，无法真实运行训练；验证手段：`py_compile` 语法检查 + `python scripts/reward_functions.py` 自测 + `yaml.safe_load` 配置校验

## 目录结构

```
deepspeed+trl分布式/
├── configs/
│   ├── accelerate_configs/
│   │   └── deepspeed_zero3.yaml      # [NEW] accelerate+DeepSpeed ZeRO-3 启动配置（bf16、CPU offload）
│   └── deepspeed/                    # 已有占位
├── recipes/
│   ├── grpo-qwen-2.5-3b-countdown.yaml      # [NEW] 全参数微调 GRPOConfig 参数
│   └── grpo-qwen-2.5-3b-countdown-qlora.yaml # [NEW] QLoRA 版（含 lora/bnb 参数）
├── scripts/
│   ├── reward_functions.py           # [MODIFY] 新增 combined_reward + 自测断言
│   └── run_grpo.py                   # [NEW] GRPO 训练主脚本
├── data/                             # 训练数据（train.jsonl 由 prepare 脚本生成）
├── output/                           # 模型输出
└── README.md                         # [MODIFY] 补充 GRPO 训练启动命令
```

## 实现注意

- 严格遵循用户给定的 recipes 参数值（lr 5e-7、max_steps 450、save_steps 100 等），不擅自改动
- `run_grpo.py` 中 `from reward_functions import combined_reward` 需先 `sys.path.insert` 脚本目录；顶层 import trl/peft 因本地环境缺失，验证仅限语法层面
- TRL 0.18 API 细节：`processing_class`（非 tokenizer）、`vllm`（非 use_vllm）、`GRPOTrainer(model_init_kwargs=...)`；若用户环境 trl 版本为 0.18.x 可直接运行
- 不修改 requirements.txt（依赖已齐全：pyyaml、tensorboard 均已列出）