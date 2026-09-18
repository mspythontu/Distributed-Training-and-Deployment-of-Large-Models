# GRPO 训练运行检查清单

> 本清单用于在**首次运行 / 环境变更 / 训练异常**时逐项排查，覆盖：
> 硬件、软件环境、数据、训练前配置，以及常见问题排查表。
> 每个检查项均给出**具体命令**与**预期结果**，按顺序执行即可。

---

## 0. 一键自检汇总

在项目根目录执行以下命令，快速判断环境是否就绪：

```bash
# ① 奖励函数自测（最快，纯 CPU，无需 GPU）
python scripts/reward_functions.py

# ② 核心依赖版本总览（torch/transformers/trl/...）
python -c "
import torch, transformers, trl, accelerate, deepspeed, peft, datasets
print('torch        :', torch.__version__)
print('transformers :', transformers.__version__)
print('trl          :', trl.__version__)
print('accelerate   :', accelerate.__version__)
print('deepspeed    :', deepspeed.__version__)
print('peft         :', peft.__version__)
print('datasets     :', datasets.__version__)
print('CUDA 可用     :', torch.cuda.is_available(), torch.version.cuda)
"

# ③ 10 条样本冒烟验证（5 步，分钟级，QLoRA 默认）
bash scripts/quick_test.sh

# ④ Megatron-LM 链路自检（可选，仅安装 --with-megatron 后需要）
bash scripts/megatron/pretrain_qwen.sh --dry-run     # 并行度预检 + 打印完整命令
```

若 ①②③ 全部通过，基本可进入正式训练：

```bash
bash scripts/launch_train.sh              # 全参数微调（默认）
bash scripts/launch_train.sh --mode qlora # QLoRA
```

---

## 1. 硬件检查

### 1.1 GPU 型号 / 数量 / 显存

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used --format=csv
# 预期：全部 GPU 可见，memory.free 充裕（至少 24GB 以上为宜，QLoRA 可放宽）
```

实时占用观察：

```bash
nvidia-smi                      # 观察每个进程的显存占用
watch -n 1 nvidia-smi           # 每秒刷新
```

> **预期结果**：训练进程占用前 N-1 卡，vLLM 进程占用最后 1 卡（显存占用明显、GPU-Util 波动）。

### 1.2 GPU 间互联（NVLink / NVSwitch）

```bash
nvidia-smi topo -m              # 拓扑矩阵：NV#"x" 表示 NVLink 直连
nvidia-smi nvlink -s            # NVLink 链路状态（Active）
nvidia-smi -q -d PERFORMANCE | grep -i "Gpu Max Clocks\|Gpu Temperature"   # 时钟/温度
```

> **预期结果**：
> - 多卡间显示 `NV#1` / `NV#2`（NVLink）或 `SYS`（PCIe 仅推荐少卡训练）
> - `nvlink -s` 各链路状态为 `Active`，无 `Inactive`/`ERROR`

### 1.3 CPU 内存 / 磁盘

```bash
free -h                          # 内存：总内存与可用内存（offload 到 CPU 时需 >= 32GB）
df -h                            # 磁盘：模型缓存/输出目录所在分区剩余空间（>= 50GB）
nproc                            # 逻辑核数（用于确认 num_processes 计算）
```

> **注意**：`configs/accelerate_configs/deepspeed_zero3.yaml` 开启了
> `offload_optimizer_device: cpu`，优化器状态会占用 CPU 内存，内存不足时训练会 OOM。

---

## 2. 软件环境检查

### 2.1 驱动 / CUDA

```bash
nvidia-smi                       # 查看 Driver Version 与 CUDA Version（驱动支持的最高版本）
nvcc --version                   # CUDA Toolkit 版本（需 >= 12.x，编译算子用）
```

> **预期结果**：驱动 CUDA >= 12.0；`nvcc` 存在且版本 >= 12.0。
> 若 `nvcc` 缺失，DeepSpeed JIT 编译算子的能力会受限（见 2.4）。

### 2.2 Python / conda

```bash
which python && python --version        # 预期 Python >= 3.10（当前 conda 环境）
conda env list                           # 确认 grpo 环境存在
conda activate grpo                      # 激活训练环境
```

### 2.3 各依赖版本核对

```bash
pip check                               # 预期输出 "No broken requirements found."
pip list | grep -iE "torch|transformers|trl|accelerate|deepspeed|vllm|peft|bitsandbytes|datasets"
```

版本要求（与 `requirements.txt` 一致）：

| 库 | 最低版本 | 备注 |
|---|---|---|
| torch / torchvision | 2.5.0 / 0.20.0 | 按集群 CUDA 版本安装（cu121/cu124/cu128） |
| transformers | 4.56.1 | |
| trl | 0.18.1 | GRPOTrainer + vLLM 后端 |
| accelerate | 1.0.0 | |
| deepspeed | 0.15.4 | ZeRO-3 |
| peft | 0.17.1 | QLoRA |
| bitsandbytes | 0.45.0 | 4bit 量化 |
| vllm | 0.8.0 | 需与 CUDA 版本匹配（12.4 默认 / 12.8 走 extra-index） |
| datasets | 3.0.0 | |

### 2.4 DeepSpeed 编译状态

```bash
deepspeed --version             # 版本号 + CUDA 版本
ds_report                       # 编译状态：JIT/AOT ops 是否全部可用
```

> **预期结果**：`ds_report` 显示 `[PASSED]` 项占绝大多数，无 `[FAILED]` 的关键算子。
> 若大量 FAILED：确认 `nvcc` 可用、`ds_report` 中的 CUDA 版本与 torch 一致，
> 必要时重装 deepspeed：`pip uninstall deepspeed && pip install deepspeed --no-cache-dir`。

### 2.5 vLLM / flash-attn / Liger（可选）

```bash
python -c "import vllm; print('vllm', vllm.__version__)"          # vLLM 已安装
python -c "import flash_attn; print('flash_attn', flash_attn.__version__)"   # flash_attention_2 后端
python -c "from trl import LigerGRPOConfig; print('Liger OK')"     # 仅从源码安装 trl 后可用
```

> **说明**：`run_grpo.py` 固定使用 `attn_implementation="flash_attention_2"`，
> 若未安装 `flash-attn`，transformers 会回退到其他实现并打印 warning（可运行但不推荐）。
> 报错 `KeyError: flash_attention_2` 时需安装：`pip install flash-attn --no-build-isolation`。

### 2.6 HuggingFace 镜像源

```bash
echo $HF_ENDPOINT                # 预期输出 https://hf-mirror.com（国内网络）
curl -sI https://hf-mirror.com | head -1    # 预期 200/302（网络可达）
```

### 2.7 wandb（可选，训练可视化）

> 安装时未加 `--with-wandb` 可跳过本节；确认开启 wandb 日志前逐项检查：

```bash
pip show wandb 2>/dev/null | grep -E "^Version"             # 预期输出 Version: 0.17.0+
python -c "import wandb; print('wandb', wandb.__version__)" # 预期无导入错误
echo ${WANDB_API_KEY:+已设置 WANDB_API_KEY}                 # 预期输出"已设置 WANDB_API_KEY"
wandb login --verify                                         # 预期输出 Valid API key（登录有效）
echo $WANDB_MODE                                             # 离线模式：预期输出 offline（如启用）
```

> **预期结果**：wandb 已安装、API Key 有效，`wandb.login()` 免交互通过。
> **说明**：本项目通过 `--report_to wandb` 启用 wandb 记录，完整说明见第 5 章。

### 2.8 Megatron-LM 环境（可选）

> 未加 `--with-megatron` 安装时可跳过本节；启用 Megatron 三条链路（预训练/SFT/GRPO）前逐项检查：

```bash
python -c "import torch; print('torch', torch.__version__)"                           # 预期 >= 2.6.0（Megatron 硬性要求）
python -c "import megatron.core; print('megatron.core', megatron.core.__version__)"   # Mcore 可用
python -c "import transformer_engine; print('TE', transformer_engine.__version__)"    # TE（mbridge 硬依赖）
python -c "import mbridge; print('mbridge OK')"                                       # HF↔Mcore 权重转换
python -c "import verl; print('verl', verl.__version__)"                              # GRPO 框架
echo $MEGATRON_PATH                                                                   # Megatron-LM 源码根目录
ls ${MEGATRON_PATH}/pretrain_gpt.py ${MEGATRON_PATH}/tools/preprocess_data.py         # 关键脚本存在
```

> **预期结果**：5 个库均可导入、torch >= 2.6.0，且 `pretrain_gpt.py` 与 `tools/preprocess_data.py` 存在。
> **常见坑**：① torch 低于 2.6.0 会导致 Megatron 导入失败（主链路只需 2.5.0）；
> ② TE 缺失会让 mbridge 转换报错（官方注明 `use_te=False` 暂不支持）。

---

## 3. 数据检查

### 3.1 数据文件存在性与行数

```bash
ls -lh data/train.jsonl data/eval.jsonl
wc -l data/train.jsonl data/eval.jsonl
```

> **预期结果**：`train.jsonl` 存在且行数符合预期（如 5000 条 → 4500 train / 500 eval）。

### 3.2 JSON 格式与字段完整性

```bash
# 首行必须是合法 JSON
head -1 data/train.jsonl | python -m json.tool

# 字段完整性统计（应输出：缺失 prompt/target/solution 的行数 = 0）
python - <<'EOF'
import json
from collections import Counter
missing = Counter()
with open("data/train.jsonl", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        rec = json.loads(line)
        for key in ("prompt", "target", "solution"):
            if not rec.get(key):
                missing[f"第{i}行 缺 {key}"] += 1
print("缺失统计:", dict(missing) or "无缺失，格式正确")
EOF
```

> **预期结果**：`{"prompt": "...", "target": "15", "solution": "..."}` 结构，
> 无缺失字段；target 均为数字字符串。

### 3.3 target 分布抽样（防数据问题）

```bash
python -c "
import json
targets = [json.loads(l)['target'] for l in open('data/train.jsonl', encoding='utf-8')]
print('样本数:', len(targets))
print('target 范围:', min(map(int, targets)), '~', max(map(int, targets)))
print('重复 target 样例:', [t for t in set(targets)][:10])
"
```

### 3.4 奖励函数自测（最关键）

```bash
python scripts/reward_functions.py
```

> **预期结果**：输出 `ALL TESTS PASSED`。
> 若 reward 异常（训练中全 0/恒值），优先排查此项与数据格式。

### 3.5 Megatron bin/idx 分词产物校验

```bash
# 确认分词产物已生成（形如 <prefix>_<key>_document.bin / .idx，必须成对出现）
ls -lh data/megatron/ | grep -E "\.bin$|\.idx$"
du -sh data/megatron/
```

> **预期结果**：`.bin` 与 `.idx` 成对存在，文件大小与样本量成正比（几百条数据约数 MB）。
> 若缺失，重新生成（注意 `--megatron-path` 指向含 `tools/preprocess_data.py` 的仓库根）：
> `python scripts/megatron/prepare_sft_data.py --mode sft --megatron-path /path/to/Megatron-LM`
>
> 训练时必须用**不带后缀**的前缀：`--data-path data/megatron/countdown_sft`。

---

## 4. 训练前检查

### 4.1 accelerate 环境

```bash
accelerate env
```

> **预期结果**：`Distributed type: DEEPSPEED`、`Mixed precision: bf16`、
> `GPU count: N-1`（与启动脚本计算一致）。

### 4.2 recipe 关键参数核对

```bash
cat recipes/grpo-qwen-2.5-3b-countdown.yaml          # 全参数
cat recipes/grpo-qwen-2.5-3b-countdown-qlora.yaml    # QLoRA
```

| 参数 | 建议值 | 检查点 |
|---|---|---|
| `vllm` | true | 训练/推理分离，加速生成 |
| `learning_rate` | 5e-7（GRPO 较敏感） | 过高易发散 |
| `max_prompt_length` / `max_completion_length` | 256 / 1024 | 与数据集长度匹配 |
| `num_generations` | 2 | 每 prompt 生成数（GRPO 优势，越大越稳但更慢） |
| `beta` | 0.001 | KL 惩罚系数 |
| `gradient_checkpointing` | true | 省显存（ZeRO-3 下强烈建议） |
| `per_device_train_batch_size` | 1 | 调大前先确认显存 |
| `max_steps` | 450 | 按实验规模调整 |
| `logging_steps` / `save_steps` | 10 / 100 | 监控与断点频率 |
| `report_to` | tensorboard | 训练曲线可视化 |

### 4.3 模型加载冒烟测试（CPU 也可）

```bash
python - <<'EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer
model_id = "Qwen/Qwen2.5-3B-Instruct"
tok = AutoTokenizer.from_pretrained(model_id)
print("tokenizer 词表:", tok.vocab_size)
# 仅加载 config + 随机权重验证架构可加载（正式训练由 run_grpo.py 加载真权重）
EOF
```

> **提示**：首次运行会从 HF Hub 下载模型（约 6GB），请确保镜像源可用或已预下载。

### 4.4 输出目录 / 磁盘

```bash
mkdir -p output && touch output/.write_test && rm output/.write_test && echo "输出目录可写"
```

### 4.5 正式启动（含 vLLM 卡预留确认）

```bash
bash scripts/launch_train.sh --mode full    # 或 --mode qlora
```

> **预期**：脚本打印 GPU 数、`num_processes = GPU 数 - 1`、完整 accelerate 命令，
> 随后进入训练；`nvidia-smi` 可见训练进程 + vLLM 进程。

### 4.6 Megatron 并行度预检（改用 Megatron 链路时）

```bash
bash scripts/megatron/pretrain_qwen.sh --dry-run     # 只做校验并打印命令，不真正启动
```

> **预期结果**：输出 `并行校验通过：TP=... PP=... CP=... EP=... -> DP=...` 并打印完整 torchrun 命令。
> 此步会提前拦截以下错误：**world_size 不能被 TP×PP×CP 整除**、层数不能被 PP 整除、
> 注意力头数/GQA 分组数不能被 TP 整除、**batch 与 DP×运行数不匹配**。
> 在此报错说明配置非法，**不要**直接提交到集群排队。

---

## 5. wandb 使用说明（可选）

> 本章适用于已用 `bash scripts/setup_env.sh --with-wandb` 安装 wandb 的场景；
> 不开启 wandb 时保持默认 `report_to: tensorboard` 即可，不影响训练。

### 5.1 安装与登录

```bash
# ① 安装（setup_env.sh 已加 --with-wandb 时跳过此步）
pip install "wandb>=0.17.0"

# ② 登录（二选一）：
#    方式 A：交互式登录，浏览器授权
wandb login
#    方式 B：无交互，使用 API Key（https://wandb.ai/authorize 获取）
export WANDB_API_KEY="your-api-key"      # 建议写入 ~/.bashrc 持久生效
```

> 登录状态可用 `wandb login --verify` 校验；多账号切换用 `wandb login --relogin`。

### 5.2 启用方式

**方式 A：CLI 覆盖（不改配置文件，临时开启）**

```bash
bash scripts/launch_train.sh --report_to wandb
```

**方式 B：recipe 写入（固定开启）**

在 `recipes/grpo-qwen-2.5-3b-countdown.yaml` 中增加：

```yaml
report_to: wandb
wandb_project: grpo-countdown      # 项目名（网页端按此分组）
wandb_entity: <用户名或团队名>      # 可省略，默认使用个人账号
```

> `run_grpo.py` 会把 recipe 中 `wandb_project` / `wandb_entity` 自动透传给 `GRPOConfig`，
> 无需修改 Python 代码；两个参数也可与方式 A 组合使用。

### 5.3 自动记录的指标

| 指标 | 含义 |
|---|---|
| `loss` | 总损失（含 KL 项） |
| `rewards` / `rewards_std` | 奖励均值 / 标准差 |
| `kl` | 与参考模型的 KL 散度 |
| `completion_length` | 生成序列平均长度 |
| `log_completions` | 采样生成的完整文本（用于人工检查输出格式） |

### 5.4 查看

- 网页端：登录 https://wandb.ai → 打开 `wandb_project` 对应项目，即可查看实时曲线（loss / rewards / kl 等）与生成样本。
- 命令同步：离线训练后，把本地 run 同步到云端：

```bash
wandb sync output/grpo-countdown/wandb/run-*   # 路径以实际输出目录为准
```

### 5.5 国内网络注意事项

- 无法直连 wandb.ai 时，推荐**离线模式**训练（曲线仍写入本地，训练后手动 `wandb sync`）：

```bash
export WANDB_MODE=offline        # 或写入 ~/.bashrc 持久生效
bash scripts/launch_train.sh --report_to wandb
```

- 登录失败 / 上传卡住的处理：
  ① `wandb login --relogin` 重新授权；
  ② 检查 `WANDB_API_KEY` 是否有效；
  ③ 改回 TensorBoard：`bash scripts/launch_train.sh --report_to tensorboard`。

---

## 6. 常见问题排查表

| 现象 | 可能原因 | 排查命令 / 处理方式 |
|---|---|---|
| **ZeRO-3 显存不足（CUDA OOM）** | batch 过大 / num_generations 过多 / offload 未生效 | ① `nvidia-smi` 确认显存占用；② 调小 `per_device_train_batch_size`、`gradient_accumulation_steps` 补偿、`num_generations`；③ 确认 `configs/accelerate_configs/deepspeed_zero3.yaml` 中 `offload_optimizer_device: cpu`、`zero3_init_flag: true`；④ 开启 `gradient_checkpointing: true` |
| **QLoRA + ZeRO-3 offload 报错** | bnb 4bit 与 offload 兼容限制 | 将 `deepspeed_zero3.yaml` 中 `offload_optimizer_device` 改为 `none`；或改用全参数模式 |
| **vLLM 生成超时 / 卡住** | 端口占用 / vLLM 显存不足 / 与训练抢卡 | ① 查看报错中端口号，`ss -tlnp \| grep <port>`；② 确认 `num_processes = GPU数-1`，最后 1 卡空闲；③ vLLM 显存不足可调小 `max_prompt_length`/`max_completion_length` |
| **vLLM 报 CUDA 版本不匹配** | vllm wheel 与 torch CUDA 不一致 | `python -c "import vllm, torch; print(vllm.__version__, torch.version.cuda)"`；CUDA 12.8 需 `pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128` |
| **loss 发散 / NaN** | 学习率过高 / warmup 不足 / bf16 溢出 | ① 调小 `learning_rate`（如 5e-7 → 1e-7）；② 增加 `warmup_ratio`；③ 检查 `beta` 是否合理；④ 确认 `bf16: true` 且 `mixed_precision: bf16` 一致 |
| **reward 全 0 或恒值** | 数据格式问题 / 模型未学会格式 / 奖励函数 bug | ① `python scripts/reward_functions.py` 自测；② 检查 `data/train.jsonl` 的 target 是否可从 prompt 解析；③ 查看 `log_completions: true` 记录的生成文本（TensorBoard 或日志）确认格式 |
| **reward 在 0~2 之间但不上升** | 训练步数不足 / KL 系数过大 | ① 确认 `max_steps` 是否过短；② `beta` 过大抑制策略更新；③ 用 TensorBoard 对比 rewards 与 kl 曲线 |
| **checkpoint 无法加载 / 目录为空** | save_steps 未到 / 磁盘满 / ZeRO 保存限制 | ① `ls output/<run_dir>/checkpoint-*`；② `df -h` 查磁盘；③ ZeRO-3 需 `zero3_save_16bit_model: true`（已配置） |
| **`flash_attention_2` 报错** | 未装 flash-attn 或 GPU 架构不支持 | `pip install flash-attn --no-build-isolation`；H100/A100/L40S 支持，老架构（如 V100）不支持时改用 `attn_implementation="sdpa"` |
| **GLIBC/CUDA 版本相关报错** | torch/vllm wheel 与系统环境不匹配 | 确认 `ldd --version` 与 torch 构建要求；建议用 conda 环境（自带兼容 glibc）重装 |
| **多机训练连接失败** | 节点间网络不通 / rdzv 配置 | `ping <其他节点IP>`、确认 `num_machines`/`machine_rank`/`main_process_ip` 配置；单机训练可不关注 |
| **数据行数过少报错** | 数据集不足 / GRPO batch 组合不满 | GRPO 每步需 `per_device_train_batch_size × num_generations` 条样本；小数据验证用 `bash scripts/quick_test.sh` |
| **下载模型/数据集超时** | 网络问题 | `export HF_ENDPOINT=https://hf-mirror.com`；或提前 `huggingface-cli download Qwen/Qwen2.5-3B-Instruct` 预下载 |
| **wandb 连接超时 / 登录失败** | 网络无法直连 / API Key 无效 | ① `wandb login --relogin` 重新授权；② 确认 `WANDB_API_KEY` 正确；③ `export WANDB_MODE=offline` 离线训练后 `wandb sync`；④ 退回 `--report_to tensorboard` |
| **TransformerEngine 编译 OOM / 长时间卡死** | 并行编译任务过多 | ① `MAX_JOBS=4` 限制并发（`--max-jobs 4`）；② 用 `--megatron-lite` 跳过 TE 编译；③ 内存不足时改用 NGC PyTorch 容器（依赖预编译） |
| **`import megatron.core` 失败 / 报 torch 版本** | torch 低于 2.6.0（主链路只需 2.5.0） | `python -c "import torch; print(torch.__version__)"`；按集群 CUDA 升级：`pip install "torch>=2.6.0" --index-url https://download.pytorch.org/whl/cu124` |
| **并行度报错：乘积不整除 / TP 非法** | TP×PP×CP 组合不匹配，或 TP 超过 GQA 分组数 | ① 确认 `world_size == TP×PP×CP×DP`；② Qwen2.5-3B 的 GQA=2，**TP 最大为 2**；③ 改扩 PP / DP；详见 `configs/megatron/README.md` |
| **mbridge 转换失败 / use_te 相关报错** | 未安装 TransformerEngine | `python -c "import transformer_engine"`；缺失则重装含 `[dev]` extras 的 megatron-core（mbridge 官方注明 `use_te=False` 暂不支持） |
| **Megatron 报数据路径不存在** | bin/idx 未生成，或 `--data-path` 写错 | ① `ls data/megatron/<prefix>*.bin`；② `--data-path` 填**不带后缀**的前缀；③ 重跑 `prepare_sft_data.py --megatron-path <repo>` |
| **veRL 报 `Could not resolve config xxx`** | veRL 版本与脚本使用的配置键不一致 | 对照安装版本的 `verl/trainer/config/ppo_trainer.yaml`，调整 `grpo_verl_megatron.sh` 中的配置键（veRL 迭代快，键名随版本变化） |

---

## 附：快速定位日志

```bash
# 训练日志（quick_test.sh 自动写入）
grep -E "'loss'|'rewards'" output/quick_test/train_log.txt | tail -n 20

# 正式训练输出（stdout 被 terminal 捕获时）
ls output/grpo-countdown/runs/
tensorboard --logdir output/grpo-countdown/runs
```

> 最后：若问题不在上述清单内，请带上 `nvidia-smi` 输出、`train_log.txt` 最后 50 行、
> 以及完整报错栈（含 `Traceback`）提交排查。
