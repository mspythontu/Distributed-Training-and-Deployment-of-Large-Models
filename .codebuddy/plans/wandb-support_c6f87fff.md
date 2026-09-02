---
name: wandb-support
overview: 为项目增加 wandb 支持：setup_env.sh 增加可选安装 wandb 的参数，docs/CHECKLIST.md 补充完整的 wandb 使用说明章节。
todos:
  - id: update-setup-env
    content: 修改 scripts/setup_env.sh：新增 --with-wandb 参数、条件安装 wandb、验证段打印版本、更新头部注释与结尾提示
    status: completed
  - id: update-checklist
    content: 修改 docs/CHECKLIST.md：新增 2.7 wandb 检查项、第 5 章 wandb 使用说明、排查表补充条目并顺延章节编号
    status: completed
  - id: verify-changes
    content: 验证：bash -n 语法检查（临时目录中转）、LF 行尾与文件路径/参数契约一致性核对
    status: completed
    dependencies:
      - update-setup-env
      - update-checklist
---

## 用户需求

用户确认需要将 wandb（Weights & Biases）支持落地到项目中，共两个改动：

1. **调整 `scripts/setup_env.sh` 增加 wandb 安装选项**

- 新增可选安装开关（默认不装，与 `requirements.txt` 中 wandb 被注释的可选定位保持一致）
- 指定版本安装（`wandb>=0.17.0`，与 requirements.txt 注释一致）
- 安装完成后在验证段打印 wandb 版本
- 更新脚本头部注释（用法示例 + 参数说明）

2. **将 wandb 使用说明补充进 `docs/CHECKLIST.md`**

- 软件环境检查部分新增 wandb 检查项（安装状态、登录状态、API Key、离线模式）
- 新增 wandb 使用说明章节：安装与登录、两种启用方式（CLI 覆盖 / recipe 配置）、自动记录的指标、查看方式、国内网络注意事项
- 常见问题排查表补充 wandb 相关条目（连接超时/登录失败的处理方式）

## 边界

- 不修改 `scripts/run_grpo.py`（已支持 `--report_to wandb`，`wandb_project`/`wandb_entity` 由 recipe YAML 自动透传）
- 不修改 `requirements.txt` 与两个 recipe 文件
- 脚本保持 bash + LF 行尾、详细中文注释、良好错误处理

## 技术栈

- Shell 脚本：bash（沿用 `setup_env.sh` 现有 `set -euo pipefail`、彩色日志、`die` 错误处理风格）
- 文档：Markdown（沿用 `docs/CHECKLIST.md` 现有章节结构与命令+预期结果格式）

## 实现方案

### 1. `scripts/setup_env.sh` — 增加 wandb 可选安装

- **参数**：新增 `--with-wandb` 布尔开关（默认 `WITH_WANDB="no"`），风格与现有 `--source-trl` 一致
- **安装位置**：在 [4/5] trl 源码安装块之后、HF 镜像配置之前插入条件块：

```
if [[ "$WITH_WANDB" == "yes" ]]; then
log_info "安装 wandb（可选，训练可视化）..."
pip install "wandb>=0.17.0"
log_ok "wandb 已安装，可配合 --report_to wandb 使用"
fi
```

- **验证段**：在 python heredoc 版本列表后追加条件打印（不并入通用 tuple，避免未安装时误报"导入失败"）：

```
if [[ "$WITH_WANDB" == "yes" ]]; then
python -c "import wandb; print(f'  wandb           : {wandb.**version**}')"
fi
```

- **头部注释**：用法区增加 `bash scripts/setup_env.sh --with-wandb` 示例；参数说明区增加 `--with-wandb 安装 wandb（训练可视化，配合 --report_to wandb 使用，默认不装）`
- **结尾提示**：`--with-wandb` 时追加一行 `bash scripts/launch_train.sh --report_to wandb  # 使用 wandb 记录训练`

### 2. `docs/CHECKLIST.md` — 补充 wandb 检查项与使用说明

- **2.7 wandb（可选）**：软件环境检查新增小节，含检查命令：
- `pip show wandb` / `python -c "import wandb; print(wandb.__version__)"` — 安装状态
- `wandb login` / `echo ${WANDB_API_KEY:+已设置}` — 登录状态
- `echo $WANDB_MODE` — 离线模式确认
- **新增第 5 章「wandb 使用说明（可选）」**（原第 5 章排查表顺延为第 6 章），内容：
- 安装与登录（`pip install wandb`、`wandb login`、`WANDB_API_KEY` 免交互）
- 启用方式 A：CLI 覆盖 `bash scripts/launch_train.sh --report_to wandb`；方式 B：recipe 写入 `report_to: wandb` + `wandb_project: grpo-countdown` + `wandb_entity`
- 自动记录指标表（loss、rewards、rewards_std、kl、completion_length、log_completions 生成的样本文本）
- 查看：网页端 wandb.ai + 离线模式 `WANDB_MODE=offline` 训练后 `wandb sync`
- 国内网络注意事项（离线优先、登录失败处理、可退回 tensorboard）
- **排查表新增条目**：wandb 连接超时/登录失败 → 处理方式（`wandb login --relogin`、`WANDB_MODE=offline`、退回 `--report_to tensorboard`）

## 设计取舍

- 采用 `--with-wandb` 可选开关而非默认安装：与 `requirements.txt` 中 wandb 被注释的可选定位一致，避免强制安装非必需依赖，符合 YAGNI
- 验证段条件打印而非并入通用 tuple：未安装时不会出现误导性的"导入失败"输出
- CHECKLIST 章节编号顺延（5→6）仅影响两处标题，改动面小、文档结构更合理