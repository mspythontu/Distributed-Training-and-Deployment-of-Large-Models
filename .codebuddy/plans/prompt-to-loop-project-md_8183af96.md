---
name: prompt-to-loop-project-md
overview: 在项目根目录创建 project.md：将本会话中用户提出的全部提示词，按 Loop Engineering（循环工程）思想重构为"工程循环"文档，每个提示词对应一个完整的 Observe→Think→Act→Verify→Feedback 闭环单元，并绘制整体闭环演进图谱。
todos:
  - id: create-skeleton
    content: 创建 project.md 骨架：文档定位、Loop Engineering 概览、项目循环架构（Harness）与交付物注册表章节
    status: completed
  - id: write-loop-1-2
    content: 编写 Loop 1（环境搭建交付）与 Loop 2（确认实施）单元，交叉引用 scripts/ 与 docs/ 真实路径
    status: completed
    dependencies:
      - create-skeleton
  - id: write-loop-3-5
    content: 编写 Loop 3（wandb 咨询）、Loop 4（wandb 落地）、Loop 5（当前转换请求）单元
    status: completed
    dependencies:
      - write-loop-1-2
  - id: finalize-verify
    content: 补全闭环演进图谱（mermaid）与循环使用指南；核对全部引用路径存在性与 Markdown 结构完整性
    status: completed
    dependencies:
      - write-loop-3-5
---

## 用户需求

将本会话中用户在该项目提出的**全部提示词**，按照 **Loop Engineering（循环工程）** 思想转换为 `project.md`，放置于项目根目录。

## 产品概述

- 转换对象：本会话中用户向 AI 提出的 5 条提示词（按时间顺序：环境搭建交付请求 → "确认" → wandb 使用方法咨询 → "需要"确认 wandb 落地 → 当前转换请求）
- 核心思想（已确认）：AI 工程四层演进范式 **Prompt → Context → Harness → Loop**；每个循环单元由 **Observe（感知）→ Think（思考）→ Act（行动）→ Verify（验证）→ Feedback（反馈）** 五段闭环组成，从"人工触发的一次性问答"升级为"可递进、可复用的自主循环"
- 交付物：根目录 `project.md`，中文撰写，定位为项目"工程循环元文档"——区别于 README（用法说明）与 CHECKLIST（操作清单），记录每条提示词从触发到闭环的完整过程

## 核心功能

- **Loop Engineering 概览**：四层范式演进说明、循环单元五段定义
- **项目循环架构（Harness）**：触发源、工具链/脚手架、验证机制、交付物注册表（交叉引用真实文件路径）
- **提示词循环集合**：5 个 Loop 单元，每个含触发 Prompt（原文/意图重建）、Observe、Think、Act、Verify、Feedback 六字段
- **闭环演进图谱**：mermaid 流程图展示 5 个 Loop 的递进关系
- **循环使用指南**：文档阅读方法、新增提示词的登记模板、与 README/CHECKLIST 的分工说明

## 实现方案

### 1. project.md 文档架构（章节规划）

```
# 项目提示词 → Loop 工程档案

## 1. Loop Engineering 思想概览
   四层范式演进表（Prompt/Context/Harness/Loop）+ 循环单元五段定义

## 2. 项目循环架构（Harness）
   触发源、工具链注册表、验证机制、交付物注册表（表格：路径 → 职责 → 所属 Loop）

## 3. 提示词循环集合（Loop 1~5）
   5 个 Loop 单元，字段统一：
   - 触发 Prompt（有原文引用原文；压缩会话缺失原文的以"意图重建 + 标注"处理）
   - Observe（感知）：项目现状 / 上下文事实
   - Think（规划）：关键决策与设计取舍
   - Act（行动）：交付物文件路径 + 核心实现要点
   - Verify（验证）：验证方式与结果
   - Feedback（反馈）：闭环结论 + 衔接下一 Loop 的线索

## 4. 闭环演进图谱
   mermaid flowchart 展示 Loop 1→5 递进关系

## 5. 循环使用指南
   阅读方法、新增提示词登记模板、与 README/CHECKLIST 分工
```

### 2. 5 个 Loop 单元内容映射（基于会话记录与已交付文件）

| Loop | 触发 Prompt | Act 交付物（真实路径） | Verify 结果 |
| --- | --- | --- | --- |
| 1 | 编写启动脚本/快速验证/检查清单/环境安装 | scripts/launch_train.sh、scripts/quick_test.sh、docs/CHECKLIST.md、scripts/setup_env.sh | bash -n 语法通过、与 run_grpo.py CLI 契约对齐 |
| 2 | "确认"（批准计划进入实施） | 无新增文件，驱动 Loop 1 执行 | 交付物落地 |
| 3 | wandb 使用方法咨询 | 知识性解答（无代码改动），识别可落地改进 | 与 run_grpo.py `--report_to wandb` 契约核对 |
| 4 | "需要"（确认 wandb 落地） | scripts/setup_env.sh 增加 `--with-wandb`；docs/CHECKLIST.md 新增 2.7/第5章/排查条目 | bash -n 通过、LF 行尾、交叉引用无冲突 |
| 5 | 当前请求（转换 project.md） | project.md（本文件） | 路径存在性 + Markdown 完整性核对 |


### 3. 关键设计决策

- **交叉引用真实路径**：每个 Loop 的 Act 字段引用项目实际文件（scripts/*.sh、docs/CHECKLIST.md、configs/、recipes/、requirements.txt），保证可追溯性
- **缺失原文处理**：会话经压缩的部分提示词（如 wandb 咨询原文）按 relative_history 汇总的意图重建并标注"（原文经会话压缩，按意图重建）"，不虚构细节
- **风格对齐**：中文撰写，表格 + 引用块 + 代码块，与 README/CHECKLIST 一致的文档风格
- **mermaid 图**：闭环演进图谱使用 ```mermaid 代码块包裹

### 4. 验证方式

- 逐一核对 project.md 引用的文件路径真实存在（read 确认）
- Markdown 结构完整性：标题层级、5 个 Loop 单元字段齐全、代码块/mermaid 块闭合、表格格式正确
- 5 条提示词覆盖完整性（与相对历史清单逐一比对）