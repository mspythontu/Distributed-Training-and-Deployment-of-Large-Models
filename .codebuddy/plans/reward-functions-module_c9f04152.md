---
name: reward-functions-module
overview: 编写 scripts/reward_functions.py：为 Countdown 倒计时游戏的 GRPO 训练实现奖励函数模块，包含正确性奖励（AST 白名单安全求值 + 数字使用规则校验 + 结果比较）与格式奖励（<thinking>/<answer> 标签校验），签名兼容 TRL GRPOTrainer（completions, **kwargs），并附带自测。
todos:
  - id: write-reward-functions
    content: 编写 scripts/reward_functions.py：工具函数、AST 白名单安全求值、correctness_reward/format_reward 及内嵌自测
    status: completed
  - id: run-self-test
    content: 本地运行 python scripts/reward_functions.py 自测，验证各场景奖励输出并修复问题
    status: completed
    dependencies:
      - write-reward-functions
---

## 产品概述

为 GRPO 训练编写 Countdown 倒计时游戏奖励函数模块 `scripts/reward_functions.py`，供 TRL `GRPOTrainer` 在训练时评估模型输出。模型需给定数字与目标值用四则运算凑出目标值，输出格式为 `<thinking>推理过程</thinking><answer>表达式 = 目标值`。

## 核心功能

- **正确性奖励** `correctness_reward`：解析 `<answer>` 中 "=" 左侧的表达式，通过 AST 白名单安全求值，校验表达式中使用的数字均来自给定集合且每个数字最多使用一次，结果与目标值匹配（浮点容差）则奖励 1.0，否则 0.0
- **格式奖励** `format_reward`：正则校验输出是否严格包含 `<thinking>...</thinking>` 与 `<answer>...</answer>` 标签，符合则奖励 1.0
- **安全求值**：不使用裸 `eval`，用 `ast` 白名单仅允许数字常量、`+ - * /` 二元运算、一元正负号与括号，杜绝任意代码注入
- **TRL 兼容**：采用新式签名 `(completions, **kwargs) -> list[float]`；目标值优先从 `kwargs["target"]` 读取（数据预处理脚本已输出该列），缺失时回退从 prompt 中解析 `Target: X`
- **工具函数**：`parse_target_from_prompt`、`parse_numbers_from_prompt`、`extract_answer`、`extract_expression`，供奖励函数复用及后续训练脚本调用
- **内嵌自测**：`if __name__ == "__main__"` 提供正确解、错误解、数字作弊、注入攻击、缺标签等场景样例，直接运行即可验证

## 技术栈

- 纯 Python 标准库实现（`ast` / `re` / `math` / `collections`），零外部依赖，无需网络，可在本地 Windows（Python 3.12）直接运行验证
- 与现有 `scripts/prepare_countdown_data.py` 保持一致的代码风格：模块 docstring、`from __future__ import annotations`、类型标注、工具函数 + 入口函数组织

## 实现方案

### 总体策略

编写独立、可复用的奖励函数模块，严格遵循 TRL 0.18.x 的 GRPO 奖励函数新式签名约定，并复现 TRL 官方 `grpo_countdown.py` 示例的奖励设计，同时用 AST 白名单替代官方示例中的裸 `eval` 以提升安全性。

### 关键设计

1. **解析链**：`extract_answer` 用正则提取 `<answer>...</answer>` 内容（无闭合标签时回退取 `<answer>` 之后文本）→ `extract_expression` 取 "=" 左侧表达式 → `safe_eval_expression` 求值
2. **AST 白名单校验**（`safe_eval_expression(expr, allowed_numbers)`）：

- 遍历 AST 节点，仅允许 `ast.Constant`（int/float 且非 bool）、`ast.BinOp`（Add/Sub/Mult/Div）、`ast.UnaryOp`（UAdd/USub）、括号节点
- 兼容 Python 3.12 新增的 `ast.Parenthesized` 节点（用 `hasattr` 检查，兼容 3.10+）
- 校验通过后用 `compile` + `eval(..., {"__builtins__": {}}, {})` 求值，捕获 `ZeroDivisionError` 等异常
- 收集表达式中出现的数字，用 `collections.Counter` 校验均来自 `allowed_numbers` 且使用次数不超过其出现次数

3. **目标值获取**：`correctness_reward` 中 `targets = kwargs.get("target")`，存在则按索引取，否则对每个样本 `parse_target_from_prompt(prompts[i])`
4. **结果比较**：`math.isclose`（`rel_tol=1e-9, abs_tol=1e-6`）处理除法产生浮点结果的情况
5. **格式奖励**：正则 `^<thinking>.*?</thinking>\s*<answer>.*?</answer>（DOTALL 模式），与 TRL 官方示例一致

### 复杂度与健壮性

- 时间 O(n × m)，n 为批次样本数、m 为表达式长度，毫秒级开销，不构成训练瓶颈
- 所有解析/求值路径均有异常兜底返回 0.0，避免单个样本异常中断训练
- 模块零外部依赖，可被 `train_grpo.py` 直接 `import`，也可独立运行自测

## 实现注意

- 返回值为 `list[float]`，元素为 0.0/1.0，与 TRL `GRPOTrainer` 的 `reward_funcs` 要求一致
- 不修改任何既有文件，本次改动仅新增 `scripts/reward_functions.py`
- 自测覆盖：正确解、错误解、数字未给出/超次数使用、注入攻击（如 `__import__('os')`）、无标签输出、除零表达式、含括号表达式、负数表达式等