#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Countdown 倒计时游戏 GRPO 奖励函数模块。

模型任务：给定一组数字和一个目标值，用四则运算（+ - * /）凑出目标值，
输出格式为：<thinking>推理过程</thinking><answer>表达式 = 目标值

本模块提供三个 GRPOTrainer 兼容的奖励函数（签名 `(completions, **kwargs) -> list[float]`）：
1. correctness_reward : 表达式合法 + 数字均来自给定集合且不超限使用 + 结果等于目标值 -> 1.0
2. format_reward      : 输出严格匹配 <thinking>...</thinking><answer>...</answer> 格式 -> 1.0
3. combined_reward    : 组合奖励 = format_reward(0/1) + correctness_reward(0/1)，范围 0~2

目标值获取：优先使用 kwargs["target"]（数据预处理脚本输出的 jsonl 包含该列），
缺失时回退从 prompt 中解析 "Target: X"。

与 TRL 官方 grpo_countdown.py 示例的差异：使用 AST 白名单安全求值，
替代裸 eval，杜绝任意代码注入风险。

用法示例（GRPOTrainer 中）：
    reward_funcs=[combined_reward]
直接运行本文件可执行自测：
    python scripts/reward_functions.py
"""

from __future__ import annotations

import ast
import math
import re
from collections import Counter
from typing import Optional, Tuple

# 表达式 = 目标值 中的 "=" 分隔
ANSWER_SPLIT_RE = re.compile(r"\s*=\s*")
# <answer>...</answer> 标签提取（无闭合标签时回退取 <answer> 之后全部）
ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
ANSWER_TAG_FALLBACK_RE = re.compile(r"<answer>(.*)", re.DOTALL)
# 格式奖励用：以 <thinking> 开头、含闭合 </thinking>，随后 <answer> 开标签直至结尾。
# 注意：按项目 prompt 约定，<answer> 为开标签（无闭合），后面直接跟"表达式 = 目标值"。
FORMAT_PATTERN = re.compile(r"^<thinking>.*?</thinking>\s*<answer>.*$", re.DOTALL)

_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)


# ---------------------------------------------------------------------------
# Prompt 解析工具
# ---------------------------------------------------------------------------
def parse_target_from_prompt(prompt: str) -> Optional[int]:
    """从 prompt 中解析目标值，例如 "Target: 15" -> 15。"""
    match = re.search(r"Target:\s*(-?\d+)", prompt)
    return int(match.group(1)) if match else None


def parse_numbers_from_prompt(prompt: str) -> list[int]:
    """从 prompt 中解析可用数字列表，例如 "Numbers: [3, 5, 2, 9]" -> [3, 5, 2, 9]。"""
    match = re.search(r"Numbers:\s*\[([^\]]*)\]", prompt)
    if not match:
        return []
    return [int(x) for x in re.findall(r"-?\d+", match.group(1))]


# ---------------------------------------------------------------------------
# 输出解析工具
# ---------------------------------------------------------------------------
def extract_answer(completion: str) -> str:
    """提取 <answer> 标签中的内容；无闭合标签时回退取 <answer> 之后全部文本。"""
    match = ANSWER_TAG_RE.search(completion)
    if match:
        return match.group(1).strip()
    match = ANSWER_TAG_FALLBACK_RE.search(completion)
    return match.group(1).strip() if match else ""


def extract_expression(answer: str) -> str:
    """从 '<表达式> = 目标值' 中提取 "=" 左侧的表达式；无 "=" 时取整个 answer。"""
    parts = ANSWER_SPLIT_RE.split(answer, maxsplit=1)
    return parts[0].strip() if parts else ""


# ---------------------------------------------------------------------------
# AST 白名单安全求值
# ---------------------------------------------------------------------------
def _validate_ast(node: ast.AST, used_numbers: list[float]) -> bool:
    """递归校验 AST 节点，仅允许数字常量与 + - * / 运算、一元正负号、括号。

    同时把表达式中的数字常量收集到 used_numbers。
    """
    if isinstance(node, ast.Expression):
        return _validate_ast(node.body, used_numbers)
    # Python 3.12+ 会生成 Parenthesized 节点，低版本无此类
    if hasattr(ast, "Parenthesized") and isinstance(node, ast.Parenthesized):
        return _validate_ast(node.body, used_numbers)
    if isinstance(node, ast.Constant):
        # 仅允许 int/float，拒绝 bool、字符串等
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            used_numbers.append(float(node.value))
            return True
        return False
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, _ALLOWED_BINOPS):
            return False
        return _validate_ast(node.left, used_numbers) and _validate_ast(node.right, used_numbers)
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, _ALLOWED_UNARYOPS):
            return False
        return _validate_ast(node.operand, used_numbers)
    return False


def _numbers_allowed(used_numbers: list[float], allowed_numbers: list[int]) -> bool:
    """校验表达式中使用的每个数字都在给定集合内，且使用次数不超过其出现次数。"""
    allowed_counter = Counter(allowed_numbers)
    for num in used_numbers:
        if not num.is_integer():  # 不允许小数（Countdown 数字均为整数）
            return False
        n = int(num)
        if allowed_counter[n] <= 0:
            return False
        allowed_counter[n] -= 1
    return True


def safe_eval_expression(
    expr: str, allowed_numbers: list[int] | None = None
) -> Optional[Tuple[float, list[float]]]:
    """AST 白名单安全求值，返回 (计算结果, 表达式用到的数字列表)；非法表达式返回 None。

    仅允许数字常量、+ - * /、一元正负号与括号，杜绝任意代码注入（如 __import__）。
    allowed_numbers 不为 None 时，额外校验表达式中的数字均来自该集合且不超限使用。
    """
    expr = expr.strip()
    if not expr:
        return None
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    used_numbers: list[float] = []
    if not _validate_ast(tree, used_numbers):
        return None
    if allowed_numbers is not None and not _numbers_allowed(used_numbers, allowed_numbers):
        return None

    try:
        # 白名单校验已通过，此处 eval 只面对数字与算术运算符
        code = compile(tree, "<string>", "eval")
        result = eval(code, {"__builtins__": {}}, {})  # noqa: S307 - AST 白名单已校验
        return float(result), used_numbers
    except (ZeroDivisionError, OverflowError, ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# GRPOTrainer 奖励函数（TRL 0.18.x 新式签名）
# ---------------------------------------------------------------------------
def correctness_reward(completions: list[str], **kwargs) -> list[float]:
    """正确性奖励：表达式合法 + 数字合规 + 结果等于目标值 -> 1.0，否则 0.0。

    目标值优先取 kwargs["target"]（与 completions 同序），缺失时从 prompt 解析。
    """
    prompts = kwargs.get("prompts") or [""] * len(completions)
    targets = kwargs.get("target")
    rewards: list[float] = []
    for i, completion in enumerate(completions):
        target = None
        if targets is not None and i < len(targets) and targets[i] is not None:
            target = int(targets[i])
        elif prompts:
            target = parse_target_from_prompt(prompts[i])
        if target is None:
            rewards.append(0.0)
            continue

        numbers = parse_numbers_from_prompt(prompts[i]) if prompts else []
        answer = extract_answer(completion)
        expr = extract_expression(answer)
        # 未能从 prompt 解析出数字时不校验数字使用（保留求值正确性判断）
        evaluated = safe_eval_expression(expr, allowed_numbers=numbers if numbers else None)
        if evaluated is None:
            rewards.append(0.0)
            continue
        result, _ = evaluated
        rewards.append(1.0 if math.isclose(result, float(target), rel_tol=1e-9, abs_tol=1e-6) else 0.0)
    return rewards


def format_reward(completions: list[str], **kwargs) -> list[float]:
    """格式奖励：输出严格匹配 <thinking>...</thinking><answer>...</answer> -> 1.0。"""
    del kwargs  # 未使用
    return [1.0 if FORMAT_PATTERN.match(c) else 0.0 for c in completions]


def combined_reward(completions: list[str], **kwargs) -> list[float]:
    """组合奖励：格式奖励(0/1) + 正确性奖励(0/1)，范围 0~2。

    同时鼓励模型"按格式输出"且"答案正确"：仅格式对得 1 分，仅答案对得 1 分，
    两者都对得满分 2 分。作为单一奖励函数传入 GRPOTrainer 即可。
    """
    format_scores = format_reward(completions, **kwargs)
    correctness_scores = correctness_reward(completions, **kwargs)
    return [fmt + corr for fmt, corr in zip(format_scores, correctness_scores)]


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
def _run_self_test() -> None:
    prompt = (
        "You are given a set of numbers and a target value. Use basic arithmetic "
        "operations (+, -, *, /) to reach the target value. You must use each number "
        "at most once.\n\n"
        "Numbers: [3, 5, 2, 9]\n"
        "Target: 15\n\n"
        "First, reason step by step inside <thinking> tags. Then provide your final "
        "answer in the format: <answer>expression = target"
    )

    # (completion, 期望 correctness, 期望 format)
    cases = [
        # 正确解：3*5 = 15
        ("<thinking>3 乘以 5 等于 15。</thinking><answer>3*5 = 15", 1.0, 1.0),
        # 正确解：含括号 + 除法（数字 9,3,5,2 各用一次）
        ("<thinking>先算括号。</thinking><answer>(9-3)*5/2 = 15", 1.0, 1.0),
        # 正确解：除法 + 负数
        ("<thinking>9 除以 3 得 3，5 加 2 得 7，3 乘 7 得 21，21 减 6... </thinking><answer>5*(9/3) = 15", 1.0, 1.0),
        # 错误解：结果不等于目标值
        ("<thinking>算了算。</thinking><answer>3+5+2 = 10", 0.0, 1.0),
        # 数字作弊：使用了未给出的数字 7
        ("<thinking>猜一个。</thinking><answer>7+8 = 15", 0.0, 1.0),
        # 数字超限使用：5 只给了 1 次，用了 2 次
        ("<thinking>重复用。</thinking><answer>5+5+5 = 15", 0.0, 1.0),
        # 表达式为空
        ("<thinking>没有答案。</thinking><answer> = 15", 0.0, 1.0),
        # 除零
        ("<thinking>除零。</thinking><answer>9/0 = 15", 0.0, 1.0),
        # 注入攻击：__import__ 不应被求值
        ("<thinking>注入。</thinking><answer>__import__('os').system('ls') = 15", 0.0, 1.0),
        # 无 <thinking> 标签
        ("<answer>3*5 = 15", 1.0, 0.0),
        # 无 <answer> 标签
        ("<thinking>3 乘以 5。</thinking>3*5 = 15", 0.0, 0.0),
        # 完全无关输出
        ("我不知道怎么算。", 0.0, 0.0),
    ]

    completions = [c for c, _, _ in cases]
    exp_correct = [c for _, c, _ in cases]
    exp_format = [f for _, _, f in cases]

    got_correct = correctness_reward(completions, prompts=[prompt] * len(completions))
    got_format = format_reward(completions)

    ok = True
    for idx, (c, gc, gf) in enumerate(zip(completions, got_correct, got_format)):
        ec, ef = exp_correct[idx], exp_format[idx]
        status = "PASS" if (gc == ec and gf == ef) else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"[{status}] correctness={gc}(expect {ec}) format={gf}(expect {ef}) | {c[:60]!r}")

    # 场景 1b：combined_reward = format + correctness（逐样本求和）
    got_combined = combined_reward(completions, prompts=[prompt] * len(completions))
    exp_combined = [c + f for c, f in zip(exp_correct, exp_format)]
    assert got_combined == exp_combined, f"combined_reward 与期望不一致: {got_combined}"
    print(f"[PASS] combined_reward: {got_combined[:6]} ... (共 {len(got_combined)} 条)")

    # 场景 2：kwargs 无 prompts，使用 kwargs["target"] 提供目标值
    got_with_target = correctness_reward(
        ["<thinking>t</thinking><answer>3*5 = 15", "<thinking>t</thinking><answer>3+5 = 7"],
        target=[15, 7],
        prompts=["", ""],  # 无 Target 可解析时用 target 列表
    )
    assert got_with_target == [1.0, 0.0], f"kwargs target 场景失败: {got_with_target}"
    print(f"[PASS] kwargs['target'] 场景: {got_with_target}")

    print("\n" + ("ALL TESTS PASSED" if ok else "SOME TESTS FAILED"))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    _run_self_test()
