#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vLLM 服务并发压测脚本

统计指标：
    - 吞吐：请求吞吐（req/s）、输出 token 吞吐（tok/s）
    - 延迟：端到端延迟的 平均 / P50 / P95 / P99
    - TTFT：首 token 延迟（Time To First Token，仅流式模式可测）
    - 成功率与错误分布

依赖：仅 Python 标准库（urllib），无需额外安装包。

用法：
    # 基础压测：16 并发、共 100 个请求
    python deploy/scripts/benchmark.py --base-url http://<网关IP>:8080/v1 \
        --concurrency 16 --requests 100

    # 流式压测（可测 TTFT，更接近真实聊天体验）
    python deploy/scripts/benchmark.py --concurrency 16 --requests 100 --stream

    # 控制生成长度（显著影响吞吐，建议与实际业务一致）
    python deploy/scripts/benchmark.py --concurrency 32 --requests 200 --max-tokens 128

    # 自定义提示词 + 结果导出 JSON
    python deploy/scripts/benchmark.py --prompt "你好，请自我介绍" \
        --concurrency 8 --requests 50 --output deploy/bench_result.json

调参建议：
    - 先小并发（4）确认服务正常，再逐步加大到出现延迟明显上升为止
    - 对比模式 A（多实例+负载均衡）与单实例，可直观看到吞吐差异
    - max-tokens 越大，吞吐越低、TTFT 占比越小；短输出场景更看并发能力
"""

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

DEFAULT_PROMPT = (
    "Countdown game: use the numbers 3, 5, 2, 9 each exactly once with basic "
    "arithmetic operations (+ - * /) to reach 15. Explain your reasoning step by step."
)

_print_lock = threading.Lock()


def percentile(values, p: float) -> float:
    """线性插值分位数（values 需已升序）。"""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    k = (len(values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def post_json(url: str, payload: dict, api_key: str, timeout: int):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    return urllib.request.urlopen(req, timeout=timeout)


def parse_sse(resp):
    for raw in resp:
        line = raw.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            break
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            continue


def single_request(base_url, model, prompt, api_key, max_tokens, temperature,
                   timeout, stream) -> dict:
    """发起单次请求并记录指标。返回 dict 结果（不抛异常，错误记录到 error 字段）。"""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    result = {
        "success": False, "error": "", "latency": 0.0, "ttft": None,
        "prompt_tokens": 0, "completion_tokens": 0,
    }

    start = time.perf_counter()
    try:
        if stream:
            payload["stream"] = True
            # 请求在流式结束时附带 usage 统计（OpenAI 兼容字段）
            payload["stream_options"] = {"include_usage": True}
            resp = post_json(f"{base_url}/chat/completions", payload, api_key, timeout)
            try:
                for obj in parse_sse(resp):
                    if result["ttft"] is None:
                        choices = obj.get("choices") or []
                        if choices:
                            delta = choices[0].get("delta", {}) or {}
                            if delta.get("content"):
                                result["ttft"] = time.perf_counter() - start
                    usage = obj.get("usage")
                    if usage:
                        result["prompt_tokens"] = usage.get("prompt_tokens", 0)
                        result["completion_tokens"] = usage.get("completion_tokens", 0)
            finally:
                resp.close()
        else:
            resp = post_json(f"{base_url}/chat/completions", payload, api_key, timeout)
            try:
                body = json.loads(resp.read().decode("utf-8"))
            finally:
                resp.close()
            usage = body.get("usage", {}) or {}
            result["prompt_tokens"] = usage.get("prompt_tokens", 0)
            result["completion_tokens"] = usage.get("completion_tokens", 0)
            if not body.get("choices"):
                raise ValueError(f"响应缺少 choices 字段: {str(body)[:200]}")

        result["success"] = True
    except Exception as exc:  # 网络/HTTP/解析异常统一记为失败
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["latency"] = time.perf_counter() - start
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="vLLM 服务并发压测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base-url", default="http://localhost:8080/v1", help="OpenAI 兼容基址")
    parser.add_argument("--model", default="qwen2.5-3b", help="模型名（与 SERVED_MODEL_NAME 一致）")
    parser.add_argument("--api-key", default="", help="鉴权令牌")
    parser.add_argument("--concurrency", type=int, default=16, help="并发数（默认 16）")
    parser.add_argument("--requests", type=int, default=100, help="总请求数（默认 100）")
    parser.add_argument("--max-tokens", type=int, default=256, help="单次最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度")
    parser.add_argument("--timeout", type=int, default=180, help="单请求超时（秒）")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="压测用提示词")
    parser.add_argument("--stream", action="store_true", help="流式压测（可测 TTFT）")
    parser.add_argument("--output", default="", help="结果 JSON 输出路径（可选）")
    args = parser.parse_args()

    if args.concurrency < 1 or args.requests < 1:
        print("[ERROR] --concurrency 与 --requests 必须 >= 1", file=sys.stderr)
        sys.exit(1)

    base_url = args.base_url.rstrip("/")
    print("=" * 62)
    print(f"  压测目标 : {base_url}")
    print(f"  模型     : {args.model}")
    print(f"  并发/总数: {args.concurrency} / {args.requests}   流式: {args.stream}")
    print(f"  max_tokens: {args.max_tokens}")
    print("=" * 62)

    counter = {"done": 0}

    def worker(_):
        res = single_request(
            base_url, args.model, args.prompt, args.api_key,
            args.max_tokens, args.temperature, args.timeout, args.stream,
        )
        with _print_lock:
            counter["done"] += 1
            done = counter["done"]
            if done % max(1, args.requests // 10) == 0 or done == args.requests:
                print(f"  进度: {done}/{args.requests}", flush=True)
        return res

    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(worker, range(args.requests)))
    wall = time.perf_counter() - wall_start

    ok = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    latencies = sorted(r["latency"] for r in ok)
    ttfts = sorted(r["ttft"] for r in ok if r["ttft"] is not None)
    total_completion = sum(r["completion_tokens"] for r in ok)
    total_prompt = sum(r["prompt_tokens"] for r in ok)

    print()
    print("=" * 62)
    print("  压测结果")
    print("=" * 62)
    print(f"  成功/总数      : {len(ok)}/{args.requests}"
          f"（失败 {len(failed)}）")
    print(f"  总墙钟耗时     : {wall:.2f} s")
    if latencies:
        print(f"  请求吞吐       : {len(ok)/wall:.2f} req/s")
        print(f"  输出 token 吞吐: {total_completion/wall:.1f} tok/s"
              f"（输入 {total_prompt} tok / 输出 {total_completion} tok）")
        print(f"  端到端延迟(avg): {statistics.mean(latencies)*1000:.0f} ms")
        print(f"  端到端延迟 P50 : {percentile(latencies, 50)*1000:.0f} ms")
        print(f"  端到端延迟 P95 : {percentile(latencies, 95)*1000:.0f} ms")
        print(f"  端到端延迟 P99 : {percentile(latencies, 99)*1000:.0f} ms")
    if ttfts:
        print(f"  首 token TTFT  : avg {statistics.mean(ttfts)*1000:.0f} ms | "
              f"P50 {percentile(ttfts, 50)*1000:.0f} ms | "
              f"P95 {percentile(ttfts, 95)*1000:.0f} ms")
    elif args.stream:
        print("  首 token TTFT  : 未采集到（vLLM 可能未返回 stream usage/增量内容）")
    if failed:
        print()
        print("  失败样例（最多 3 条）:")
        for r in failed[:3]:
            print(f"    - {r['error'][:160]}")
    print("=" * 62)

    if args.output:
        summary = {
            "base_url": base_url,
            "model": args.model,
            "concurrency": args.concurrency,
            "requests": args.requests,
            "stream": args.stream,
            "max_tokens": args.max_tokens,
            "wall_seconds": round(wall, 3),
            "success": len(ok),
            "failed": len(failed),
            "requests_per_second": round(len(ok) / wall, 3) if wall > 0 else 0,
            "output_tokens_per_second": round(total_completion / wall, 1) if wall > 0 else 0,
            "latency_ms": {
                "avg": round(statistics.mean(latencies) * 1000, 1) if latencies else 0,
                "p50": round(percentile(latencies, 50) * 1000, 1),
                "p95": round(percentile(latencies, 95) * 1000, 1),
                "p99": round(percentile(latencies, 99) * 1000, 1),
            },
            "ttft_ms": {
                "avg": round(statistics.mean(ttfts) * 1000, 1) if ttfts else None,
                "p50": round(percentile(ttfts, 50) * 1000, 1) if ttfts else None,
                "p95": round(percentile(ttfts, 95) * 1000, 1) if ttfts else None,
            },
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
        }
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
        print(f"  结果已导出: {args.output}")

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
