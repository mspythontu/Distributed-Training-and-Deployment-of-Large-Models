#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vLLM OpenAI 兼容服务调用示例

vLLM 的 OpenAI 服务端点：
    POST /v1/chat/completions    对话补全（推荐）
    POST /v1/completions         文本补全
    GET  /v1/models              模型列表
    GET  /health                 健康检查

本脚本仅依赖 Python 标准库（urllib），无需安装 openai SDK，开箱即用。

用法：
    # 单次对话（直连某节点）
    python deploy/scripts/client_example.py --base-url http://worker1:8000/v1

    # 经 Nginx 网关（多实例负载均衡）
    python deploy/scripts/client_example.py --base-url http://<网关IP>:8080/v1

    # 流式输出（逐 token 打印）
    python deploy/scripts/client_example.py --stream

    # 并发 8 路请求，观察服务端并行处理能力
    python deploy/scripts/client_example.py --concurrent 8

    # 自定义提示词与生成参数
    python deploy/scripts/client_example.py \
        --prompt "用 Python 实现快速排序" --max-tokens 512 --temperature 0.7

    # 启用鉴权（与部署时的 API_KEY 对应）
    python deploy/scripts/client_example.py --api-key sk-xxx
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

DEFAULT_PROMPT = (
    "Countdown game: use the numbers 3, 5, 2, 9 each exactly once with "
    "basic arithmetic operations (+ - * /) to reach 15. Show your reasoning."
)


def build_request(url: str, payload: dict, api_key: str, timeout: int):
    """构造 POST 请求（JSON），可选带上 Bearer 鉴权头。"""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    return urllib.request.urlopen(req, timeout=timeout)


def parse_sse_lines(resp):
    """解析 SSE 流式响应，逐块产出 data 字段的 JSON 对象。

    vLLM 流式格式为：
        data: {"choices":[{"delta":{"content":"..."}}]}
        data: [DONE]
    """
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


def chat_stream(base_url: str, model: str, prompt: str, api_key: str,
                max_tokens: int, temperature: float, timeout: int, tag: str = "") -> dict:
    """流式对话：逐 token 打印，并统计首 token 延迟（TTFT）。"""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    start = time.perf_counter()
    ttft = None
    chunks = 0
    text_parts = []

    resp = build_request(f"{base_url}/chat/completions", payload, api_key, timeout)
    try:
        for obj in parse_sse_lines(resp):
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta", {}) or {}
            content = delta.get("content")
            if content:
                if ttft is None:
                    ttft = time.perf_counter() - start
                chunks += 1
                text_parts.append(content)
                print(content, end="", flush=True)
    finally:
        resp.close()

    total = time.perf_counter() - start
    print()
    return {
        "tag": tag,
        "mode": "stream",
        "ttft": ttft or 0.0,
        "total": total,
        "chunks": chunks,
        "chars": sum(len(p) for p in text_parts),
    }


def chat_once(base_url: str, model: str, prompt: str, api_key: str,
              max_tokens: int, temperature: float, timeout: int, tag: str = "") -> dict:
    """非流式对话：一次性拿到完整结果，并打印 usage 统计。"""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    start = time.perf_counter()
    resp = build_request(f"{base_url}/chat/completions", payload, api_key, timeout)
    try:
        body = json.loads(resp.read().decode("utf-8"))
    finally:
        resp.close()
    total = time.perf_counter() - start

    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        content = json.dumps(body, ensure_ascii=False)[:500]
    usage = body.get("usage", {}) or {}

    print(f"----- 响应{tag} -----")
    print(content.strip())
    print(f"----- usage: prompt={usage.get('prompt_tokens', '-')} "
          f"completion={usage.get('completion_tokens', '-')} "
          f"total={usage.get('total_tokens', '-')} -----")
    return {
        "tag": tag,
        "mode": "chat",
        "ttft": 0.0,
        "total": total,
        "chunks": 1,
        "chars": len(content),
        "usage": usage,
    }


def list_models(base_url: str, api_key: str, timeout: int) -> None:
    """打印服务端已加载的模型列表。"""
    req = urllib.request.Request(f"{base_url}/models", method="GET")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    print("已加载模型:")
    for item in body.get("data", []):
        print(f"  - {item.get('id')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="vLLM OpenAI 兼容服务调用示例",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base-url", default="http://localhost:8080/v1",
                        help="OpenAI 兼容基址（网关或单节点，默认 http://localhost:8080/v1）")
    parser.add_argument("--model", default="qwen2.5-3b", help="模型名（需与 SERVED_MODEL_NAME 一致）")
    parser.add_argument("--api-key", default="", help="鉴权令牌（与部署时 API_KEY 一致，可留空）")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="提示词")
    parser.add_argument("--max-tokens", type=int, default=512, help="最大生成长度")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度")
    parser.add_argument("--timeout", type=int, default=120, help="单次请求超时（秒）")
    parser.add_argument("--stream", action="store_true", help="流式输出（逐 token 打印）")
    parser.add_argument("--concurrent", type=int, default=1, help="并发请求数（默认 1）")
    parser.add_argument("--list-models", action="store_true", help="只列出服务端模型后退出")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")

    try:
        if args.list_models:
            list_models(base_url, args.api_key, args.timeout)
            return

        print(f"目标服务: {base_url}  模型: {args.model}  并发: {args.concurrent}")
        if args.concurrent <= 1:
            if args.stream:
                stat = chat_stream(base_url, args.model, args.prompt, args.api_key,
                                   args.max_tokens, args.temperature, args.timeout)
                print(f"\n[统计] 首 token 延迟 {stat['ttft']*1000:.0f} ms，"
                      f"总耗时 {stat['total']:.2f} s，共 {stat['chunks']} 个分块")
            else:
                stat = chat_once(base_url, args.model, args.prompt, args.api_key,
                                 args.max_tokens, args.temperature, args.timeout)
                print(f"[统计] 总耗时 {stat['total']:.2f} s")
            return

        # 并发示例：线程池发起多请求，观察服务端并行处理
        print(f"\n===== 并发 {args.concurrent} 路请求 =====")
        start = time.perf_counter()

        def worker(idx: int):
            tag = f"#{idx}"
            if args.stream:
                # 流式并发下逐字打印会交错，这里只收集不打印
                return chat_stream(base_url, args.model, args.prompt, args.api_key,
                                   args.max_tokens, args.temperature, args.timeout, tag)
            return chat_once(base_url, args.model, args.prompt, args.api_key,
                             args.max_tokens, args.temperature, args.timeout, tag)

        with ThreadPoolExecutor(max_workers=args.concurrent) as pool:
            results = list(pool.map(worker, range(args.concurrent)))

        wall = time.perf_counter() - start
        ok = [r for r in results if r["chars"] > 0]
        print(f"\n===== 并发汇总 =====")
        print(f"  完成请求: {len(ok)}/{args.concurrent}    总墙钟耗时: {wall:.2f} s")
        print(f"  平均单请求耗时: {sum(r['total'] for r in ok)/max(len(ok),1):.2f} s")
        print(f"  聚合吞吐: {len(ok)/wall:.2f} req/s")

    except urllib.error.HTTPError as exc:
        print(f"[ERROR] HTTP {exc.code}: {exc.reason}", file=sys.stderr)
        print("  排查：模型名是否与 SERVED_MODEL_NAME 一致？API_KEY 是否匹配？", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as exc:
        print(f"[ERROR] 连接失败: {exc.reason}", file=sys.stderr)
        print("  排查：服务是否已启动？地址/端口是否正确？可参考 deploy/scripts/health_check.sh", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
