"""Простой стрим-тест: печатает размышления и ответ по мере поступления.

    python test_api.py minimaxai/minimax-m2
"""
import asyncio
import json
import sys

import httpx

from _keyhelper import API_BASE, headers

MODEL = sys.argv[1] if len(sys.argv) > 1 else "minimaxai/minimax-m2"


async def main():
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Сколько будет 2+2? Рассуждай кратко."}],
        "stream": True,
        "temperature": 0.6,
        "max_tokens": 1000,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", f"{API_BASE}/chat/completions", headers=headers(), json=payload) as resp:
            print(f"Статус: {resp.status_code}")
            if resp.status_code != 200:
                print((await resp.aread()).decode("utf-8", "replace")[:600])
                return
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    delta = (json.loads(data).get("choices") or [{}])[0].get("delta", {})
                except json.JSONDecodeError:
                    continue
                if delta.get("reasoning_content"):
                    print(f"\033[90m{delta['reasoning_content']}\033[0m", end="", flush=True)
                if delta.get("content"):
                    print(delta["content"], end="", flush=True)
    print()


asyncio.run(main())
