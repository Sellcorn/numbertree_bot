"""Разведка формата стрима: какие поля модель реально шлёт в delta.

Нужен, чтобы понять, как модель отдаёт размышления — отдельным полем
`reasoning_content` или тегами <think> внутри `content`.

    python test_flash.py minimaxai/minimax-m2
"""
import asyncio
import json
import sys
from collections import Counter

import httpx

from _keyhelper import API_BASE, headers

MODEL = sys.argv[1] if len(sys.argv) > 1 else "minimaxai/minimax-m2"
PROMPT = "Сколько будет 17 * 23? Порассуждай, потом дай ответ."


async def main():
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": True,
        "temperature": 0.6,
        "max_tokens": 1200,
    }

    seen_keys = Counter()
    raw_samples = []
    reasoning, content = [], []
    usage = {}

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", f"{API_BASE}/chat/completions", headers=headers(), json=payload) as resp:
            print(f"Модель: {MODEL}")
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
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                seen_keys.update(k for k, v in delta.items() if v)
                if len(raw_samples) < 3 and delta:
                    raw_samples.append(delta)
                if delta.get("reasoning_content"):
                    reasoning.append(delta["reasoning_content"])
                if delta.get("content"):
                    content.append(delta["content"])

    joined_content = "".join(content)
    print(f"\n--- Непустые поля delta: {dict(seen_keys)}")
    print(f"--- Первые чанки: {json.dumps(raw_samples, ensure_ascii=False)[:500]}")
    print(f"--- reasoning_content: {len(''.join(reasoning))} символов")
    print(f"--- content: {len(joined_content)} символов")
    print(f"--- <think> внутри content: {'ДА' if '<think>' in joined_content else 'нет'}")
    print(f"--- usage: {usage}")
    print(f"\n=== REASONING ===\n{''.join(reasoning)[:800]}")
    print(f"\n=== CONTENT ===\n{joined_content[:800]}")


asyncio.run(main())
