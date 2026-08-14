import os
import httpx
import json
import asyncio

os.environ['NVIDIA_API_KEY'] = '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'

async def test():
    headers = {
        'Authorization': f'Bearer {os.environ["NVIDIA_API_KEY"]}',
        'Content-Type': 'application/json',
    }
    payload = {
        'model': 'deepseek-ai/deepseek-r1',
        'messages': [{'role': 'user', 'content': 'Сколько будет 2+2? Рассуждай кратко.'}],
        'stream': True,
        'temperature': 0.6,
        'max_tokens': 1000,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream('POST', 'https://integrate.api.nvidia.com/v1/chat/completions', headers=headers, json=payload) as resp:
            print(f'Status: {resp.status_code}')
            async for line in resp.aiter_lines():
                if line.startswith('data: '):
                    data = line[6:]
                    if data == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk.get('choices', [{}])[0].get('delta', {})
                        r = delta.get('reasoning_content', '')
                        c = delta.get('content', '')
                        if r: print(f'REASONING: {r}', end='', flush=True)
                        if c: print(f'CONTENT: {c}', end='', flush=True)
                    except: pass

asyncio.run(test())