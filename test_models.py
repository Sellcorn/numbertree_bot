import os
import httpx
import json

os.environ['NVIDIA_API_KEY'] = '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'

headers = {
    'Authorization': f'Bearer {os.environ["NVIDIA_API_KEY"]}',
    'Content-Type': 'application/json',
}

models = [
    'deepseek-ai/deepseek-r1',
    'deepseek-r1',
    'nvidia/deepseek-r1',
    'deepseek-ai/deepseek-r1-distill-llama-70b',
    'nvidia/nemotron-3-ultra',
]

for model in models:
    payload = {'model': model, 'messages': [{'role': 'user', 'content': 'test'}], 'max_tokens': 10}
    r = httpx.post('https://integrate.api.nvidia.com/v1/chat/completions', headers=headers, json=payload, timeout=30)
    print(f'{model}: {r.status_code} - {r.text[:200]}')