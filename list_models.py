import os
import httpx
import json

os.environ['NVIDIA_API_KEY'] = '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'

headers = {
    'Authorization': f'Bearer {os.environ["NVIDIA_API_KEY"]}',
    'Content-Type': 'application/json',
}

r = httpx.get('https://integrate.api.nvidia.com/v1/models', headers=headers, timeout=30)
data = r.json()

# Find DeepSeek models
for model in data['data']:
    if 'deepseek' in model['id'].lower():
        print(model['id'])

print("\n--- All models ---")
for model in data['data']:
    print(model['id'])