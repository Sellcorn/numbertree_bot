import os
import httpx

os.environ['NVIDIA_API_KEY'] = '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'

headers = {
    'Authorization': f'Bearer {os.environ["NVIDIA_API_KEY"]}',
    'Content-Type': 'application/json',
}

urls = [
    'https://integrate.api.nvidia.com/v1/models',
    'https://integrate.api.nvidia.com/v1/chat/completions',
    'https://api.nvidia.com/v1/chat/completions',
    'https://integrate.api.nvidia.com/v1/completions',
]

for url in urls:
    r = httpx.get(url, headers=headers, timeout=30)
    print(f'{url}: {r.status_code} - {r.text[:300]}')