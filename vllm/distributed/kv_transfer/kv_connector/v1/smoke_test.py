import requests
import sys

VLLM_URL = "http://localhost:8192/v1/completions"
prompt = "2 + 2 = "

response = requests.post(
    VLLM_URL,
    json={
        "model": "Qwen/Qwen3-0.6B",
        "prompt": prompt,
        "max_tokens": 10,
        "temperature": 0.0
    },
    timeout=10,
)

if response.status_code != 200:
    print(f"❌ Request failed: {response.status_code} {response.text}")
    sys.exit(1)

text = response.json()["choices"][0]["text"].strip()
print(f"Response: {text!r}")

if text.startswith("4"):
    print("✅ Smoke test passed (2+2=4)")
else:
    print("❌ Smoke test failed")
    sys.exit(1)
