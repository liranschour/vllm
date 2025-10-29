import requests
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Simple smoke test for vLLM HTTP server.")
    parser.add_argument("--model", required=True, help="Model name to query (e.g. Qwen/Qwen3-0.6B)")
    parser.add_argument("--url", default="http://localhost:8192/v1/completions", help="vLLM completion endpoint URL")
    args = parser.parse_args()

    prompt = "2 + 2 = "

    response = requests.post(
        args.url,
        json={
            "model": args.model,
            "prompt": prompt,
            "max_tokens": 10,
            "temperature": 0.0,
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


if __name__ == "__main__":
    main()
