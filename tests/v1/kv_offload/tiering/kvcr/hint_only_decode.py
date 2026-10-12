# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Send a decode-only request carrying a KVCR kv_hints envelope.

Unlike the proxy, this does NOT prefill first, so it can ask the decoder to
fetch a prefix the prefiller computed earlier and has since evicted from its
own primary CPU tier. Block hashes come from the proxy's own exact-mode code
so the two agree by construction.

Env:
  PROMPT_FILE   file holding the prompt text
  DECODER_URL   e.g. http://10.0.3.38:8200
  SOURCE_HOST   prefiller KVCR control host
  SOURCE_PORT   prefiller KVCR control port
  MODEL         model id
Run with the engines' PYTHONHASHSEED.
"""

import importlib.util
import json
import os
import sys
import urllib.request

PROXY = os.environ.get("PROXY_PATH", "/tmp/kvcr_connector_proxy.py")
spec = importlib.util.spec_from_file_location("kvcrproxy", PROXY)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load the proxy module from {PROXY}")
proxy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proxy)

model = os.environ["MODEL"]
with open(os.environ["PROMPT_FILE"]) as handle:
    prompt = handle.read()
source_host = os.environ["SOURCE_HOST"]
source_port = os.environ["SOURCE_PORT"]

saved_argv = sys.argv
sys.argv = [
    "kvcr_connector_proxy.py",
    "--hash-mode",
    "exact",
    "--tokenizer",
    model,
    "--block-size",
    "16",
    "--hash-algo",
    "sha256",
    "--kvcr-control-host",
    source_host,
    "--kvcr-control-port",
    source_port,
]
proxy.global_args = proxy.parse_args()  # type: ignore[attr-defined]
sys.argv = saved_argv

hashes = proxy._block_hashes_for({"prompt": prompt, "model": model})
envelope = proxy._kv_hints("retention-probe", 0, hashes)
print("hint blocks:", len(hashes), "source:", proxy._source_endpoint(0))

body = json.dumps(
    {
        "model": model,
        "prompt": prompt,
        "max_tokens": 8,
        "temperature": 0,
        "kv_hints": envelope,
    }
).encode()
req = urllib.request.Request(
    os.environ["DECODER_URL"] + "/v1/completions",
    data=body,
    headers={"content-type": "application/json"},
)
resp = urllib.request.urlopen(req, timeout=180)
print("decode-only status", resp.status, "bodylen", len(resp.read()))
