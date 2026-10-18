# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import glob
import json
import sys

for d in sys.argv[1:]:
    print("==", d.rsplit("/", 1)[-1])
    print(" phase   wall_s  bench_s  out_tok/s  req/s  ttft_p50  itl_p50")
    ph = {
        p: (int(b) - int(a)) / 1e9
        for p, a, b in (l.strip().split(",") for l in open(f"{d}/phases.csv"))
    }
    for f in sorted(
        glob.glob(f"{d}/bench_c*.json"), key=lambda f: int(f.split("_c")[-1][:-5])
    ):
        c = f.split("_c")[-1][:-5]
        j = json.load(open(f))
        print(
            f" c{c:<5} {ph.get('c' + c, 0):7.1f} {j['duration']:8.1f} {j['output_throughput']:10.0f} "
            f"{j['request_throughput']:6.2f} {j['median_ttft_ms']:9.0f} {j['median_itl_ms']:8.1f}"
        )
