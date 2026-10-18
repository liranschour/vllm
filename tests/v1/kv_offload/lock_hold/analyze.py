# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Summarize a lock trace: hold times, duty cycle, free gaps, probe waits."""

import glob
import sys
from collections import defaultdict

import numpy as np


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def load(run_dir):
    rows = []
    for f in glob.glob(f"{run_dir}/trace.*"):
        for line in open(f):
            p = line.rstrip("\n").split(",")
            rows.append((p[0], p[1], int(p[3]), int(p[4]), int(p[5]), p[6]))
    rows.sort(key=lambda r: r[3])
    phases = [
        (p, int(a), int(b))
        for p, a, b in (l.strip().split(",") for l in open(f"{run_dir}/phases.csv"))
    ]
    return rows, phases


def summarize(run_dir, out=sys.stdout):
    rows, phases = load(run_dir)
    hdr = (
        f"{'phase':>6} {'steps/s':>8} {'step p50':>9} {'p90':>8} {'p99':>8} {'max':>8} "
        f"{'duty%':>6} {'gap p50':>8} {'gap p10':>8} {'short/s':>8} {'sh p99':>7} "
        f"{'probe wait p50':>14} {'p99':>8} {'max':>8} {'tok/step':>8}"
    )
    print(f"== {run_dir.rsplit('/', 1)[-1]}   (times in ms unless noted)", file=out)
    print(hdr, file=out)
    res = {}
    for phase, t0, t1 in phases:
        r = [x for x in rows if t0 <= x[3] < t1]
        span = (t1 - t0) / 1e9
        step = [x for x in r if x[0] == "step"]
        short = [x for x in r if x[0] == "short"]
        probe = [x for x in r if x[0] == "probe"]
        sh = np.array([(x[4] - x[3]) / 1e6 for x in step])
        shh = np.array([(x[4] - x[3]) / 1e6 for x in short])
        # Non-probe holds -> scheduler-thread occupancy and the free gaps between them.
        sched = sorted([(x[3], x[4]) for x in step + short])
        held = sum(b - a for a, b in sched) / 1e9
        gaps = np.array(
            [(sched[i + 1][0] - sched[i][1]) / 1e6 for i in range(len(sched) - 1)]
        )
        # Gaps between step-region holds only (the window available during model exec).
        st = sorted([(x[3], x[4]) for x in step])
        sgaps = np.array([(st[i + 1][0] - st[i][1]) / 1e6 for i in range(len(st) - 1)])
        pw = np.array([(x[3] - x[2]) / 1e6 for x in probe])
        sw = np.array([(x[3] - x[2]) / 1e6 for x in step + short])
        ph_hold = np.array([(x[4] - x[3]) / 1e6 for x in probe])
        res_extra = dict(
            sw50=pct(sw, 50),
            sw99=pct(sw, 99),
            swmax=sw.max() if len(sw) else 0,
            swsum=sw.sum() / span / 10 if len(sw) else 0,  # % of wall
            probe_rate=len(probe) / span,
            probe_duty=100 * ph_hold.sum() / 1e3 / span if len(ph_hold) else 0,
        )
        tok = [
            int(x[5].split(";")[0].split("|")[0]) for x in step if x[5].split(";")[0]
        ]
        res[phase] = dict(
            steps=len(step) / span,
            p50=pct(sh, 50),
            p90=pct(sh, 90),
            p99=pct(sh, 99),
            max=sh.max() if len(sh) else 0,
            duty=100 * held / span,
            gap50=pct(sgaps, 50),
            gap10=pct(sgaps, 10),
            short_rate=len(short) / span,
            short_p99=pct(shh, 99),
            pw50=pct(pw, 50),
            pw99=pct(pw, 99),
            pwmax=pw.max() if len(pw) else 0,
            tok=np.mean(tok) if tok else 0,
            n_probe=len(probe),
            **res_extra,
        )
        d = res[phase]
        print(
            f"{phase:>6} {d['steps']:8.1f} {d['p50']:9.3f} {d['p90']:8.3f} {d['p99']:8.3f} "
            f"{d['max']:8.2f} {d['duty']:6.2f} {d['gap50']:8.2f} {d['gap10']:8.2f} "
            f"{d['short_rate']:8.1f} {d['short_p99']:7.3f} {d['pw50']:14.4f} {d['pw99']:8.3f} "
            f"{d['pwmax']:8.2f} {d['tok']:8.0f}",
            file=out,
        )
        if len(probe):
            print(
                f"{'':>6} sched-wait p50={d['sw50']:.4f} p99={d['sw99']:.3f} max={d['swmax']:.2f} "
                f"total={d['swsum']:.2f}% of wall | probes/s={d['probe_rate']:.1f} "
                f"probe duty={d['probe_duty']:.2f}%",
                file=out,
            )
    # Where the step-region time goes: per hook, summed over all steps, per phase.
    print("  step-region composition (% of step-region hold, mean ms/step):", file=out)
    for phase, t0, t1 in phases:
        step = [x for x in rows if x[0] == "step" and t0 <= x[3] < t1]
        if not step:
            continue
        tot = sum(x[4] - x[3] for x in step)
        acc = defaultdict(int)
        for x in step:
            hooks = x[5].split(";")[1] if ";" in x[5] else ""
            for h in filter(None, hooks.split("|")):
                k, n, t = h.split(":")
                acc[k] += int(t)
        other = tot - sum(acc.values())
        parts = [
            f"{k}={100 * v / tot:.0f}%({v / len(step) / 1e6:.3f})"
            for k, v in sorted(acc.items(), key=lambda kv: -kv[1])
        ]
        parts.append(
            f"between-hooks={100 * other / tot:.0f}%({other / len(step) / 1e6:.3f})"
        )
        print(f"    {phase:>5}: " + "  ".join(parts), file=out)
    # Per-hook breakdown of what opened the step region / short regions.
    by = defaultdict(list)
    for x in rows:
        if x[0] != "probe":
            by[(x[0], x[1])].append((x[4] - x[3]) / 1e6)
    print("  hook breakdown (all phases):", file=out)
    for k, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
        v = np.array(v)
        print(
            f"    {k[0]:5} {k[1]:32} n={len(v):6d} p50={pct(v, 50):.3f} p99={pct(v, 99):.3f} "
            f"max={v.max():.2f}",
            file=out,
        )
    return res


if __name__ == "__main__":
    for d in sys.argv[1:]:
        summarize(d)
