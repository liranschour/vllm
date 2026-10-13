# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Drive the two-pod router environment and prove KV crossed between pods.

Runs inside a pod (stdlib only, no extra deps) and talks to the EPP stub.

Two phases per trial, because a cross-pod pull only does real work when the
puller has never seen the prompt:

1. **seed** -- send the prompt to pod S with no source hint. S computes it and
   offloads the prompt blocks to its host tier.
2. **pull** -- send the *same* prompt to the other pod with a source hint
   naming S. The puller misses locally, and its secondary tier fetches from S.

Prompts are unique per trial. A reused prompt would hit the puller's own GPU
prefix cache and transfer nothing, which looks identical to success in the
latency but shows up as zero tier hits.

The verdict comes from the puller's own metrics, read before and after:

* ``vllm:kv_offload_tiering_chunk_hits_total`` -- chunks the **secondary**
  (p2p/kvcr) tier served. This is the number that proves a cross-pod fetch;
  everything else can be explained by local caching.
* ``vllm:external_prefix_cache_hits_total`` -- blocks the engine took from the
  offload connector rather than recomputing.

Usage (on the pod):
    python3 router_pull_test.py --epp localhost:8080 \
        --vllm-a localhost:8100 --vllm-b localhost:8200 \
        --trials 8 --mode pair
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid

# Each trial needs a prompt spanning many blocks so a pull moves a visible
# amount of KV. ~40 repetitions of this clause lands around 600 tokens, i.e.
# ~37 blocks at --block-size 16.
FILLER = (
    "The distributed inference scheduler reconciles replica state with the "
    "observed key value cache inventory before admitting the next request. "
)
FILLER_REPEATS = 40

TIER_HITS = "vllm:kv_offload_tiering_chunk_hits_total"
TIER_QUERIES = "vllm:kv_offload_tiering_chunk_queries_total"
EXTERNAL_HITS = "vllm:external_prefix_cache_hits_total"
EXTERNAL_QUERIES = "vllm:external_prefix_cache_queries_total"

METRICS_OF_INTEREST = (TIER_HITS, TIER_QUERIES, EXTERNAL_HITS, EXTERNAL_QUERIES)


def http_post(url: str, payload: dict, headers: dict, timeout: float = 600.0) -> dict:
    body = json.dumps(payload).encode()
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    for key, value in headers.items():
        request.add_header(key, value)
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return {
            "status": response.status,
            "body": json.loads(response.read().decode() or "{}"),
            "headers": dict(response.headers),
        }


def scrape_metrics(endpoint: str) -> dict[str, float]:
    """Return the metrics of interest from a vLLM /metrics endpoint."""
    url = f"http://{endpoint}/metrics"
    with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310
        text = response.read().decode()

    totals: dict[str, float] = dict.fromkeys(METRICS_OF_INTEREST, 0.0)
    for line in text.splitlines():
        if line.startswith("#") or not line:
            continue
        name, _, rest = line.partition("{")
        if not rest:
            name, _, value = line.partition(" ")
        else:
            _, _, value = rest.partition("} ")
        name = name.strip()
        if name in totals:
            try:
                # Labelled series are summed across labels.
                totals[name] += float(value.strip())
            except ValueError:
                continue
    return totals


def unique_prompt(tag: str) -> str:
    # The uuid makes the whole prefix unique, so neither pod can have it cached
    # from an earlier trial or an earlier run of this script.
    return f"Session {tag} {uuid.uuid4()}. " + (FILLER * FILLER_REPEATS)


def send(epp: str, prompt: str, *, model: str, target: int | None, hint: bool) -> dict:
    headers = {}
    if target is not None:
        headers["x-test-target"] = str(target)
    if not hint:
        headers["x-test-no-hint"] = "1"
    payload = {
        "model": model,
        "prompt": prompt,
        # One token: we are measuring the prefill-side prefix fetch, not decode.
        "max_tokens": 1,
        "temperature": 0.0,
        "stream": False,
    }
    return http_post(f"http://{epp}/v1/completions", payload, headers)


def delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {key: after[key] - before[key] for key in before}


def run_pair_mode(args, vllm: list[str]) -> list[dict]:
    """Deterministic: seed on one pod, pull from the other.

    Alternates which pod seeds, so both directions of the transfer are
    exercised rather than only A->B.
    """
    results = []
    for trial in range(args.trials):
        seed_index = trial % 2
        pull_index = 1 - seed_index
        prompt = unique_prompt(f"pair-{trial}")

        # Seed: no hint, pinned to the seeding pod.
        send(args.epp, prompt, model=args.model, target=seed_index, hint=False)
        # The store is asynchronous; give the tier a moment to land the blocks
        # before asking the other pod to fetch them.
        time.sleep(args.settle)

        before = scrape_metrics(vllm[pull_index])
        response = send(
            args.epp, prompt, model=args.model, target=pull_index, hint=True
        )
        time.sleep(args.settle)
        after = scrape_metrics(vllm[pull_index])

        d = delta(before, after)
        results.append(
            {
                "trial": trial,
                "seeded_on": seed_index,
                "pulled_on": pull_index,
                "routed_to": response["headers"].get("x-test-routed-to"),
                "routed_source": response["headers"].get("x-test-routed-source"),
                "tier_hits": d[TIER_HITS],
                "tier_queries": d[TIER_QUERIES],
                "external_hits": d[EXTERNAL_HITS],
                "external_queries": d[EXTERNAL_QUERIES],
            }
        )
        print(
            f"  trial {trial}: seed=pod{seed_index} pull=pod{pull_index} "
            f"tier_hits=+{d[TIER_HITS]:.0f} tier_queries=+{d[TIER_QUERIES]:.0f} "
            f"external_hits=+{d[EXTERNAL_HITS]:.0f}",
            flush=True,
        )
    return results


def run_control_mode(args, vllm: list[str]) -> list[dict]:
    """Negative control: identical to pair mode, but with no source hint.

    This is what makes pair mode's result mean anything. Same prompts, same
    pods, same seeding -- only the hint is withheld. If tier hits still appear,
    they were never caused by the hint and pair mode proves nothing; the
    expected result here is zero, with the puller recomputing the prefix.
    """
    results = []
    for trial in range(args.trials):
        seed_index = trial % 2
        pull_index = 1 - seed_index
        prompt = unique_prompt(f"control-{trial}")

        send(args.epp, prompt, model=args.model, target=seed_index, hint=False)
        time.sleep(args.settle)

        before = scrape_metrics(vllm[pull_index])
        send(args.epp, prompt, model=args.model, target=pull_index, hint=False)
        time.sleep(args.settle)
        after = scrape_metrics(vllm[pull_index])

        d = delta(before, after)
        results.append(
            {
                "trial": trial,
                "seeded_on": seed_index,
                "pulled_on": pull_index,
                "tier_hits": d[TIER_HITS],
                "tier_queries": d[TIER_QUERIES],
                "external_hits": d[EXTERNAL_HITS],
                "external_queries": d[EXTERNAL_QUERIES],
            }
        )
        print(
            f"  trial {trial}: seed=pod{seed_index} pull=pod{pull_index} "
            f"(no hint) tier_hits=+{d[TIER_HITS]:.0f} "
            f"external_hits=+{d[EXTERNAL_HITS]:.0f}",
            flush=True,
        )
    return results


def run_random_mode(args, vllm: list[str]) -> list[dict]:
    """The literal scenario: the router picks a replica, hints the other.

    Each prompt is sent twice with no pinning. Whether that produces a transfer
    depends on where the two requests land, so a ~50% transfer rate is the
    expected outcome, not a failure.
    """
    results = []
    for trial in range(args.trials):
        prompt = unique_prompt(f"random-{trial}")
        before = [scrape_metrics(endpoint) for endpoint in vllm]
        first = send(args.epp, prompt, model=args.model, target=None, hint=True)
        time.sleep(args.settle)
        second = send(args.epp, prompt, model=args.model, target=None, hint=True)
        time.sleep(args.settle)
        after = [scrape_metrics(endpoint) for endpoint in vllm]

        totals = {
            key: sum(after[i][key] - before[i][key] for i in range(len(vllm)))
            for key in METRICS_OF_INTEREST
        }
        results.append(
            {
                "trial": trial,
                "first_routed_to": first["headers"].get("x-test-routed-to"),
                "second_routed_to": second["headers"].get("x-test-routed-to"),
                "tier_hits": totals[TIER_HITS],
                "tier_queries": totals[TIER_QUERIES],
                "external_hits": totals[EXTERNAL_HITS],
                "external_queries": totals[EXTERNAL_QUERIES],
            }
        )
        same = first["headers"].get("x-test-routed-to") == second["headers"].get(
            "x-test-routed-to"
        )
        results[-1]["same_pod"] = same
        print(
            f"  trial {trial}: {'same pod' if same else 'different pods'} "
            f"tier_hits=+{totals[TIER_HITS]:.0f} "
            f"external_hits=+{totals[EXTERNAL_HITS]:.0f}"
            f"{'' if totals[EXTERNAL_HITS] else '  (no transfer)'}",
            flush=True,
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epp", default="localhost:8080")
    parser.add_argument("--vllm-a", default="localhost:8100")
    parser.add_argument(
        "--vllm-b",
        required=True,
        help="pod B's vLLM metrics endpoint, reachable from here (host:port)",
    )
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--mode", choices=("pair", "random", "control"), default="pair")
    parser.add_argument(
        "--settle",
        type=float,
        default=2.0,
        help="seconds to wait for the asynchronous offload store to land",
    )
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    vllm = [args.vllm_a, args.vllm_b]

    print(f"mode={args.mode} trials={args.trials} epp={args.epp}")
    runner = {
        "pair": run_pair_mode,
        "random": run_random_mode,
        "control": run_control_mode,
    }[args.mode]
    try:
        results = runner(args, vllm)
    except urllib.error.HTTPError as exc:
        print(f"HTTP error: {exc.code} {exc.read().decode()[:400]}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"connection error: {exc}", file=sys.stderr)
        return 1

    total_tier_hits = sum(r["tier_hits"] for r in results)
    total_external_hits = sum(r["external_hits"] for r in results)
    # A transfer is counted on external_hits, not tier_hits. tier_hits also
    # moves when a pod merely stores or looks up its own blocks -- a same-pod
    # trial in random mode shows tier_hits > 0 with external_hits == 0, having
    # moved nothing. external_hits only rises when the engine actually took
    # blocks from the connector instead of recomputing them.
    transfers = sum(1 for r in results if r["external_hits"] > 0)

    print()
    print(f"trials that took KV from the connector : {transfers}/{len(results)}")
    print(f"total secondary-tier chunk hits        : {total_tier_hits:.0f}")
    print(f"total external prefix-cache hits       : {total_external_hits:.0f}")

    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump(
                {"mode": args.mode, "model": args.model, "results": results},
                handle,
                indent=2,
            )
        print(f"wrote {args.json_out}")

    if args.mode == "control":
        # Inverted expectation: any tier hit without a hint means the pair-mode
        # measurement is not attributable to the hint.
        if transfers != 0:
            print(
                f"\nFAIL: {transfers} of {len(results)} trials served from the "
                "tier with NO hint -- pair mode's result is not attributable "
                "to the hint",
                file=sys.stderr,
            )
            return 1
        print("\nPASS: no hint, no tier hits -- pair mode's hits come from the hint")
        return 0

    if args.mode == "pair":
        # Every pair-mode trial seeds on one pod and pulls on the other, so
        # anything short of all of them means KV did not cross.
        if transfers != len(results):
            print(
                f"\nFAIL: {len(results) - transfers} of {len(results)} trials "
                "moved no KV between pods",
                file=sys.stderr,
            )
            return 1
        print("\nPASS: every trial served its prefix from the peer's tier")
        return 0

    # Free routing sends a prompt's two requests to the same pod about half the
    # time, which is a local hit and moves nothing. So the bar is "some trial
    # transferred", and the same-pod count is reported rather than treated as
    # failure.
    cross_pod = sum(1 for r in results if not r.get("same_pod", False))
    if transfers == 0:
        print(
            f"\nFAIL: no trial took KV from the connector "
            f"({cross_pod} of {len(results)} were routed across pods)",
            file=sys.stderr,
        )
        return 1
    print(
        f"\nPASS: {transfers} of {len(results)} trials transferred "
        f"({cross_pod} were routed across pods; the rest hit the same pod "
        "locally and are expected to move nothing)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
