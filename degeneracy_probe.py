"""Show what Kimi-K3 actually generates from a random-token prompt.

The DSpark grid runs on `--dataset-name random` with temperature 0 and
ignore_eos, so the model is forced to emit exactly random_output_len tokens from
a meaningless prompt. If that output collapses into a loop, the ~3.8 accept
length those cells report says more about degenerate text being trivially
predictable than about how DSpark behaves on real traffic. This prints the text
and a repetition measure so the question is settled by looking rather than
guessing.
"""

import argparse
import collections
import json
import urllib.request


def repetition_stats(tokens: list) -> dict:
    n = len(tokens)
    uniq = len(set(tokens))
    # Longest run of a single repeated token, and the most common 8-gram count:
    # a healthy sample has a top 8-gram count of 1.
    longest_run, run, prev = 0, 0, object()
    for t in tokens:
        run = run + 1 if t == prev else 1
        longest_run = max(longest_run, run)
        prev = t
    grams = collections.Counter(
        tuple(tokens[i : i + 8]) for i in range(max(0, n - 7))
    )
    top_gram, top_count = ("", 0)
    if grams:
        top_gram, top_count = grams.most_common(1)[0]
    return {
        "n_tokens": n,
        "unique_tokens": uniq,
        "unique_ratio": round(uniq / n, 4) if n else 0.0,
        "longest_single_token_run": longest_run,
        "most_repeated_8gram_count": top_count,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:30000")
    p.add_argument("--model", default="moonshotai/Kimi-K3")
    p.add_argument("--input-len", type=int, default=1024)
    p.add_argument("--output-len", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    from sglang.benchmark.serving import get_tokenizer  # noqa: F401  (path check)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Same construction bench_serving uses: uniformly sampled token ids.
    import random

    rng = random.Random(args.seed)
    vocab = tok.vocab_size
    prompt_ids = [rng.randint(0, vocab - 1) for _ in range(args.input_len)]
    prompt = tok.decode(prompt_ids)

    body = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": args.output_len,
            "ignore_eos": True,
        },
    }
    req = urllib.request.Request(
        f"{args.base_url}/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())

    text = out["text"] if isinstance(out, dict) else out[0]["text"]
    out_ids = tok.encode(text, add_special_tokens=False)

    print("=" * 70)
    print("PROMPT: 1024 uniformly random token ids (same as --dataset-name random)")
    print("SAMPLING: temperature 0, ignore_eos=True, max_new_tokens", args.output_len)
    print("=" * 70)
    print("--- first 1200 chars of generated text ---")
    print(text[:1200])
    print("...")
    print("--- last 400 chars ---")
    print(text[-400:])
    print()
    print("--- repetition stats of the generated tokens ---")
    for k, v in repetition_stats(out_ids).items():
        print(f"  {k}: {v}")
    print()
    print("Read: unique_ratio near 1.0 and most_repeated_8gram_count == 1 means the")
    print("output is genuinely varied, so the accept length is real. A low unique")
    print("ratio or an 8-gram repeating many times means the model is looping and")
    print("the random-dataset accept length is inflated.")


if __name__ == "__main__":
    main()
