"""Build an sglang agentic-trace JSON from real OpenHands trajectories.

sglang's AgenticTraceDataset replays a conversation round by round and feeds the
server's own reply back into the next round's history, which is the only way to
reproduce the thing that actually characterises agentic serving: a prompt that
grows by a small delta each turn on top of a very large shared prefix.

Expected shape:
    {"metadata": {...},
     "conversations": [
        [ {"messages": [ {role, content}, ... ], "prompt_tokens": N}, ... ],
        ...
     ]}
where each turn's `messages` holds only the NEW non-assistant messages for that
turn; the assistant replies come from the server during replay.

Source: nebius/SWE-rebench-openhands-trajectories (Qwen3-Coder-480B on OpenHands
v0.54.0 against real GitHub issues).
"""

import argparse
import json
import os
from pathlib import Path


def to_turns(trajectory: list) -> list:
    """Group a flat role/content trace into per-turn message deltas.

    A turn is the run of non-assistant messages that precedes an assistant
    reply. Assistant messages are dropped: during replay the server produces
    them, which is what makes the measured prefill delta realistic.
    """
    turns, pending = [], []
    for msg in trajectory:
        role = msg.get("role")
        content = msg.get("content")
        if not isinstance(content, str) or not content:
            # Tool calls can carry structured content; flatten what we can so
            # the token volume stays representative.
            if isinstance(content, list):
                content = "\n".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            if not content:
                continue
        if role == "assistant":
            if pending:
                turns.append(pending)
                pending = []
            continue
        # `tool` has no standalone slot in a plain chat template; fold
        # observations into the user turn, which preserves both the token
        # volume and the turn boundaries.
        pending.append({"role": "system" if role == "system" else "user",
                        "content": content})
    if pending:
        turns.append(pending)
    return turns


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/sgl-workspace/workspace/data/agentic_trace.json")
    p.add_argument("--dataset", default="nebius/SWE-rebench-openhands-trajectories")
    p.add_argument("--num-conversations", type=int, default=64)
    p.add_argument("--max-turns", type=int, default=24)
    p.add_argument("--min-turns", type=int, default=6)
    p.add_argument("--model", default="moonshotai/Kimi-K3")
    args = p.parse_args()

    os.environ.pop("HF_HUB_OFFLINE", None)
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"streaming {args.dataset} ...")
    ds = load_dataset(args.dataset, split="train", streaming=True)

    conversations = []
    scanned = 0
    for row in ds:
        scanned += 1
        if len(conversations) >= args.num_conversations:
            break
        traj = row.get("trajectory")
        if not isinstance(traj, list) or len(traj) < 4:
            continue
        turns = to_turns(traj)
        if len(turns) < args.min_turns:
            continue
        turns = turns[: args.max_turns]

        # prompt_tokens is informational for the loader, but we want it accurate
        # so the reported ISL growth is real rather than assumed.
        conv, running = [], 0
        for msgs in turns:
            delta = sum(len(tok.encode(m["content"], add_special_tokens=False))
                        for m in msgs)
            running += delta
            conv.append({"messages": msgs, "prompt_tokens": running})
            # The assistant reply that will be generated also joins the history;
            # 220 tokens is the OpenHands average sglang itself assumes.
            running += 220
        conversations.append(conv)
        if len(conversations) % 8 == 0:
            print(f"  {len(conversations)} conversations (scanned {scanned})")

    if not conversations:
        raise SystemExit("no usable conversations found")

    first_tokens = [c[0]["prompt_tokens"] for c in conversations]
    last_tokens = [c[-1]["prompt_tokens"] for c in conversations]
    n_turns = [len(c) for c in conversations]

    out = {
        "metadata": {
            "source": args.dataset,
            "scaffold": "OpenHands v0.54.0",
            "num_conversations": len(conversations),
            "tokenizer": args.model,
            "assumed_assistant_reply_tokens": 220,
        },
        "conversations": conversations,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f)

    def stats(name, xs):
        xs = sorted(xs)
        print(f"  {name}: min={xs[0]} p50={xs[len(xs)//2]} max={xs[-1]}")

    print(f"\nwrote {args.out}  ({len(conversations)} conversations)")
    stats("turns per conversation", n_turns)
    stats("prompt_tokens at turn 1 ", first_tokens)
    stats("prompt_tokens at last turn", last_tokens)


if __name__ == "__main__":
    main()
