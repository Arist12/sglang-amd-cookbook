#!/usr/bin/env python3
"""Write the phase-5 accuracy verdict from the eval outputs on disk.

    gate-k3.py <accuracy_dir> > accuracy_gate.md

Standalone (rather than inline in accuracy-k3.sh) so the verdict can be recomputed
after the fact -- e.g. when the degeneration thresholds are corrected -- without
re-running six hours of evals.

Tolerances are set against the published Day-0 baselines: a config only ships if it
is indistinguishable from them.
"""

from __future__ import annotations

import os
import re
import sys

BASE = {
    "nospec": {"gsm8k": 97.49, "aime25": 93.33, "aime_sigma": 4.36},
    "dspark": {"gsm8k": 97.64, "aime25": 94.58, "aime_sigma": 3.05},
}
GSM8K_TOL_PP = 1.0


def read(path: str) -> str:
    if not os.path.exists(path):
        return ""
    return open(path, encoding="utf-8", errors="replace").read()


def pct(v: float) -> float:
    """Eval harnesses report either a fraction or a percentage."""
    return v * 100 if v <= 1.0 else v


def main() -> int:
    acc = sys.argv[1] if len(sys.argv) > 1 else "."
    out = ["# Phase 5 accuracy gate", ""]
    out.append(
        "Each tuned config re-run through exactly the protocol that produced the "
        "published Day-0 numbers, so the comparison is direct."
    )
    out.append("")
    verdicts: list[tuple[str, bool]] = []

    for lane in ("nospec", "dspark"):
        base = BASE[lane]
        out.append(f"## lane `{lane}`")
        out.append("")

        g = read(f"{acc}/gsm8k-{lane}.txt")
        # The [METRIC] line is exact. The fallback anchors on a colon so it cannot
        # pick up the 64 from "--max-tokens 64000", which a looser pattern does.
        m = re.findall(r"gsm8k_score=([\d.]+)", g) or re.findall(
            r"^\s*Score:\s*([\d.]+)", g, re.M
        )
        if m:
            v = pct(float(m[-1]))
            d = v - base["gsm8k"]
            ok = abs(d) <= GSM8K_TOL_PP
            verdicts.append((f"{lane}/gsm8k", ok))
            out.append(
                f"- GSM8K greedy n=1319: **{v:.3f}%** vs baseline {base['gsm8k']}% "
                f"(delta {d:+.3f} pp, tolerance +/-{GSM8K_TOL_PP} pp) -> "
                f"**{'PASS' if ok else 'FAIL'}**"
            )
        else:
            out.append("- GSM8K: no result")

        a = read(f"{acc}/aime25-{lane}.txt")
        # Anchor on the "=" rather than scanning forward from "pass@1": the label is
        # `pass@1[avg-of-8]`, so a lazy match happily returns the 8 from "avg-of-8".
        m = re.findall(r"pass@1\[avg-of-\d+\]\s*=\s*([\d.]+)%?(?:\s*\+/-\s*([\d.]+))?", a)
        if m:
            v = pct(float(m[-1][0]))
            own_sigma = m[-1][1]
            d = v - base["aime25"]
            ok = abs(d) <= base["aime_sigma"]
            verdicts.append((f"{lane}/aime25", ok))
            out.append(
                f"- AIME25 pass@1 avg-of-8: **{v:.2f}%**"
                + (f" +/-{own_sigma}%" if own_sigma else "")
                + f" vs baseline {base['aime25']}% "
                f"(delta {d:+.2f} pp, baseline 1 sigma = {base['aime_sigma']}) -> "
                f"**{'PASS' if ok else 'FAIL'}**"
            )
            for field in ("stop_rate", "truncated", "no_answer", "error_rate"):
                fm = re.search(rf"{field}\D{{0,12}}([\d.]+)", a)
                if fm:
                    out.append(f"  - {field} = {fm.group(1)}")
        else:
            out.append("- AIME25: not run")

        gb = read(f"{acc}/gibberish-{lane}.txt")
        if gb:
            vm = re.search(r"verdict=(\w+)", gb)
            rmax = re.search(r"rep\d+gram_max=([\d.]+)", gb)
            rmean = re.search(r"rep\d+gram_mean=([\d.]+)", gb)
            runmax = re.search(r"longest_token_run_max=(\d+)", gb)
            empty = re.search(r"empty=(\d+)", gb)
            if vm:
                ok = vm.group(1) == "PASS"
                verdicts.append((f"{lane}/degeneration", ok))
                out.append(
                    f"- degeneration probe: **{vm.group(1)}** "
                    f"(mean n-gram repetition {rmean.group(1) if rmean else '?'}, "
                    f"max {rmax.group(1) if rmax else '?'}, longest consecutive token run "
                    f"{runmax.group(1) if runmax else '?'}, empty outputs "
                    f"{empty.group(1) if empty else '?'})"
                )
        out.append("")

    out.append("## Verdict")
    out.append("")
    if not verdicts:
        out.append("INCONCLUSIVE - no results parsed.")
    else:
        failed = [name for name, ok in verdicts if not ok]
        if not failed:
            out.append(
                f"**PASS** - all {len(verdicts)} checks within tolerance of the published "
                "baseline. The tuned recipes change scheduling and memory, not output."
            )
        else:
            out.append(
                f"**FAIL** - {len(failed)} of {len(verdicts)} checks outside tolerance: "
                + ", ".join(f"`{f}`" for f in failed)
                + ". Do not publish the affected config as recommended."
            )
        out.append("")
        for name, ok in verdicts:
            out.append(f"- `{name}`: {'PASS' if ok else 'FAIL'}")

    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
