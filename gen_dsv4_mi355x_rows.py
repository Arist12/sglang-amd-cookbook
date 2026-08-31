#!/usr/bin/env python3
"""Regenerate DeepSeek-V4 MI355X rows from the repeated serving records.

Each published point is the median of three complete fixed-length runs.  The
script fails closed on a missing repeat, token-accounting mismatch, unexpected
server configuration/provenance, or more than 5% total-throughput spread.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


SGLANG_SHA = "71de97b264b04dcd514cf904003028aefe9775c8"
AITER_SHA = "d9e5ef7ce08ee7045d583aed768cff41aa9210fe"
DATASET_SHA = "35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4"
CONCURRENCIES = (1, 8, 32)
REPEATS = (1, 2, 3)

VARIANTS = {
    "flash": {
        "model_id": "deepseek-v4-flash-0731",
        "model_path": "/data/DeepSeek-V4-Flash-0731",
        "served_name": "deepseek-v4-flash-0731",
        "revision": "7872f01b1d1fe23eabc4c98b48bffcef5a386062",
        "index_sha256": "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b",
        "results_subdir": "flash-0731/perf-final",
        "source": (
            "dsv4_flash_playbook.md section 5 (median of 3 runs; 2026-08-31; "
            "8x MI355X; DeepSeek-V4-Flash-0731 7872f01b1d; SGLang 71de97b264; "
            "AITER d9e5ef7ce0; unified_kv_triton)"
        ),
    },
    "pro": {
        "model_id": "deepseek-v4-pro-0813",
        "model_path": "/data/DeepSeek-V4-Pro-0813",
        "served_name": "deepseek-v4-pro-0813",
        "revision": "72e1d3230f6c080a530b0a1d46f8eb4602340597",
        "index_sha256": "2de2ac1e43134f8b03bf6156067715b7c3c73b1a507329e606023c601a56d30a",
        "results_subdir": "pro-0813/perf-final",
        "source": (
            "dsv4_pro_playbook.md section 5 (median of 3 runs; 2026-08-31; "
            "8x MI355X; DeepSeek-V4-Pro-0813 72e1d3230f; SGLang 71de97b264; "
            "AITER d9e5ef7ce0; unified_kv_triton)"
        ),
    },
}


def read_text(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"missing provenance file: {path}")
    return path.read_text().strip()


def load_record(path: Path) -> dict:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(records) != 1:
        raise ValueError(f"{path}: expected exactly one JSONL record, found {len(records)}")
    return records[0]


def recorded_sha(path: Path) -> str:
    fields = read_text(path).split()
    if not fields:
        raise ValueError(f"empty checksum file: {path}")
    return fields[0]


def validate_quiescent(path: Path) -> None:
    payload = json.loads(read_text(path))
    gpus = payload.get("gpus") or []
    if payload.get("samples") != 30 or len(gpus) != 8:
        raise ValueError(f"{path}: expected 30 samples on 8 GPUs")
    for gpu in gpus:
        gfx = gpu.get("gfx_samples") or []
        umc = gpu.get("umc_samples") or []
        if len(gfx) != 30 or len(umc) != 30:
            raise ValueError(f"{path}: incomplete samples for GPU {gpu.get('smi_index')}")
        if not gpu.get("quiescent") or max(gfx) != 0 or max(umc) != 0:
            raise ValueError(f"{path}: GPU {gpu.get('smi_index')} was not fully idle")


def validate_provenance(root: Path, spec: dict) -> None:
    expected = {
        "model-revision.txt": spec["revision"],
        "sglang-sha.txt": SGLANG_SHA,
        "aiter-sha.txt": AITER_SHA,
    }
    for filename, value in expected.items():
        actual = read_text(root / filename)
        if actual != value:
            raise ValueError(f"{root / filename}: {actual!r}, expected {value!r}")

    checksums = {
        "model-index.sha256": spec["index_sha256"],
        "dataset.sha256": DATASET_SHA,
    }
    for filename, expected_sha in checksums.items():
        actual_sha = recorded_sha(root / filename)
        if actual_sha != expected_sha:
            raise ValueError(
                f"{root / filename}: sha256 {actual_sha}, expected {expected_sha}"
            )

    for phase in ("before", "after"):
        validate_quiescent(root / f"gpu-quiescent-{phase}.json")


def rows_for(root: Path, spec: dict) -> list[dict]:
    validate_provenance(root, spec)
    rows = []
    for concurrency in CONCURRENCIES:
        records = [
            load_record(root / f"perf-c{concurrency}-r{repeat}.jsonl")
            for repeat in REPEATS
        ]
        for repeat, record in zip(REPEATS, records):
            expected_benchmark = {
                "backend": "sglang",
                "dataset_name": "random",
                "max_concurrency": concurrency,
                "random_input_len": 8192,
                "random_output_len": 1024,
                "random_range_ratio": 1.0,
            }
            for key, value in expected_benchmark.items():
                if record.get(key) != value:
                    raise ValueError(
                        f"{spec['model_id']} c{concurrency} r{repeat}: "
                        f"{key}={record.get(key)!r}, expected {value!r}"
                    )
            expected_tokens = (
                4 * concurrency,
                4 * concurrency * 8192,
                4 * concurrency * 1024,
            )
            actual_tokens = (
                record.get("completed"),
                record.get("total_input_tokens"),
                record.get("total_output_tokens"),
            )
            if actual_tokens != expected_tokens:
                raise ValueError(
                    f"{spec['model_id']} c{concurrency} r{repeat}: token accounting "
                    f"{actual_tokens}, expected {expected_tokens}"
                )
            info = record.get("server_info") or {}
            expected_info = {
                "model_path": spec["model_path"],
                "served_model_name": spec["served_name"],
                "tp_size": 8,
                "dp_size": 1,
                "attention_backend": "dsv4",
                "kv_cache_dtype": "fp8_e4m3",
                "trust_remote_code": True,
                "mem_fraction_static": 0.9,
                "disable_radix_cache": True,
                "page_size": 256,
                "chunked_prefill_size": 8192,
                "max_running_requests": 256,
                "swa_full_tokens_ratio": 0.1,
                "disable_shared_experts_fusion": True,
                "enable_dp_attention": False,
                "speculative_algorithm": None,
                "tool_call_parser": "deepseekv4",
                "reasoning_parser": "deepseek-v4",
            }
            for key, value in expected_info.items():
                if info.get(key) != value:
                    raise ValueError(
                        f"{spec['model_id']} c{concurrency} r{repeat}: "
                        f"server_info.{key}={info.get(key)!r}, expected {value!r}"
                    )

        totals = [float(record["total_throughput"]) for record in records]
        spread = (max(totals) - min(totals)) / statistics.mean(totals)
        if spread > 0.05:
            raise ValueError(
                f"{spec['model_id']} c{concurrency}: total-throughput spread "
                f"{spread:.2%} exceeds 5%"
            )

        med = lambda key: statistics.median(float(record[key]) for record in records)
        tpot = med("median_tpot_ms")
        row = {
            "isl": 8192,
            "osl": 1024,
            "concurrency": concurrency,
            "ttft_ms": round(med("median_ttft_ms"), 2),
            "tpot_ms": round(tpot, 2),
        }
        if concurrency == 1:
            row["decode_tok_s"] = round(1000 / tpot, 1)
        total = med("total_throughput")
        row.update(
            {
                "output_tok_s": round(med("output_throughput"), 2),
                "total_tok_s": round(total, 2),
                "tok_s_per_gpu": round(total / 8, 1),
                "source": spec["source"],
            }
        )
        rows.append(row)
    return rows


def load_models(path: Path) -> list[dict]:
    source = path.read_text()
    marker = "window.MODELS = "
    if marker not in source:
        raise ValueError(f"{path}: {marker!r} not found")
    payload = source.split(marker, 1)[1].strip()
    if not payload.endswith(";"):
        raise ValueError(f"{path}: MODELS assignment does not end in a semicolon")
    return json.loads(payload[:-1])


def check_models(path: Path, generated: dict[str, list[dict]]) -> None:
    models = load_models(path)
    for variant, rows in generated.items():
        model_id = VARIANTS[variant]["model_id"]
        model = next((item for item in models if item.get("id") == model_id), None)
        if model is None:
            raise ValueError(f"{path}: {model_id} entry missing")
        cell = next(
            (
                item
                for item in model.get("configs", [])
                if item.get("gfx") == "gfx950" and item.get("strategy") == "low-latency"
            ),
            None,
        )
        if cell is None:
            raise ValueError(f"{path}: {model_id} gfx950 low-latency cell missing")
        if cell.get("benchmarks") != rows:
            raise ValueError(f"{path}: {model_id} benchmark rows differ from raw records")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("/results/dsv4-mi355x-20260831"),
    )
    parser.add_argument("--variant", choices=["flash", "pro", "both"], default="both")
    parser.add_argument("--check-models", type=Path)
    args = parser.parse_args()

    names = VARIANTS if args.variant == "both" else {args.variant: VARIANTS[args.variant]}
    generated = {
        name: rows_for(args.results_root / spec["results_subdir"], spec)
        for name, spec in names.items()
    }
    if args.check_models:
        check_models(args.check_models, generated)
    print(json.dumps(generated, indent=2))


if __name__ == "__main__":
    main()
