# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the prefill->decode KV-transfer term in disagg planning.

See ai-dynamo/dynamo#10863 (p_storage_d_network path). The transfer term is
off by default (fabric_bandwidth_GBps=0 or kv_bytes_per_token=0), in which case
the disagg row must be bit-identical to the pre-feature output.
"""

import pytest

from aiconfigurator.sdk.picking import (
    _build_disagg_summary_dict,
    _kv_transfer_latency_ms,
)

pytestmark = pytest.mark.unit


def _prefill_dict(**overrides) -> dict:
    base = {
        "model": "test-model",
        "isl": 4000,
        "osl": 500,
        "prefix": 0,
        "concurrency": 1,
        "bs": 1,
        "global_bs": 1,
        "tp": 4,
        "pp": 1,
        "dp": 1,
        "moe_tp": 1,
        "moe_ep": 1,
        "cp": 1,
        "parallel": "tp4",
        "ttft": 80.0,
        "tpot": 0.0,
        "seq/s": 10.0,
        "tokens/s/user": 0.0,
        "gemm": "fp8",
        "kvcache": "fp8",
        "fmha": "fp8",
        "moe": "fp8",
        "comm": "half",
        "memory": 12.3,
        "backend": "trtllm",
        "version": "1.3.0",
        "system": "h200_sxm",
        "power_w": 500.0,
        "encoder_latency": 0.0,
        "encoder_memory": 0.0,
    }
    base.update(overrides)
    return base


def _decode_dict(**overrides) -> dict:
    base = {
        "model": "test-model",
        "isl": 4000,
        "osl": 500,
        "prefix": 0,
        "concurrency": 32,
        "bs": 32,
        "global_bs": 32,
        "tp": 4,
        "pp": 1,
        "dp": 1,
        "moe_tp": 1,
        "moe_ep": 1,
        "cp": 1,
        "parallel": "tp4",
        "ttft": 0.0,
        "tpot": 10.0,
        "seq/s": 100.0,  # decode not the bottleneck, so prefill occupancy drives seq/s
        "tokens/s/user": 100.0,
        "gemm": "fp8",
        "kvcache": "fp8",
        "fmha": "fp8",
        "moe": "fp8",
        "comm": "half",
        "memory": 40.0,
        "backend": "trtllm",
        "version": "1.3.0",
        "system": "h200_sxm",
        "power_w": 600.0,
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# _kv_transfer_latency_ms                                                      #
# --------------------------------------------------------------------------- #


def test_transfer_latency_off_by_default() -> None:
    assert _kv_transfer_latency_ms(4000, 0.0, 100.0, 0.0) == 0.0
    assert _kv_transfer_latency_ms(4000, 24576.0, 0.0, 0.0) == 0.0


def test_transfer_latency_units_and_value() -> None:
    # 4000 tokens * 24576 B/token = 98.304 MB over 100 GB/s = 0.983 ms.
    ms = _kv_transfer_latency_ms(4000, 24576.0, 100.0, 0.0)
    assert ms == pytest.approx(4000 * 24576.0 / (100.0 * 1e9) * 1e3)
    assert ms == pytest.approx(0.983, abs=1e-3)


def test_transfer_latency_scales_with_uncached_fraction() -> None:
    full = _kv_transfer_latency_ms(4000, 24576.0, 100.0, 0.0)
    half = _kv_transfer_latency_ms(4000, 24576.0, 100.0, 0.5)
    none = _kv_transfer_latency_ms(4000, 24576.0, 100.0, 1.0)
    assert half == pytest.approx(full * 0.5)
    assert none == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# _build_disagg_summary_dict integration                                      #
# --------------------------------------------------------------------------- #


def test_disagg_row_unchanged_when_transfer_off() -> None:
    """Feature off -> row identical to the pre-feature output."""
    base = _build_disagg_summary_dict(_prefill_dict(), 1, _decode_dict(), 1)
    off = _build_disagg_summary_dict(
        _prefill_dict(),
        1,
        _decode_dict(),
        1,
        fabric_bandwidth_GBps=0.0,
        kv_bytes_per_token=24576.0,  # ignored because bw==0
    )
    assert off["ttft"] == base["ttft"]
    assert off["seq/s"] == base["seq/s"]
    assert off["request_latency"] == base["request_latency"]


def test_transfer_adds_to_ttft_and_request_latency() -> None:
    p, d = _prefill_dict(), _decode_dict()
    row = _build_disagg_summary_dict(
        p, 1, d, 1, fabric_bandwidth_GBps=100.0, kv_bytes_per_token=24576.0
    )
    expected_transfer = _kv_transfer_latency_ms(p["isl"], 24576.0, 100.0, 0.0)
    assert row["ttft"] == pytest.approx(p["ttft"] + expected_transfer)
    assert row["request_latency"] == pytest.approx(
        p["ttft"] + expected_transfer + d["tpot"] * (p["osl"] - 1)
    )


def test_transfer_lowers_prefill_bound_seq_s() -> None:
    """When prefill is the rate-match bottleneck, transfer reduces system seq/s."""
    # decode seq/s huge (100) so prefill (10) binds.
    base = _build_disagg_summary_dict(_prefill_dict(), 1, _decode_dict(), 1)
    # A slow fabric with big KV -> noticeable transfer occupancy.
    slow = _build_disagg_summary_dict(
        _prefill_dict(),
        1,
        _decode_dict(),
        1,
        fabric_bandwidth_GBps=5.0,
        kv_bytes_per_token=200_000.0,
    )
    assert slow["seq/s"] < base["seq/s"]
    # occupancy math: eff_seq_s = 10 * ttft/(ttft+transfer); *0.9 degradation
    p = _prefill_dict()
    transfer = _kv_transfer_latency_ms(p["isl"], 200_000.0, 5.0, 0.0)
    eff = p["seq/s"] * p["ttft"] / (p["ttft"] + transfer)
    assert slow["seq/s"] == pytest.approx(eff * 0.9)


def test_hit_rate_reduces_transfer_penalty() -> None:
    cold = _build_disagg_summary_dict(
        _prefill_dict(),
        1,
        _decode_dict(),
        1,
        fabric_bandwidth_GBps=5.0,
        kv_bytes_per_token=200_000.0,
        kv_hit_rate=0.0,
    )
    warm = _build_disagg_summary_dict(
        _prefill_dict(),
        1,
        _decode_dict(),
        1,
        fabric_bandwidth_GBps=5.0,
        kv_bytes_per_token=200_000.0,
        kv_hit_rate=0.5,
    )
    assert warm["ttft"] < cold["ttft"]
    assert warm["seq/s"] > cold["seq/s"]
