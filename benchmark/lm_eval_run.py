#!/usr/bin/env python
"""Thin lm-eval launcher that hardens the vLLM backend against a known crash.

Some chat models (e.g. THUDM/glm-4-9b-chat) surface an empty string in the
per-request ``stop``/``until`` list that lm-eval hands to vLLM. Recent vLLM
rejects this in ``SamplingParams._verify_args`` with::

    ValueError: stop cannot contain an empty string.

which aborts the whole benchmark run. We monkeypatch the ``SamplingParams``
reference used inside ``lm_eval.models.vllm_causallms`` so empty/None stop
strings are dropped before construction, then defer to the normal lm-eval CLI.

All CLI arguments are passed through unchanged, so this is a drop-in
replacement for the ``lm_eval`` entry point.
"""
from __future__ import annotations

import lm_eval.models.vllm_causallms as _vllm_mod
from lm_eval.__main__ import cli_evaluate

_RealSamplingParams = _vllm_mod.SamplingParams


def _sanitize_stop(value):
    """Drop empty/None entries from a stop spec; return None if nothing left."""
    if value is None:
        return None
    if isinstance(value, str):
        return value if value else None
    cleaned = [s for s in value if s]
    return cleaned or None


def _SamplingParamsSafe(*args, **kwargs):
    if "stop" in kwargs:
        kwargs["stop"] = _sanitize_stop(kwargs["stop"])
    return _RealSamplingParams(*args, **kwargs)


# Patch the name bound inside vllm_causallms (it does `from vllm import
# SamplingParams`), which is what both generate_until code paths call.
_vllm_mod.SamplingParams = _SamplingParamsSafe


if __name__ == "__main__":
    cli_evaluate()
