# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy

import pytest

from vllm.models.hy_v4.dfly_config import _validate_checkpoint_quantization

_MXFP8_ARGS = dict(
    num_bits=8,
    type="float",
    symmetric=True,
    group_size=32,
    strategy="group",
    scale_dtype="torch.uint8",
)


def _ct_mxfp8_config() -> dict:
    # quantization_config exported by AutoRound (llm_compressor format).
    return {
        "quant_method": "compressed-tensors",
        "format": "mxfp8-quantized",
        "kv_cache_scheme": None,
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": dict(_MXFP8_ARGS, dynamic=False),
                "input_activations": dict(_MXFP8_ARGS, dynamic=True),
                "output_activations": None,
                "format": None,
            }
        },
    }


def _group(config: dict) -> dict:
    return config["config_groups"]["group_0"]


def test_accepts_compressed_tensors_mxfp8():
    _validate_checkpoint_quantization(_ct_mxfp8_config())


def test_accepts_mixed_precision_with_mxfp8_group():
    config = _ct_mxfp8_config()
    config["format"] = "mixed-precision"
    _group(config)["format"] = "mxfp8-quantized"
    _validate_checkpoint_quantization(config)


def _static_activations(c):
    _group(c)["input_activations"]["dynamic"] = False


def _weight_only(c):
    _group(c)["input_activations"] = None


def _group_size_16(c):
    _group(c)["weights"]["group_size"] = 16


def _fp32_scales(c):
    _group(c)["weights"]["scale_dtype"] = "torch.float32"


def _kv_cache_scheme(c):
    c["kv_cache_scheme"] = {"num_bits": 8, "type": "float"}


@pytest.mark.parametrize(
    "mutate",
    [
        _static_activations,
        _weight_only,
        _group_size_16,
        _fp32_scales,
        _kv_cache_scheme,
    ],
)
def test_rejects_unsupported_mxfp8_variants(mutate):
    config = copy.deepcopy(_ct_mxfp8_config())
    mutate(config)
    with pytest.raises(ValueError, match="MXFP8"):
        _validate_checkpoint_quantization(config)
