import pytest

from rkv import R1KV


def test_serving_config_defaults_match_vllm_port_without_changing_legacy_defaults():
    legacy = R1KV()
    serving = R1KV.from_serving_config(
        {
            "budget": 128,
            "buffer": 128,
        }
    )

    assert legacy.mix_lambda == 0.07
    assert serving.budget == 128
    assert serving.buffer == 128
    assert serving.window_size == 8
    assert serving.kernel_size == 7
    assert serving.mix_lambda == 0.1
    assert serving.retain_ratio == 0.1
    assert serving.retain_direction == "last"

def test_serving_config_overrides_match_vllm_port_algorithm_knobs():
    policy = R1KV.from_serving_config(
        {
            "budget": 64,
            "buffer": 40,
            "window_size": 4,
            "kernel_size": 5,
            "mix_lambda": 0.25,
            "retain_ratio": 0.2,
            "retain_direction": "first",
        }
    )

    assert policy.budget == 64
    assert policy.buffer == 40
    assert policy.window_size == 4
    assert policy.kernel_size == 5
    assert policy.mix_lambda == 0.25
    assert policy.retain_ratio == 0.2
    assert policy.retain_direction == "first"

@pytest.mark.parametrize(
    ("config", "match"),
    [
        ([], "mapping"),
        ({"budget": 32, "buffer": 16, "unused": 0}, "Unsupported"),
        ({"budget": 0, "buffer": 16}, "positive integer"),
        ({"budget": True, "buffer": 16}, "positive integer"),
        ({"budget": 8, "buffer": 16}, "greater than window_size"),
        ({"budget": 32, "buffer": 4}, ">= window_size"),
        ({"budget": 32, "buffer": 16, "window_size": 0}, "window_size"),
        ({"budget": 32, "buffer": 16, "kernel_size": 4}, "odd integer"),
        ({"budget": 32, "buffer": 16, "mix_lambda": 2.0}, "mix_lambda"),
        ({"budget": 32, "buffer": 16, "retain_ratio": 0}, "retain_ratio"),
        (
            {"budget": 32, "buffer": 16, "retain_direction": "middle"},
            "retain_direction",
        ),
        ({"budget": 32, "buffer": 16, "unknown": 1}, "Unsupported"),
    ],
)
def test_serving_config_validation(config, match):
    with pytest.raises(ValueError, match=match):
        R1KV.from_serving_config(config)


def test_empty_serving_config_uses_rkv_owned_defaults():
    policy = R1KV.from_serving_config({})
    assert policy.budget == 128
    assert policy.buffer == 128
    assert policy.window_size == 8
