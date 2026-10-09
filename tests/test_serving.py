import pytest

from rkv import R1KV
from rkv.serving import RKVServing


def test_serving_policy_inherits_legacy_rkv():
    assert issubclass(RKVServing, R1KV)
    assert RKVServing().mix_lambda == R1KV().mix_lambda


def test_serving_config_defaults_do_not_change_legacy_algorithm():
    policy = RKVServing.from_serving_config({})

    assert type(policy) is RKVServing
    assert (policy.budget, policy.buffer, policy.window_size, policy.kernel_size) == (
        128, 128, 8, 7
    )
    assert policy.mix_lambda == 0.1
    assert policy.retain_ratio == 0.1
    assert policy.retain_direction == "last"
    assert R1KV().mix_lambda == 0.07
    assert policy._serving_query_history == {}
    assert policy._serving_layer_order is None


def test_serving_config_overrides():
    policy = RKVServing.from_serving_config({
        "budget": 64,
        "buffer": 40,
        "window_size": 4,
        "kernel_size": 5,
        "mix_lambda": 0.25,
        "retain_ratio": 0.2,
        "retain_direction": "first",
    })

    assert policy.budget == 64
    assert policy.buffer == 40
    assert policy.window_size == 4
    assert policy.kernel_size == 5
    assert policy.mix_lambda == 0.25
    assert policy.retain_ratio == 0.2
    assert policy.retain_direction == "first"


def test_serving_config_instances_have_independent_request_state():
    first = RKVServing.from_serving_config({})
    second = RKVServing.from_serving_config({})

    assert first is not second
    assert first._serving_query_history is not second._serving_query_history
    first._serving_query_history["layer0"] = object()
    assert second._serving_query_history == {}


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ([], "must be a mapping"),
        ({"unknown": 1}, "Unsupported R-KV serving config keys"),
        ({"budget": 0}, "budget must be a positive integer"),
        ({"budget": True}, "budget must be a positive integer"),
        ({"buffer": 0}, "buffer must be a positive integer"),
        ({"buffer": False}, "buffer must be a positive integer"),
        ({"window_size": 0}, "window_size must be a positive integer"),
        ({"window_size": True}, "window_size must be a positive integer"),
        ({"budget": 8}, "budget must be greater than window_size"),
        ({"buffer": 4}, "buffer must be >= window_size"),
        ({"kernel_size": 4}, "kernel_size must be a positive odd integer"),
        ({"mix_lambda": True}, "mix_lambda must be numeric"),
        ({"mix_lambda": 2.0}, "mix_lambda must be in"),
        ({"retain_ratio": False}, "retain_ratio must be numeric"),
        ({"retain_ratio": 0}, "retain_ratio must be in"),
        ({"retain_direction": "middle"}, "Unsupported retain_direction"),
    ],
)
def test_serving_config_rejects_invalid_settings(config, message):
    with pytest.raises(ValueError, match=message):
        RKVServing.from_serving_config(config)
