from rkv import R1KV
from rkv.serving import RKVServing


def test_serving_policy_inherits_legacy_rkv():
    assert issubclass(RKVServing, R1KV)
    assert RKVServing().mix_lambda == R1KV().mix_lambda
