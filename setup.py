from setuptools import setup, find_packages

setup(
    name="rkv",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        "lmcache.token_drop_algorithms": [
            "rkv = rkv.serving:RKVServing.from_serving_config",
        ],
    },
)
