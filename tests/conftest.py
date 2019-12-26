import torch


def pytest_configure(config):
    config.addinivalue_line("markers", "seed_rng: seed the pytorch RNG")


def pytest_runtest_setup(item):
    seed_marker = item.get_closest_marker("seed_rng")
    if seed_marker is not None:
        if len(seed_marker.args) != 0:
            seed = seed_marker.args[0]
        else:
            seed = 1337
        torch.manual_seed(seed)
