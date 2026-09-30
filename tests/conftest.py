"""Fast/slow split: ``slow`` tests (the four twin rebuilds) need --runslow.

    pytest                   the fast suite
    pytest --runslow         everything, twins included
    pytest -m slow --runslow just the twins
"""

import pytest


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False,
                     help="also run the slow end-to-end tests (the four twins)")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: end-to-end, minutes not seconds (needs --runslow)")
    config.addinivalue_line(
        "markers", "needs_rl: needs mjlab/torch, which the generator does not")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="slow; pass --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
