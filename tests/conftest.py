"""Pytest fixtures and collection hooks for the discrete1 test suite.

This module defines shared pytest fixtures and command-line options
used by the tests. The fixtures are intentionally lightweight and are
applied automatically where appropriate.
"""

import pytest


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def pytest_addoption(parser):
    """Add custom command-line options to pytest.

    Options
    -------
    --mg : bool
        When set, enable (do not skip) multigroup one-dimensional tests.
    --ml : bool
        When set, enable (do not skip) machine learning tests.
    """
    parser.addoption(
        "--mg",
        action="store_true",
        default=False,
        help="Runs one-dimensional multigroup tests if True",
    )
    parser.addoption(
        "--ml",
        action="store_true",
        default=False,
        help="Runs machine learning tests if True",
    )


def pytest_collection_modifyitems(config, items):
    """Modify collected test items based on provided options.

    If the ``--mg`` option is not provided, mark tests that have the
    "multigroup" keyword to be skipped. This avoids running long
    multigroup tests by default in CI.

    If the ``--ml`` option is not provided, mark tests that have the
    "machine_learning" keyword to be skipped. This avoids running
    machine learning tests by default in CI.
    """
    # One dimensional multigroup
    if not config.getoption("--mg"):
        multigroup1d = pytest.mark.skip(reason="Run on --mg option")
        for item in items:
            if "multigroup" in item.keywords:
                item.add_marker(multigroup1d)

    # Machine learning tests
    if not config.getoption("--ml"):
        machine_learning = pytest.mark.skip(reason="Run on --ml option")
        for item in items:
            if "machine_learning" in item.keywords:
                item.add_marker(machine_learning)
