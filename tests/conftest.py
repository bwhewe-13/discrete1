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
    """
    parser.addoption(
        "--mg",
        action="store_true",
        default=False,
        help="Runs one-dimensional multigroup problems if True",
    )


def pytest_collection_modifyitems(config, items):
    """Modify collected test items based on provided options.

    If the ``--mg`` option is not provided, mark tests that have the
    "multigroup" keyword to be skipped. This avoids running long
    multigroup tests by default in CI.
    """
    # One dimensional multigroup
    if config.getoption("--mg"):
        # --mg given in cli: do not skip multigroup tests
        return
    multigroup1d = pytest.mark.skip(reason="Run on --mg option")
    for item in items:
        if "multigroup" in item.keywords:
            item.add_marker(multigroup1d)
            item.add_marker(multigroup1d)
            item.add_marker(multigroup1d)
