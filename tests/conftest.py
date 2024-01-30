
from pathlib import Path
import pytest


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def pytest_addoption(parser):
    parser.addoption("--mg", action="store_true", default=False, \
                     help="Runs one-dimensional multigroup problems if True")

def pytest_collection_modifyitems(config, items):
    # One dimensional multigroup
    if config.getoption("--mg"):
        # --mg given in cli: do not skip multigroup tests
        return
    multigroup1d = pytest.mark.skip(reason="Run on --mg option")
    for item in items:
        if "multigroup" in item.keywords:
            item.add_marker(multigroup1d)
