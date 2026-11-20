import importlib
import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))


@pytest.fixture(scope="session")
def breeze():
    name = "_breeze_tests"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    return package


@pytest.fixture(scope="session")
def native(breeze):
    return importlib.import_module(f"{breeze.__name__}.native")


@pytest.fixture(scope="session")
def loader(breeze):
    return importlib.import_module(f"{breeze.__name__}.loader")


@pytest.fixture(scope="session")
def runtime(breeze):
    return importlib.import_module(f"{breeze.__name__}.runtime")
