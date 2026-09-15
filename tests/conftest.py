import pathlib

import pytest

from awkward_monetizer.adl import events_from_root
from awkward_monetizer.sampledata import write_nanoaod

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="session")
def cms_root():
    """Path to the real dimuon file, or skip if it isn't present."""
    p = DATA / "cms.root"
    if not p.exists():
        pytest.skip("data/cms.root not present (see data/README.md)")
    return str(p)


@pytest.fixture(scope="session")
def nano_file(tmp_path_factory):
    """A freshly generated synthetic NanoAOD file."""
    p = tmp_path_factory.mktemp("data") / "nano.root"
    write_nanoaod(str(p), n_events=1500, seed=1)
    return str(p)


@pytest.fixture(scope="session")
def nano_events(nano_file):
    return events_from_root(nano_file)
