import importlib.util

import pytest

from awkward_monetizer.roundtrip import run

_has_monetdbe = importlib.util.find_spec("monetdbe") is not None


@pytest.mark.skipif(not _has_monetdbe, reason="monetdbe not installed")
def test_embedded_roundtrip(cms_root):
    """Full ingest -> embedded MonetDB -> SQL slice -> reconstruct -> validate."""
    res = run(root_file=cms_root, dataset="dimuon", backend="embedded",
              where="mass BETWEEN 60 AND 120")
    assert res["passed"]
    assert res["max_abs_diff"] < 1e-6
    assert res["n_events"] > 0
