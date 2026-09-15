import importlib.util

import uproot

from awkward_monetizer import benchmark

_has_monetdbe = importlib.util.find_spec("monetdbe") is not None


def test_awkward_and_hybrid_agree(cms_root):
    """The awkward and hybrid backends must produce the same Z→μμ selection."""
    aw = benchmark.backend_awkward(cms_root, "dimuon", repeats=1)
    assert aw["metric"]["n"] > 0
    assert 80 < aw["metric"]["mean"] < 100  # Z peak region

    if _has_monetdbe:
        hy = benchmark.backend_hybrid(cms_root, "dimuon", repeats=1,
                                      hybrid_backend="embedded")
        assert hy["metric"]["n"] == aw["metric"]["n"]
        assert abs(hy["metric"]["mean"] - aw["metric"]["mean"]) < 1e-6


def test_scale_root_tiles_events(tmp_path, cms_root):
    out = benchmark.scale_root(cms_root, 3, str(tmp_path / "big.root"), "dimuon")
    with uproot.open(out) as f:
        assert f["events"].num_entries == 2304 * 3


def test_monetdb_backend_agrees(cms_root):
    """The in-database SQL-UDF backend must match the awkward selection."""
    if not _has_monetdbe:
        import pytest
        pytest.skip("monetdbe not installed")
    aw = benchmark.backend_awkward(cms_root, "dimuon", repeats=1)
    md = benchmark.backend_monetdb(cms_root, "dimuon", repeats=1,
                                   hybrid_backend="embedded")
    assert md["metric"]["n"] == aw["metric"]["n"]
    assert abs(md["metric"]["mean"] - aw["metric"]["mean"]) < 1e-6
