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


def test_trijet_awkward_and_monetdb_agree(nano_file):
    """ADL Q6 trijet: the in-DB SQL (self-join + UDFs + window) must match
    ak.combinations."""
    aw = benchmark.trijet_awkward(nano_file, repeats=1)
    assert aw["metric"]["n"] > 0
    if _has_monetdbe:
        md = benchmark.trijet_monetdb(nano_file, repeats=1)
        assert md["metric"]["n"] == aw["metric"]["n"]
        assert abs(md["metric"]["mean"] - aw["metric"]["mean"]) < 1e-6


def test_zmumu_nano_awkward_and_monetdb_agree(nano_file):
    """Z→μμ on jagged NanoAOD: the in-DB SQL self-join must match ak.combinations
    (this is the real-CMS-NanoAOD benchmark path)."""
    aw = benchmark.zmumu_nano_awkward(nano_file, repeats=1)
    assert aw["metric"]["n"] >= 0
    if aw["metric"]["n"]:
        assert 60 < aw["metric"]["mean"] < 120        # inside the Z window
    if _has_monetdbe:
        md = benchmark.zmumu_nano_monetdb(nano_file, repeats=1)
        assert md["metric"]["n"] == aw["metric"]["n"]
        if aw["metric"]["n"]:
            assert abs(md["metric"]["mean"] - aw["metric"]["mean"]) < 1e-6


def test_zmumu_nano_metric_only_matches_full_fetch(nano_file):
    """--metric-only aggregates count/mean inside MonetDB; it must equal the
    full-fetch path (and hence the awkward selection)."""
    if not _has_monetdbe:
        import pytest
        pytest.skip("monetdbe not installed")
    full = benchmark.zmumu_nano_monetdb(nano_file, repeats=1, metric_only=False)
    mo = benchmark.zmumu_nano_monetdb(nano_file, repeats=1, metric_only=True)
    assert mo["metric"]["n"] == full["metric"]["n"]
    if full["metric"]["n"]:
        assert abs(mo["metric"]["mean"] - full["metric"]["mean"]) < 1e-9


def test_zmumu_nano_hybrid_matches(nano_file):
    """Hybrid (SQL event cut + Awkward combinatorics) must equal the pure
    awkward selection."""
    aw = benchmark.zmumu_nano_awkward(nano_file, repeats=1)
    if not _has_monetdbe:
        import pytest
        pytest.skip("monetdbe not installed")
    hy = benchmark.zmumu_nano_hybrid(nano_file, repeats=1)
    assert hy["metric"]["n"] == aw["metric"]["n"]
    if aw["metric"]["n"]:
        assert abs(hy["metric"]["mean"] - aw["metric"]["mean"]) < 1e-6
