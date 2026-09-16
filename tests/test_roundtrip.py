import importlib.util
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from awkward_monetizer import roundtrip
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


def test_unsupported_dataset_rejected_before_io():
    with patch.object(roundtrip, "read_root") as read, \
            patch.object(roundtrip, "open_server") as connect, \
            pytest.raises(ValueError, match="only the dimuon"):
        run(root_file="unused.root", dataset="nanoaod")
    read.assert_not_called()
    connect.assert_not_called()


def test_empty_selection_is_not_a_success(capsys):
    conn = MagicMock()
    with patch.object(roundtrip, "read_root", return_value=(None, "events")), \
            patch.object(roundtrip, "build_tables", return_value={}), \
            patch.object(roundtrip, "open_server", return_value=conn), \
            patch.object(roundtrip, "create_schema"), \
            patch.object(roundtrip, "load_tables"), \
            patch.object(roundtrip, "fetch_tables",
                         return_value=(pd.DataFrame(), pd.DataFrame())):
        result = run(root_file="unused.root", where="1=0")
    assert result["n_events"] == 0
    assert result["passed"] is False
    assert np.isnan(result["max_abs_diff"])
    assert np.isnan(result["mean_abs_diff"])
    assert "validation did not run" in capsys.readouterr().out
    conn.close.assert_called_once()
