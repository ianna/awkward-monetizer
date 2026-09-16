import importlib.util

import pytest

from awkward_monetizer import db
from awkward_monetizer.datasets import DATASETS
from awkward_monetizer.ingest import ingest_root_chunked
from awkward_monetizer.sampledata import write_nanoaod

_has_monetdbe = importlib.util.find_spec("monetdbe") is not None


@pytest.mark.skipif(not _has_monetdbe, reason="monetdbe not installed")
def test_chunked_nanoaod_ingest_unique_event_ids(tmp_path):
    """Streaming a NanoAOD file in chunks yields globally-unique event_ids and
    no orphan object rows (the real-file ingestion path)."""
    f = write_nanoaod(str(tmp_path / "nano.root"), n_events=2500, seed=2)

    conn = db.open_embedded(":memory:")
    db.create_schema(conn, db.read_schema("nanoaod"))
    total = ingest_root_chunked(f, DATASETS["nanoaod"], conn,
                                step_size=1000, use_copy_into=False)
    cur = conn.cursor()
    cur.execute("SELECT count(*), count(DISTINCT event_id) FROM events")
    n_events, distinct = cur.fetchall()[0]
    cur.execute("SELECT count(*) FROM jets "
                "WHERE event_id NOT IN (SELECT event_id FROM events)")
    orphans = cur.fetchall()[0][0]
    conn.close()

    assert total == 2500
    assert n_events == 2500 and distinct == 2500   # unique across chunks
    assert orphans == 0                            # jets reference real events
