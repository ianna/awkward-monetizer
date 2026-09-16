import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from awkward_monetizer import db
from awkward_monetizer.cli import build_parser, main


class _Cursor(sqlite3.Cursor):
    def executemany(self, sql, rows):
        return super().executemany(sql.replace("%s", "?"), rows)


class _Connection(sqlite3.Connection):
    def cursor(self):
        return super().cursor(factory=_Cursor)


@pytest.mark.parametrize("step_size", ["500", "100 MB"])
def test_chunked_dry_run_never_connects(nano_file, step_size, capsys):
    with patch("awkward_monetizer.db.open_server") as connect:
        main(["ingest", nano_file, "--dataset", "nanoaod", "--step-size", step_size,
              "--dry-run", "--create-schema", "--truncate"])
    connect.assert_not_called()
    assert "dry run -- 1500 events" in capsys.readouterr().out


@pytest.mark.parametrize("entry_stop", [0, 1100])
def test_chunked_truncate_replaces_all_tables_once(nano_file, entry_stop):
    # SQLite exercises the DB-API INSERT path without requiring a MonetDB server.
    conn = sqlite3.connect(":memory:", factory=_Connection)
    try:
        db.create_schema(conn, db.read_schema("nanoaod"))
        conn.execute("INSERT INTO events (event_id) VALUES (-1)")
        conn.execute("INSERT INTO jets (event_id) VALUES (-1)")
        conn.execute("INSERT INTO muons (event_id) VALUES (-1)")
        conn.commit()
        proxy = MagicMock(wraps=conn)
        proxy.close.side_effect = lambda: None
        with patch("awkward_monetizer.db.open_server", return_value=proxy):
            main(["ingest", nano_file, "--dataset", "nanoaod", "--step-size", "500",
                  "--entry-stop", str(entry_stop), "--truncate", "--no-copy-into"])
        assert conn.execute("SELECT count(*) FROM events").fetchone()[0] == entry_stop
        for table in ("events", "jets", "muons"):
            assert conn.execute(
                f"SELECT count(*) FROM {table} WHERE event_id = -1"
            ).fetchone()[0] == 0
        if entry_stop:
            assert conn.execute("SELECT max(event_id) FROM events").fetchone()[0] == 1099
            for table in ("jets", "muons"):
                assert conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] > 0
    finally:
        conn.close()


@pytest.mark.parametrize("value", ["0", "-1"])
def test_nonpositive_chunk_sizes_are_rejected(value):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["ingest", "--step-size", value])
