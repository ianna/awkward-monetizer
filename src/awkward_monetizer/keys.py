"""Globally-unique, collision-free ``event_id`` for parallel/distributed ingest.

``event_id`` is a signed 64-bit BIGINT that packs two numbers::

    event_id = (file_id << entry_bits) | entry

* ``file_id`` -- a stable integer per input file, assigned up front (see
  :func:`build_manifest`).
* ``entry``   -- the event's entry number *within that file* (the TTree/RNTuple
  entry, not the chunk-local row).

The split is a :class:`KeyLayout`. The default (40 entry bits, 23 file bits)
allows 8.4 million files; :data:`EXABYTE_LAYOUT` (31 entry bits, 32 file bits)
allows 4.3 billion files of up to 2.1e9 entries. The layout travels with the
manifest and shard map, and :func:`migrate_tables` converts stored keys.

Because the key depends only on (file, entry), it is identical no matter how a
file is chunked, which worker ingests it, or in what order files are loaded, so
independent ingest jobs can never collide. ``file_id = 0`` reproduces the
original single-file numbering (``event_id == entry``), so existing databases
and tests are unaffected.

Each file owns the contiguous range ``[file_id << entry_bits,
(file_id + 1) << entry_bits)`` (see :func:`file_id_range`), which is what makes
range-partitioning ``events``/``jets``/``muons`` by file across MonetDB shards
straightforward: all of an event's rows land on the same shard.

In SQL the parts can be recovered with integer arithmetic (MonetDB needs the
derived columns in a subquery before you can GROUP BY them)::

    SELECT file_id, count(*) FROM (
        SELECT event_id / 1099511627776 AS file_id,   -- 2**entry_bits (here 40)
               event_id % 1099511627776 AS entry
        FROM events) AS t
    GROUP BY file_id
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KeyLayout:
    """How a signed 64-bit ``event_id`` is split into file id and entry number.

    ``entry_bits`` low bits hold the entry number; the remaining
    ``63 - entry_bits`` bits hold the file id (the sign bit stays clear).

    * ``KeyLayout(40)`` -- the original layout: 8.4 million files of up to
      1.1e12 entries each (:data:`DEFAULT_LAYOUT`).
    * ``KeyLayout(31)`` -- 4.3 billion files of up to 2.1e9 entries each
      (:data:`EXABYTE_LAYOUT`); at ~1.5 GiB per file that is several exabytes.

    The layout is recorded in manifests and shard maps; inputs with different
    layouts are never mixed implicitly (see :meth:`convert_event_ids` and
    :func:`migrate_tables`).
    """
    entry_bits: int = 40

    def __post_init__(self):
        if not 1 <= int(self.entry_bits) <= 62:
            raise ValueError(f"entry_bits must be in [1, 62], got {self.entry_bits}")

    @property
    def file_bits(self) -> int:
        return 63 - self.entry_bits

    @property
    def max_entry(self) -> int:
        return (1 << self.entry_bits) - 1

    @property
    def max_file_id(self) -> int:
        return (1 << self.file_bits) - 1

    def check_file_id(self, file_id: int) -> int:
        file_id = int(file_id)
        if not 0 <= file_id <= self.max_file_id:
            raise ValueError(
                f"file_id must be in [0, {self.max_file_id}], got {file_id}")
        return file_id

    def make_event_ids(self, file_id: int, entries) -> np.ndarray:
        """Pack ``file_id`` and an array of per-file entry numbers."""
        file_id = self.check_file_id(file_id)
        entries = np.asarray(entries, dtype=np.int64)
        if entries.size and (entries.min() < 0 or entries.max() > self.max_entry):
            raise ValueError(f"entry numbers must be in [0, {self.max_entry}]")
        return (np.int64(file_id) << np.int64(self.entry_bits)) | entries

    def split_event_ids(self, event_ids):
        """Inverse of :meth:`make_event_ids`: ``(file_ids, entries)``."""
        eid = np.asarray(event_ids, dtype=np.int64)
        return eid >> np.int64(self.entry_bits), eid & np.int64(self.max_entry)

    def file_id_range(self, file_id: int) -> tuple[int, int]:
        """Half-open ``[lo, hi)`` event_id range owned by one file."""
        file_id = self.check_file_id(file_id)
        return file_id << self.entry_bits, (file_id + 1) << self.entry_bits

    def convert_event_ids(self, event_ids, to: KeyLayout) -> np.ndarray:
        """Re-encode keys of this layout in layout ``to``.

        Deterministic, but only possible when every file id and entry number
        fits the target fields; otherwise raises ValueError and converts
        nothing.
        """
        fids, entries = self.split_event_ids(event_ids)
        if fids.size:
            if fids.min() < 0 or fids.max() > to.max_file_id:
                raise ValueError(
                    f"file ids up to {int(fids.max())} do not fit {to.file_bits} bits")
            if entries.max() > to.max_entry:
                raise ValueError(
                    f"entry numbers up to {int(entries.max())} do not fit "
                    f"{to.entry_bits} bits")
        return (fids << np.int64(to.entry_bits)) | entries


DEFAULT_LAYOUT = KeyLayout(40)
EXABYTE_LAYOUT = KeyLayout(31)

# Backwards-compatible constants and functions (the default layout).
ENTRY_BITS = DEFAULT_LAYOUT.entry_bits
FILE_BITS = DEFAULT_LAYOUT.file_bits
MAX_ENTRY = DEFAULT_LAYOUT.max_entry
MAX_FILE_ID = DEFAULT_LAYOUT.max_file_id
ENTRY_MASK = MAX_ENTRY


def make_event_ids(file_id: int, entries, layout: KeyLayout = DEFAULT_LAYOUT) -> np.ndarray:
    """Pack ``file_id`` and an array of per-file entry numbers into event_ids."""
    return layout.make_event_ids(file_id, entries)


def make_event_id(file_id: int, entry: int, layout: KeyLayout = DEFAULT_LAYOUT) -> int:
    """Scalar version of :func:`make_event_ids`."""
    return int(layout.make_event_ids(file_id, [entry])[0])


def split_event_ids(event_ids, layout: KeyLayout = DEFAULT_LAYOUT):
    """Inverse of :func:`make_event_ids`: return ``(file_ids, entries)``."""
    return layout.split_event_ids(event_ids)


def file_id_range(file_id: int, layout: KeyLayout = DEFAULT_LAYOUT) -> tuple[int, int]:
    """Half-open ``[lo, hi)`` event_id range owned by one file (for sharding)."""
    return layout.file_id_range(file_id)


def migrate_tables(conn, tables, old: KeyLayout, new: KeyLayout, *,
                   check_only: bool = False) -> dict[str, int]:
    """Convert stored ``event_id`` keys in ``tables`` from layout ``old`` to
    ``new``, in place, inside one transaction.

    All tables are checked first: if any file id or entry number would not fit
    the new fields, nothing is changed and ValueError is raised. File ids are
    preserved, so manifests and shard file-id ranges stay valid (shard *key*
    ranges follow from the layout). Returns rows updated per table. On any
    error the transaction is rolled back. With ``check_only=True`` the checks
    run and nothing is modified (use it across all shards before converting any).
    """
    if old == new:
        return {t: 0 for t in tables}
    old_div, new_mul = 1 << old.entry_bits, 1 << new.entry_bits
    cur = conn.cursor()
    try:
        for t in tables:
            cur.execute(f"SELECT count(*), min(event_id), max(event_id) FROM {t}")
            n, lo, hi = cur.fetchall()[0]
            if not n:
                continue
            if int(lo) < 0:
                raise ValueError(f"{t}: negative event_id {lo}; not a packed key")
            if int(hi) // old_div > new.max_file_id:
                raise ValueError(
                    f"{t}: file ids up to {int(hi) // old_div} do not fit "
                    f"{new.file_bits} bits")
            # CAST: MonetDB may widen BIGINT arithmetic to HUGEINT
            cur.execute(f"SELECT CAST(max(event_id % {old_div}) AS BIGINT) FROM {t}")
            max_entry = int(cur.fetchall()[0][0])
            if max_entry > new.max_entry:
                raise ValueError(
                    f"{t}: entry numbers up to {max_entry} do not fit "
                    f"{new.entry_bits} bits")
        counts = {}
        for t in tables:
            cur.execute(f"SELECT count(*) FROM {t}")
            counts[t] = int(cur.fetchall()[0][0])
        if check_only:
            return counts
        for t in tables:
            # Two steps through negative values: a new key can equal another
            # row's old key, and some engines check uniqueness row by row.
            cur.execute(
                f"UPDATE {t} SET event_id = CAST(-((event_id / {old_div}) * {new_mul} "
                f"+ (event_id % {old_div})) - 1 AS BIGINT)")
            cur.execute(f"UPDATE {t} SET event_id = -event_id - 1")
        conn.commit()
        return counts
    except Exception:
        conn.rollback()
        raise


# --------------------------------------------------------------------------
# File manifest: path -> stable file_id (+ entry count)
# --------------------------------------------------------------------------
@dataclass
class FileEntry:
    file_id: int
    path: str
    n_entries: int | None = None


class Manifest:
    """Stable mapping of input files to ``file_id``.

    Build it once for a dataset, save it next to the data, and give it to every
    ingest worker; appending new files never renumbers existing ones.
    """

    def __init__(self, files: list[FileEntry] | None = None,
                 layout: KeyLayout = DEFAULT_LAYOUT):
        self.files: list[FileEntry] = list(files or [])
        self.layout = layout
        self._check()

    def _check(self) -> None:
        ids = [f.file_id for f in self.files]
        keys = [_key(f.path) for f in self.files]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate file_id in manifest")
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate path in manifest")
        for f in self.files:
            self.layout.check_file_id(f.file_id)
            if f.n_entries is not None and f.n_entries - 1 > self.layout.max_entry:
                raise ValueError(
                    f"{f.path}: {f.n_entries} entries do not fit "
                    f"{self.layout.entry_bits} entry bits")

    def file_id(self, path: str) -> int:
        k = _key(path)
        for f in self.files:
            if _key(f.path) == k:
                return f.file_id
        raise KeyError(f"{path!r} is not in the manifest")

    def add(self, paths, *, count_entries: bool = False,
            tree: str | None = None) -> Manifest:
        """Append new paths with the next free file_ids (existing ones kept)."""
        known = {_key(f.path) for f in self.files}
        next_id = max((f.file_id for f in self.files), default=-1) + 1
        for p in paths:
            if _key(p) in known:
                continue
            n = _count_entries(p, tree) if count_entries else None
            self.files.append(FileEntry(next_id, p, n))
            known.add(_key(p))
            next_id += 1
        self._check()
        return self

    def to_json(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump({"entry_bits": self.layout.entry_bits,
                       "files": [f.__dict__ for f in self.files]}, fh, indent=2)
            fh.write("\n")

    @classmethod
    def from_json(cls, path: str) -> Manifest:
        with open(path) as fh:
            data = json.load(fh)
        layout = KeyLayout(int(data.get("entry_bits", ENTRY_BITS)))
        return cls([FileEntry(**f) for f in data["files"]], layout=layout)

    def with_layout(self, layout: KeyLayout) -> Manifest:
        """Same files and file ids under another key layout. Raises if a file
        id or a recorded entry count does not fit the new layout."""
        return Manifest([FileEntry(f.file_id, f.path, f.n_entries)
                         for f in self.files], layout=layout)


def build_manifest(paths, *, count_entries: bool = False,
                   tree: str | None = None,
                   layout: KeyLayout = DEFAULT_LAYOUT) -> Manifest:
    """Assign file_ids 0..N-1 to ``paths`` in the order given."""
    return Manifest(layout=layout).add(paths, count_entries=count_entries, tree=tree)


def _key(path: str) -> str:
    # Remote URLs (root://, https://) are kept verbatim; local paths normalised.
    return path if "://" in path else os.path.abspath(path)


def _count_entries(path: str, tree: str | None) -> int:
    import uproot

    from .ingest import open_tree
    with uproot.open(path) as f:
        return int(open_tree(f, tree).num_entries)
