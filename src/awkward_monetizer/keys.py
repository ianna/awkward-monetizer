"""Globally-unique, collision-free ``event_id`` for parallel/distributed ingest.

``event_id`` is a signed 64-bit BIGINT that packs two numbers::

    event_id = (file_id << ENTRY_BITS) | entry

* ``file_id`` -- a stable integer per input file, assigned up front (see
  :func:`build_manifest`). 23 bits -> up to 8,388,607 files.
* ``entry``   -- the event's entry number *within that file* (the TTree/RNTuple
  entry, not the chunk-local row). 40 bits -> ~1.1e12 events per file.

Because the key depends only on (file, entry), it is identical no matter how a
file is chunked, which worker ingests it, or in what order files are loaded, so
independent ingest jobs can never collide. ``file_id = 0`` reproduces the
original single-file numbering (``event_id == entry``), so existing databases
and tests are unaffected.

Each file owns the contiguous range ``[file_id << ENTRY_BITS,
(file_id + 1) << ENTRY_BITS)`` (see :func:`file_id_range`), which is what makes
range-partitioning ``events``/``jets``/``muons`` by file across MonetDB shards
straightforward: all of an event's rows land on the same shard.

In SQL the parts can be recovered with integer arithmetic (MonetDB needs the
derived columns in a subquery before you can GROUP BY them)::

    SELECT file_id, count(*) FROM (
        SELECT event_id / 1099511627776 AS file_id,   -- 2**40
               event_id % 1099511627776 AS entry
        FROM events) AS t
    GROUP BY file_id
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

ENTRY_BITS = 40
FILE_BITS = 63 - ENTRY_BITS           # keep the sign bit clear (BIGINT is signed)
MAX_ENTRY = (1 << ENTRY_BITS) - 1
MAX_FILE_ID = (1 << FILE_BITS) - 1
ENTRY_MASK = MAX_ENTRY


def _check_file_id(file_id: int) -> int:
    file_id = int(file_id)
    if not 0 <= file_id <= MAX_FILE_ID:
        raise ValueError(f"file_id must be in [0, {MAX_FILE_ID}], got {file_id}")
    return file_id


def make_event_ids(file_id: int, entries) -> np.ndarray:
    """Pack ``file_id`` and an array of per-file entry numbers into event_ids."""
    file_id = _check_file_id(file_id)
    entries = np.asarray(entries, dtype=np.int64)
    if entries.size and (entries.min() < 0 or entries.max() > MAX_ENTRY):
        raise ValueError(f"entry numbers must be in [0, {MAX_ENTRY}]")
    return (np.int64(file_id) << np.int64(ENTRY_BITS)) | entries


def make_event_id(file_id: int, entry: int) -> int:
    """Scalar version of :func:`make_event_ids`."""
    return int(make_event_ids(file_id, [entry])[0])


def split_event_ids(event_ids):
    """Inverse of :func:`make_event_ids`: return ``(file_ids, entries)``."""
    eid = np.asarray(event_ids, dtype=np.int64)
    return eid >> np.int64(ENTRY_BITS), eid & np.int64(ENTRY_MASK)


def file_id_range(file_id: int) -> tuple[int, int]:
    """Half-open ``[lo, hi)`` event_id range owned by one file (for sharding)."""
    file_id = _check_file_id(file_id)
    return file_id << ENTRY_BITS, (file_id + 1) << ENTRY_BITS


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

    def __init__(self, files: list[FileEntry] | None = None):
        self.files: list[FileEntry] = list(files or [])
        self._check()

    def _check(self) -> None:
        ids = [f.file_id for f in self.files]
        keys = [_key(f.path) for f in self.files]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate file_id in manifest")
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate path in manifest")
        for i in ids:
            _check_file_id(i)

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
            json.dump({"entry_bits": ENTRY_BITS,
                       "files": [f.__dict__ for f in self.files]}, fh, indent=2)
            fh.write("\n")

    @classmethod
    def from_json(cls, path: str) -> Manifest:
        with open(path) as fh:
            data = json.load(fh)
        if data.get("entry_bits", ENTRY_BITS) != ENTRY_BITS:
            raise ValueError(
                f"manifest uses entry_bits={data['entry_bits']}, "
                f"this version uses {ENTRY_BITS}")
        return cls([FileEntry(**f) for f in data["files"]])


def build_manifest(paths, *, count_entries: bool = False,
                   tree: str | None = None) -> Manifest:
    """Assign file_ids 0..N-1 to ``paths`` in the order given."""
    return Manifest().add(paths, count_entries=count_entries, tree=tree)


def _key(path: str) -> str:
    # Remote URLs (root://, https://) are kept verbatim; local paths normalised.
    return path if "://" in path else os.path.abspath(path)


def _count_entries(path: str, tree: str | None) -> int:
    import uproot

    from .ingest import open_tree
    with uproot.open(path) as f:
        return int(open_tree(f, tree).num_entries)
