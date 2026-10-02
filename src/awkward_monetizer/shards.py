"""Scatter-gather over several MonetDB shards, partitioned by ``file_id``.

Each shard owns a contiguous, half-open range of file ids ``[first, stop)`` and
therefore -- because ``event_id = (file_id << 40) | entry`` (see :mod:`.keys`)
-- a contiguous ``event_id`` range. Every event and all of its objects live on
exactly one shard, so:

* the ``events JOIN objects`` in :func:`~.reconstruct.fetch_tables` is local
  to each shard,
* :func:`~.reconstruct.reconstruct_multi` runs unchanged on each shard's slice,
* per-event results simply concatenate, and aggregates merge by addition.

Typical use::

    smap = ShardMap.from_json("shards.json")
    ingest_manifest(Manifest.from_json("manifest.json"), smap, DATASETS["nanoaod"])
    events = fetch_events(smap, where="met_pt > 50")            # one ak.Array
    met = map_events(smap, adl.q4_met_ge2jets40)                # merged result

``shards.json``::

    {"shards": [
      {"name": "s0", "file_ids": [0, 50],   "host": "node0", "database": "hep"},
      {"name": "s1", "file_ids": [50, 100], "host": "node1", "database": "hep"}
    ]}

A shard with ``"backend": "embedded"`` and a ``"target"`` directory uses
in-process monetdbe instead of a server. monetdbe allows only one open database
per process, so embedded shards are always queried from a process pool.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

import awkward as ak
import numpy as np

from .keys import DEFAULT_LAYOUT, KeyLayout, Manifest


# --------------------------------------------------------------------------
# Shard map
# --------------------------------------------------------------------------
@dataclass
class Shard:
    name: str
    file_ids: tuple[int, int]          # half-open [first, stop)
    backend: str = "server"            # "server" (pymonetdb) | "embedded" (monetdbe)
    host: str = "localhost"
    port: int = 50000
    database: str = "hep"
    user: str = "monetdb"
    password: str = "monetdb"
    target: str | None = None          # embedded: database directory
    # testing / custom drivers: a zero-arg callable returning a DB-API connection
    connector: Callable[[], Any] | None = field(default=None, repr=False,
                                               compare=False)

    def __post_init__(self):
        first, stop = (int(x) for x in self.file_ids)
        if not 0 <= first < stop:
            raise ValueError(f"shard {self.name!r}: bad file_ids {self.file_ids}")
        self.file_ids = (first, stop)
        if self.backend not in ("server", "embedded"):
            raise ValueError(f"shard {self.name!r}: unknown backend {self.backend!r}")
        if self.backend == "embedded" and not self.target and not self.connector:
            raise ValueError(f"embedded shard {self.name!r} needs a 'target' dir")

    def owns(self, file_id: int) -> bool:
        return self.file_ids[0] <= file_id < self.file_ids[1]

    def key_range(self, layout: KeyLayout = DEFAULT_LAYOUT) -> tuple[int, int]:
        """Half-open ``[lo, hi)`` event_id range owned by this shard."""
        return (layout.file_id_range(self.file_ids[0])[0],
                layout.file_id_range(self.file_ids[1] - 1)[1])

    @property
    def event_id_range(self) -> tuple[int, int]:
        """:meth:`key_range` under the default key layout."""
        return self.key_range()

    def connect(self):
        if self.connector is not None:
            return self.connector()
        from . import db
        if self.backend == "embedded":
            return db.open_embedded(self.target)
        return db.open_server(database=self.database, host=self.host,
                              port=self.port, user=self.user,
                              password=self.password)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("connector")
        d["file_ids"] = list(self.file_ids)
        return d


class ShardMap:
    """Ordered, non-overlapping shards covering (part of) the file_id space."""

    def __init__(self, shards: Sequence[Shard],
                 layout: KeyLayout = DEFAULT_LAYOUT):
        self.layout = layout
        self.shards = sorted(shards, key=lambda s: s.file_ids[0])
        for s in self.shards:
            if s.file_ids[1] > layout.max_file_id + 1:
                raise ValueError(
                    f"shard {s.name!r}: file_ids {s.file_ids} exceed the "
                    f"{layout.file_bits}-bit file-id range of the key layout")
        names = [s.name for s in self.shards]
        if len(set(names)) != len(names):
            raise ValueError("duplicate shard names")
        for a, b in zip(self.shards, self.shards[1:], strict=False):
            if b.file_ids[0] < a.file_ids[1]:
                raise ValueError(f"shards {a.name!r} and {b.name!r} overlap")
        if not self.shards:
            raise ValueError("a ShardMap needs at least one shard")

    def __len__(self) -> int:
        return len(self.shards)

    def __iter__(self):
        return iter(self.shards)

    def shard_for(self, file_id: int) -> Shard:
        for s in self.shards:
            if s.owns(file_id):
                return s
        raise KeyError(f"no shard owns file_id {file_id}")

    @property
    def embedded(self) -> bool:
        return any(s.backend == "embedded" and s.connector is None
                   for s in self.shards)

    @classmethod
    def from_json(cls, path: str) -> ShardMap:
        with open(path) as fh:
            data = json.load(fh)
        layout = KeyLayout(int(data.get("entry_bits", DEFAULT_LAYOUT.entry_bits)))
        return cls([Shard(**s) for s in data["shards"]], layout=layout)

    def to_json(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump({"entry_bits": self.layout.entry_bits,
                       "shards": [s.to_dict() for s in self.shards]}, fh, indent=2)
            fh.write("\n")

    @classmethod
    def split(cls, n_files: int, shards: Sequence[dict],
              layout: KeyLayout = DEFAULT_LAYOUT) -> ShardMap:
        """Spread file ids ``0..n_files-1`` evenly over the given shard specs
        (dicts of :class:`Shard` fields without ``file_ids``). The last shard
        is left open-ended so files appended to the manifest still have a home.
        """
        n = len(shards)
        if n == 0 or n_files < n:
            raise ValueError("need at least one file per shard")
        bounds = ([round(i * n_files / n) for i in range(n)]
                  + [layout.max_file_id + 1])
        return cls([Shard(**spec, file_ids=(bounds[i], bounds[i + 1]))
                    for i, spec in enumerate(shards)], layout=layout)


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------
class ShardError(RuntimeError):
    """One or more shards failed; ``errors`` maps shard name -> exception."""

    def __init__(self, errors: dict[str, BaseException], n_shards: int):
        self.errors = errors
        lines = [f"{len(errors)} of {n_shards} shard(s) failed:"]
        lines += [f"  {name}: {type(e).__name__}: {e}" for name, e in errors.items()]
        super().__init__("\n".join(lines))


def scatter(smap: ShardMap, task: Callable[[Shard, Any], Any], *,
            executor: str = "auto", max_workers: int | None = None) -> list:
    """Run ``task(shard, conn)`` on every shard in parallel; results in shard
    order. Each call gets its own connection, closed afterwards.

    ``executor``: "thread", "process", or "auto" (process if any shard is
    embedded monetdbe, which is one-database-per-process; else thread). With
    "process", ``task`` must be picklable (a module-level function).
    """
    if executor == "auto":
        executor = "process" if smap.embedded else "thread"
    if executor == "thread" and smap.embedded and len(smap) > 1:
        raise ValueError("embedded (monetdbe) shards need executor='process'")
    workers = max_workers or len(smap)
    pool_cls = {"thread": ThreadPoolExecutor,
                "process": ProcessPoolExecutor}[executor]
    results, errors = [], {}
    if workers == 1 or len(smap) == 1:
        for s in smap:
            try:
                results.append(_run_on_shard(s, task))
            except Exception as e:  # noqa: BLE001 - reported per shard below
                errors[s.name] = e
    else:
        with pool_cls(max_workers=workers) as pool:
            futures = [(s, pool.submit(_run_on_shard, s, task)) for s in smap]
            for s, f in futures:
                try:
                    results.append(f.result())
                except Exception as e:  # noqa: BLE001
                    errors[s.name] = e
    if errors:
        raise ShardError(errors, len(smap)) from next(iter(errors.values()))
    return results


def _run_on_shard(shard: Shard, task):
    conn = shard.connect()
    try:
        return task(shard, conn)
    finally:
        conn.close()


def combine(partials: Sequence[Any]):
    """Merge per-shard partial results (shard order is preserved).

    * ``numpy.ndarray`` -> concatenated;  ``ak.Array`` -> ``ak.concatenate``
    * ``list``/``tuple`` -> concatenated
    * ``dict`` -> merged key by key (recursively)
    * anything else supporting ``+`` (numbers, ``hist.Hist``, ``Counter``) -> sum
    * ``None`` partials (e.g. an empty shard) are skipped
    """
    parts = [p for p in partials if p is not None]
    if not parts:
        return None
    first = parts[0]
    if isinstance(first, ak.Array):
        return ak.concatenate(parts)
    if isinstance(first, np.ndarray):
        return np.concatenate(parts)
    if isinstance(first, dict):
        keys = list(dict.fromkeys(k for p in parts for k in p))
        return {k: combine([p.get(k) for p in parts]) for k in keys}
    if isinstance(first, (list, tuple)):
        out = [x for p in parts for x in p]
        return type(first)(out) if isinstance(first, tuple) else out
    total = first
    for p in parts[1:]:
        total = total + p
    return total


# --------------------------------------------------------------------------
# Query: SQL slice per shard -> Awkward NF2 per shard -> gather
# --------------------------------------------------------------------------
@dataclass
class _FetchTask:
    """Picklable per-shard fetch + reconstruct (+ optional map function)."""
    where: str | None
    collections: tuple[str, ...]
    event_columns: tuple[str, ...] | None
    fn: Callable[[ak.Array], Any] | None = None

    def __call__(self, shard: Shard, conn):
        events = fetch_shard_events(conn, self.where, self.collections,
                                    event_columns=self.event_columns)
        if self.fn is None:
            return events
        # an empty selection on this shard contributes nothing (and has no
        # usable types for fn to work on)
        return None if len(events) == 0 else self.fn(events)


def fetch_shard_events(conn, where: str | None = None,
                       collections: Sequence[str] = ("jets", "muons"), *,
                       event_columns: Sequence[str] | None = None) -> ak.Array:
    """Fetch one shard's selected events plus their collections and rebuild NF2."""
    from .reconstruct import fetch_tables, reconstruct_multi
    ev_cols = tuple(event_columns) if event_columns is not None else None
    colls = tuple(collections)
    if not colls:
        import pandas as pd
        proj = "*" if ev_cols is None else ", ".join(ev_cols)
        sel = f" WHERE {where}" if where else ""
        cur = conn.cursor()
        try:
            cur.execute(f"SELECT {proj} FROM events{sel}")
            events_df = pd.DataFrame(cur.fetchall(),
                                     columns=[d[0] for d in cur.description])
        finally:
            cur.close()
        return reconstruct_multi(events_df, {})
    events_df, first = fetch_tables(conn, where, colls[0], event_columns=ev_cols)
    objects = {colls[0]: first}
    for coll in colls[1:]:
        # the events query is repeated with a minimal projection only
        _, objects[coll] = fetch_tables(conn, where, coll,
                                        event_columns=("event_id",))
    return reconstruct_multi(events_df, objects)


def fetch_events(smap: ShardMap, where: str | None = None,
                 collections: Sequence[str] = ("jets", "muons"), *,
                 event_columns: Sequence[str] | None = None,
                 executor: str = "auto",
                 max_workers: int | None = None) -> ak.Array:
    """Scatter a SQL selection to every shard, rebuild Awkward NF2 on each, and
    gather into one array ordered by ``event_id`` (shards are range-ordered).

    ``where`` is evaluated in the ``events`` scope on every shard, exactly as in
    :func:`~.reconstruct.fetch_tables`.
    """
    task = _FetchTask(where, tuple(collections),
                      tuple(event_columns) if event_columns else None)
    parts = scatter(smap, task, executor=executor, max_workers=max_workers)
    nonempty = [p for p in parts if len(p)]
    return ak.concatenate(nonempty) if nonempty else parts[0]


def map_events(smap: ShardMap, fn: Callable[[ak.Array], Any],
               where: str | None = None,
               collections: Sequence[str] = ("jets", "muons"), *,
               event_columns: Sequence[str] | None = None,
               reduce: Callable[[list], Any] | None = combine,
               executor: str = "auto", max_workers: int | None = None):
    """Run ``fn(events)`` on each shard's reconstructed events *next to the
    shard*, and merge only the (small) results with ``reduce`` (default
    :func:`combine`; pass ``reduce=None`` to get the per-shard list).

    ``fn`` must only look at events within one shard (per-event selections,
    combinatorics, histogram filling, counts, sums). Cross-event operations
    such as a global top-k or normalisation belong in ``reduce``.
    """
    task = _FetchTask(where, tuple(collections),
                      tuple(event_columns) if event_columns else None, fn)
    partials = scatter(smap, task, executor=executor, max_workers=max_workers)
    return partials if reduce is None else reduce(partials)


# --------------------------------------------------------------------------
# Ingest routing and consistency checks
# --------------------------------------------------------------------------
def ingest_manifest(manifest: Manifest, smap: ShardMap, ds, *,
                    step_size="100 MB", tree: str | None = None,
                    use_copy_into: bool | None = None, create_schema: bool = False,
                    max_workers: int | None = None, executor: str = "auto",
                    method: str = "auto") -> dict[str, int]:
    """Load every manifest file into the shard that owns its ``file_id``.

    Files are grouped per shard and loaded shard-parallel (files for the same
    shard go sequentially over one connection). Each file is loaded with
    ``replace=True``, so re-running is idempotent. Returns events per shard.

    ``method="auto"`` picks per shard: ``copy`` (temp CSV + server-side COPY
    INTO) for a server on this host, ``binary`` (columns streamed over the
    connection with ``COPY BINARY ... ON CLIENT``) for remote servers, and
    ``insert`` for embedded monetdbe (which cannot COPY INTO). Any method from
    :data:`.upload.METHODS` can be forced for all shards. ``use_copy_into=False``
    is kept for compatibility and means ``method="insert"``.
    """
    if manifest.layout != smap.layout:
        raise ValueError(
            f"key layouts differ: manifest uses {manifest.layout.entry_bits} "
            f"entry bits, shard map uses {smap.layout.entry_bits}; convert one "
            "explicitly (Manifest.with_layout, keys.migrate_tables)")
    check_files(manifest, ds, tree=tree)
    by_shard: dict[str, list] = {s.name: [] for s in smap}
    for f in manifest.files:
        by_shard[smap.shard_for(f.file_id).name].append((f.path, f.file_id))
    if use_copy_into is False:
        method = "insert"
    task = _IngestTask(by_shard, ds, step_size, tree, method, create_schema,
                       manifest.layout)
    counts = scatter(smap, task, executor=executor, max_workers=max_workers)
    return {s.name: n for s, n in zip(smap, counts, strict=True)}


def check_files(manifest: Manifest, ds, *, tree: str | None = None) -> None:
    """Open every manifest file and check it has the dataset's tree and
    branches, before anything is loaded. Raises ValueError listing all bad
    files (e.g. a dimuon ntuple in a NanoAOD manifest)."""
    import uproot

    from .ingest import open_tree
    wanted = ds.all_branches()
    bad = []
    for f in manifest.files:
        try:
            with uproot.open(f.path) as fh:
                t = open_tree(fh, tree if tree is not None else ds.tree)
                missing = [b for b in wanted if b not in set(t.keys())]
            if missing:
                bad.append(f"{f.path}: missing branches {missing[:5]}"
                           + (" ..." if len(missing) > 5 else ""))
        except Exception as e:  # noqa: BLE001 - collect every problem
            bad.append(f"{f.path}: {type(e).__name__}: {e}")
    if bad:
        raise ValueError(
            f"{len(bad)} file(s) in the manifest don't match dataset "
            f"{ds.name!r}; nothing was loaded:\n  " + "\n  ".join(bad))


@dataclass
class _IngestTask:
    by_shard: dict
    ds: Any
    step_size: Any
    tree: str | None
    method: str
    create_schema: bool
    layout: KeyLayout = DEFAULT_LAYOUT

    def _method(self, shard: Shard) -> str:
        from .upload import LOCAL_HOSTS, METHODS
        if self.method not in METHODS:
            raise ValueError(f"unknown load method {self.method!r}")
        if shard.backend == "embedded" or shard.connector is not None:
            if self.method not in ("auto", "insert"):
                raise ValueError(f"shard {shard.name!r}: only INSERT is possible "
                                 f"on a non-server connection, not {self.method!r}")
            return "insert"
        if self.method == "auto":
            return "copy" if shard.host in LOCAL_HOSTS else "binary"
        return self.method

    def __call__(self, shard: Shard, conn) -> int:
        from . import db
        from .ingest import ingest_root_chunked
        if self.create_schema:
            db.create_schema(conn, db.read_schema(self.ds.name))
        total = 0
        for path, file_id in self.by_shard[shard.name]:
            total += ingest_root_chunked(
                path, self.ds, conn, tree=self.tree, step_size=self.step_size,
                file_id=file_id, replace=True, method=self._method(shard),
                layout=self.layout)
        conn.commit()
        return total


def _verify_task(tables, layout: KeyLayout = DEFAULT_LAYOUT):
    def task(shard: Shard, conn) -> dict:
        lo, hi = shard.key_range(layout)
        cur = conn.cursor()
        report = {"shard": shard.name, "problems": []}
        for t in tables:
            cur.execute(f"SELECT count(*), min(event_id), max(event_id) FROM {t}")
            n, mn, mx = cur.fetchall()[0]
            report[t] = int(n)
            if n and not (lo <= int(mn) and int(mx) < hi):
                report["problems"].append(
                    f"{t}: event_id [{mn}, {mx}] outside shard range [{lo}, {hi})")
            if t != "events":
                cur.execute(f"SELECT count(*) FROM {t} WHERE event_id NOT IN "
                            "(SELECT event_id FROM events)")
                orphans = int(cur.fetchall()[0][0])
                if orphans:
                    report["problems"].append(f"{t}: {orphans} orphan rows")
        cur.close()
        return report
    return task


def verify(smap: ShardMap, tables: Sequence[str] = ("events", "jets", "muons"),
           ) -> list[dict]:
    """Check each shard only holds its own event_id range and has no orphan
    objects (i.e. no event is split across shards). Returns one report per
    shard; ``report["problems"]`` is empty when the shard is consistent."""
    # thread executor: the closure is not picklable; embedded shards are
    # checked one at a time in this process instead.
    if smap.embedded:
        task = _verify_task(tuple(tables), smap.layout)
        return [_run_on_shard(s, task) for s in smap]
    return scatter(smap, _verify_task(tuple(tables), smap.layout), executor="thread")
