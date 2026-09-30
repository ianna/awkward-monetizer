"""awkward-monetizer: a hybrid HEP analysis engine (Awkward + MonetDB + Arrow)."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .datasets import DATASETS, Dataset
from .ingest import build_tables, load_tables, read_root
from .keys import Manifest, build_manifest, file_id_range, make_event_ids, split_event_ids
from .physics import invariant_mass, opposite_charge_pair
from .reconstruct import (
    fetch_tables,
    reconstruct_events,
    reconstruct_multi,
    tables_from_root,
)

try:
    __version__ = version("awkward-monetizer")
except PackageNotFoundError:  # not installed (e.g. running from a source tree)
    __version__ = "0.0.0"

__all__ = [
    "DATASETS",
    "Dataset",
    "Manifest",
    "__version__",
    "build_manifest",
    "build_tables",
    "fetch_tables",
    "file_id_range",
    "invariant_mass",
    "load_tables",
    "make_event_ids",
    "opposite_charge_pair",
    "read_root",
    "reconstruct_events",
    "reconstruct_multi",
    "split_event_ids",
    "tables_from_root",
]
