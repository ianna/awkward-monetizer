"""awkward-monetizer: a hybrid HEP analysis engine (Awkward + MonetDB + Arrow)."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .datasets import DATASETS, Dataset
from .ingest import build_tables, load_tables, read_root
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
    "__version__",
    "build_tables",
    "fetch_tables",
    "invariant_mass",
    "load_tables",
    "opposite_charge_pair",
    "read_root",
    "reconstruct_events",
    "reconstruct_multi",
    "tables_from_root",
]
