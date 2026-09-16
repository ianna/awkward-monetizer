"""Synthetic NanoAOD-style sample data for tests and examples.

Random toy data — not physics — with the right structure: jagged ``Jet_*`` and
``Muon_*`` collections plus event-level MET.
"""

from __future__ import annotations

import awkward as ak
import numpy as np


def _jagged(counts: np.ndarray, values: np.ndarray) -> ak.Array:
    return ak.unflatten(values, counts)


def make_nanoaod(n_events: int = 2000, seed: int = 0) -> dict:
    """Return a dict of branches suitable for ``uproot.recreate``'s TTree write."""
    rng = np.random.default_rng(seed)

    n_jet = rng.integers(0, 8, n_events)      # 0..7 jets
    n_mu = rng.integers(0, 5, n_events)       # 0..4 muons
    tj, tm = int(n_jet.sum()), int(n_mu.sum())

    return {
        "run": np.full(n_events, 1, dtype=np.int32),
        "luminosityBlock": rng.integers(1, 2000, n_events).astype(np.int32),
        "event": np.arange(n_events, dtype=np.int64),
        "MET_pt": rng.exponential(30.0, n_events).astype(np.float64),
        "MET_phi": rng.uniform(-np.pi, np.pi, n_events).astype(np.float64),

        "Jet_pt": _jagged(n_jet, rng.exponential(40.0, tj) + 15.0),
        "Jet_eta": _jagged(n_jet, rng.uniform(-4.0, 4.0, tj)),
        "Jet_phi": _jagged(n_jet, rng.uniform(-np.pi, np.pi, tj)),
        "Jet_mass": _jagged(n_jet, rng.uniform(2.0, 25.0, tj)),

        "Muon_pt": _jagged(n_mu, rng.exponential(20.0, tm) + 5.0),
        "Muon_eta": _jagged(n_mu, rng.uniform(-2.5, 2.5, tm)),
        "Muon_phi": _jagged(n_mu, rng.uniform(-np.pi, np.pi, tm)),
        "Muon_mass": _jagged(n_mu, np.full(tm, 0.10566)),
        "Muon_charge": _jagged(n_mu, rng.choice([-1, 1], tm).astype(np.int32)),
    }


def write_nanoaod(path: str, n_events: int = 2000, seed: int = 0) -> str:
    """Write a synthetic NanoAOD ROOT file (tree ``Events``) and return its path."""
    import uproot
    with uproot.recreate(path) as f:
        f["Events"] = make_nanoaod(n_events, seed)
    return path
