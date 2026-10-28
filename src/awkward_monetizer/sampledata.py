"""Synthetic NanoAOD-style sample data for tests and examples.

Random toy data — not physics — with the right structure: jagged ``Jet_*`` and
``Muon_*`` collections, event-level MET, and HLT trigger bits.
"""

from __future__ import annotations

import awkward as ak
import numpy as np

FORMATS = ("rntuple", "ttree")


def _jagged(counts: np.ndarray, values: np.ndarray) -> ak.Array:
    return ak.unflatten(values, counts)


def make_nanoaod(n_events: int = 2000, seed: int = 0, *,
                 triggers: bool = True) -> dict:
    """Return a dict of NanoAOD-named branches (``Jet_pt``, ``HLT_IsoMu24``, ...).

    The trigger bits are drawn *after* everything else, so for a given seed the
    kinematics are identical with and without ``triggers``. They are loosely
    tied to the muons (IsoMu24 needs a muon above 24 GeV, ~90% efficient), and
    ``HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ`` is deliberately left out, so the
    sample also exercises a trigger path that is missing from the file.
    """
    rng = np.random.default_rng(seed)

    n_jet = rng.integers(0, 8, n_events)      # 0..7 jets
    n_mu = rng.integers(0, 5, n_events)       # 0..4 muons
    tj, tm = int(n_jet.sum()), int(n_mu.sum())

    out = {
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
    if triggers:
        lead = ak.to_numpy(ak.fill_none(ak.max(out["Muon_pt"], axis=1), 0.0))
        fired = rng.random(n_events) < 0.9
        out["HLT_IsoMu24"] = (lead > 24.0) & fired
        out["HLT_IsoMu27"] = (lead > 27.0) & fired
    return out


def _ttree_layout(branches: dict) -> dict:
    """Group ``Prefix_field`` jagged branches into records, so uproot writes the
    real NanoAOD TTree layout: an ``nPrefix`` counter plus ``Prefix_field``."""
    flat, groups = {}, {}
    for name, v in branches.items():
        if isinstance(v, ak.Array) and v.ndim == 2 and "_" in name:
            prefix, fld = name.split("_", 1)
            groups.setdefault(prefix, {})[fld] = v
        else:
            flat[name] = v
    return {**flat, **{p: ak.zip(fs) for p, fs in groups.items()}}


def write_nanoaod(path: str, n_events: int = 2000, seed: int = 0, *,
                  fmt: str = "rntuple", triggers: bool = True) -> str:
    """Write a synthetic NanoAOD ROOT file (``Events``) and return its path.

    ``fmt="rntuple"`` (uproot's default since 5.7) or ``"ttree"`` (the classic
    NanoAOD layout with ``nJet``/``nMuon`` counters).
    """
    import uproot
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}; one of {FORMATS}")
    branches = make_nanoaod(n_events, seed, triggers=triggers)
    with uproot.recreate(path) as f:
        if fmt == "rntuple":
            f["Events"] = branches
        else:
            data = _ttree_layout(branches)
            f.mktree("Events", {k: (v.type if isinstance(v, ak.Array) else v.dtype)
                                for k, v in data.items()})
            f["Events"].extend(data)
    return path
