"""ADL benchmark queries (Q1–Q8) on reconstructed NanoAOD-style events.

The standard IRIS-HEP "Analysis Description Language" benchmark tasks, on Awkward
arrays with scikit-hep ``vector`` for the 4-momentum math. Each query returns a
1-D numpy array of the quantity you would histogram.

Expected event record: ``events.met_pt``, ``events.met_phi``, ``events.jets``
(pt, eta, phi, mass) and ``events.muons`` (pt, eta, phi, mass, charge; used as
the lepton collection).

Scope notes: only muons are available, so Q7/Q8 treat muons as leptons (Q8's
SFOS pair is an opposite-charge muon pair); Q6 omits the "max b-tag" part of the
canonical query because the schema has no b-tag branch.
"""

from __future__ import annotations

import awkward as ak
import numpy as np

from .datasets import DATASETS
from .ingest import build_tables, read_root
from .physics import TOP_MASS, Z_MASS, jets_p4, muons_p4
from .reconstruct import reconstruct_multi


# --------------------------------------------------------------------------
# Q1–Q8
# --------------------------------------------------------------------------
def q1_met(events: ak.Array) -> np.ndarray:
    """MET of all events."""
    return ak.to_numpy(events.met_pt)


def q2_jet_pt(events: ak.Array) -> np.ndarray:
    """pT of all jets."""
    return ak.to_numpy(ak.flatten(events.jets.pt, axis=1))


def q3_jet_pt_central(events: ak.Array) -> np.ndarray:
    """pT of jets with |eta| < 1."""
    j = events.jets
    return ak.to_numpy(ak.flatten(j.pt[abs(j.eta) < 1.0], axis=1))


def q4_met_ge2jets40(events: ak.Array) -> np.ndarray:
    """MET of events with >= 2 jets of pT > 40 GeV."""
    n40 = ak.sum(events.jets.pt > 40, axis=1)
    return ak.to_numpy(events.met_pt[n40 >= 2])


def q5_met_osmumu(events: ak.Array) -> np.ndarray:
    """MET of events with an opposite-charge muon pair, 60 < m(μμ) < 120 GeV."""
    mu = muons_p4(events)
    pair = ak.combinations(mu, 2, fields=["a", "b"])
    mass = (pair.a + pair.b).mass
    good = (pair.a.charge != pair.b.charge) & (mass > 60) & (mass < 120)
    return ak.to_numpy(events.met_pt[ak.any(good, axis=1)])


def q6_trijet_pt(events: ak.Array) -> np.ndarray:
    """pT of the 3-jet system with mass closest to 172.5 GeV (events with >=3 jets)."""
    ev = events[ak.num(events.jets, axis=1) >= 3]
    jets = jets_p4(ev)
    tri = ak.combinations(jets, 3, fields=["a", "b", "c"])
    p4 = tri.a + tri.b + tri.c
    best = ak.argmin(abs(p4.mass - TOP_MASS), axis=1, keepdims=True)
    return ak.to_numpy(ak.flatten(p4.pt[best]))


def q7_ht_cleaned(events: ak.Array) -> np.ndarray:
    """Scalar sum of pT of jets (pT>30, |eta|<2.4) not within ΔR<0.4 of a muon (pT>10)."""
    jets = jets_p4(events)
    jets = jets[(jets.pt > 30) & (abs(jets.eta) < 2.4)]
    leps = muons_p4(events)
    leps = leps[leps.pt > 10]

    pair = ak.cartesian({"j": jets, "l": leps}, nested=True)   # per jet: all leptons
    dr = pair.j.deltaR(pair.l)
    min_dr = ak.min(dr, axis=2)
    min_dr = ak.fill_none(min_dr, np.inf)                       # jets with no lepton: keep
    clean = jets[min_dr > 0.4]
    ht = ak.sum(clean.pt, axis=1)
    return ak.to_numpy(ak.fill_none(ht, 0.0))


def q8_mt_met_lepton(events: ak.Array) -> np.ndarray:
    """Events with >=3 muons and a same-flavor opposite-charge (SFOS) pair:
    pick the SFOS pair with mass closest to the Z, then the transverse mass of
    MET + the highest-pT muon not in that pair."""
    ev = events[ak.num(events.muons, axis=1) >= 3]
    mu = muons_p4(ev)

    pairs = ak.argcombinations(mu, 2, fields=["i", "j"])
    mi, mj = mu[pairs.i], mu[pairs.j]
    sfos = mi.charge != mj.charge
    zmass = (mi + mj).mass
    metric = abs(ak.where(sfos, zmass, np.inf) - Z_MASS)

    has_sfos = ak.any(sfos, axis=1)
    best = ak.argmin(metric, axis=1, keepdims=True)
    bi = ak.flatten(pairs.i[best])
    bj = ak.flatten(pairs.j[best])

    idx = ak.local_index(mu, axis=1)
    other = mu[(idx != bi) & (idx != bj)]
    lead = other[ak.argmax(other.pt, axis=1, keepdims=True)]
    lead_pt = ak.flatten(lead.pt)
    lead_phi = ak.flatten(lead.phi)

    dphi = lead_phi - ev.met_phi
    mt = np.sqrt(2.0 * lead_pt * ev.met_pt * (1.0 - np.cos(dphi)))

    valid = ak.to_numpy(has_sfos & (ak.num(other, axis=1) >= 1))
    return ak.to_numpy(mt)[valid]


QUERIES = {
    "Q1: MET (all events)": q1_met,
    "Q2: jet pT (all jets)": q2_jet_pt,
    "Q3: jet pT, |eta|<1": q3_jet_pt_central,
    "Q4: MET, >=2 jets pT>40": q4_met_ge2jets40,
    "Q5: MET, OS dimuon 60-120": q5_met_osmumu,
    "Q6: trijet pT (m~172.5)": q6_trijet_pt,
    "Q7: HT of cleaned jets": q7_ht_cleaned,
    "Q8: MT(MET, lead lepton)": q8_mt_met_lepton,
}


def run_all(events: ak.Array) -> dict[str, np.ndarray]:
    return {name: fn(events) for name, fn in QUERIES.items()}


def events_from_root(root_file: str) -> ak.Array:
    """Load a NanoAOD ROOT file and reconstruct events with nested jets + muons."""
    ds = DATASETS["nanoaod"]
    events_ak, _ = read_root(root_file, ds, None)
    t = build_tables(events_ak, ds)
    return reconstruct_multi(t["events"], {"jets": t["jets"], "muons": t["muons"]})
