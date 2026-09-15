"""Physics helpers: 4-vectors (scikit-hep ``vector``) and simple quantities."""

from __future__ import annotations

import awkward as ak
import numpy as np
import vector

vector.register_awkward()

Z_MASS = 91.1876
TOP_MASS = 172.5


def jets_p4(events: ak.Array) -> ak.Array:
    """Momentum4D array of jets from (pt, eta, phi, mass)."""
    j = events.jets
    return ak.zip({"pt": j.pt, "eta": j.eta, "phi": j.phi, "mass": j.mass},
                  with_name="Momentum4D")


def muons_p4(events: ak.Array) -> ak.Array:
    """Momentum4D array of muons, carrying charge as an extra field."""
    m = events.muons
    return ak.zip({"pt": m.pt, "eta": m.eta, "phi": m.phi, "mass": m.mass,
                   "charge": m.charge}, with_name="Momentum4D")


def invariant_mass(objects: ak.Array) -> ak.Array:
    """Invariant mass of the summed 4-momenta of all objects in each event,
    from Cartesian components (E, px, py, pz).

    Used on reconstructed dimuon muons (which carry e/px/py/pz); for the dimuon
    dataset each event has two muons, so this is the dimuon invariant mass.
    """
    E = ak.sum(objects.e, axis=1)
    px = ak.sum(objects.px, axis=1)
    py = ak.sum(objects.py, axis=1)
    pz = ak.sum(objects.pz, axis=1)
    m2 = E * E - (px * px + py * py + pz * pz)
    return np.sqrt(np.clip(m2, 0.0, None))


def opposite_charge_pair(events: ak.Array, collection: str = "muons") -> ak.Array:
    """Boolean mask: events with exactly two opposite-charge objects."""
    objs = events[collection]
    n = ak.num(objs, axis=1)
    charge_sum = ak.sum(objs.charge, axis=1)
    return (n == 2) & (charge_sum == 0)
