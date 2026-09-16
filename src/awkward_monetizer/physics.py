"""Physics helpers: 4-vectors (scikit-hep ``vector``) and simple quantities."""

from __future__ import annotations

import awkward as ak
import numpy as np
import vector

vector.register_awkward()

Z_MASS = 91.1876
TOP_MASS = 172.5


def jets_p4(events: ak.Array) -> ak.Array:
    """Momentum4D array of jets from (pt, eta, phi, mass).

    Kinematics are promoted to float64: NanoAOD stores them as float32, but the
    MonetDB tables are DOUBLE, and invariant mass is a difference of large
    squares (E^2 - p^2) that loses all significance in float32 for high-momentum
    objects. Computing in float64 keeps the Awkward and in-DB results in
    agreement (and is simply more accurate).
    """
    j = events.jets
    def f64(x): return ak.values_astype(x, np.float64)
    return ak.zip({"pt": f64(j.pt), "eta": f64(j.eta),
                   "phi": f64(j.phi), "mass": f64(j.mass)},
                  with_name="Momentum4D")


def muons_p4(events: ak.Array) -> ak.Array:
    """Momentum4D array of muons, carrying charge as an extra field.

    Kinematics are promoted to float64 (see :func:`jets_p4`) so the invariant
    mass matches the DOUBLE-precision in-database computation.
    """
    m = events.muons
    def f64(x): return ak.values_astype(x, np.float64)
    return ak.zip({"pt": f64(m.pt), "eta": f64(m.eta), "phi": f64(m.phi),
                   "mass": f64(m.mass), "charge": m.charge},
                  with_name="Momentum4D")


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
