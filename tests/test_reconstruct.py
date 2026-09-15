import awkward as ak
import numpy as np

from awkward_monetizer.physics import invariant_mass
from awkward_monetizer.reconstruct import reconstruct_events, tables_from_root


def test_dimuon_mass_roundtrip(cms_root):
    """Reconstructed dimuon mass must match the file's stored M (no DB)."""
    events_df, muons_df = tables_from_root(cms_root, "dimuon")
    events = reconstruct_events(events_df, muons_df)

    assert len(events) == 2304
    assert ak.all(ak.num(events.muons, axis=1) == 2)

    recon = ak.to_numpy(invariant_mass(events.muons))
    stored = ak.to_numpy(events.mass)
    assert np.max(np.abs(recon - stored)) < 1e-6


def test_reconstruct_multi_nests_jets_and_muons(nano_events):
    """NanoAOD reconstruction nests both collections."""
    assert "jets" in nano_events.fields
    assert "muons" in nano_events.fields
    assert ak.all(ak.num(nano_events.jets, axis=1) >= 0)
