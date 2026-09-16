import awkward as ak
import numpy as np
import vector

from awkward_monetizer import adl

vector.register_awkward()


def _muon(pt, phi, q, eta=0.0):
    return dict(pt=pt, eta=eta, phi=phi, mass=0.10566, charge=q)


def _jet(pt, eta, phi, mass=5.0):
    return dict(pt=pt, eta=eta, phi=phi, mass=mass)


def test_q8_matches_hand_calculation():
    """3 muons; (mu0,mu1) is the ~90 GeV Z pair, mu2 is the lead remaining lepton."""
    mus = [_muon(45, 0.0, +1), _muon(45, np.pi, -1), _muon(30, 0.3, +1)]
    events = ak.Array([{"met_pt": 100.0, "met_phi": 0.0, "jets": [], "muons": mus}])
    got = adl.q8_mt_met_lepton(events)
    expected = np.sqrt(2 * 30 * 100 * (1 - np.cos(0.3)))
    assert got.size == 1
    assert abs(got[0] - expected) < 1e-6


def test_q6_single_trijet_is_exact():
    jets = [_jet(80, 0.5, 0.0, 10.0), _jet(70, -0.3, 1.2, 8.0), _jet(60, 0.1, -2.0, 12.0)]
    events = ak.Array([{"met_pt": 0.0, "met_phi": 0.0, "jets": jets, "muons": []}])
    p4 = sum((vector.obj(**j) for j in jets[1:]), vector.obj(**jets[0]))
    got = adl.q6_trijet_pt(events)
    assert abs(got[0] - p4.pt) < 1e-6

    two_jets = ak.Array([{"met_pt": 0.0, "met_phi": 0.0, "jets": jets[:2], "muons": []}])
    assert adl.q6_trijet_pt(two_jets).size == 0


def test_q7_removes_jet_overlapping_muon():
    events = ak.Array([{
        "met_pt": 0.0, "met_phi": 0.0,
        "jets": [_jet(50, 0.0, 0.0), _jet(60, 0.0, 2.0)],
        "muons": [_muon(20, 0.02, 1, eta=0.02)],
    }])
    ht = adl.q7_ht_cleaned(events)          # jet near muon (dR~0.03) removed -> HT=60
    assert abs(ht[0] - 60.0) < 1e-6


def test_query_invariants(nano_events):
    ev = nano_events
    assert adl.q1_met(ev).size == len(ev)
    assert adl.q2_jet_pt(ev).size == int(ak.sum(ak.num(ev.jets, axis=1)))
    exp3 = int(ak.sum(ak.num(ev.jets.pt[abs(ev.jets.eta) < 1.0], axis=1)))
    assert adl.q3_jet_pt_central(ev).size == exp3
    exp4 = int(ak.sum(ak.sum(ev.jets.pt > 40, axis=1) >= 2))
    assert adl.q4_met_ge2jets40(ev).size == exp4
    q7 = adl.q7_ht_cleaned(ev)
    assert q7.size == len(ev) and np.all(q7 >= 0)
