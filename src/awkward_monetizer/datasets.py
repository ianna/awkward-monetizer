"""Dataset registry: how each ROOT layout maps onto flat relational tables.

A :class:`Dataset` describes the event-level scalar columns plus one or more
object collections. A collection is either *jagged* (NanoAOD-style per-event
lists such as ``Jet_pt``) or *wide* (a fixed set of objects stored as parallel
suffixed columns, e.g. the dimuon ntuple's ``pt1``/``pt2``). Adding support for
a new file layout is a matter of declaring one more ``Dataset`` here.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Scalars:
    """Event-level scalar columns: ``{destination column: ROOT branch}``."""
    fields: dict


@dataclass
class JaggedCollection:
    """A NanoAOD-style jagged collection (per-event variable-length lists)."""
    table: str
    index_col: str
    count_branch: str          # a jagged branch used to derive per-event counts
    branches: dict             # {dest column: jagged ROOT branch}
    kind: str = "jagged"


@dataclass
class WideCollection:
    """A fixed set of objects stored as parallel suffixed columns (pt1, pt2)."""
    table: str
    index_col: str
    slots: list                # e.g. ["1", "2"]
    branches: dict             # {dest column: template with "{}" for the slot}
    kind: str = "wide"


@dataclass
class Dataset:
    name: str
    events: Scalars
    collections: list
    tree: str | None = None    # explicit TTree name, or None to auto-detect
    event_id: str = "row"      # "row" -> synthetic 0..N-1, else an event branch
    # derived event columns: list of (column, kind, ref)
    #   kind "count" -> ak.num of a jagged ref branch
    #   kind "const" -> a constant value ref
    derived: list = field(default_factory=list)

    def all_branches(self) -> list[str]:
        names: list[str] = list(self.events.fields.values())
        if self.event_id != "row":
            names.append(self.event_id)
        for c in self.collections:
            if isinstance(c, JaggedCollection):
                names.append(c.count_branch)
                names.extend(c.branches.values())
            else:  # WideCollection
                for tmpl in c.branches.values():
                    names.extend(tmpl.format(s) for s in c.slots)
        for _col, kind, ref in self.derived:
            if kind == "count":
                names.append(ref)
        seen, out = set(), []
        for n in names:
            if n not in seen:
                seen.add(n)
                out.append(n)
        return out


# The flat CMS dimuon teaching ntuple (Run, Event, Type, M, and two muons).
DIMUON = Dataset(
    name="dimuon",
    tree=None,                 # file's tree is "events"; auto-detect finds it
    event_id="row",            # Event is not guaranteed unique -> synthesize
    events=Scalars({
        "run": "Run",
        "event_number": "Event",
        "muon_type": "Type",
        "mass": "M",
    }),
    collections=[
        WideCollection(
            table="muons", index_col="muon_index", slots=["1", "2"],
            branches={
                "e": "E{}", "px": "px{}", "py": "py{}", "pz": "pz{}",
                "pt": "pt{}", "eta": "eta{}", "phi": "phi{}", "charge": "Q{}",
            },
        ),
    ],
    derived=[("n_muons", "const", 2)],
)

# The NanoAOD-style design target (jagged jets + muons, plus MET).
NANOAOD = Dataset(
    name="nanoaod",
    tree="Events",
    # `event` alone is NOT unique in real NanoAOD (the key is run+lumi+event), so
    # synthesize event_id from the row index and keep run/lumi/event as columns.
    event_id="row",
    events=Scalars({"run": "run", "lumi": "luminosityBlock",
                    "event_number": "event",
                    "met_pt": "MET_pt", "met_phi": "MET_phi"}),
    collections=[
        JaggedCollection(
            table="jets", index_col="jet_index", count_branch="Jet_pt",
            branches={"pt": "Jet_pt", "eta": "Jet_eta",
                      "phi": "Jet_phi", "mass": "Jet_mass"},
        ),
        JaggedCollection(
            table="muons", index_col="muon_index", count_branch="Muon_pt",
            branches={"pt": "Muon_pt", "eta": "Muon_eta", "phi": "Muon_phi",
                      "mass": "Muon_mass", "charge": "Muon_charge"},
        ),
    ],
    derived=[("n_jets", "count", "Jet_pt")],
)

DATASETS = {ds.name: ds for ds in (DIMUON, NANOAOD)}
