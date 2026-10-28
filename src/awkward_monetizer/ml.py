"""ML consumption of the reconstructed NF2 events — the "AI-ready" last mile.

This module turns the jagged, variable-multiplicity event structure produced by
:mod:`awkward_monetizer.reconstruct` into fixed-shape, batched PyTorch tensors,
and exercises that path two ways:

  * :func:`throughput` — how many events/second the pipeline pushes through a
    real PyTorch training step, separated into three honest numbers: the NF2 →
    tensor *bridge*, pure *data-loading* (DataLoader collation), and the full
    *training* step (data + forward + backward).
  * :func:`train` — a small end-to-end regression (predict event MET from its
    set of jets with a permutation-invariant DeepSets network), reporting that
    the loss actually decreases and an R² on held-out events — i.e. the pipeline
    feeds a real optimizer, not just a shape.

The bridge is deliberately explicit. ``ak.pad_none`` → ``ak.fill_none`` →
``ak.to_numpy`` turns a jagged per-event list of jets into a dense
``(n_events, max_jets, n_features)`` block plus a boolean validity mask — exactly
the shape a set/DeepSets model consumes. (``ak.to_torch`` covers already-regular
arrays; padding is what a variable-length collection needs first, and the mask is
what keeps the padding from polluting the physics.)

``torch`` is an optional dependency, imported lazily and gated like the ROOT
backends in :mod:`awkward_monetizer.benchmark`, so importing this module never
requires it. Install with the ``ml`` extra: ``pip install awkward-monetizer[ml]``.

The functions take an already-reconstructed Awkward array (``events`` with a
nested ``jets`` collection and an ``met_pt`` field), so they are source-agnostic:
feed them the output of :func:`reconstruct_multi` from MonetDB, of
:func:`awkward_monetizer.adl.events_from_root` from a ROOT file, or of
:func:`synthetic_events` for a self-contained, data-free demo.
"""

from __future__ import annotations

import statistics
import time

import awkward as ak
import numpy as np

from .timing import stage_time

DEFAULT_JET_FIELDS = ("pt", "eta", "phi", "mass")


# --------------------------------------------------------------------------
# The bridge: jagged NF2 collection -> dense (events, objects, features) block
# --------------------------------------------------------------------------
def jets_to_tensor(events: ak.Array, fields: tuple[str, ...] = DEFAULT_JET_FIELDS,
                   max_jets: int | None = None, *, collection: str = "jets",
                   pad_value: float = 0.0,
                   timings: dict[str, float] | None = None,
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad a jagged per-event collection into dense NumPy arrays.

    Returns ``(X, mask, counts)``:
      * ``X``      — ``(n_events, max_jets, n_features)`` float32, ``pad_value``
                     in the padded slots.
      * ``mask``   — ``(n_events, max_jets)`` bool, True where a real object sits.
      * ``counts`` — ``(n_events,)`` int64 per-event object multiplicity.

    ``max_jets`` defaults to the largest multiplicity present (``clip=True`` keeps
    the block rectangular either way). This is the single conversion the
    "AI-ready" claim rests on, so it is timed as its own stage.
    """
    with stage_time(timings, "count"):
        coll = events[collection]
        counts = ak.to_numpy(ak.num(coll, axis=1)).astype("int64")
    if max_jets is None:
        max_jets = int(counts.max()) if counts.size else 0

    with stage_time(timings, "pad_fill"):
        columns = []
        for f in fields:
            padded = ak.pad_none(coll[f], max_jets, axis=1, clip=True)
            filled = ak.fill_none(padded, pad_value, axis=1)
            columns.append(ak.to_numpy(filled).astype("float32"))
    with stage_time(timings, "stack"):
        # (n_events, max_jets, n_features): the DeepSets/transformer input shape.
        X = np.stack(columns, axis=-1) if columns else np.empty((len(counts), 0, 0),
                                                                 dtype="float32")
        mask = np.arange(max_jets)[None, :] < counts[:, None]
    return X, mask, counts


def event_targets(events: ak.Array, target: str = "met_pt") -> np.ndarray:
    """Per-event regression target as a float32 vector (default: MET pT)."""
    return ak.to_numpy(events[target]).astype("float32")


# --------------------------------------------------------------------------
# A self-contained synthetic source (no ROOT file, no database) so the loop
# runs anywhere — used by the tests, the reproducibility notebook, and CI.
# --------------------------------------------------------------------------
def synthetic_events(n_events: int = 100_000, avg_jets: float = 4.0,
                     seed: int = 0) -> ak.Array:
    """A NanoAOD-shaped Awkward array with jagged ``jets`` and an ``met_pt`` label.

    Jet multiplicity is Poisson(``avg_jets``); kinematics are plausible ranges.
    ``met_pt`` is a smooth function of the per-event jet activity plus noise, so a
    model *can* learn it and the training loss demonstrably decreases — the point
    is to prove the data path drives a real optimizer, not to model detector MET.
    """
    rng = np.random.default_rng(seed)
    counts = rng.poisson(avg_jets, size=n_events).astype("int64")
    total = int(counts.sum())
    pt = rng.exponential(30.0, total).astype("float32") + 15.0
    eta = rng.normal(0.0, 2.0, total).astype("float32")
    phi = rng.uniform(-np.pi, np.pi, total).astype("float32")
    mass = rng.exponential(8.0, total).astype("float32") + 2.0
    jets = ak.unflatten(
        ak.zip({"pt": pt, "eta": eta, "phi": phi, "mass": mass}), counts)
    # A learnable label: vector-summed jet pT imbalance + soft term + noise.
    px = ak.sum(jets.pt * np.cos(jets.phi), axis=1)
    py = ak.sum(jets.pt * np.sin(jets.phi), axis=1)
    recoil = np.hypot(ak.to_numpy(px), ak.to_numpy(py)).astype("float32")
    met = 0.6 * recoil + 4.0 * counts + rng.normal(0.0, 6.0, n_events).astype("float32")
    met = np.clip(met, 0.0, None).astype("float32")
    return ak.Array({
        "event_id": np.arange(n_events, dtype="int64"),
        "met_pt": met,
        "n_jets": counts,
        "jets": jets,
    })


# --------------------------------------------------------------------------
# A permutation-invariant DeepSets regressor (built lazily so torch stays optional)
# --------------------------------------------------------------------------
def build_model(n_features: int, hidden: int = 64):
    """A DeepSets network: per-jet MLP -> masked sum-pool -> per-event MLP.

    Sum-pooling over the jet axis makes the output invariant to jet ordering — the
    right inductive bias for an unordered set of reconstructed objects, and the
    reason the padding mask must be threaded through the pool.
    """
    import torch
    from torch import nn

    class DeepSets(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.phi = nn.Sequential(
                nn.Linear(n_features, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU())
            self.rho = nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1))

        def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            h = self.phi(x)                          # (B, J, H)
            h = h * mask.unsqueeze(-1)               # zero the padded jets
            pooled = h.sum(dim=1)                    # (B, H), order-invariant
            return self.rho(pooled).squeeze(-1)      # (B,)

    return DeepSets()


def make_dataset(events: ak.Array, fields: tuple[str, ...] = DEFAULT_JET_FIELDS,
                 target: str = "met_pt", max_jets: int | None = None, *,
                 standardize: bool = True,
                 timings: dict[str, float] | None = None):
    """Reconstructed events -> a torch ``TensorDataset`` of ``(X, mask, y)``.

    Standardizes features over real (unmasked) jets and the target over events, so
    the regression is well-conditioned; returns ``(dataset, stats)`` where stats
    holds the means/stds needed to invert the target scaling for reporting.
    """
    import torch
    from torch.utils.data import TensorDataset

    X, mask, _counts = jets_to_tensor(events, fields, max_jets, timings=timings)
    y = event_targets(events, target)

    stats: dict[str, np.ndarray | float] = {}
    if standardize:
        flat = X[mask]                               # (n_real_jets, n_features)
        mu = flat.mean(axis=0) if flat.size else np.zeros(X.shape[-1], "float32")
        sd = flat.std(axis=0) if flat.size else np.ones(X.shape[-1], "float32")
        sd = np.where(sd < 1e-6, 1.0, sd).astype("float32")
        X = ((X - mu) * mask[..., None]) / sd        # keep padded slots exactly 0
        y_mu, y_sd = float(y.mean()), float(y.std() or 1.0)
        y = (y - y_mu) / y_sd
        stats = {"x_mean": mu, "x_std": sd, "y_mean": y_mu, "y_std": y_sd}

    ds = TensorDataset(torch.from_numpy(np.ascontiguousarray(X)),
                       torch.from_numpy(np.ascontiguousarray(mask)),
                       torch.from_numpy(np.ascontiguousarray(y)))
    return ds, stats


# --------------------------------------------------------------------------
# Throughput: events/second through the bridge, the loader, and a full step
# --------------------------------------------------------------------------
def throughput(events: ak.Array, *, batch_size: int = 1024, epochs: int = 3,
               fields: tuple[str, ...] = DEFAULT_JET_FIELDS, target: str = "met_pt",
               device: str = "cpu", num_workers: int = 0,
               hidden: int = 64) -> dict:
    """Measure how fast the pipeline feeds a PyTorch training loop.

    Reports three events/second figures on the same data:
      * ``bridge`` — Awkward NF2 -> ``(X, mask, y)`` tensors (the AI-ready step).
      * ``loader`` — pure DataLoader iteration over the materialized tensors
                     (batching/collation/host->device), no compute.
      * ``train``  — the full step: data + forward + backward + optimizer.
    """
    import torch
    from torch.utils.data import DataLoader

    n = len(events)
    dev = torch.device(device)

    # 1) Bridge throughput: the jagged -> dense conversion, timed on its own.
    t0 = time.perf_counter()
    ds, _stats = make_dataset(events, fields, target)
    bridge_s = time.perf_counter() - t0

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=(dev.type == "cuda"))
    n_features = ds.tensors[0].shape[-1]

    # 2) Data-loading throughput: iterate batches to the device, no model.
    t0 = time.perf_counter()
    for _ in range(epochs):
        for xb, mb, yb in loader:
            xb.to(dev); mb.to(dev); yb.to(dev)
    loader_s = time.perf_counter() - t0

    # 3) Full training-step throughput.
    model = build_model(n_features, hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    model.train()
    t0 = time.perf_counter()
    for _ in range(epochs):
        for xb, mb, yb in loader:
            xb, mb, yb = xb.to(dev), mb.to(dev).float(), yb.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(xb, mb), yb)
            loss.backward()
            opt.step()
    train_s = time.perf_counter() - t0

    def eps(seconds: float, passes: int) -> float:
        return (n * passes) / seconds if seconds > 0 else float("nan")

    return {
        "n_events": n, "batch_size": batch_size, "epochs": epochs,
        "n_features": n_features, "device": str(dev),
        "bridge": {"seconds": bridge_s, "events_per_s": eps(bridge_s, 1)},
        "loader": {"seconds": loader_s, "events_per_s": eps(loader_s, epochs)},
        "train": {"seconds": train_s, "events_per_s": eps(train_s, epochs)},
    }


# --------------------------------------------------------------------------
# End-to-end: a short training run that actually converges
# --------------------------------------------------------------------------
def train(events: ak.Array, *, epochs: int = 15, batch_size: int = 1024,
          lr: float = 1e-3, val_frac: float = 0.2, hidden: int = 64,
          fields: tuple[str, ...] = DEFAULT_JET_FIELDS, target: str = "met_pt",
          device: str = "cpu", seed: int = 0, verbose: bool = True) -> dict:
    """Train the DeepSets regressor end-to-end on the reconstructed events.

    Returns per-epoch train/val loss, the final held-out R² (in physical units),
    and the training-step throughput — a single, reproducible result that
    substantiates "AI-ready data" with a measurement.
    """
    import torch
    from torch.utils.data import DataLoader, random_split

    torch.manual_seed(seed)
    ds, stats = make_dataset(events, fields, target)
    n_val = int(len(ds) * val_frac)
    n_train = len(ds) - n_val
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=gen)

    dev = torch.device(device)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    model = build_model(ds.tensors[0].shape[-1], hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    history: list[dict] = []
    step_times: list[float] = []
    for epoch in range(epochs):
        model.train()
        running, seen, t0 = 0.0, 0, time.perf_counter()
        for xb, mb, yb in train_loader:
            xb, mb, yb = xb.to(dev), mb.to(dev).float(), yb.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(xb, mb), yb)
            loss.backward()
            opt.step()
            running += loss.item() * len(yb); seen += len(yb)
        step_times.append(time.perf_counter() - t0)
        train_loss = running / max(seen, 1)

        model.eval()
        vrun, vseen = 0.0, 0
        preds, trues = [], []
        with torch.no_grad():
            for xb, mb, yb in val_loader:
                xb, mb, yb = xb.to(dev), mb.to(dev).float(), yb.to(dev)
                out = model(xb, mb)
                vrun += loss_fn(out, yb).item() * len(yb); vseen += len(yb)
                preds.append(out.cpu().numpy()); trues.append(yb.cpu().numpy())
        val_loss = vrun / max(vseen, 1)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss})
        if verbose:
            print(f"epoch {epoch:2d}  train {train_loss:.4f}  val {val_loss:.4f}")

    # Held-out R², reported in physical MET units (undo the target scaling).
    p = np.concatenate(preds) * stats.get("y_std", 1.0)
    t = np.concatenate(trues) * stats.get("y_std", 1.0)
    ss_res = float(np.sum((t - p) ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2)) or 1.0
    r2 = 1.0 - ss_res / ss_tot
    n = len(events)
    train_eps = (n_train / statistics.median(step_times)
                 if step_times else float("nan"))

    return {
        "history": history,
        "final_train_loss": history[-1]["train_loss"] if history else float("nan"),
        "final_val_loss": history[-1]["val_loss"] if history else float("nan"),
        "val_r2": r2,
        "n_events": n, "n_train": n_train, "n_val": n_val,
        "train_events_per_s": train_eps,
        "device": str(dev),
    }


# --------------------------------------------------------------------------
# Convenience source loaders (thin wrappers over the existing pipeline)
# --------------------------------------------------------------------------
def events_from_root(nano_file: str) -> ak.Array:
    """Reconstruct NanoAOD events (jets + muons) straight from a ROOT file."""
    from .adl import events_from_root as _from_root
    return _from_root(nano_file)


def events_from_monetdb(*, database: str = "hep", where: str | None = None,
                        **conn_kwargs) -> ak.Array:
    """Reconstruct events from a running MonetDB `hep` database.

    Pushes ``where`` down to SQL (e.g. ``"met_pt > 50"``), fetching only the
    selected slice, then rebuilds the NF2 ``jets`` structure in Awkward.
    """
    import pandas as pd

    from .db import open_server
    from .reconstruct import reconstruct_multi

    conn = open_server(database=database, **conn_kwargs)
    try:
        cur = conn.cursor()

        def fetch(sql: str) -> pd.DataFrame:
            cur.execute(sql)
            return pd.DataFrame(cur.fetchall(),
                                columns=[c[0] for c in cur.description])

        sel = f" WHERE {where}" if where else ""
        events_df = fetch(f"SELECT * FROM events{sel}")
        join = (f" JOIN (SELECT event_id FROM events{sel}) s "
                "ON jets.event_id = s.event_id") if where else ""
        jets_df = fetch(f"SELECT jets.* FROM jets{join} "
                        "ORDER BY jets.event_id, jets.jet_index")
        return reconstruct_multi(events_df, {"jets": jets_df})
    finally:
        conn.close()
