"""Edge-type read-out, the single code path used by experiments/analysis/a3_source_readout.py and
edge_b_compare.py. Truth-free: edges are the LEARNED edges (2-means on |W|); each edge's coupling is evaluated with the
model's own VectorField.edge_coupling along u_j = u_i + du (other state coordinates at the observed median), averaged
over a few receiver states, sign-gauge fixed and unit-normalised.
  per-edge   : select_k on the edge shapes;
  per-source : average the shapes of each source node's learned out-edges -> select_k on the source curves; an edge
               gets its source's group (design option B's read-out; right when the coupling form is set by the source).
Ground truth is used only for the scores (ARI against the true edge types / source classes, known-K references)."""
import json
from typing import Dict

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from .gen.kuramoto import load_dataset
from .model import build
from .readout import binarize_2means, select_k
from .snapshot import array_path, metrics_of, state_path


def load_run(run: str):
    m = json.load(open(run + "/metrics.json")); a = m["args"]; d = load_dataset(a["data"])
    N, dim = d["A"].shape[0], d["data"].shape[-1]
    gp, ode = build(a["model"], N, dim, a["hidden"], a["d_z"], float(d["meta"]["dt"]), 1e-3, 1e-3, "rk4", bool(a["mask_diag"]),
                    feat=a.get("feat", "raw"), coupling=a.get("coupling", "free"), z_norm=a.get("z_norm", "none"),
                    edge_z=bool(a.get("edge_z", 0)), edge_emb=int(a.get("edge_emb", 0) or 0))
    st = torch.load(state_path(run, m), map_location="cpu"); gp.load_state_dict(st["gp"]); ode.load_state_dict(st["vf"])
    ode.vf.eval()
    return m, d, ode.vf, np.load(array_path(run, "W", m))


def oriented_sign_accuracy(run: str, n_states: int = 256) -> float:
    """Sign accuracy on true edges after the model's own gauge (model.gauge_sign: one orientation per edge for design
    option B, the global one otherwise). Used to rescore saved runs trained before the per-edge gauge existed,
    without retraining. Gauge states: node states of the training trajectories only (as in train.py); the true signs
    enter only the score."""
    from .data import select_trajectories, trajectory_split
    from .metrics import sign_accuracy
    from .model import gauge_sign
    m, d, vf, W = load_run(run)
    a = m["args"]; X = np.asarray(d["data"], np.float32)
    X = X[select_trajectories(X.shape[0], a.get("n_traj"), a["seed"])]
    tr, _, _ = trajectory_split(X.shape[0], a["seed"])
    xs = X[tr].reshape(-1, X.shape[-2], X.shape[-1])                       # (snapshots, N, dim)
    rng = np.random.default_rng(0); xs = xs[rng.choice(len(xs), size=min(len(xs), max(1, n_states // X.shape[-2] + 1)), replace=False)]
    g = gauge_sign(vf, torch.as_tensor(xs))
    return sign_accuracy(W * np.asarray(g, dtype=W.dtype), d["S"])


def edge_type_readout(run: str, n_du: int = 41, n_recv: int = 5, keep_curves: bool = False) -> Dict:
    """keep_curves (for the figures): also return du, the per-source curves and their read-out groups."""
    m, d, vf, W = load_run(run)
    B = binarize_2means(W); src, dst = np.nonzero(B)
    X = d["data"].reshape(-1, d["data"].shape[-1])
    lo, hi = np.percentile(X[:, 0], 5), np.percentile(X[:, 0], 95); med = np.median(X, 0)
    du = np.linspace(-(hi - lo), hi - lo, n_du); ui = np.linspace(lo, hi, n_recv)
    shapes = []
    with torch.no_grad():
        for j, i in zip(src, dst):
            acc = np.zeros_like(du)
            for u0 in ui:
                xi = np.tile(med, (n_du, 1)); xj = xi.copy(); xi[:, 0] = u0; xj[:, 0] = u0 + du
                acc += vf.edge_coupling(torch.tensor(xi, dtype=torch.float32), torch.tensor(xj, dtype=torch.float32), int(i), int(j))[:, 0].numpy()
            shapes.append(acc / n_recv)
    out = dict(n_learned=int(len(src)), n_true=int((d["A"] > 0).sum()), n_correct=int((B & (d["A"] > 0)).sum()),
               auroc=metrics_of(m)["auroc"], auprc=metrics_of(m)["auprc"], val=m["val_mse_best"])
    if len(src) < 3:
        return out
    S = np.array(shapes); S = S * np.sign(S[:, -1:] - S[:, :1])
    Sn = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12)
    K_e, lab_e, _ = select_k(Sn, k_max=5, curves=True)
    srcs = np.unique(src)
    C = np.stack([Sn[src == j].mean(0) for j in srcs]); C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    K_s, lab_s, _ = select_k(C, k_max=5, curves=True) if len(srcs) > 2 else (1, np.zeros(len(srcs), int), [])
    out.update(K_edge=int(K_e), K_source=int(K_s), n_sources=int(len(srcs)))
    if keep_curves:
        out.update(du=du, sources=srcs, source_curves=C, source_groups=lab_s, edge_src=src, edge_dst=dst, edge_curves=Sn,
                   edge_groups=lab_s[np.searchsorted(srcs, src)])
    E = d.get("etypes")
    if E is not None:                                                        # scoring only
        t = E[src, dst]; cls = d["classes"]
        out.update(ari_edge=adjusted_rand_score(t, lab_e), ari_source_edges=adjusted_rand_score(t, lab_s[np.searchsorted(srcs, src)]),
                   ari_source_nodes=adjusted_rand_score(cls[srcs], lab_s),
                   ref_knownK_edge=adjusted_rand_score(t, KMeans(2, n_init=20, random_state=0).fit_predict(Sn)),
                   ref_knownK_source=adjusted_rand_score(cls[srcs], KMeans(2, n_init=20, random_state=0).fit_predict(C)))
    return out
