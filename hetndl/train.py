"""Single-run trainer.

Model selection never sees ground truth. Three snapshots are evaluated and saved:
  by_mse  - epoch with the lowest validation single-step MSE (default `best`)
  by_obj  - epoch with the lowest validation MSE + lambda_l1 * ||W||_1
  last    - the final epoch (fixed-budget training, as in the NOGI paper)
Ground truth (A, S, omega, classes) is read only inside `evaluate()` for reporting.
"""
import argparse
import csv
import json
import math
import os
import shutil
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as Fnn

from .data import build_loaders
from .gen.kuramoto import load_dataset
from .metrics import (auroc_auprc, sign_accuracy, spurious_per_node, spearman_z, cluster_ari,
                      rollout_nrmse)
from .model import VARIANTS, build, gauge_sign
from .readout import binarize_2means
from .snapshot import state_path


def release_sign(opt_w: torch.optim.Optimizer, reset: bool, lr: Optional[float]) -> None:
    """Sign release after the warm-up: optionally clear W's Adam moments and set W's learning rate."""
    if reset:
        opt_w.state.clear()
    if lr is not None:
        for pg in opt_w.param_groups:
            pg["lr"] = lr


def support_2means(W: np.ndarray) -> np.ndarray:
    """Edge set = upper group of an exact 1-D 2-means split of the off-diagonal |W| (the edge set of
    readout.binarize_2means, computed by a sorted scan so that it is cheap enough to run every epoch)."""
    W = np.abs(np.asarray(W, np.float64)); N = W.shape[0]; off = ~np.eye(N, dtype=bool)
    v = W[off]; srt = np.sort(v); n = len(srt)
    c, c2 = np.cumsum(srt), np.cumsum(srt * srt); k = np.arange(1, n)
    sse = (c2[k - 1] - c[k - 1] ** 2 / k) + ((c2[-1] - c2[k - 1]) - (c[-1] - c[k - 1]) ** 2 / (n - k))
    B = np.zeros_like(W, dtype=bool)
    B[off] = v > srt[int(np.argmin(sse))]
    return B


def support_stable(B: np.ndarray, B_prev: Optional[np.ndarray], thr: float) -> bool:
    """Adaptive sign release: the edge set counts as settled when its Jaccard overlap with the edge set
    of K epochs earlier is at least thr."""
    if B_prev is None:
        return False
    union = int((B | B_prev).sum())
    return (int((B & B_prev).sum()) / union if union else 1.0) >= thr


def _w_stats(gp) -> Tuple[float, float]:
    with torch.no_grad():
        W = gp.W().detach()
        a = W.abs()
        return float(a[W < 0].sum() / max(float(a.sum()), 1e-12)), float(a.sum())


def parse(argv: Optional[List[str]]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="HetNDL single run")
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--d_z", type=int, default=1)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr_w", type=float, default=1e-2)
    ap.add_argument("--lr_f", type=float, default=1e-3)
    ap.add_argument("--lambda_l1", type=float, default=1e-8)
    ap.add_argument("--prox_tau", type=float, default=0.0,
                    help="absolute soft-threshold applied to W after every optimiser step (0 = off)")
    ap.add_argument("--lambda_z", type=float, default=0.0,
                    help="L2 penalty on Z; keep 0 (a penalty removes the node-level information, see the ablations); the earliest runs used 1e-4")
    ap.add_argument("--edge_emb", type=int, default=0,
                    help="edge-function heterogeneity, design option B: dimension of separate outgoing / incoming edge "
                         "embeddings in G with per-edge unit-RMS normalisation (0 = shared G)")
    ap.add_argument("--edge_z", type=int, default=0,
                    help="1: G(x_i, x_j, z_i, z_j), coupling form depends on both endpoints' latents")
    ap.add_argument("--z_norm", choices=["none", "standardize"], default="none",
                    help="feed F the per-dimension standardised z (fixes the z scale/shift gauge)")
    ap.add_argument("--noise_snr", type=float, default=None)
    ap.add_argument("--noise_std", type=float, default=None, help="absolute observation-noise std (for phases)")
    ap.add_argument("--n_traj", type=int, default=None, help="use only this many trajectories (data-size curves)")
    ap.add_argument("--smooth_window", type=int, default=None,
                    help="Savitzky-Golay window (odd, time steps) applied to the observed series after noise")
    ap.add_argument("--weight_decay", type=float, default=0.0,
                    help="decoupled weight decay (AdamW) on the vector-field parameters; 0 = plain Adam")
    ap.add_argument("--time_range", choices=["dt", "unit"], default="dt")
    ap.add_argument("--mask_diag", type=int, default=1)
    ap.add_argument("--split", choices=["traj", "pairs"], default="traj")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--max_minutes", type=float, default=0.0)
    ap.add_argument("--method", type=str, default="dopri5")
    ap.add_argument("--step_size", type=float, default=None, help="fixed-step solvers: sub-step over [0, t_end]")
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--atol", type=float, default=1e-3)
    ap.add_argument("--rollout_steps", type=int, default=100)
    ap.add_argument("--init_scale", type=float, default=1.0)
    ap.add_argument("--sign_warmup", type=int, default=0,
                    help="signed W only: keep W >= 0 (nonneg projection) for the first N epochs, then release the sign")
    ap.add_argument("--release_reset", type=int, default=0,
                    help="at the sign release, clear W's Adam state (moments built up while entries were clamped)")
    ap.add_argument("--release_lr_scale", type=float, default=1.0,
                    help="at the sign release, multiply W's learning rate by this factor for the signed phase")
    ap.add_argument("--release_stable", type=int, default=0,
                    help="adaptive sign release: end the warm-up early once the 2-means edge set has Jaccard >= "
                         "--release_jaccard with the edge set of this many epochs earlier (0 = off; --sign_warmup is the cap)")
    ap.add_argument("--release_jaccard", type=float, default=0.95)
    ap.add_argument("--log_w_stats", type=int, default=0,
                    help="add w_neg_frac (share of |W| mass on negative entries) and w_l1 to log.csv")
    ap.add_argument("--init_from", type=str, default=None,
                    help="start from the reported snapshot of this run directory (same model settings)")
    ap.add_argument("--w_support", choices=["none", "2means"], default="none",
                    help="prune-and-refit: fix W's support to the 2-means edge set of the initial |W| (needs --init_from)")
    ap.add_argument("--cluster_k", type=int, default=0, help="GMM components for ARI; 0 = #true classes")
    ap.add_argument("--coupling", choices=["free", "diffusive"], default="free",
                    help="additive variants: G(x_i,x_j) (free) or G(x_i,x_j)-G(x_i,x_i) (diffusive)")
    ap.add_argument("--feat", choices=["raw", "trig"], default="raw",
                    help="MLP input features: raw states (NOGI) or (cos x, sin x) for unwrapped phases")
    ap.add_argument("--select", choices=["val_mse", "val_obj"], default="val_mse",
                    help="which snapshot is reported as `best`")
    ap.add_argument("--gauge_scale", choices=["none", "ref"], default="none",
                    help="ref: after every step rescale (W, G) -> (r W, G / r) so that the shared coupling has unit RMS on "
                         "fixed reference states; the model is unchanged and the L1 penalty on W gets a definite scale")
    ap.add_argument("--edge_scale", choices=["batch", "epoch"], default="batch",
                    help="edge_emb: per-edge RMS of the current batch (batch) or of fixed reference states, refreshed at "
                         "the start of every epoch and used unchanged in training and evaluation (epoch)")
    ap.add_argument("--ref_states", type=int, default=32, help="reference states for --gauge_scale ref / --edge_scale epoch")
    ap.add_argument("--init_seed", type=int, default=None,
                    help="seed of the initialisation and of the batch order; default = --seed, which also fixes the data split")
    return ap.parse_args(argv)


def _cpu_state(gp, ode) -> dict:
    return {"gp": {k: v.detach().cpu().clone() for k, v in gp.state_dict().items()},
            "vf": {k: v.detach().cpu().clone() for k, v in ode.state_dict().items()}}


def _load_state(gp, ode, st: dict) -> None:
    gp.load_state_dict(st["gp"])
    ode.load_state_dict(st["vf"])


@torch.no_grad()
def _mean_mse(ode, gp, loader, device) -> float:
    tot, n = 0.0, 0
    W = gp.W()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = ode(x, W)
        tot += float(Fnn.mse_loss(pred, y, reduction="sum"))
        n += y.numel()
    return tot / max(n, 1)


def _try(errors: List[str], tag: str, fn: Callable, default):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 - one broken metric must not sink the run
        errors.append("%s: %s: %s" % (tag, type(e).__name__, e))
        return default


def evaluate(gp, ode, d: dict, te_idx: np.ndarray, te_loader, device, rollout_steps: int, cluster_k: int,
             x_gauge: Optional[torch.Tensor] = None, errors: Optional[List[str]] = None, tag: str = "") -> dict:
    errors = [] if errors is None else errors
    A, S, omega, classes, data = d["A"], d["S"], d["omega"], d["classes"], d["data"]
    with torch.no_grad():
        W_t = gp.W().detach()
    W = W_t.cpu().numpy()
    Z = ode.vf.z_eff().detach().cpu().numpy() if ode.vf.Z is not None else None
    nan = float("nan")
    out: Dict = {}
    g = 1
    if gp.mode in ("signed", "pm") and x_gauge is not None:
        g = _try(errors, tag + ".gauge", lambda: gauge_sign(ode.vf, x_gauge), 1)
    if np.ndim(g) == 0:
        g = int(g); out["gauge"] = g
    else:                                     # design option B: one orientation per edge, W's [j, i] indexing
        g = np.asarray(g, dtype=W.dtype); out["gauge"] = "per_edge"; out["gauge_flipped_frac"] = float((g < 0).mean())
    au_ap = _try(errors, tag + ".auroc_auprc", lambda: auroc_auprc(W, A), (nan, nan))
    out["auroc"], out["auprc"] = float(au_ap[0]), float(au_ap[1])
    out["sign_acc"] = _try(errors, tag + ".sign_acc", lambda: sign_accuracy(W * g, S), nan)
    out["has_neg_edges"] = bool((S < 0).any())
    out["spurious_per_node"] = _try(errors, tag + ".spurious", lambda: [int(v) for v in spurious_per_node(W, A)], [])
    out["spearman_z"] = nan
    if Z is not None and np.isfinite(omega).all() and float(np.std(omega)) > 0:
        out["spearman_z"] = _try(errors, tag + ".spearman_z", lambda: spearman_z(Z, omega), nan)
    out["cluster_ari"] = nan
    if Z is not None and int(classes.max()) >= 0:
        K = cluster_k if cluster_k > 0 else int(len(np.unique(classes)))
        out["cluster_ari"] = _try(errors, tag + ".cluster_ari", lambda: cluster_ari(Z, classes, K), nan)
    out["test_mse"] = _try(errors, tag + ".test_mse", lambda: _mean_mse(ode, gp, te_loader, device), nan)
    out["rollout_nrmse"] = _try(errors, tag + ".rollout",
                                lambda: rollout_nrmse(ode, W_t, data, te_idx, rollout_steps, device), nan)
    out["rollout_steps"] = int(min(rollout_steps, data.shape[1] - 1))
    return out


def main(argv: Optional[List[str]] = None) -> dict:
    a = parse(argv)
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed if a.init_seed is None else a.init_seed)
    np.random.seed(a.seed)
    use_cuda = a.device.startswith("cuda") and torch.cuda.is_available()
    device = torch.device(a.device if use_cuda else "cpu")

    d = load_dataset(a.data)
    data, meta = d["data"], d["meta"]
    _, _, N, node_dim = data.shape
    dt = float(meta["dt"])
    t_end = dt if a.time_range == "dt" else 1.0
    tr, va, te, te_idx = build_loaders(data, a.seed, a.batch, a.noise_snr, a.num_workers, split=a.split,
                                       n_traj=a.n_traj, noise_std=a.noise_std, smooth_window=a.smooth_window)

    gp, ode = build(a.model, N, node_dim, a.hidden, a.d_z, t_end, a.rtol, a.atol, a.method,
                    bool(a.mask_diag), a.init_scale, feat=a.feat,
                    coupling=a.coupling, z_norm=a.z_norm, edge_z=bool(a.edge_z), step_size=a.step_size,
                    edge_emb=a.edge_emb, edge_scale=a.edge_scale)
    gp.to(device)
    ode.to(device)
    if a.gauge_scale == "ref" and (not ode.vf.additive or ode.vf.edge_z or ode.vf.edge_emb):
        raise ValueError("--gauge_scale ref needs an additive model with a shared coupling")
    if a.edge_scale == "epoch" and not ode.vf.edge_emb:
        raise ValueError("--edge_scale epoch needs --edge_emb")
    # fixed reference states (evenly spaced over the training pairs) for the two scale conventions
    ref_x = None
    if a.gauge_scale == "ref" or a.edge_scale == "epoch":
        idx = np.linspace(0, len(tr.dataset) - 1, min(a.ref_states, len(tr.dataset))).astype(int)
        ref_x = torch.stack([tr.dataset[int(i)][0] for i in idx]).to(device)

    def fix_scale() -> float:
        r = float(ode.vf.coupling_rms(ref_x))
        if not (math.isfinite(r) and r > 0):
            raise FloatingPointError("coupling RMS %r on the reference states" % r)
        with torch.no_grad():
            gp.raw.mul_(r)
            if gp.raw_neg is not None:
                gp.raw_neg.mul_(r)
        ode.vf.rescale_coupling(r)
        return r
    support_size = None
    if a.init_from:
        m0 = json.load(open(os.path.join(a.init_from, "metrics.json")))
        _load_state(gp, ode, torch.load(state_path(a.init_from, m0), map_location=device))
    if a.w_support == "2means":
        if not a.init_from:
            raise ValueError("--w_support needs --init_from")
        B = binarize_2means(gp.W().detach().cpu().numpy())
        gp.set_support(B)
        support_size = int(B.sum())
    if a.gauge_scale == "ref":
        fix_scale()                                   # the initial model, written in the gauge "coupling RMS = 1"
    opt_w = torch.optim.Adam(gp.parameters(), lr=a.lr_w)
    opt_f = (torch.optim.AdamW(ode.parameters(), lr=a.lr_f, weight_decay=a.weight_decay) if a.weight_decay > 0
             else torch.optim.Adam(ode.parameters(), lr=a.lr_f))

    log_path = os.path.join(a.out, "log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_mse", "val_obj", "auroc", "auprc", "sec"]
                               + (["w_neg_frac", "w_l1"] if a.log_w_stats else []))

    nan = float("nan")
    best = {k: {"crit": math.inf, "epoch": 0, "val_mse": nan, "val_obj": nan, "state": None}
            for k in ("by_mse", "by_obj")}
    sel_key = "by_mse" if a.select == "val_mse" else "by_obj"
    last_val, last_obj = nan, nan
    failed, reason = False, ""
    epoch_times: List[float] = []
    release_at = a.sign_warmup + 1 if a.sign_warmup > 0 else None      # first epoch with a free sign
    sup_hist: Dict[int, np.ndarray] = {}
    t_start = time.time()
    epoch = 0
    try:
        for epoch in range(1, a.epochs + 1):
            gp.clamp_signed = release_at is not None and epoch < release_at
            if release_at is not None and epoch == release_at:
                release_sign(opt_w, bool(a.release_reset), a.lr_w * a.release_lr_scale if a.release_lr_scale != 1.0 else None)
            t0 = time.time()
            if a.edge_scale == "epoch":
                ode.vf.refresh_edge_scale(ref_x)
            gp.train()
            ode.train()
            losses = []
            for x, y in tr:
                x, y = x.to(device), y.to(device)
                opt_w.zero_grad()
                opt_f.zero_grad()
                W = gp.W()
                pred = ode(x, W)
                loss = Fnn.mse_loss(pred, y) + a.lambda_l1 * gp.l1()
                if ode.vf.Z is not None:
                    loss = loss + a.lambda_z * (ode.vf.Z ** 2).sum()
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite loss at epoch %d" % epoch)
                loss.backward()
                opt_w.step()
                opt_f.step()
                gp.prox(a.prox_tau)
                gp.project()
                if a.gauge_scale == "ref":
                    fix_scale()
                losses.append(float(loss))
            gp.eval()
            ode.eval()
            val = _mean_mse(ode, gp, va, device)
            if not np.isfinite(val):
                raise FloatingPointError("non-finite validation MSE at epoch %d" % epoch)
            with torch.no_grad():
                val_obj = val + a.lambda_l1 * float(gp.l1())
            last_val, last_obj = val, val_obj
            if a.release_stable > 0 and gp.clamp_signed:
                B = support_2means(gp.W().detach().cpu().numpy()); sup_hist[epoch] = B
                if support_stable(B, sup_hist.pop(epoch - a.release_stable, None), a.release_jaccard):
                    release_at = epoch + 1
            sec = time.time() - t0
            epoch_times.append(sec)
            for key, crit in (("by_mse", val), ("by_obj", val_obj)):
                if crit < best[key]["crit"]:
                    best[key].update(crit=crit, epoch=epoch, val_mse=val, val_obj=val_obj, state=_cpu_state(gp, ode))
            if epoch == 1 or epoch % a.eval_every == 0 or epoch == a.epochs:
                with torch.no_grad():
                    au, ap = auroc_auprc(gp.W().detach().cpu().numpy(), d["A"])
                with open(log_path, "a", newline="") as f:
                    csv.writer(f).writerow([epoch, float(np.mean(losses)), val, val_obj, au, ap, round(sec, 3)]
                                           + (list(_w_stats(gp)) if a.log_w_stats else []))
            if epoch - best[sel_key]["epoch"] >= a.patience:
                break
            if a.max_minutes > 0 and (time.time() - t_start) > a.max_minutes * 60:
                break
    except Exception as e:  # noqa: BLE001 - a failed run must still write metrics.json
        failed, reason = True, "%s: %s" % (type(e).__name__, e)

    result: Dict = {"args": vars(a), "meta": meta, "N": int(N), "device": str(device),
                    "n_train_pairs": int(len(tr.dataset)), "failed": failed, "reason": reason,
                    "epochs_run": int(epoch), "sec_per_epoch": float(np.mean(epoch_times)) if epoch_times else 0.0,
                    "wall_sec": time.time() - t_start, "selection": a.select, "eval_errors": [],
                    "support_size": support_size, "release_epoch": release_at,
                    "test_idx": [int(i) for i in te_idx]}
    if a.gauge_scale == "ref" and not failed:
        result["coupling_rms_last"] = float(ode.vf.coupling_rms(ref_x))
    errors: List[str] = result["eval_errors"]
    try:
        last_state = _cpu_state(gp, ode)
        snaps = {"last": (last_state, epoch, last_val, last_obj)}
        for key in ("by_mse", "by_obj"):
            b = best[key]
            snaps[key] = (b["state"] or last_state, b["epoch"] or epoch, b["val_mse"], b["val_obj"])
        x_gauge = next(iter(tr))[0][:64].to(device) if len(tr.dataset) else None
        for name in ("last", "by_mse", "by_obj"):
            st, ep, vm, vo = snaps[name]
            _load_state(gp, ode, st)
            gp.eval()
            ode.eval()
            np.save(os.path.join(a.out, "W_%s.npy" % name), gp.W().detach().cpu().numpy())
            if ode.vf.Z is not None:
                np.save(os.path.join(a.out, "Z_%s.npy" % name), ode.vf.z_eff().detach().cpu().numpy())
            torch.save(st, os.path.join(a.out, "state_%s.pt" % name))
            snap = evaluate(gp, ode, d, te_idx, te, device, a.rollout_steps, a.cluster_k, x_gauge, errors, name)
            snap.update(epoch=int(ep), val_mse=vm, val_obj=vo)
            result[name] = snap
        result["best"] = dict(result[sel_key])
        result["best_epoch"] = result["best"]["epoch"]
        result["val_mse_best"] = result["best"]["val_mse"]
        result["val_obj_best"] = result["best"]["val_obj"]
        result["test_mse"] = result["best"]["test_mse"]
        shutil.copyfile(os.path.join(a.out, "W_%s.npy" % sel_key), os.path.join(a.out, "W_best.npy"))
        if ode.vf.Z is not None:
            shutil.copyfile(os.path.join(a.out, "Z_%s.npy" % sel_key), os.path.join(a.out, "Z_best.npy"))
    except Exception as e:  # noqa: BLE001
        result["failed"], result["reason"] = True, (reason + " | eval: %s: %s" % (type(e).__name__, e)).strip(" |")
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump(result, f, indent=1, default=float)
    return result


if __name__ == "__main__":
    r = main()
    print(json.dumps({k: r.get(k) for k in ("failed", "reason", "epochs_run", "best_epoch", "sec_per_epoch")}),
          json.dumps(r.get("best", {}).get("auroc")))
