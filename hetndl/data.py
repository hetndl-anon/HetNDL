"""Trajectory-level data pipeline for single-step training.

Noise (if any) is added to the full observed series once, so inputs and targets
are both noisy. Splits are by trajectory, never by (x_t, x_{t+1}) pair.
"""
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def add_noise_snr(data: np.ndarray, snr_db: Optional[float], rng: np.random.Generator) -> np.ndarray:
    if snr_db is None:
        return data.copy()
    noisy = data.copy()
    S, _, N, _ = data.shape
    lin = 10 ** (snr_db / 10.0)
    for s in range(S):
        for n in range(N):
            sig = data[s, :, n, :]
            noise_power = float(np.mean(sig ** 2)) / lin
            noisy[s, :, n, :] = sig + rng.normal(0.0, np.sqrt(noise_power), sig.shape)
    return noisy.astype(np.float32)


def add_noise(data: np.ndarray, rng: np.random.Generator, snr_db: Optional[float] = None,
              noise_std: Optional[float] = None) -> np.ndarray:
    """Observation noise, added once to the whole series. snr_db: per-node SNR relative to the signal power
    (meaningless for unbounded phases); noise_std: absolute Gaussian std in state units (use this for phases)."""
    if noise_std is not None:
        if snr_db is not None:
            raise ValueError("give snr_db or noise_std, not both")
        return (data + rng.normal(0.0, float(noise_std), data.shape)).astype(np.float32)
    return add_noise_snr(data, snr_db, rng)


def smooth_series(obs: np.ndarray, window: Optional[int], polyorder: int = 3) -> np.ndarray:
    """Savitzky-Golay smoothing along time (axis 1), per trajectory, node and state dimension: a truth-free
    preprocessing for noisy observations (the same filter the tuned ARNI baseline receives). window None or 0 =
    identity; an even window is widened to the next odd one."""
    if not window:
        return obs
    window = int(window) + (1 - int(window) % 2)
    if window <= polyorder:
        raise ValueError("smooth_window must exceed the polynomial order %d" % polyorder)
    from scipy.signal import savgol_filter
    return savgol_filter(obs, window, polyorder, axis=1).astype(np.float32)


def phase_features(obs: np.ndarray, feat: str = "raw") -> np.ndarray:
    """Observed states as a model input: 'raw' unchanged; 'trig' every coordinate as (cos x, sin x), the periodic feature
    rule for unwrapped phases (HetNDL / NOGI feat='trig', MTGNN on Kuramoto); 'nri' / 'nri_omega' (the input
    check on the NRI-paper Kuramoto only) the increment x_{t+1} - x_t and sin x_t, as in the Kuramoto experiments of the NRI paper ('nri_omega': the caller
    appends the node frequency). Applied after any observation noise."""
    if feat == "raw":
        return obs
    if feat in ("nri", "nri_omega"):        # NRI-paper Kuramoto inputs (forward difference, sine); one step shorter
        return np.concatenate([np.diff(obs, axis=1), np.sin(obs[:, :-1])], -1).astype(obs.dtype)
    if feat != "trig":
        raise ValueError(feat)
    return np.concatenate([np.cos(obs), np.sin(obs)], -1).astype(obs.dtype)


def select_trajectories(n_total: int, n_traj: Optional[int], seed: int) -> np.ndarray:
    """Indices of the trajectories used at all (data-size experiments): the first n_traj of a fixed permutation,
    so that a smaller budget is a subset of a larger one for the same seed. None = every trajectory, in order."""
    if n_traj is None or n_traj >= n_total:
        return np.arange(n_total)
    if n_traj < 3:
        raise ValueError("need at least 3 trajectories for a train/val/test split")
    return np.sort(np.random.default_rng(seed + 777).permutation(n_total)[:n_traj])


def trajectory_split(n_traj: int, seed: int, ratios=(0.6, 0.2, 0.2)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    perm = np.random.default_rng(seed).permutation(n_traj)
    n_val = max(1, int(round(ratios[1] * n_traj)))
    n_test = max(1, int(round(ratios[2] * n_traj)))
    n_train = n_traj - n_val - n_test
    if n_train < 1:
        raise ValueError("need at least 3 trajectories, got %d" % n_traj)
    return perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]


def make_pairs(data: np.ndarray, traj_idx: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = [data[s, :-1] for s in traj_idx]
    ys = [data[s, 1:] for s in traj_idx]
    x = torch.from_numpy(np.concatenate(xs, 0).astype(np.float32))
    y = torch.from_numpy(np.concatenate(ys, 0).astype(np.float32))
    return x, y


def _loader(x, y, batch, shuffle, num_workers):
    return DataLoader(TensorDataset(x, y), batch_size=batch, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=False, drop_last=False)


def build_loaders(data: np.ndarray, seed: int, batch: int, snr_db: Optional[float], num_workers: int = 0,
                  split: str = "traj", n_traj: Optional[int] = None, noise_std: Optional[float] = None,
                  smooth_window: Optional[int] = None):
    """split='traj' (default): hold out whole trajectories. split='pairs': the NOGI reference
    protocol, a random 6:2:2 split of (x_t, x_{t+1}) pairs; test_idx is then every trajectory.
    n_traj limits the trajectories used (see select_trajectories); the returned test indices always index the
    ORIGINAL array. Noise is added to the kept trajectories only; smooth_window then smooths the observed series."""
    keep = select_trajectories(data.shape[0], n_traj, seed)
    rng = np.random.default_rng(seed + 12345)
    obs = smooth_series(add_noise(data[keep], rng, snr_db, noise_std), smooth_window)
    n_kept = len(keep)
    if split == "pairs":
        x, y = make_pairs(obs, np.arange(n_kept))
        n = x.shape[0]
        perm = torch.from_numpy(np.random.default_rng(seed).permutation(n))
        n_tr, n_va = int(0.6 * n), int(0.2 * n)
        parts = (perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:])
        tr_l, va_l, te_l = [_loader(x[i], y[i], batch, k == 0, num_workers) for k, i in enumerate(parts)]
        return tr_l, va_l, te_l, keep
    if split != "traj":
        raise ValueError("unknown split: %s" % split)
    tr, va, te = trajectory_split(n_kept, seed)
    loaders = []
    for idx, shuffle in ((tr, True), (va, False), (te, False)):
        x, y = make_pairs(obs, idx)
        loaders.append(_loader(x, y, batch, shuffle, num_workers))
    return loaders[0], loaders[1], loaders[2], keep[te]
