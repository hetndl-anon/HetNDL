"""Netsim (Smith et al., NeuroImage 2011): simulated fMRI on known small networks, the real-data benchmark of NOGI.

convert_sim turns sim<k>.mat into our npz: data (S, T, N, 1) standardised per node over all subjects and
times, A = binarised off-diagonal part of net[0] (the DCM connection matrix; its diagonal holds self-decay and
is dropped), S = sign(net[0]) on those edges, omega = nan, classes = -1 (no node labels), dt = 1 (one TR).
sim13 varies the topology across subjects and is skipped by convert_all, as in NOGI."""
import argparse
import json
import os
from typing import Dict

import numpy as np
import scipy.io

from .kuramoto import save_dataset

SKIP = {13}


def convert_sim(mat_path: str, out_path: str, sim_id: int) -> Dict:
    m = scipy.io.loadmat(mat_path)
    N, S, T = (int(m[k][0, 0]) for k in ("Nnodes", "Nsubjects", "Ntimepoints"))
    ts = np.asarray(m["ts"], dtype=np.float64).reshape(S, T, N)
    ts = (ts - ts.reshape(-1, N).mean(0)) / (ts.reshape(-1, N).std(0) + 1e-12)
    net0 = np.asarray(m["net"][0], dtype=np.float64)
    A = (net0 != 0).astype(np.float32)
    np.fill_diagonal(A, 0.0)
    Sg = (np.sign(net0) * A).astype(np.float32)
    meta = dict(system="netsim", dt=1.0, T=float(T), sim=int(sim_id), N=N, samples=S, n_edges=int(A.sum()),
                n_neg_edges=int((Sg < 0).sum()), source=os.path.basename(mat_path))
    save_dataset(out_path, dict(data=ts[..., None].astype(np.float32), A=A, S=Sg,
                                omega=np.full(N, np.nan, dtype=np.float32),
                                classes=np.full(N, -1, dtype=np.int64), meta=meta))
    return meta


def convert_all(raw_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for k in range(1, 29):
        if k in SKIP:
            continue
        p = os.path.join(raw_dir, "sim%d.mat" % k)
        if os.path.exists(p):
            print(json.dumps(convert_sim(p, os.path.join(out_dir, "netsim_sim%d.npz" % k), k)))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Netsim .mat -> HetNDL npz")
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    convert_all(a.raw, a.out)


if __name__ == "__main__":
    main()
