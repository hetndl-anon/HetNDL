"""A coarser sampling interval for an existing dataset: every k-th stored state of the same trajectories, dt x k;
nothing else changes (graph, node parameters, number of trajectories, time span). The supplement uses it for the
sampling-interval check (<source>_sub2 and <source>_sub4 of Kuramoto-± with p⁻ = 0.5 and of MixRelax).

Usage: python -m hetndl.gen.subsample --src experiments/data/a1v2_N30_seed1.npz --k 2 \
           --out experiments/data/a1v2_N30_seed1_sub2.npz"""
import argparse
import os

import numpy as np

from .kuramoto import load_dataset, save_dataset


def subsample(d: dict, k: int, source: str) -> dict:
    """Every k-th state of d["data"] (S, T, N, D); meta records dt x k, the factor and the source dataset."""
    d = dict(d)
    d["data"] = np.ascontiguousarray(d["data"][:, ::k])
    d["meta"] = dict(d["meta"], dt=float(d["meta"]["dt"]) * k, subsample=int(k), source=source)
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description="Subsample the trajectories of a dataset in time")
    ap.add_argument("--src", type=str, required=True, help="source dataset (.npz written by a hetndl.gen generator)")
    ap.add_argument("--k", type=int, required=True, help="keep every k-th stored state (dt is multiplied by k)")
    ap.add_argument("--out", type=str, required=True)
    a = ap.parse_args(argv)
    assert a.k >= 1, a.k
    d = subsample(load_dataset(a.src), a.k, os.path.splitext(os.path.basename(a.src))[0])
    save_dataset(a.out, d)
    print("saved", a.out, d["data"].shape, "dt", d["meta"]["dt"])


if __name__ == "__main__":
    main()
