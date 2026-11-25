"""Per-node symbolic distillation of the self-fields, stage 1 of 2 (hetndl env); stage 2 is pernode_distill.py (env with
PySR). Sect. 4.7 of the paper and Sect. S10 of the supplement.

One shared equation form per read-out class and state coordinate, with constants per node, on every benchmark; the
only benchmark-specific element is the documented Grid-118 grid (phase at the node's mean, frequency grid).
 Runs, grids  the distilled runs of the paper and the grids written by distill_export.py:
              experiments/distill_sel/*/grids.npz (MixRelax, MixOsc N=30 / N=60, Grid-118, Kuramoto-± p⁻ 0.5) and
              experiments/distill_glv_1106/*/grids.npz (GLV-±), paper seeds 1-3 (paper_seeds.py); the learned per-node
              self-fields F_node on those grids.
 Classes      the read-out classes stored there by distill_export.py (cluster), for every system from the same
              select_k read-out (Kuramoto-± and GLV-±: one class on all six runs).
This stage writes experiments/distill_pernode/<tag>/data.npz: grid, F_node, Z, labels, true class per label, and the
true per-node fields T_node (used for scoring only), and info.json. Run from the repository root with PYTHONPATH=."""
import glob, json, os, re, numpy as np
from hetndl.gen.kuramoto import load_dataset
from hetndl.gen.a1_mixed import self_term
from hetndl.gen.a2_mixed import self_field
from experiments.analysis.paper_seeds import PAPER_SEEDS

OUT = "experiments/distill_pernode"
NAMES = {"a1": ["x"], "a2": ["u", "v"], "powergrid": ["omega"], "kuramoto": ["theta"], "glv": ["x"]}


def main():
    paths = sorted(glob.glob("experiments/distill_sel/*/grids.npz")) + sorted(glob.glob("experiments/distill_glv_1106/*/grids.npz"))
    paths = [p for p in paths if int(re.search(r"_seed(\d+)", p).group(1)) in PAPER_SEEDS]
    for p in paths:
        dd = os.path.dirname(p); tag = os.path.basename(dd)
        info = json.load(open(os.path.join(dd, "info.json"))); sysn = info["system"]
        z = np.load(p); grid, Fn, Z = z["grid"], z["F_node"], z["Z"]; N = Fn.shape[0]
        a = json.load(open(info["run"] + "/metrics.json"))["args"]; d = load_dataset(a["data"]); meta = d["meta"]
        lab = np.unique(z["cluster"], return_inverse=True)[1]; to_true = z["cluster_to_true"]
        cls = np.asarray(d["classes"])
        # ---- truth, used only by the scores of stage 2
        if sysn == "a1":
            T = np.stack([self_term(grid[:, 0], np.full(len(grid), cls[i]))[:, None] for i in range(N)])
        elif sysn == "a2":
            T = np.stack([self_field(grid[:, 0], grid[:, 1], np.full(len(grid), cls[i])) for i in range(N)])
        elif sysn == "powergrid":
            M = np.where(cls == 1, meta["m_gen"], meta["m_load"]); D = float(meta["damping"]); om = grid[:, 0]
            T = np.stack([np.stack([om, (float(d["omega"][i]) - D * om) / M[i]], -1) for i in range(N)])
        elif sysn == "kuramoto":
            T = np.repeat(np.asarray(d["omega"], float)[:, None, None], len(grid), 1)
        else:
            T = z["F_true_node"]
        assert T.shape == Fn.shape, (tag, T.shape, Fn.shape)
        od = os.path.join(OUT, tag); os.makedirs(od, exist_ok=True)
        np.savez(os.path.join(od, "data.npz"), grid=grid, F_node=Fn, Z=Z, labels=lab, to_true=to_true, T_node=T)
        json.dump(dict(system=sysn, run=info["run"], names=NAMES[sysn]), open(os.path.join(od, "info.json"), "w"), indent=1)
        print("%-66s %-9s N %3d  grid %s  classes %s" % (tag, sysn, N, grid.shape, np.bincount(lab).tolist()), flush=True)


if __name__ == "__main__":
    main()
