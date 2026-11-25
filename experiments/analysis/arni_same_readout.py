"""Does ARNI separate the node mechanisms when its learned functions go through EXACTLY the read-out of HetNDL?
arni_mechanism.py clusters ARNI's per-node basis coefficients (each coefficient standardised separately), which mixes
function quality with the feature representation. Here ARNI's fitted functions are sampled on the same grids and clustered
with the same pipeline as HetNDL's self-fields. Fixed before the first run:
  ARNI fit   the "full" joint least-squares fit of the main-table ARNI run (runs/arni/<ds>/full_seed<k>), refitted with its
             stored arguments by experiments/analysis/rollout_baselines.arni_fit (same rows, basis, ridge); the refit is
             checked against the stored self_coef.npy (assert).
  Two curves per node, both reported:
   (A) self block       Theta_self(x) a_i: ARNI's own self term (its node "self-function"); its pair basis is not
                        diffusive, so this term also carries the x_i-part of the incoming coupling (Lemma 1).
   (B) self-field       ARNI's whole fitted vector field of node i with every candidate source at the node's state,
                        x_j = x_i for all j: the same quantity as HetNDL's self-field ("the node's rate of change when the
                        neighbours share its state"). This is the like-for-like comparison and the number the paper
                        quotes; (A) goes to the supplement.
  Grids and clustering: identical to the paper's read-out (readout_split_check.readout with all trajectories):
   MixRelax / MixRelax (BA)  64 points over the 5-95 % range of all states, select_k(curves=True);
   MixOsc N=30 / N=60        16 x 16 grid over the 5-95 % range of (u, v), both field components, select_k(curves=True);
   Grid-118                  slope in omega of the frequency component at the node's mean phase, omega on 33 points over
                             its 2-98 % range, select_k on the slope.
  Data seeds 1-3 (MixOsc N=30: experiments/analysis/paper_seeds.PAPER_SEEDS). HetNDL's own read-out is recomputed with
  the same code as a check (it must equal the "all" column of readout_split_check.py).
Reading: the paper reports ARNI's ARI under this read-out (Sect. 5.3 / Table 3), whatever it is; the coefficient
clustering stays in the supplement next to it. Truth (classes) only in the ARI.
Run from the project root with PYTHONPATH=. (-> results/supplement/arni_same_readout.txt)."""
import os, numpy as np
from sklearn.metrics import adjusted_rand_score
from hetndl.baselines.arni import self_basis, pair_basis
from hetndl.edge_readout import load_run
from hetndl.gen.kuramoto import load_dataset
from hetndl.readout import select_k
from experiments.analysis.baseline_table import B, PAPER_NAME, sel, sel_key
from experiments.analysis.readout_split_check import readout
from experiments.analysis.rollout_baselines import arni_fit
SETTINGS = ("MixRelax", "MixRelax (BA)", "MixOsc N=30", "MixOsc N=60", "Grid-118")


class ArniField:
    """Duck-types the parts of HetNDL's vector field that readout() uses: phi, F, z_eff, N. F(cat[h, z]) returns node i's
    curve, where node i is encoded in the 'latent' (a one-hot index)."""
    def __init__(self, basis, fits, d, mode):
        self.basis, self.fits, self.N, self.d, self.mode = basis, fits, len(fits), d, mode

    def phi(self, x):
        return x

    def z_eff(self):
        import torch
        return torch.eye(self.N)

    def F(self, hz):
        import torch
        h = hz[:, :self.d].double().numpy(); i = int(hz[0, self.d:].argmax())
        cand, beta, off = self.fits[i]; S = self_basis(h, self.basis); m_s = S.shape[1]
        if self.mode == "A":
            out = S @ beta[:m_s] - off
        else:
            out = np.concatenate([S] + [pair_basis(h, h, self.basis)] * len(cand), 1) @ beta - off
        return torch.tensor(out, dtype=torch.float32)


def coef_readout(C, cls):
    K, lab, _ = select_k(C, k_max=5); return K, adjusted_rand_score(cls, lab)


def main():
    print("ARNI functions through HetNDL's read-out (same grids, centring, global scale, PCA, GMM-BIC); classes: K and ARI")
    print("%-14s %4s | %-12s | %-12s | %-12s | %-12s" % ("setting", "seed", "HetNDL", "ARNI coef.", "ARNI (A) self", "ARNI (B) field"))
    res = {}
    for lab, cells in B:
        name = PAPER_NAME[lab]
        if name not in SETTINGS:
            continue
        for ds, s in cells:
            m, d, vf, W = load_run(sel[sel_key(lab, ds, s)]["run"]); X = np.asarray(d["data"], np.float32)
            system = d["meta"]["system"]; cls = d["classes"]
            Kh, lh, _ = readout(vf, X, system)
            run = os.path.join("experiments/runs/arni", ds, "full_seed%d" % s)
            basis, fits, _, _, _ = arni_fit(run, load_dataset("experiments/data/%s.npz" % ds))
            C = np.load(os.path.join(run, "self_coef.npy"))
            m_s = self_basis(X[0, :1, 0].astype(np.float64), basis).shape[1]
            Cre = []
            for cand, beta, off in fits:                     # as hetndl.baselines.arni._full_node: constant row minus mu @ beta
                sc = beta[:m_s].copy(); sc[0] -= off; Cre.append(sc.T.ravel())
            Cre = np.array(Cre)
            assert np.allclose(Cre, C, rtol=1e-6, atol=1e-8), (run, np.abs(Cre - C).max())
            Kc, ac = coef_readout(C, cls)
            out = [(Kh, adjusted_rand_score(cls, lh)), (Kc, ac)]
            for mode in ("A", "B"):
                K, l, _ = readout(ArniField(basis, fits, X.shape[-1], mode), X, system); out.append((K, adjusted_rand_score(cls, l)))
            res.setdefault(name, []).append(out)
            print("%-14s %4d | %s" % (name, s, " | ".join("K %d ARI %.3f" % o for o in out)))
    print("\nmeans over data seeds (minimum in parentheses)")
    print("%-14s | %-14s | %-14s | %-14s | %-14s" % ("setting", "HetNDL", "ARNI coef.", "ARNI (A) self", "ARNI (B) field"))
    for name, rows in res.items():
        a = np.array([[o[1] for o in r] for r in rows])
        print("%-14s | %s" % (name, " | ".join("%.3f (%.3f)" % (a[:, k].mean(), a[:, k].min()) for k in range(4))))


if __name__ == "__main__":
    main()
