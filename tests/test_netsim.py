import numpy as np
import scipy.io

from hetndl.gen.netsim import convert_sim
from hetndl.gen.kuramoto import load_dataset


def test_convert_sim_to_npz(tmp_path):
    N, S, T = 5, 3, 20
    rng = np.random.default_rng(0)
    net = np.zeros((S, N, N))
    net[:, 0, 1] = 0.4; net[:, 1, 2] = -0.3; net[:, 2, 2] = -0.5          # self-loop must be dropped
    mat = dict(Nnodes=np.array([[N]]), Nsubjects=np.array([[S]]), Ntimepoints=np.array([[T]]),
               ts=rng.normal(size=(S * T, N)), net=net)
    p = str(tmp_path / "sim1.mat"); scipy.io.savemat(p, mat)
    out = str(tmp_path / "netsim_sim1.npz")
    meta = convert_sim(p, out, sim_id=1)
    d = load_dataset(out)
    assert d["data"].shape == (S, T, N, 1) and d["data"].dtype == np.float32
    assert d["A"][0, 1] == 1 and d["A"][1, 2] == 1 and d["A"][2, 2] == 0 and d["A"].sum() == 2
    assert d["S"][1, 2] == -1 and d["S"][0, 1] == 1                       # sign kept from the DCM weights
    assert meta["system"] == "netsim" and meta["dt"] == 1.0 and meta["sim"] == 1 and meta["n_edges"] == 2
    assert np.isnan(d["omega"]).all() and (d["classes"] == -1).all()
    # per-node standardisation over all subjects and times
    x = d["data"][..., 0].reshape(-1, N)
    assert np.allclose(x.mean(0), 0, atol=1e-5) and np.allclose(x.std(0), 1, atol=1e-4)
