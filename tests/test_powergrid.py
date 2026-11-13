import numpy as np

from hetndl.gen.powergrid import parse_matpower, build_grid, simulate_grid
from hetndl.gen.kuramoto import save_dataset, load_dataset

SNIPPET = """function mpc = toy
mpc.version = '2';
mpc.baseMVA = 100;
%% bus data
mpc.bus = [
	1	3	0	0	0	0	1	1.0	0	138	1	1.06	0.94;
	2	1	50	10	0	0	1	1.0	0	138	1	1.06	0.94;
	5	1	30	5	0	0	1	1.0	0	138	1	1.06	0.94;
];
%% generator data
mpc.gen = [
	1	80	0	50	-50	1.0	100	1	100	0;
];
%% branch data
mpc.branch = [
	1	2	0.01	0.1	0	0	0	0	0	0	1	-360	360;
	2	5	0.02	0.2	0	0	0	0	0	0	1	-360	360;
	2	5	0.02	0.2	0	0	0	0	0	0	1	-360	360;
	1	5	0.01	0.5	0	0	0	0	0	0	1	-360	360;
];
"""


def test_parse_and_build_grid(tmp_path):
    p = tmp_path / "toy.m"; p.write_text(SNIPPET)
    case = parse_matpower(str(p))
    assert case["bus"].shape == (3, 13) and case["branch"].shape == (4, 13) and case["gen"].shape == (1, 10)
    g = build_grid(case)
    K, P, is_gen = g["K"], g["P"], g["is_gen"]
    assert K.shape == (3, 3) and np.allclose(K, K.T) and np.diag(K).sum() == 0
    assert np.isclose(K[0, 1], 10.0) and np.isclose(K[1, 2], 10.0) and np.isclose(K[0, 2], 2.0)   # parallel lines summed: 5 + 5
    assert np.allclose(P, [0.8, -0.5, -0.3]) and is_gen.tolist() == [True, False, False]       # p.u., balanced
    assert g["bus_ids"] == [1, 2, 5]


def test_simulate_grid_rate_and_sync(tmp_path):
    p = tmp_path / "toy.m"; p.write_text(SNIPPET)
    g = build_grid(parse_matpower(str(p)))
    d = simulate_grid(g, kappa=1.0, m_gen=1.0, m_load=0.1, damping=0.5, samples=3, T=8.0, dt=0.05, seed=0)
    x = d["data"]
    assert x.shape == (3, 160, 3, 2) and np.isfinite(x).all()
    assert np.abs(x[:, -1, :, 1]).max() < 0.05                       # frequencies settle (synchronous state)
    assert d["classes"].tolist() == [1, 0, 0] and np.allclose(d["omega"], g["P"])
    assert d["meta"]["system"] == "powergrid" and d["A"].sum() == 6 and (d["S"] == d["A"]).all()
    # the rate at a given state follows the swing equation with W_true[j, i] = kappa K_ij / M_i
    Wt = d["W_true"]
    M = np.where(g["is_gen"], 1.0, 0.1)
    assert np.allclose(Wt, 1.0 * g["K"] / M[None, :])
    x0 = np.array([[0.3, 0.1], [0.0, -0.2], [-0.2, 0.0]])
    d2 = simulate_grid(g, kappa=1.0, m_gen=1.0, m_load=0.1, damping=0.5, samples=1, T=0.002, dt=0.001, seed=0, x0=x0)
    rate = (d2["data"][0, 1] - d2["data"][0, 0]) / 0.001
    th, om = x0[:, 0], x0[:, 1]
    coup = (g["K"] * np.sin(th[None, :] - th[:, None])).sum(1)    # sum_j K_ij sin(theta_j - theta_i)
    expect = np.stack([om, (g["P"] - 0.5 * om + 1.0 * coup) / M], -1)
    assert np.allclose(rate, expect, atol=0.05, rtol=0.01)     # forward difference of a fast (M = 0.1) load
    out = str(tmp_path / "g.npz"); save_dataset(out, d)
    back = load_dataset(out)
    assert np.allclose(back["W_true"], Wt) and back["meta"]["system"] == "powergrid"
