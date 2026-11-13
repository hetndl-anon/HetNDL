"""Cost scaling with the number of nodes: time of one training iteration and peak GPU memory for HetNDL
(signed member) and NOGI with their Kuramoto settings (copied from the main-table runs: RK4 over one sampling step,
trigonometric features, hidden width 16; HetNDL d_z 2, diffusive coupling; NOGI d_z 1), N in {100, 200, 400, 800}.
One iteration = forward over one step for a batch of state pairs, MSE + lambda_1 |W|_1, backward through the adjoint,
Adam step — exactly the training step of hetndl/train.py; random phases (the cost does not depend on the values).
Per epoch = ceil(11988 / batch) iterations (Kuramoto protocol: 12 training trajectories x 999 pairs). Batch 256 as in
the protocol; on out-of-memory the batch is halved until it fits, and this is reported. Every configuration runs alone
on the given device. Output runs/scale/bench.json. Usage: PYTHONPATH=. python experiments/analysis/scale_bench.py cuda:0"""
import json, math, os, sys, time
import torch
import torch.nn.functional as Fnn
from hetndl.model import build
R = "experiments/runs"; PAIRS = 12 * 999
REF = {"HetNDL": "sel_dz2/nowu/kura_ER100_K2_s0.3_n0.5/add_z_signed_diff_lz0_seed1", "NOGI": "motiv_trig/kura_ER100_K2_s0.3_n0.5/orig_seed1"}


def one(args, N, B, dev, warm=2, reps=5):
    a = args
    gp, ode = build(a["model"], N, 1, a["hidden"], a["d_z"], 0.01, a.get("rtol", 1e-3), a.get("atol", 1e-3), a["method"],
                    bool(a["mask_diag"]), feat=a.get("feat", "raw"), coupling=a.get("coupling", "free"),
                    z_norm=a.get("z_norm", "none"), step_size=a.get("step_size"))
    gp.to(dev); ode.to(dev)
    opt = torch.optim.Adam([{"params": gp.parameters(), "lr": a["lr_w"]}, {"params": ode.parameters(), "lr": a["lr_f"]}])
    x = (2 * math.pi * torch.rand(B, N, 1)).to(dev); y = (2 * math.pi * torch.rand(B, N, 1)).to(dev)
    torch.cuda.reset_peak_memory_stats(dev); ts = []
    for k in range(warm + reps):
        torch.cuda.synchronize(dev); t = time.time()
        opt.zero_grad(); W = gp.W(); loss = Fnn.mse_loss(ode(x, W), y) + a["lambda_l1"] * gp.l1(); loss.backward(); opt.step()
        torch.cuda.synchronize(dev); ts.append(time.time() - t)
    return sum(ts[warm:]) / reps, torch.cuda.max_memory_allocated(dev) / 2 ** 30


def main(dev):
    out = {}
    for name, ref in REF.items():
        a = json.load(open(os.path.join(R, ref, "metrics.json")))["args"]
        for N in (100, 200, 400, 800):
            B = int(a["batch"])
            while B >= 8:
                try:
                    t, mem = one(a, N, B, dev); break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache(); B //= 2
            else:
                t, mem = float("nan"), float("nan")
            it = math.ceil(PAIRS / B)
            out["%s/N%d" % (name, N)] = dict(batch=B, sec_per_iter=t, iters_per_epoch=it, sec_per_epoch=t * it, peak_gib=mem,
                                            protocol_epochs=int(a["epochs"]), est_train_h=t * it * int(a["epochs"]) / 3600)
            print("%-6s N %4d batch %3d  %.3f s/iter  %.1f s/epoch  peak %.1f GiB  ~%.1f h for %d epochs" % (
                name, N, B, t, t * it, mem, t * it * int(a["epochs"]) / 3600, int(a["epochs"])), flush=True)
            torch.cuda.empty_cache()
    os.makedirs(os.path.join(R, "scale"), exist_ok=True)
    json.dump(dict(device=torch.cuda.get_device_name(dev), results=out), open(os.path.join(R, "scale", "bench.json"), "w"), indent=1)


if __name__ == "__main__":
    main(torch.device(sys.argv[1]))
