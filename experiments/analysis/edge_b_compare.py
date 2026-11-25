"""Design option B vs the old endpoint-latent coupling vs a shared G. Per cell and coupling kind the default
procedure picks the sign member (non-negative / plain signed) by validation MSE; then the coupling kind itself is
picked by validation MSE between shared G and option B (truth-free); the nested choice equals the minimum over all four
candidates, printed per cell. Reported: val MSE, topology (AUROC / AUPRC), sign accuracy on true edges after the model's own gauge (per edge for
option B; hetndl.edge_readout.oriented_sign_accuracy),
learned-edge precision, edge-type read-out (hetndl.edge_readout: truth-free K per edge and per source, ARI), the
self-field read-out (a2_fields.py, truth-free K, ARI) and the 100-step rollout NRMSE.
A3 (edge types set by the source class; N = 30 seeds 1-5 and N = 60 seeds 1-3, the cells outside the main manifest
from experiments/launch/edge_b_ext.py) and the no-harm checks A2 N30 and signed Kuramoto p- 0.5 (one coupling form).
Run from the project root with PYTHONPATH=. (hetndl env)."""
import json, os, re, subprocess, sys, numpy as np
from hetndl.edge_readout import edge_type_readout, oriented_sign_accuracy
from hetndl.snapshot import metrics_of
from experiments.analysis.paper_seeds import PAPER_SEEDS
R = "experiments/runs"; PY = sys.executable
man = json.load(open(os.path.join(R, "sel", "manifest.json")))


def met(run):
    p = os.path.join(run, "metrics.json")
    if not os.path.exists(p):
        return None
    m = json.load(open(p)); return None if m.get("failed") else m


def pick(runs):
    c = [(met(r)["val_mse_best"], r) for r in runs if met(r) is not None]
    return min(c)[1] if len(c) == len(runs) else None


def fields(run):
    out = subprocess.run([PY, "experiments/analysis/a2_fields.py", run], capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=".")).stdout
    g = re.search(r"K_sel (\d+) ARI\(K_sel\) (-?[0-9.]+)", out)
    return (int(g.group(1)), float(g.group(2))) if g else (None, float("nan"))


for ds_tpl, seeds, has_fields in (("a3_N30_seed{s}", PAPER_SEEDS, True), ("a3_N60_seed{s}", (1, 2, 3), True), ("a2_N30_seed{s}", (1, 2, 3), True),
                                  ("kura_ER100_K2_s0.3_n0.5{g}", (1, 2, 3), False)):
    print("==", ds_tpl.split("{")[0])
    for s in seeds:
        ds = ds_tpl.format(s=s, g="" if s == 1 else "_g%d" % s)
        cell = "%s/add_z_signed_diff_lz0_seed%d" % (ds, s)
        shared = [man[cell]["nonneg"], man[cell]["nowu"]] if cell in man else \
            [os.path.join(R, "edge_b", v, ds, "%s_diff_lz0_seed%d" % (mo, s)) for v, mo in (("nonneg", "add_z"), ("nowu", "add_z_signed"))]  # edge_b_ext.py
        cand = {"shared G": shared,
                "edge_z": [os.path.join(R, "edge_b", v, ds, "%s_diff_lz0_ez_seed%d" % (mo, s)) for v, mo in (("nonneg", "add_z"), ("nowu", "add_z_signed"))],
                "option B": [os.path.join(R, "edge_b", v, ds, "%s_diff_lz0_ee2_seed%d" % (mo, s)) for v, mo in (("nonneg", "add_z"), ("nowu", "add_z_signed"))]}
        chosen = {}
        for kind, runs in cand.items():
            if not runs or not all(os.path.exists(r) for r in runs):
                continue
            r = pick(runs)
            if r is None:
                print("  s%d %-9s pending" % (s, kind)); continue
            chosen[kind] = r; m = met(r); b = metrics_of(m)
            e = edge_type_readout(r) if kind != "shared G" or ds.startswith("a3") else {}
            fk, fa = fields(r) if has_fields else (None, float("nan"))
            sg = oriented_sign_accuracy(r)
            print("  s%d %-9s %-6s val %.2e | AUROC %.3f AUPRC %.3f sign %.3f | learned edges %s/%s correct | edge types: per-edge K %s ARI %s, per-source K %s ARI %s | fields K %s ARI %.3f | rollout %.3f" % (
                s, kind, "nonneg" if "/nonneg/" in r or "add_z_diff" in os.path.basename(r) else "signed", m["val_mse_best"], b["auroc"], b["auprc"], sg,
                e.get("n_correct", "-"), e.get("n_learned", "-"), e.get("K_edge", "-"), "%.3f" % e["ari_edge"] if "ari_edge" in e else "-",
                e.get("K_source", "-"), "%.3f" % e["ari_source_edges"] if "ari_source_edges" in e else "-", fk, fa, b["rollout_nrmse"]))
        if "shared G" in chosen and "option B" in chosen:
            w = min(("shared G", "option B"), key=lambda k: met(chosen[k])["val_mse_best"])
            print("  s%d default-procedure choice between shared G and option B (val MSE): %s" % (s, w))
            four = {(k, mem): met(r)["val_mse_best"] for k in ("shared G", "option B") for mem, r in zip(("nonneg", "signed"), cand[k]) if met(r)}
            best4 = min(four, key=four.get)
            print("  s%d four candidates (val MSE): %s -> %s %s; val(shared G) / val(option B) = %.2f" % (
                s, ", ".join("%s/%s %.2e" % (k, mem, v) for (k, mem), v in four.items()), best4[0], best4[1],
                met(chosen["shared G"])["val_mse_best"] / met(chosen["option B"])["val_mse_best"]))
