"""The reported snapshot of a run is the validation-MSE-selected one (`best`), never the last epoch and never anything
that looks at ground truth. hetndl.snapshot centralises that choice for every summary / read-out script."""
import json, os
import numpy as np

from hetndl.snapshot import metrics_of, state_path, array_path, SNAPSHOT


def _run(tmp_path, selection="val_mse"):
    d = str(tmp_path / "run"); os.makedirs(d)
    m = {"selection": selection, "best_epoch": 7,
         "best": {"auroc": 0.9, "epoch": 7}, "last": {"auroc": 0.6, "epoch": 600},
         "by_mse": {"auroc": 0.9, "epoch": 7}, "by_obj": {"auroc": 0.8, "epoch": 9}}
    json.dump(m, open(d + "/metrics.json", "w"))
    for k in ("last", "by_mse", "by_obj"):
        open(d + "/state_%s.pt" % k, "w").close(); np.save(d + "/W_%s.npy" % k, np.zeros(2))
    np.save(d + "/W_best.npy", np.ones(2))
    return d, m


def test_default_snapshot_is_val_mse_best(tmp_path):
    d, m = _run(tmp_path)
    assert SNAPSHOT == "best"
    assert metrics_of(m)["auroc"] == 0.9 and metrics_of(m)["epoch"] == 7
    assert state_path(d, m).endswith("state_by_mse.pt")
    assert array_path(d, "W", m).endswith("W_best.npy") and np.load(array_path(d, "W", m)).sum() == 2


def test_val_obj_selection_maps_to_by_obj_state(tmp_path):
    d, m = _run(tmp_path, selection="val_obj")
    assert state_path(d, m).endswith("state_by_obj.pt")


def test_last_snapshot_can_be_requested_explicitly(tmp_path):
    d, m = _run(tmp_path)
    assert metrics_of(m, "last")["auroc"] == 0.6
    assert state_path(d, m, "last").endswith("state_last.pt") and array_path(d, "Z", m, "last").endswith("Z_last.npy")


def test_metrics_without_best_falls_back_to_selection_key(tmp_path):
    d, m = _run(tmp_path); del m["best"]
    assert metrics_of(m)["auroc"] == 0.9
