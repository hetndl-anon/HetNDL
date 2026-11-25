"""Which snapshot of a trained run is *reported*.

Reporting protocol: the validation-MSE-selected snapshot (`best` in metrics.json, saved
during training without looking at ground truth), never the last epoch. Every summary / read-out script goes through
these helpers; set HETNDL_SNAPSHOT=last to reproduce the pre-protocol tables."""
import os

SNAPSHOT = os.environ.get("HETNDL_SNAPSHOT", "best")

_KEY = {"val_mse": "by_mse", "val_obj": "by_obj"}


def _select_key(m):
    return _KEY.get(m.get("selection", "val_mse"), "by_mse")


def metrics_of(m, which=SNAPSHOT):
    """Metric dict of the reported snapshot. `best` falls back to the selection key when a run lacks a `best` block."""
    if which == "best":
        return m["best"] if "best" in m else m[_select_key(m)]
    return m[which]


def state_path(run, m, which=SNAPSHOT):
    """Path of the state_*.pt file holding the reported snapshot (no state_best.pt is written, so map best → key)."""
    key = _select_key(m) if which == "best" else which
    return os.path.join(run, "state_%s.pt" % key)


def array_path(run, name, m, which=SNAPSHOT):
    """Path of W_*.npy / Z_*.npy for the reported snapshot (W_best.npy / Z_best.npy are written by train.py)."""
    return os.path.join(run, "%s_%s.npy" % (name, which))
