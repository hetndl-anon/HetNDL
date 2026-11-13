"""Print the run directories that the default procedure selected (experiments/runs/sel/selected.json, written by
experiments/analysis/sel_tables.py) for the main-table cells (not the noise / trajectory-count variants) whose dataset
name matches a regular expression.
Usage: python experiments/reproduce/selected_runs.py 'a2_N(30|60)_seed[123]$'"""
import json
import re
import sys

sel = json.load(open("experiments/runs/sel/selected.json"))
pat = re.compile(sys.argv[1])
main_cell = re.compile(r"^[^/]+/add_z_signed_diff_lz0_seed\d+$")          # main-table cells, not robustness variants
runs = [v["run"] for k, v in sorted(sel.items()) if main_cell.match(k) and pat.search(k.split("/")[0])]
print(" ".join(runs))
