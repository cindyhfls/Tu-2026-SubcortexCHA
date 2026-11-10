"""Build the HCP 7T cortical connectivity template (cortical seeds, Fig 7 comparison condition).

Searchlight (20 mm, onavg-ico32 centres) template of the connectivity of every cortical vertex to
the onavg-ico8 targets, fitted on the same 144 HCP subjects / four movie runs as
template_hcp_subcortical.py. The searchlight blocks are distance-weighted when the
template is aggregated (default; `--no-dist-weighted` uses uniform weights).

Usage:  python template_hcp_cortical.py            # both hemispheres
        python template_hcp_cortical.py --hemi r
Input:  data/HCP_YA/prep_oneregression/onavg-ico32/{l,r}-cerebrum  (00_cache_inputs/hcp_resample/hcp_prep.py)
Output: results/HCP/onavg-ico32/templates_distweighted/targets_onavg-ico8_{l,r}h.npy
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import neuroboros as nb
import numpy as np
from hyperalignment import searchlight_template
from utils.data_io_utils import get_HCP
from utils.hyperalignment_utils import calculate_connectivity
from utils.paths import DATA, RESULTS, resolve  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--hemi", nargs="+", default=["l", "r"], choices=["l", "r"])
parser.add_argument("--sids-file", default="config/HCP_movie_v1_complete.txt")
parser.add_argument("--prep-root", default=str(DATA / "HCP_YA" / "prep_oneregression"))
parser.add_argument("--searchlight-radius", type=float, default=20, help="mm")
parser.add_argument("--no-dist-weighted", action="store_true",
                    help="Uniform instead of distance-weighted searchlight aggregation")
parser.add_argument("--n-jobs", type=int, default=4)
args = parser.parse_args()

task = "MOVIE"
runs = [1, 2, 3, 4]
surface_space = "onavg-ico32"
targets = "onavg-ico8"

with open(resolve(args.sids_file)) as f:
    sids = f.read().splitlines()
prep_root = str(resolve(args.prep_root))
template_dir = RESULTS / "HCP" / surface_space / "templates_distweighted"
os.makedirs(template_dir, exist_ok=True)

mappings = {lr: nb.mapping(lr, surface_space, targets, mask=True) for lr in "lr"}
target_ts = get_HCP(prep_root, sids, ["l", "r"], task, runs, surface_space, xfms=mappings)

for hemi in args.hemi:
    tpl_fn = template_dir / f"targets_{targets}_{hemi}h.npy"
    if tpl_fn.exists():
        print(f"{tpl_fn} exists, skipping")
        continue
    seed_ts = get_HCP(prep_root, sids, [hemi], task, runs, surface_space)
    dm = np.stack([calculate_connectivity(target_ts[i], seed_ts[i]) for i in range(len(sids))], axis=0)
    sls, dists = nb.sls(hemi, args.searchlight_radius, space=surface_space,
                        center_space=surface_space, mask=True, return_dists=True)
    radius = args.searchlight_radius
    if args.no_dist_weighted:
        dists, radius = None, None
    nb.record(str(tpl_fn), searchlight_template)(dm, sls, dists, radius, n_jobs=args.n_jobs, tpl_kind="pca")
    print(f"Template saved: {tpl_fn}")
