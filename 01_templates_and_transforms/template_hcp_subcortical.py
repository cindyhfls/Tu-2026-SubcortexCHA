"""Build the HCP 7T connectivity template for the subcortical Tian atlases (Fig 7).

Same procedure as template_budapest_subcortical.py (PCA + Procrustes
`searchlight_template`, Methods "Connectivity hyperalignment (CHA) procedure"), but fitted on the
144 HCP subjects with complete movie data (config/HCP_movie_v1_complete.txt), all four 7T movie
runs, seeds in the 7T Tian atlas (mni-1pt6mm) and cortical targets on onavg-ico8. Each Tian
parcel is one searchlight.

Usage:
    python template_hcp_subcortical.py                       # TianS1..TianS4
    python template_hcp_subcortical.py --atlas TianS1 --n-jobs 8

Input:  data/HCP_YA/prep_oneregression/{mni-1pt6mm/Tian_Subcortex,onavg-ico32/{l,r}-cerebrum}
        (made by 00_cache_inputs/hcp_resample/hcp_prep.py)
Output: results/HCP/mni-1pt6mm/templates_<atlas>_wholebrain/targets_onavg-ico8_Tian_Subcortex.npy
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import neuroboros as nb
import numpy as np
from hyperalignment import searchlight_template
from utils.data_io_utils import get_Tian_sls
from utils.data_io_utils import get_HCP
from utils.hyperalignment_utils import calculate_connectivity
from utils.paths import DATA, RESULTS, resolve  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--atlas", nargs="+", default=["TianS1", "TianS2", "TianS3", "TianS4"],
                    choices=["TianS1", "TianS2", "TianS3", "TianS4"])
parser.add_argument("--sids-file", default="config/HCP_movie_v1_complete.txt",
                    help="Template subjects (144 HCP subjects with all four movie runs)")
parser.add_argument("--prep-root", default=str(DATA / "HCP_YA" / "prep_oneregression"))
parser.add_argument("--n-jobs", type=int, default=4)
args = parser.parse_args()

task = "MOVIE"
runs = [1, 2, 3, 4]
volume_space = "mni-1pt6mm"
surface_space = "onavg-ico32"
targets = "onavg-ico8"
roi = "Tian_Subcortex"

with open(resolve(args.sids_file)) as f:
    sids = f.read().splitlines()
prep_root = str(resolve(args.prep_root))

seed_ts = get_HCP(prep_root, sids, [roi], task, runs, volume_space)
mappings = {lr: nb.mapping(lr, surface_space, targets, mask=True) for lr in "lr"}
target_ts = get_HCP(prep_root, sids, ["l", "r"], task, runs, surface_space, xfms=mappings)
dm = np.stack([calculate_connectivity(target_ts[i], seed_ts[i]) for i in range(len(sids))], axis=0)

for atlas in args.atlas:
    template_dir = RESULTS / "HCP" / volume_space / f"templates_{atlas}_wholebrain"
    os.makedirs(template_dir, exist_ok=True)
    tpl_fn = template_dir / f"targets_{targets}_{roi}.npy"
    if tpl_fn.exists():
        print(f"{tpl_fn} exists, skipping")
        continue
    sls = get_Tian_sls(atlas, is_7T=True)
    nb.record(str(tpl_fn), searchlight_template)(dm, sls, None, None, n_jobs=args.n_jobs, tpl_kind="pca")
    print(f"Template saved: {tpl_fn}")
