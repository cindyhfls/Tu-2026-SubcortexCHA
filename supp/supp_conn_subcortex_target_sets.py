"""Local-target ablation for subcortical connectivity ISC (data for Supp Fig 5).

Connectivity to the whole aseg atlas has the seed nucleus itself in its target set. Under AA
(no transform) the leave-one-out ISC is recomputed with progressively stricter target sets:
  exclude_own                  drop the seed ROI's own voxels from the targets
  contralateral                keep only the opposite hemisphere (also drops every structure adjacent
                               to the seed; every seed of a hemisphere gets the same targets)
  contralateral_no_homologue   as above, also dropping the seed's mirror structure
                               (l-amygdala <-> r-amygdala)
The 'all targets' case is the ordinary AA connectivity_targets-aseg ISC and is read from the 06_
summary by supp_plot_ablation_contralateral.py. AA only: with no transform the comparison side is the
reference side, so connectivity is computed once per ROI. Targets are raw/untransformed.

Usage:  python supp_conn_subcortex_target_sets.py --targets-mode exclude_own   # and the two others
Input:  data/cache/Raiders_raiders/run34/{sids.npy,ts_aseg_Subcortex_float32.npy}  (cache_subcortical_raiders.py)
Output: results/Raiders/mni-2mm_1step_linear_overlap/conn_subcortex_target_sets/results_<mode>.csv
        (+ args_<mode>.json)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import CACHE, RESULTS, resolve  # noqa: E402

import argparse
import json
import os

import numpy as np

from utils.data_io_utils import get_aseg_sls, return_aseg_labels
from utils.isc_utils import FisherAvg, conn_features_all_subjects, loo_corrsim_dot_xy

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="Raiders")
parser.add_argument("--task", type=str, default="raiders")
parser.add_argument("--run-tag", type=str, default="run34")
parser.add_argument("--targets-mode", type=str, default="exclude_own",
                    choices=["exclude_own", "contralateral", "contralateral_no_homologue"])
parser.add_argument("--out", type=str, default=None,
                    help="default: results/<dataset>/mni-2mm_1step_linear_overlap/conn_subcortex_target_sets/results_<mode>.csv")
args = parser.parse_args()
if args.out is None:
    outdir = RESULTS / args.dataset / 'mni-2mm_1step_linear_overlap' / 'conn_subcortex_target_sets'
    args.out = outdir / f'results_{args.targets_mode}.csv'
args.out = resolve(args.out)
args.out.parent.mkdir(parents=True, exist_ok=True)

cache = CACHE / f"{args.dataset}_{args.task}" / args.run_tag
rois = return_aseg_labels(atlas_name='mni-2mm')
sls = get_aseg_sls()
sids = np.load(cache / 'sids.npy').astype(str)
N = len(sids)
raw = np.load(cache / 'ts_aseg_Subcortex_float32.npy', mmap_mode='r')
Vfull = raw.shape[2]
print(f"AA | N={N} | Vfull={Vfull} | {len(rois)} ROIs | targets_mode={args.targets_mode}")

rows = ["condition,roi,nvox,targets_mode,n_targets,isc"]
for iroi, roi in enumerate(rois):
    sl = sls[iroi]
    if args.targets_mode == "exclude_own":
        keep = np.ones(Vfull, bool)
        keep[sl] = False
    else:
        other = 'r-' if roi.startswith('l-') else 'l-'
        homologue = other + roi[2:]          # l-amygdala -> r-amygdala
        keep = np.zeros(Vfull, bool)
        for j, rj in enumerate(rois):
            if not rj.startswith(other):
                continue
            if args.targets_mode == "contralateral_no_homologue" and rj == homologue:
                continue
            keep[sls[j]] = True
    tgt = np.asarray(raw[:, :, keep])
    seed = np.asarray(raw[:, :, sl])
    # AA: no transform, so the comparison side is the same array as the reference side
    F = conn_features_all_subjects(tgt, seed, zscore_axis=0, flatten=False)
    isc = float(FisherAvg(loo_corrsim_dot_xy(F, F)))   # Fisher average across subjects
    rows.append(f"AA,{roi},{len(sl)},{args.targets_mode},{int(keep.sum())},{isc:.6f}")
    print(f"  {roi:<24} nVox={len(sl):>5}  targets={int(keep.sum()):>6}  ISC={isc:.4f}", flush=True)

with open(args.out.parent / f'args_{args.targets_mode}.json', 'w') as f:
    json.dump(dict(vars(args), script=os.path.basename(__file__), n_subjects=int(N), rois=rois),
              f, indent=2, default=str)
with open(args.out, 'w') as f:
    f.write('\n'.join(rows) + '\n')
print(f"wrote {args.out} ({len(rows) - 1} rows)")
