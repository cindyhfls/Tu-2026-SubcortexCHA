"""Intersubject point-spread function (PSF) of residual fine-grained ISC (Fig 4, Supp Figs 13-15).

For each aseg nucleus, subject s's voxel-wise features (response time series, or connectivity to the
targets) are correlated, over voxel pairs (a, b) separated by a given inter-voxel distance, with the mean
of the OTHER subjects (voxel a of s vs voxel b of the others); the correlations are Fisher-averaged to one
value per subject. Distance 0 is the usual same-voxel ISC. "fine" = nucleus mean removed from every voxel
(residual); "coarsefine" = original data. Features per metric:
  resp_fine / resp_coarsefine   response time series
  conn_fine / conn_coarsefine   connectivity of the seed voxels to the --targets (z-scored per target)
Fig 4 uses conn_fine to onavg-ico32 cortex (AA: onavg-ico32, CHA: CHA+onavg-ico32), conn_fine to the
contralateral subcortex (--targets aseg|CHA+aseg --targets-hemisphere contralateral) and resp_fine.
Run once on the anatomically aligned data (AA) and once on the CHA-transformed data.

Usage (needs the Raiders caches from 00_cache_inputs/cache_*_raiders.py and fig4_calc_voxel_distances.py):
  python fig4_psf.py --targets onavg-ico32                                  # AA
  python fig4_psf.py --targets CHA+onavg-ico32 \
      --transformation-folder ridgeCV/transformation_to_template_targets_onavg-ico8 \
      --targets-transformation-folder <tree with dms_lr_float32.npy>                             # CHA
Output: results/Raiders/mni-2mm_1step_linear_overlap/<AA|transformation folder>/
        memmap_ISC_loo_psf_euclidean_binned_aseg_raiders_summary/sl<a>-<b>/[connectivity_targets-<targets>[-contralateral]/]
        isc_<metric>_<distance>mm.dat   (float32 memmap, nuclei x subjects); read by fig4_plot.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.paths import resolve  # noqa: E402

import os
import argparse
import numpy as np
from utils.data_io_utils import ensure_memmap, row_done, roi_hemisphere
from utils.isc_utils import parse_data_coarse_fine, conn_features_all_subjects, psf_corrsim
from fig4_helper import add_common_args, load_inputs, make_outdir, RES

ALL_METRICS = ["resp_coarsefine", "resp_fine", "conn_coarsefine", "conn_fine"]

parser = argparse.ArgumentParser()
add_common_args(parser)
# Restrict the connectivity targets to the hemisphere OPPOSITE each seed nucleus: removes the seed nucleus
# and every adjacent structure from the target set. Output goes to connectivity_targets-<targets>-contralateral/.
parser.add_argument("--targets-hemisphere", type=str, default="both", choices=["both", "contralateral"])
# Only the requested metrics get a memmap, so a partial run leaves no all-NaN files behind.
parser.add_argument("--metrics", type=str, nargs="+", default=ALL_METRICS, choices=ALL_METRICS)
args = parser.parse_args()

inp = load_inputs(args)
sids, dms, rois, sls, nSL = inp["sids"], inp["dms"], inp["rois"], inp["sls"], inp["nSL"]
sl_start, sl_stop = inp["sl_start"], inp["sl_stop"]
targets_ts, voxel_pairs = inp["targets_ts"], inp["voxel_pairs"]
N = len(sids)
distances = inp["distances"] + [0]

CONTRA = args.targets_hemisphere == "contralateral"
if CONTRA:
    assert targets_ts.shape[2] == dms.shape[2], (
        "contralateral restriction indexes the target by seed-atlas voxel, so the targets must be "
        "the seed atlas itself (aseg, or CHA+aseg for a transformed tree)")
    hemis = [roi_hemisphere(r) for r in rois]
    assert None not in hemis, f"cannot parse a hemisphere from every ROI in {rois}"
    contra_idx = {}
    for i, h in enumerate(hemis):
        keep = np.zeros(targets_ts.shape[2], bool)
        for j, hj in enumerate(hemis):
            if hj != h:
                keep[sls[j]] = True
        contra_idx[i] = np.where(keep)[0]
    print(f"contralateral targets: {len(contra_idx[0])} / {len(contra_idx[len(rois)//2])} voxels "
          f"(left / right seed), instead of {targets_ts.shape[2]}")
CONN_SUBDIR = f"connectivity_targets-{args.targets}" + ("-contralateral" if CONTRA else "")

outdir = make_outdir(inp)
METRICS = list(args.metrics)
mm = {}
paths = []
for dist in distances:
    for metric in METRICS:
        folder = os.path.join(outdir, CONN_SUBDIR) if metric.startswith("conn") else outdir
        fn = os.path.join(folder, f"isc_{metric}_{dist*RES}mm.dat")
        mm[f"{metric}_{dist*RES}mm"] = ensure_memmap(fn, (sl_stop - sl_start, N))
        paths.append(fn)
print(f"metrics: {METRICS}")
print(paths)

# ---- compute nucleus by nucleus (checkpointed per row) ----
print(f"Computing {len(paths)} metrics per nucleus (checkpointed)...")
for curr_row, isl in enumerate(range(sl_start, sl_stop)):
    roi = rois[isl]
    X = dms[:, :, sls[isl]]  # (N,T,Vsl)
    X_coarse, X_fine = parse_data_coarse_fine(X)
    data = {"resp_coarsefine": X, "resp_fine": X_fine}
    # connectivity features do not depend on distance, so they are built once per nucleus
    tgt_ts = np.asarray(targets_ts[:, :, contra_idx[isl]]) if CONTRA else targets_ts
    if "conn_coarsefine" in METRICS:
        data["conn_coarsefine"] = conn_features_all_subjects(tgt_ts, X, zscore_axis=0, flatten=False)
    if "conn_fine" in METRICS:
        data["conn_fine"] = conn_features_all_subjects(tgt_ts, X_fine, zscore_axis=0, flatten=False)
    for dist in distances:
        for metric in METRICS:
            key = f"{metric}_{dist*RES}mm"
            if not row_done(mm[key][curr_row]):
                mm[key][curr_row] = psf_corrsim(data[metric], dist, roi, voxel_pairs)
                mm[key].flush()
    print(f"  {roi} ({isl+1}/{nSL}) done")

print("All done.")
