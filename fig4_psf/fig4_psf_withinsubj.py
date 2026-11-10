"""Within-subject control of the point-spread function (Fig 4 control, Supp Figs 13-15).

Same features, nuclei and distance bins as fig4_psf.py, but the voxel-pair correlation is
computed WITHIN each subject (voxel a vs voxel b of the same subject's data, no comparison to other
subjects), giving the spatial smoothness of a subject's own fine-grained signal as reference for the
intersubject PSF. Distance 0 is omitted (it would be identically 1).

Usage:  python fig4_psf_withinsubj.py --targets onavg-ico32                  # AA
        python fig4_psf_withinsubj.py --targets CHA+onavg-ico32 \
            --transformation-folder ridgeCV/transformation_to_template_targets_onavg-ico8 \
            --targets-transformation-folder <tree with dms_lr_float32.npy>                        # CHA
Output: same chunk folder as fig4_psf.py (see its docstring); files isc_<metric>_<distance>mm_withinsubj.dat
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.paths import resolve  # noqa: E402

import os
import argparse
import numpy as np
from utils.data_io_utils import ensure_memmap, row_done
from utils.isc_utils import parse_data_coarse_fine, conn_features_all_subjects, psf_corrsim_withinsubj
from fig4_helper import add_common_args, load_inputs, make_outdir, RES

parser = argparse.ArgumentParser()
add_common_args(parser)
args = parser.parse_args()

inp = load_inputs(args)
sids, dms, rois, sls, nSL = inp["sids"], inp["dms"], inp["rois"], inp["sls"], inp["nSL"]
sl_start, sl_stop = inp["sl_start"], inp["sl_stop"]
targets_ts, voxel_pairs, distances = inp["targets_ts"], inp["voxel_pairs"], inp["distances"]
N = len(sids)

outdir = make_outdir(inp)
conn_dir = os.path.join(outdir, f"connectivity_targets-{args.targets}")
mm = {}
paths = []
for dist in distances:
    for metric in ["resp_coarsefine", "resp_fine", "conn_coarsefine", "conn_fine"]:
        folder = conn_dir if metric.startswith("conn") else outdir
        fn = os.path.join(folder, f"isc_{metric}_{dist*RES}mm_withinsubj.dat")
        mm[f"{metric}_{dist*RES}mm"] = ensure_memmap(fn, (sl_stop - sl_start, N))
        paths.append(fn)
print(paths)

# ---- compute nucleus by nucleus (checkpointed per row) ----
print(f"Computing {len(paths)} metrics per nucleus (checkpointed)...")
for curr_row, isl in enumerate(range(sl_start, sl_stop)):
    roi = rois[isl]
    X = dms[:, :, sls[isl]]  # (N,T,Vsl)
    X_coarse, X_fine = parse_data_coarse_fine(X)
    data = {"resp_coarsefine": X, "resp_fine": X_fine,
            "conn_coarsefine": conn_features_all_subjects(targets_ts, X, zscore_axis=0, flatten=False),
            "conn_fine": conn_features_all_subjects(targets_ts, X_fine, zscore_axis=0, flatten=False)}
    for dist in distances:
        for metric, feats in data.items():
            key = f"{metric}_{dist*RES}mm"
            if not row_done(mm[key][curr_row]):
                mm[key][curr_row] = psf_corrsim_withinsubj(feats, dist, roi, voxel_pairs)
                mm[key].flush()
    print(f"  {roi} ({isl+1}/{nSL}) done")

print("All done.")
