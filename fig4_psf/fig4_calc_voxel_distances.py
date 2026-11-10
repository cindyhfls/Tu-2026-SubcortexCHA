"""Pre-compute the voxel-pair lists used by the point-spread-function (PSF) scripts.

For every aseg nucleus (MNI152NLin2009cAsym, 2 mm) all within-nucleus voxel pairs are binned by Euclidean
distance (in voxel units; bins of width 0.5 voxel centred on 1.0, 1.5, ..., 4.0 voxels = 2 ... 8 mm).
`voxel_pairs[nucleus][bin_centre]` is an int16 array (2, nPairs) of within-nucleus voxel indices (same order
as the columns of the `ts_aseg_Subcortex_float32.npy` caches). Voxel coordinates come from the aseg lookup
table shipped in utils/support_files, so no atlas image is needed.

Usage:  python fig4_calc_voxel_distances.py
Output: results/voxel_pairs/voxel_pairs_subcortical_euclidean_MNI152NLin2009cAsym_res-02_seg-aseg_dseg_binned.npy
        (read by fig4_psf.py and fig4_psf_withinsubj.py)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import RESULTS, resolve  # noqa: E402

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

RES = 2
DISTANCE_METRIC = "euclidean"
BIN_CENTERS = np.arange(1, 4.5, 0.5)  # voxels
HALF_WIDTH = 0.25                    # half the bin size; bins are [centre-0.25, centre+0.25)

HERE = Path(__file__).resolve().parents[1] / "utils" / "support_files"
lut = pd.read_csv(HERE / f"atlas-aseg_space-MNI152NLin2009cAsym_res-{RES:02d}.lut.tsv", sep="\t")

voxel_pairs = {}
for label in lut.label.unique():
    voxels = lut[lut.label == label].sort_values("within_idx")
    print(f"ROI: {label} ({len(voxels)} voxels)")
    dist_mat = squareform(pdist(voxels[["i", "j", "k"]].values, metric=DISTANCE_METRIC))
    voxel_pairs[label] = {}
    for center in BIN_CENTERS:
        mask = (dist_mat >= center - HALF_WIDTH) & (dist_mat < center + HALF_WIDTH)
        voxel_pairs[label][center] = np.int16(np.where(mask))

out_dir = RESULTS / "voxel_pairs"
out_dir.mkdir(exist_ok=True)
np.save(out_dir / f"voxel_pairs_subcortical_{DISTANCE_METRIC}_MNI152NLin2009cAsym_res-{RES:02d}_seg-aseg_dseg_binned.npy",
        voxel_pairs)
