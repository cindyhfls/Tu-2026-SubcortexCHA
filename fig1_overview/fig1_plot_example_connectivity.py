"""Example connectivity maps of single thalamus voxels to the cortical onavg-ico8 targets (Fig 1B).

For one example subject (anatomically aligned data, Raiders runs 1-2 concatenated) and two l-thalamus voxels
it draws
  (a) the location of the voxel in the l-thalamus (MNI slices through the voxel, cross-hair), and
  (b) the voxel's connectivity (z-scored across targets, 'bwr', -3..3) to the 1210 cortical ico8 targets
      on the inflated surface.
The voxel index is the index within the nucleus' voxels (the order of utils.data_io_utils.get_aseg_sls). Voxels 346
and 1234 of l-thalamus are at (x, y) = (-16.5, -28.5) and (-6.5, -2.5) mm, the two locations given on Fig 1B; the
slice shown is z = 9.5 mm.

Source: the legacy A8_connectivity_pattern_plot_and_prediction notebook (allcons[0, :, sls[6][1234]] with nb.plot,
where allcons is the connectivity computed on the whole cached time series).

Usage:  python fig1_plot_example_connectivity.py [--sid sid000012] [--voxels 1234 346]
Input:  data/cache/Raiders_raiders/run12/ (cache_subcortical_raiders.py, cache_cortical_raiders.py)
Output: results/figures/fig1B_Example_thalamus_voxel<v>.png, fig1B_conn_ico8_l-thalamus_voxel<v>_<sid>.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import CACHE, FIGURES, resolve  # noqa: E402

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import neuroboros as nb
import numpy as np
import pandas as pd

from utils.data_io_utils import apply_dm_transform, get_aseg_sls, return_aseg_labels
from utils.hyperalignment_utils import calculate_connectivity
from utils.plotting_utils import map_subcortical_roi_to_img

parser = argparse.ArgumentParser()
parser.add_argument("--sid", default="sid000012")
parser.add_argument("--roi", default="l-thalamus")
parser.add_argument("--voxels", type=int, nargs="+", default=[1234, 346],
                    help="voxel indices within the nucleus")
parser.add_argument("--run-tag", default="run12", help="cache folder under data/cache/Raiders_raiders (runs 1-2 concatenated)")
parser.add_argument("--cut-z", type=float, default=9.5, help="z (mm) of the slice shown in the location panel")
parser.add_argument("--vlim", type=float, default=3.0)
args = parser.parse_args()

plt.rcParams.update({'font.size': 12})
FIGURES.mkdir(exist_ok=True)
cache_dir = CACHE / "Raiders_raiders" / args.run_tag
surface_space = 'onavg-ico32'

rois = return_aseg_labels(atlas_name='mni-2mm')
sls = get_aseg_sls()
isl = rois.index(args.roi)
sids = np.load(cache_dir / "sids.npy").astype(str)
isubj = int(np.where(sids == args.sid)[0][0])

seed_ts = np.load(cache_dir / "ts_aseg_Subcortex_float32.npy", mmap_mode="r")[isubj]   # (T, Vseed), AA
ts_lr = np.concatenate([np.load(cache_dir / "ts_l_float32.npy", mmap_mode="r")[isubj],
                        np.load(cache_dir / "ts_r_float32.npy", mmap_mode="r")[isubj]],
                       axis=1).astype(np.float32, copy=False)[None]   # (1, T, Vall)
targets_ts = apply_dm_transform(ts_lr, nb.mapping('lr', surface_space, 'onavg-ico8', mask=True))[0]

conn = calculate_connectivity(targets_ts, seed_ts)   # (n_targets, Vseed), whole cached time series

lut = pd.read_csv(Path(__file__).resolve().parents[1] / "utils" / "support_files"
                  / "atlas-aseg_space-MNI152NLin2009cAsym_res-02.lut.tsv", sep="\t")
lut_roi = lut[lut.label == args.roi].reset_index(drop=True)

for v in args.voxels:
    # (a) voxel location
    cut = (float(lut_roi.loc[v, 'x']), float(lut_roi.loc[v, 'y']), args.cut_z)
    f = plt.figure(figsize=(3, 1))
    map_subcortical_roi_to_img(np.ones(len(sls[isl])), 'aseg', rois[isl], cut_coords=cut, figure=f,
                               interactive=False, colorbar=False, draw_cross=True)
    f.savefig(FIGURES / f"fig1B_Example_thalamus_voxel{v}.png", dpi=300)
    plt.close(f)
    # (b) connectivity of this voxel to the cortical targets
    nb.plot(conn[:, sls[isl][v]], cmap='bwr', vmin=-args.vlim, vmax=args.vlim, colorbar=True, labelsize=20,
            fn=str(FIGURES / f"fig1B_conn_ico8_{args.roi}_voxel{v}_{args.sid}.png"))
