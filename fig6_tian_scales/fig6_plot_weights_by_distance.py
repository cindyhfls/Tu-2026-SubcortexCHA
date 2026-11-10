"""Transformation weights of a nucleus as a function of voxel distance (Fig 6B-C).

Uses the Tian S1 transforms into the template (ridgeCV, pooled alpha; transformation_to_template_raiders.py
with config_align_to_template_ridgeCV_raiders_TianS1_wholebrain.json), which are sparse block-diagonal
matrices with one dense block per nucleus.

  Fig 6B  weights of one example output voxel (column --voxel of the --sid transform block of
          --nucleus), shown as a brain map of the source voxels, and the location of that voxel.
  Fig 6C  mean weight against inter-voxel distance within the nucleus: one dot per subject and the
          across-subject mean (voxel-pair distances in 2 mm voxels, binned in 0.5-voxel bins, the
          first bin being the diagonal; plotted in mm).

Usage:  python fig6_plot_weights_by_distance.py                       # Fig 6B-C, aTHA-rh
Output: results/figures/fig6B_weights_spatial_example_map.png, fig6C_weights_spatial_distribution_example.png
"""
import argparse
import os
import sys
from glob import glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import neuroboros as nb
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from utils.data_io_utils import get_Tian_sls, return_Tian_labels
from utils.paths import FIGURES, RESULTS, ROOT, resolve  # noqa: E402
from utils.plotting_utils import map_subcortical_roi_to_img

parser = argparse.ArgumentParser()
parser.add_argument("--scale", type=int, default=1, choices=[1, 2, 3, 4], help="Tian scale")
parser.add_argument("--nucleus", default="aTHA-rh", help="Tian label of the example nucleus")
parser.add_argument("--sid", default=None, help="Example subject for Fig 6B (default: first transform file)")
parser.add_argument("--voxel", type=int, default=808, help="Index of the example output voxel within the nucleus")
parser.add_argument("--cut-z", type=float, default=8, help="z coordinate (mm) of the slice shown in Fig 6B")
parser.add_argument("--sids-file", default="config/raiders_subj_not_in_budapest_template.txt")
parser.add_argument("--xfm-dir", default=None,
                    help="Folder of <sid>_targets-onavg-ico8_Tian_Subcortex_ridgeCV.npz; default "
                         "results/Raiders/mni-2mm_1step_linear_overlap/ridgeCV_TianS<scale>_wholebrain/"
                         "transformation_to_template_targets_onavg-ico8")
parser.add_argument("--outdir", default=str(FIGURES))
args = parser.parse_args()

atlas = f"TianS{args.scale}"
distance_metric = 'euclidean'
bin_centers = np.concatenate([[0], np.arange(1, 4.5, 0.5)])  # in voxels (2 mm)
half_width = 0.25  # half of the bin size
voxel_mm = 2

xfm_dir = resolve(args.xfm_dir) if args.xfm_dir else (
    RESULTS / "Raiders" / "mni-2mm_1step_linear_overlap" / f"ridgeCV_{atlas}_wholebrain" /
    "transformation_to_template_targets_onavg-ico8")
outdir = resolve(args.outdir)
os.makedirs(outdir, exist_ok=True)

with open(resolve(args.sids_file)) as f:
    sids = f.read().strip().split('\n')
rois = return_Tian_labels(atlas)
sls = get_Tian_sls(atlas)
assert args.nucleus in rois, f"{args.nucleus} not in {atlas}"

# voxel coordinates (in voxel indices) of every nucleus, same voxel order as the transform blocks
lut = pd.read_csv(ROOT / "utils" / "support_files" /
                  f"atlas-{atlas}_space-MNI152NLin2009cAsym_res-02.lut.tsv", sep="\t").sort_values("flat_idx")


def dist_matrix(roi):
    coords = lut.loc[lut.label == roi, ["i", "j", "k"]].to_numpy()
    return squareform(pdist(coords, metric=distance_metric))


def load_block(sid, roi):
    xfm = nb.load(str(xfm_dir / f"{sid}_targets-onavg-ico8_Tian_Subcortex_ridgeCV.npz"))
    idx = sls[rois.index(roi)]
    return xfm[np.ix_(idx, idx)].toarray()  # toarray(): otherwise it would be a (sparse) matrix product


def binned_mean_weights(xfm0, dist_mat):
    out = np.full(len(bin_centers), np.nan)
    for ic, center in enumerate(bin_centers):
        mask = (dist_mat >= center - half_width) & (dist_mat < center + half_width)
        out[ic] = np.sum(mask * xfm0) / np.sum(mask)
    return out


# ---- Fig 6B: weights onto one output voxel ----
sid0 = args.sid or os.path.basename(sorted(glob(str(xfm_dir / "*.npz")))[0]).split('_targets-')[0]
xfm0 = load_block(sid0, args.nucleus)
weights = xfm0[:, args.voxel]

f = plt.figure(figsize=(4, 2))
map_subcortical_roi_to_img([weights], atlas, [args.nucleus], display_mode='z', cut_coords=[args.cut_z], figure=f,
                           interactive=False, colorbar=True, draw_cross=False, annotate=False, bg_img=None,
                           cbar_tick_format="%.2f")
cbar_ax = f.get_axes()[-1]
cbar_ax.set_yticks([-0.01, 0.01])
cbar_ax.tick_params(labelsize=20)
f.savefig(outdir / "fig6B_weights_spatial_example_map.png", dpi=300, bbox_inches='tight')
plt.close(f)

# ---- Fig 6C: mean weight vs distance, all subjects ----
roi_names = [args.nucleus]
dist_mats = {roi: dist_matrix(roi) for roi in roi_names}
all_mean_weights = {roi: np.full((len(sids), len(bin_centers)), np.nan) for roi in roi_names}
for isubj, sid in enumerate(sids):
    for roi in roi_names:
        all_mean_weights[roi][isubj] = binned_mean_weights(load_block(sid, roi), dist_mats[roi])

x = bin_centers * voxel_mm
plt.rcParams['font.size'] = 11
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab20.colors)
data = all_mean_weights[args.nucleus]
f = plt.figure(figsize=(3, 2))
for isubj in range(data.shape[0]):
    plt.plot(x, data[isubj], '.', markersize=5)
plt.plot(x, data.mean(axis=0), 'k*', label='average', markersize=8, alpha=0.5)
plt.legend()
plt.xlabel('Distance (mm)')
plt.ylabel('Average weights')
plt.title(args.nucleus)
f.savefig(outdir / "fig6C_weights_spatial_distribution_example.png", dpi=300, bbox_inches='tight')
plt.close(f)
