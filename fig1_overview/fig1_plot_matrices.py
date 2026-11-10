"""Matrices of the hyperalignment schematic (Fig 1C), for one example subject and the left thalamus.

  template                 connectivity PCA template of l-thalamus (Budapest, 10 subjects, runs 1-5)
  connectivity run 1-2     the subject's l-thalamus connectivity to the cortical onavg-ico8 targets, Raiders runs 1-2
                           concatenated (the data the transformation is fitted on), z-scored over the voxels
  weights                  the subject's transformation into the template (ridgeCV, Raiders runs 1-2)
  response run 3-4         the subject's l-thalamus response, anatomically aligned and after the transformation
                           (hyperaligned), Raiders runs 3-4

All shown as in the legacy A8_connectivity_pattern_plot_and_prediction notebook (imshow, 'bwr', +-3; weights +-0.01
with a horizontal colour bar); the voxel axis of the connectivity and response matrices is vertical.

Usage:  python fig1_plot_matrices.py [--sid sid000012]
Input:  results/Budapest/.../templates_aseg, results/Raiders/.../ridgeCV transformation, data/cache/Raiders_raiders/{run12,run34}
Output: results/figures/fig1C_{template,connectivity_run12,weights,response_AA_run34,response_CHA_run34}.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import CACHE, FIGURES, RESULTS  # noqa: E402

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import neuroboros as nb
import numpy as np

from utils.data_io_utils import apply_dm_transform, get_aseg_sls, return_aseg_labels
from utils.hyperalignment_utils import calculate_connectivity

parser = argparse.ArgumentParser()
parser.add_argument("--sid", default="sid000012")
parser.add_argument("--roi", default="l-thalamus")
parser.add_argument("--transformation", default="ridgeCV/transformation_to_template_targets_onavg-ico8",
                    help="transformation folder (CHA) under results/Raiders/mni-2mm_1step_linear_overlap")
args = parser.parse_args()

FIGURES.mkdir(exist_ok=True)
VOL = "mni-2mm_1step_linear_overlap"
rois = return_aseg_labels(atlas_name="mni-2mm")
sl = get_aseg_sls()[rois.index(args.roi)]
cache = CACHE / "Raiders_raiders"
sids = np.load(cache / "run12" / "sids.npy").astype(str)
isubj = int(np.where(sids == args.sid)[0][0])


def save(matrix, name, vlim, colorbar=False):
    fig = plt.figure()
    plt.imshow(matrix, aspect="equal", cmap="bwr", vmin=-vlim, vmax=vlim)
    plt.axis("off")
    if colorbar:
        cbar = plt.colorbar(ticks=[-vlim, 0, vlim], orientation="horizontal")
        cbar.ax.tick_params(labelsize=20)
    fig.savefig(FIGURES / name, dpi=300, bbox_inches="tight")
    plt.close(fig)


# template (targets x voxels, as stored)
tpl = np.load(RESULTS / "Budapest" / VOL / "templates_aseg" / f"targets_onavg-ico8_{args.roi}.npy")
save(tpl, "fig1C_template.png", 3)

# connectivity of the voxels to the cortical targets, runs 1-2 concatenated -> (voxels, targets)
run12 = cache / "run12"
seed = np.load(run12 / "ts_aseg_Subcortex_float32.npy", mmap_mode="r")[isubj][:, sl]
ctx = np.concatenate([np.load(run12 / "ts_l_float32.npy", mmap_mode="r")[isubj],
                      np.load(run12 / "ts_r_float32.npy", mmap_mode="r")[isubj]], axis=1).astype(np.float32)[None]
targets = apply_dm_transform(ctx, nb.mapping("lr", "onavg-ico32", "onavg-ico8", mask=True))[0]
conn = calculate_connectivity(targets, np.asarray(seed))          # (targets, voxels)
save(conn.T, "fig1C_connectivity_run12.png", 3)

# transformation weights
xfm = np.load(RESULTS / "Raiders" / VOL / args.transformation / f"{args.sid}_targets-onavg-ico8_{args.roi}_ridgeCV.npy")
save(xfm, "fig1C_weights.png", 0.01, colorbar=True)

# response of runs 3-4, anatomically aligned and hyperaligned -> (voxels, time points)
run34 = cache / "run34"
save(np.load(run34 / "ts_aseg_Subcortex_float32.npy", mmap_mode="r")[isubj][:, sl].T, "fig1C_response_AA_run34.png", 3)
save(np.load(run34 / VOL / args.transformation / "dms_aseg_Subcortex_float32.npy", mmap_mode="r")[isubj][:, sl].T,
     "fig1C_response_CHA_run34.png", 3)
print("wrote the Fig 1C matrices to", FIGURES)
