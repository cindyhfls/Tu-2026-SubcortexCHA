"""Matrices of the cross-run self-transform schematic (Supp Fig 9A), for one example subject and the left thalamus.

The null transformation is fitted between two runs of the SAME subject (Raiders run 1 -> run 2), with the same
regularisation as CHA but no cross-subject alignment, and applied to the test runs (3-4):

  connectivity run 1 / run 2   l-thalamus connectivity to the cortical onavg-ico8 targets in each run
                               (the two 840-frame halves of the run 1-2 cache), z-scored over the voxels
  weights                      the cross-run transformation (run 1 -> 2, alpha matched to the CHA fit)
  response run 3-4             the subject's l-thalamus response, anatomically aligned and "denoised"
                               (after the cross-run transformation)

Same display as fig1_plot_matrices.py (imshow, 'bwr', +-3; weights +-0.01 with a horizontal colour bar).
Reconstructed: the notebook that saved the original images was not found; the data and settings are those of
supp_transformation_to_self.py (--cross-run --run 1 2 --align-method ridge, matched alpha).

Usage:  python supp_plot_crossrun_matrices.py [--sid sid000012]
Input:  results/Raiders/.../cross_run1to2_shrinkage_matched/transformation_to_self_targets_onavg-ico8, data/cache/Raiders_raiders/{run12,run34}
Output: results/figures/suppfig9A_{connectivity_run1,connectivity_run2,weights,response_AA_run34,response_denoised_run34}.png
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
parser.add_argument("--run-ends", type=int, nargs=2, default=[840, 1680], help="end frame of run 1 and of run 2 in the run 1-2 cache")
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


# connectivity of the voxels to the cortical targets in run 1 and in run 2 -> (voxels, targets)
run12 = cache / "run12"
seed = np.asarray(np.load(run12 / "ts_aseg_Subcortex_float32.npy", mmap_mode="r")[isubj][:, sl])
ctx = np.concatenate([np.load(run12 / "ts_l_float32.npy", mmap_mode="r")[isubj],
                      np.load(run12 / "ts_r_float32.npy", mmap_mode="r")[isubj]], axis=1).astype(np.float32)[None]
targets = apply_dm_transform(ctx, nb.mapping("lr", "onavg-ico32", "onavg-ico8", mask=True))[0]
starts, ends = [0, args.run_ends[0]], args.run_ends
for irun, (a, b) in enumerate(zip(starts, ends), start=1):
    conn = calculate_connectivity(targets[a:b], seed[a:b])      # (targets, voxels)
    save(conn.T, f"suppfig9A_connectivity_run{irun}.png", 3)

# cross-run transformation (run 1 -> run 2), matched alpha
xfm = np.load(RESULTS / "Raiders" / VOL / "cross_run1to2_shrinkage_matched" / "transformation_to_self_targets_onavg-ico8"
              / f"{args.sid}_targets-onavg-ico8_{args.roi}_ridge.npy")
save(xfm, "suppfig9A_weights.png", 0.01, colorbar=True)

# response of runs 3-4: anatomically aligned and denoised (after the cross-run transformation) -> (voxels, time points)
x = np.asarray(np.load(cache / "run34" / "ts_aseg_Subcortex_float32.npy", mmap_mode="r")[isubj][:, sl], dtype=np.float32)
save(x.T, "suppfig9A_response_AA_run34.png", 3)
save((x @ xfm).T, "suppfig9A_response_denoised_run34.png", 3)
print("wrote the Supp Fig 9A matrices to", FIGURES)
