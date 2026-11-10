"""Supp Figs 1-3: representational geometry consistency across subjects, per subcortical nucleus.

For every aseg nucleus (anatomically aligned data, AA), each subject's representational
dissimilarity matrix (RDM, correlation distance) is built over the nucleus' voxels, with either
  response     the response time series of runs 1-2 (samples = time points), or
  connectivity the connectivity of each voxel to the cortical onavg-ico8 targets
               (samples = cortical targets),
and the flattened upper triangles of the RDMs are correlated across all subject pairs. This gives a
subject x subject matrix per nucleus (lower triangle shown for the within-run versions):
  Supp Fig 1  response geometry, runs 1-2 (subject i vs subject j)
  Supp Fig 2  connectivity geometry, runs 1-2 (subject i vs subject j)
  Supp Fig 3  connectivity geometry, runs 1-2 (rows) vs runs 3-4 (columns), all pairs incl. the
              diagonal = each subject's own split-half reliability
The summary statistics (mean/min/max across nuclei of the Fisher-averaged between-subject values and,
for run 1-2 vs 3-4, of the within-subject diagonal) are written to a text file.

Usage:  python supp_plot_geometry_consistency.py [--n-jobs 8]
Input:  data/cache/Raiders_raiders/{run12,run34}/ (cache_{subcortical,cortical}_raiders.py)
Output: results/figures/suppfig{1,2}_geometry_{response,connectivity}_consistency_run12.png,
        results/figures/suppfig3_geometry_connectivity_consistency_run12_vs_run34.png, results/stats/geometry_consistency_summary.txt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import CACHE, FIGURES, STATS, resolve  # noqa: E402

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import neuroboros as nb
import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import squareform

from utils.classification_utils import classification_RDM
from utils.data_io_utils import apply_dm_transform, get_aseg_sls, return_aseg_labels
from utils.hyperalignment_utils import calculate_connectivity
from utils.isc_utils import FisherAvg, pairwise_corrsim_dot, pairwise_corrsim_dot_xy

parser = argparse.ArgumentParser()
parser.add_argument("--n-jobs", type=int, default=8)
parser.add_argument("--targets", default="onavg-ico8", help="cortical connectivity targets")
args = parser.parse_args()

plt.rcParams.update({'font.size': 12})
FIGURES.mkdir(exist_ok=True)
STATS.mkdir(exist_ok=True)

dataset, task = "Raiders", "raiders"
surface_space = 'onavg-ico32'
cache_root = CACHE / f"{dataset}_{task}"
RUN_DIRS = {'run12': cache_root / "run12", 'run34': cache_root / "run34"}

rois = return_aseg_labels(atlas_name='mni-2mm')
sls = get_aseg_sls()
sids = np.load(RUN_DIRS['run12'] / "sids.npy").astype(str)
Nsubj = len(sids)
print(f"N subjects: {Nsubj}, N nuclei: {len(sls)}")

dms_by_run, targets_ts_by_run, allcons_by_run = {}, {}, {}
for run_name, run_dir in RUN_DIRS.items():
    assert np.array_equal(np.load(run_dir / "sids.npy").astype(str), sids), \
        f"{run_name} sids do not match"
    dms_by_run[run_name] = np.load(run_dir / "ts_aseg_Subcortex_float32.npy", mmap_mode="r")
    ts = np.concatenate([np.load(run_dir / "ts_l_float32.npy"), np.load(run_dir / "ts_r_float32.npy")],
                        axis=2).astype(np.float32, copy=False)   # (N, T, Vall)
    assert args.targets == 'onavg-ico8', "only the ico8 targets of the manuscript are implemented"
    targets_ts_by_run[run_name] = apply_dm_transform(ts, nb.mapping('lr', surface_space, args.targets,
                                                                    mask=True))
    allcons_by_run[run_name] = np.stack(
        [calculate_connectivity(targets_ts_by_run[run_name][i], dms_by_run[run_name][i])
         for i in range(Nsubj)], axis=0)   # (Nsubj, Vtargets, Vseed)
    print(run_name, "dms:", dms_by_run[run_name].shape, "allcons:", allcons_by_run[run_name].shape)


def rdm_triu_by_subject(voxel_by_feature, sl):
    """voxel_by_feature: (Nsubj, Vall, Nfeatures). Returns (Nsubj, Ntriu): each subject's
    Nfeatures x Nfeatures RDM within `sl` (voxels as the description of each feature item),
    flattened to its upper triangle including the diagonal."""
    triu_idx = np.triu_indices(voxel_by_feature.shape[2])
    out = np.zeros((voxel_by_feature.shape[0], len(triu_idx[0])), dtype=np.float32)
    for isubj in range(voxel_by_feature.shape[0]):
        X = voxel_by_feature[isubj, sl, :].T   # (Nfeatures, n_voxels_in_ROI)
        out[isubj] = classification_RDM(X, X)[triu_idx]
    return out


def rdm_consistency_matrix(voxel_by_feature, sl):
    """(Nsubj, Nsubj) correlation between subjects' RDMs within `sl` (diagonal not filled)."""
    return squareform(pairwise_corrsim_dot(rdm_triu_by_subject(voxel_by_feature, sl)))


def run_contrast_consistency_matrix(feat_a, feat_b, sl):
    """M[i, j] = corr(subject i's RDM from run A, subject j's RDM from run B), incl. i == j."""
    return pairwise_corrsim_dot_xy(rdm_triu_by_subject(feat_a, sl), rdm_triu_by_subject(feat_b, sl))


def plot_consistency_grid(matrices, title, out_fn, vmin=0, vmax=1, lower_only=False,
                          xlabel=None, ylabel_right=None):
    """4 x 4 grid of subject x subject matrices: left-hemisphere nuclei in the top two rows, right-hemisphere
    nuclei in the bottom two. lower_only blanks the diagonal and the upper triangle."""
    ncols = 4
    nrows = int(np.ceil(len(matrices) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2 * nrows), constrained_layout=True)
    for isl, (M, ax) in enumerate(zip(matrices, axes.flat)):
        M = M.copy()
        if lower_only:
            M[np.triu_indices_from(M, k=0)] = np.nan   # k=0 also blanks the diagonal
        im = ax.imshow(M, cmap='viridis', vmin=vmin, vmax=vmax, aspect='equal')
        ax.set_title(rois[isl][2:])
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.flat[len(matrices):]:
        ax.axis('off')
    cbar = fig.colorbar(im, ax=axes, shrink=0.6, label="r")
    fig.suptitle(title)
    fig.canvas.draw()   # resolve the constrained layout before placing the labels
    half = nrows // 2
    for name, rows in (("Left hemisphere", range(0, half)), ("Right hemisphere", range(half, nrows))):
        top = axes[rows[0], 0].get_position().y1
        bottom = axes[rows[-1], 0].get_position().y0
        fig.text(0.0, (top + bottom) / 2, name, rotation=90, va='center', ha='right', fontsize=16)
    if xlabel:
        fig.text(0.5 * (axes[-1, 0].get_position().x0 + axes[-1, -1].get_position().x1),
                 axes[-1, 0].get_position().y0 - 0.035, xlabel, ha='center', va='top')
    if ylabel_right:
        fig.text(cbar.ax.get_position().x1 + 0.1, 0.5, ylabel_right, rotation=270, va='center', ha='center')
    fig.savefig(out_fn, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("wrote", out_fn)


def between_subject_mean(M):
    return FisherAvg(squareform(M))


def parallel(feat_fn):
    return Parallel(n_jobs=args.n_jobs, backend="threading")(delayed(feat_fn)(sl) for sl in sls)


response_features = np.transpose(np.asarray(dms_by_run['run12']), (0, 2, 1))   # (Nsubj, Vall, T)
conn_features = {r: np.transpose(allcons_by_run[r], (0, 2, 1)) for r in allcons_by_run}   # (Nsubj, Vall, Vtargets)

response_consistency = parallel(lambda sl: rdm_consistency_matrix(response_features, sl))
conn_consistency = parallel(lambda sl: rdm_consistency_matrix(conn_features['run12'], sl))
conn_run_contrast = parallel(lambda sl: run_contrast_consistency_matrix(conn_features['run12'],
                                                                        conn_features['run34'], sl))

plot_consistency_grid(response_consistency, "Representational geometry: subject x subject consistency",
                      FIGURES / "suppfig1_geometry_response_consistency_run12.png", vmax=0.5, lower_only=True)
plot_consistency_grid(conn_consistency, "Connectivity geometry: subject x subject consistency",
                      FIGURES / "suppfig2_geometry_connectivity_consistency_run12.png", lower_only=True,
                      xlabel="subjects")
plot_consistency_grid(conn_run_contrast, "Connectivity geometry: run12 v.s. run34 consistency",
                      FIGURES / "suppfig3_geometry_connectivity_consistency_run12_vs_run34.png",
                      xlabel="subjects run12", ylabel_right="subjects run34")

# ---- summary numbers ----
lines = []
for name, mats in (("response (run12)", response_consistency), ("connectivity (run12)", conn_consistency)):
    c = [between_subject_mean(M) for M in mats]
    lines.append(f"{name}: between-subject mean {FisherAvg(c):.2f}, min {np.min(c):.2f}, max {np.max(c):.2f}")
b, w = [], []
for M in conn_run_contrast:
    b.append(FisherAvg(np.concatenate([M[np.tril_indices_from(M, k=-1)], M[np.triu_indices_from(M, k=1)]])))
    w.append(FisherAvg(np.diag(M)))
lines.append(f"connectivity run12 vs run34, within-subject (diagonal): mean {FisherAvg(w):.2f}, "
             f"{rois[int(np.argmin(w))]} min {np.min(w):.2f}, {rois[int(np.argmax(w))]} max {np.max(w):.2f}")
lines.append(f"connectivity run12 vs run34, between-subject (off-diagonal): mean {FisherAvg(b):.2f}, "
             f"{rois[int(np.argmin(b))]} min {np.min(b):.2f}, {rois[int(np.argmax(b))]} max {np.max(b):.2f}")
(STATS / "geometry_consistency_summary.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
