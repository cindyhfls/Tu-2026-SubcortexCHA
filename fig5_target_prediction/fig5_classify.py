"""Cross-subject prediction of cortical connectivity targets from residual fine-grained patterns (Fig 5, Supp Fig 17).

Per nucleus the connectivity of every seed voxel to the cortical targets (1210 onavg-ico8 vertices) is computed
separately in the two test runs (runs 3 and 4 of the cache; --run-ends gives the frame where run 1 ends and
run 2 ends). The seed signal is the "fine" (residual) signal: the nucleus mean is subtracted from every voxel
before computing connectivity. For each subject, the connectivity profile of each seed voxel (run A, a vector
over targets) is classified among the nucleus' voxels using the mean profiles of all OTHER subjects (run B),
by correlation distance (multiclass: identify the voxel; binary: pairwise discrimination; the code passes the
voxel axis as classes, i.e. (nVoxels x nTargets) matrices). Baseline (= within-subject ceiling, Supp Fig 17)
classifies against the same subject's run B.
Run on the anatomically aligned data (AA, empty --transformation-folder) and on the CHA-transformed data.

Usage:  python fig5_classify.py --targets onavg-ico8                         # AA
        python fig5_classify.py --targets CHA+onavg-ico8 \
            --transformation-folder ridgeCV/transformation_to_template_targets_onavg-ico8 \
            --targets-transformation-folder ridge_alpha1e3/transformation_to_template_targets_onavg-ico8  # CHA
        (add --patterns fine coarse for the nucleus-mean "coarse" control, a single seed signal per nucleus)
Output: results/Raiders/mni-2mm_1step_linear_overlap/<AA|transformation folder>/
        connectivity_target-<targets>_prediction_splitruns_residualfine/acc_{multiclass,binary}[_baseline].npy
        (subjects x nuclei; the "coarse" control goes to ..._splitruns_coarse/); read by fig5_plot.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import CACHE, RESULTS, resolve  # noqa: E402

import argparse
import os
import numpy as np
import neuroboros as nb
from utils.data_io_utils import get_aseg_sls, apply_dm_transform
from utils.hyperalignment_utils import calculate_connectivity
from utils.classification_utils import classification_accuracy
from utils.isc_utils import parse_data_coarse_fine

DATASET = "Raiders"
TASK = "raiders"
VOLUME_SPACE = "mni-2mm"
VOLUME_RESAMPLE = "1step_linear_overlap"
SURFACE_SPACE = "onavg-ico32"
SURFACE_RESAMPLE = "1step_pial_overlap"

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=int, nargs="+", default=[3, 4], help="runs of the cache (paper: test runs 3 4)")
parser.add_argument("--run-ends", type=int, nargs=2, default=[840, 1680],
                    help="frame index at which the first / second cached run ends (840 frames per Raiders run)")
parser.add_argument("--targets", type=str, default="onavg-ico8",
                    help="connectivity targets: onavg-ico8 (anatomically aligned) or CHA+onavg-ico8 (hyperaligned)")
parser.add_argument("--transformation-folder", type=str, default="",
                    help="empty = AA; else the transformed seed caches under mni-2mm_1step_linear_overlap/<folder>")
parser.add_argument("--targets-transformation-folder", type=str, default="",
                    help="CHA+ targets only: tree with dms_lr_float32.npy, e.g. "
                         "ridge_alpha1e3/transformation_to_template_targets_onavg-ico8")
parser.add_argument("--patterns", type=str, nargs="+", default=["fine"], choices=["fine", "coarse"])
args = parser.parse_args()

run_ends = args.run_ends
cache_dir = CACHE / f"{DATASET}_{TASK}" / ("run" + "".join(map(str, args.run)))
source_folder = f"{VOLUME_SPACE}_{VOLUME_RESAMPLE}"
transformation_folder = args.transformation_folder
out_root = f"{source_folder}/{transformation_folder}" if transformation_folder else f"{source_folder}/AA/"
targets = args.targets

sids = np.load(cache_dir / "sids.npy").astype(str)
print("N total in file:", len(sids))

if transformation_folder:
    dms = np.load(os.path.join(cache_dir, out_root, "dms_aseg_Subcortex_float32.npy"), mmap_mode="r")
else:
    dms = np.load(cache_dir / "ts_aseg_Subcortex_float32.npy", mmap_mode="r")
sls = get_aseg_sls()

if targets.startswith("CHA+"):  # connectivity-hyperaligned targets
    assert args.targets_transformation_folder, "CHA+ targets need --targets-transformation-folder"
    targets_out_root = f"{SURFACE_SPACE}_{SURFACE_RESAMPLE}/{args.targets_transformation_folder}"
    targets_ts = np.load(os.path.join(cache_dir, targets_out_root, "dms_lr_float32.npy"), mmap_mode="r")
else:  # anatomically aligned targets
    ts_l = np.load(cache_dir / "ts_l_float32.npy")
    ts_r = np.load(cache_dir / "ts_r_float32.npy")
    targets_ts = np.concatenate([ts_l, ts_r], axis=2).astype(np.float32, copy=False)  # (N,T,Vall)
assert "onavg-ico8" in targets, "only the sparse onavg-ico8 targets are supported"
targets_ts = apply_dm_transform(targets_ts, nb.mapping("lr", SURFACE_SPACE, "onavg-ico8", mask=True))
print("targets_ts:", targets_ts.shape)

output_paths = {"fine": os.path.join(RESULTS, DATASET, out_root, f"connectivity_target-{targets}_prediction_splitruns_residualfine"),
                "coarse": os.path.join(RESULTS, DATASET, out_root, f"connectivity_target-{targets}_prediction_splitruns_coarse")}
result_names = ["acc_multiclass_baseline", "acc_binary_baseline", "acc_multiclass", "acc_binary"]
todo = [p for p in args.patterns
        if not all(os.path.exists(os.path.join(output_paths[p], f"{n}.npy")) for n in result_names)]
print(f"patterns to compute: {todo}")

nS, nSL = len(sids), len(sls)
acc = {p: {n: np.full([nS, nSL], np.nan) for n in result_names} for p in todo}

# "coarse" (nucleus-mean) connectivity: only 1 seed signal per nucleus, so there is no within-nucleus voxel
# pattern to classify (that comparison collapses to nc=1). Instead classify along the target axis: the
# connectivity values of the nTargets targets (a scalar per target) are matched to the other subjects' values;
# correlation distance is undefined for 1-D features, so this uses metric='euclidean' and no transpose.
for isl, sl in enumerate(sls):
    if not todo:
        break
    X_coarse, X_fine = parse_data_coarse_fine(dms[:, :, sl])
    seeds = {"fine": X_fine, "coarse": X_coarse}
    cons0, cons1 = {}, {}
    for p in todo:
        c0 = [calculate_connectivity(targets_ts[i, :run_ends[0], :], seeds[p][i, :run_ends[0], :]) for i in range(nS)]
        c1 = [calculate_connectivity(targets_ts[i, run_ends[0]:run_ends[1], :], seeds[p][i, run_ends[0]:run_ends[1], :]) for i in range(nS)]
        cons0[p], cons1[p] = np.stack(c0, axis=0), np.stack(c1, axis=0)  # (N, nTargets, nVoxels or 1)
    for isubj in range(nS):
        for p in todo:
            avg_others = np.mean(np.delete(cons1[p], isubj, axis=0), axis=0)
            kw, tr = ({}, True) if p == "fine" else ({"metric": "euclidean"}, False)
            a, b = (cons0[p][isubj].T, avg_others.T) if tr else (cons0[p][isubj], avg_others)
            acc[p]["acc_multiclass"][isubj, isl], acc[p]["acc_binary"][isubj, isl], _ = classification_accuracy(a, b, **kw)
            a, b = (cons0[p][isubj].T, cons1[p][isubj].T) if tr else (cons0[p][isubj], cons1[p][isubj])
            acc[p]["acc_multiclass_baseline"][isubj, isl], acc[p]["acc_binary_baseline"][isubj, isl], _ = classification_accuracy(a, b, **kw)
    print(f"  nucleus {isl+1}/{nSL} done")

for p in todo:
    os.makedirs(output_paths[p], exist_ok=True)
    for n in result_names:
        np.save(os.path.join(output_paths[p], f"{n}.npy"), acc[p][n])
for p in set(args.patterns) - set(todo):
    print(f"skipped {p}: already exists at {output_paths[p]}")
