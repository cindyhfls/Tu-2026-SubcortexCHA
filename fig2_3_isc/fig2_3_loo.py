"""Leave-one-out intersubject correlation (ISC) of subcortical responses and connectivity.

For every seed region (aseg nucleus / Tian parcel; or a cortical searchlight) and every subject, the
subject's feature is correlated with the mean of the other subjects' features (leave-one-out ISC).
Features are
  resp_*  the seed-region response time series (voxels x time),
  conn_*  the connectivity of each seed voxel to a set of target time series (--targets),
  rdm_*   the time-point RDM of the seed region,
each computed on the 'coarsefine' data (all voxels), the 'coarse' data (nucleus mean) or the 'fine'
data (nucleus mean subtracted = "residual fine"). `*_voxelwise` gives a per-voxel ISC map instead.

Manuscript: Fig 2/3 (ISC of connectivity to hyperaligned cortical ico32 targets / to contralateral
subcortical voxels, and of responses; coarse vs residual-fine), Fig 6 and Supp Fig 18 (Tian S1-S4,
pooled / per-nucleus alpha), Supp Figs 4-5 (Procrustes / ridge / ridgeCV, run 1-2 'training' ISC).
Conditions are selected with --transformation-folder (empty = anatomical alignment, AA) and
--targets / --targets-transformation-folder; see slurm/run_isc.sh for the command lines.

Target labels: 'onavg-ico32' / 'onavg-ico8' (anatomically aligned cortex), 'CHA+onavg-ico32' (cortex
hyperaligned by the transform in --targets-transformation-folder), 'aseg' / 'Tian_Subcortex' (the seed
atlas itself, voxelwise), 'CHA+aseg' / 'CHA+Tian_Subcortex' (transformed subcortex), 'aseg-mean',
'TianS<n>' (parcel means). --targets-hemisphere contralateral keeps only the targets in the opposite
hemisphere of each seed (subcortical targets only).

Usage:  python fig2_3_loo.py --atlas aseg --targets CHA+onavg-ico32 \
            --transformation-folder ridgeCV/transformation_to_template_targets_onavg-ico8/ \
            --targets-transformation-folder ridge_alpha1e3/transformation_to_template_targets_onavg-ico8/
        add `--run 1 2 --out-subdir run12` for the training runs (default: test runs 3-4).
Input:  data/cache/<dataset>_<task>/<run_tag>/ (written by cache_subcortical_raiders.py and cache_cortical_raiders.py)
Output: results/<dataset>/<space>_<resample>/<transformation folder or AA>/<out-subdir>/
        memmap_ISC_loo_<atlas>_<task>_summary/sl<start>-<stop>/[connectivity_targets-<targets>/]isc_*.dat
        (float32 memmaps, rows = seed regions, columns = subjects)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import CACHE, RESULTS, resolve  # noqa: E402

import os
import argparse
import numpy as np
import neuroboros as nb
from utils.data_io_utils import ensure_memmap,row_done,return_aseg_labels,get_aseg_sls,return_Tian_labels,get_Tian_sls,roi_hemisphere,untransformed_target_label
from utils.hyperalignment_utils import get_target_ts
from utils.isc_utils import loo_corrsim_dot,loo_voxelwise_corrsim_dot,parse_data_coarse_fine,conn_features_all_subjects,rdm_features_all_subjects,compute_targets_ts_from_mapping

parser = argparse.ArgumentParser()
parser.add_argument("--sl-start", type=int, default=0, help="first seed region of this chunk")
parser.add_argument("--sl-stop", type=int, default=None, help="end (exclusive) of this chunk; default all")
parser.add_argument("--targets", type=str, default="onavg-ico32",
                    help="connectivity targets used for the ISC (not the targets used for alignment)")
parser.add_argument("--atlas", type=str, default="aseg", help="seed atlas: aseg, TianS1-S4, or lr (cortex)")
parser.add_argument("--source-space", type=str, default="mni-2mm")
parser.add_argument("--transformation-folder", type=str, default="",
                    help="transform applied to the seeds (relative to <space>_<resample>/); empty = AA")
parser.add_argument("--targets-transformation-folder", type=str, default="",
                    help="transform the CHA+ targets went through; must be empty for untransformed targets")
parser.add_argument("--targets-hemisphere", type=str, default="both",
                    choices=["both", "contralateral"],
                    help="contralateral: restrict each seed's subcortical targets to the opposite "
                         "hemisphere (removes the seed's own nucleus and its neighbours)")
parser.add_argument("--dataset", type=str, default="Raiders")
parser.add_argument("--task", type=str, default="raiders")
parser.add_argument("--run", type=int, nargs="+", default=[3, 4])
default_metrics = ["resp_coarse", "resp_coarsefine", "resp_fine",
                   "conn_coarse", "conn_coarsefine", "conn_fine", "rdm_coarsefine",
                   "resp_voxelwise", "conn_voxelwise"]
parser.add_argument("--metrics", type=str, nargs="+", default=default_metrics, choices=default_metrics,
                    help="metrics to compute; memmaps are created only for these")
parser.add_argument("--run-tag", type=str, default=None,
                    help="Cache subdirectory; defaults to run<runs>, e.g. run34.")
parser.add_argument("--out-subdir", type=str, default="",
                    help="Folder inserted between the transform folder and memmap_ISC_loo_*. "
                         "Use 'run12' for runs 1-2; empty keeps the run 3-4 layout.")
args = parser.parse_args()

dataset = args.dataset
task = args.task
run = args.run
run_tag = args.run_tag or f"run{''.join(map(str, run))}"
# --run only changes which cache is READ; the output path has a run level only through --out-subdir.
if dataset == "Raiders" and list(run) != [3, 4] and not args.out_subdir:
    raise SystemExit(f"--run {run} on Raiders needs --out-subdir (e.g. run12); without it the "
                     f"output would land in the run34 folders")
cache_dir = f"{CACHE}/{dataset}_{task}/{run_tag}"

volume_space = 'mni-2mm'
surface_space = 'onavg-ico32'
surface_resample = '1step_pial_overlap'
volume_resample = '1step_linear_overlap'
transformation_folder = args.transformation_folder
source_space = args.source_space
atlas = args.atlas
targets = args.targets
sl_start = args.sl_start
sl_stop = args.sl_stop

if 'mni' in source_space:
    source_folder = f"{volume_space}_{volume_resample}"
else:
    source_folder = f"{surface_space}_{surface_resample}"
if transformation_folder:
    out_root = f"{source_folder}/{transformation_folder}"
else:
    out_root = f"{source_folder}/AA/"
targets_transformation_folder = args.targets_transformation_folder
if 'onavg' in targets:
    targets_out_root = f"{surface_space}_{surface_resample}/{targets_transformation_folder}"
else:
    targets_out_root = f"{volume_space}_{volume_resample}/{targets_transformation_folder}"

# Naming guard: the target label goes straight into the output folder name, so a CHA+ label on
# untransformed targets would be wrong on disk. An untransformed target is expressed by an EMPTY
# --targets-transformation-folder; 'AA/' is a real directory of untransformed data and is rejected.
_tt = targets_transformation_folder.strip('/')
assert _tt != 'AA', (
    f"--targets-transformation-folder={targets_transformation_folder!r} is not a valid value: "
    f"use '' for an untransformed target.")
if targets.startswith('CHA+'):
    assert _tt, (
        f"--targets {targets} says the targets are connectivity-hyperaligned, but "
        f"--targets-transformation-folder={targets_transformation_folder!r} reads untransformed data. "
        f"Pass the transform the TARGETS went through, or relabel as "
        f"--targets {untransformed_target_label(targets)}.")
elif _tt:
    print(f"WARNING: --targets-transformation-folder={targets_transformation_folder!r} is ignored for "
          f"the untransformed target '{targets}'; raw data will be used. Use --targets CHA+{targets} "
          f"if you meant the transformed targets.")

sids = np.load(os.path.join(cache_dir, "sids.npy")).astype(str)
N = len(sids)
print("N total in file:", N)

# seed data (N, T, V)
if transformation_folder:
    if atlas.startswith("Tian"):
        dms_fn = os.path.join(cache_dir, out_root, "dms_Tian_Subcortex_float32.npy")
    elif atlas.startswith("aseg"):
        dms_fn = os.path.join(cache_dir, out_root, "dms_aseg_Subcortex_float32.npy")
    else:
        dms_fn = os.path.join(cache_dir, out_root, f"dms_{atlas}_float32.npy")
    dms = np.load(dms_fn, mmap_mode="r")
else:
    if atlas.startswith("Tian"):
        dms = np.load(os.path.join(cache_dir, "ts_Tian_Subcortex_float32.npy"), mmap_mode="r")
    elif atlas.startswith("aseg"):
        dms = np.load(os.path.join(cache_dir, "ts_aseg_Subcortex_float32.npy"), mmap_mode="r")
    else:
        dms = np.load(os.path.join(cache_dir, f"ts_{atlas}_float32.npy"), mmap_mode="r")

if 'mni' in source_space:
    if atlas == 'aseg':
        rois = return_aseg_labels(atlas_name=source_space)
        sls = get_aseg_sls()
    elif atlas.startswith('Tian'):
        rois = return_Tian_labels(atlas)
        sls = get_Tian_sls(atlas)
else:
    radius = 20  # cortical searchlight radius (mm)
    sls = nb.sls(atlas, radius, space=surface_space, center_space='onavg-ico8', mask=True, return_dists=False)
nSL = len(sls)
sl_stop = nSL if sl_stop is None else min(sl_stop, nSL)

# connectivity targets (N, T, nTargets)
if targets == atlas:
    targets_ts = dms  # the seed atlas itself; for a transformed condition this is the transformed atlas
else:
    if targets.startswith('CHA+'):  # connectivity-hyperaligned targets
        if 'Tian' in targets:
            target_dms_fn = os.path.join(cache_dir, targets_out_root, "dms_Tian_Subcortex_float32.npy")
        elif 'onavg' in targets:
            target_dms_fn = os.path.join(cache_dir, targets_out_root, "dms_lr_float32.npy")
        elif 'aseg' in targets:
            target_dms_fn = os.path.join(cache_dir, targets_out_root, "dms_aseg_Subcortex_float32.npy")
        targets_ts = np.load(target_dms_fn, mmap_mode="r")
    else:  # anatomically aligned targets
        if 'Tian' in targets:
            targets_ts = np.load(os.path.join(cache_dir, "ts_Tian_Subcortex_float32.npy"))
        elif 'onavg' in targets:
            ts_l = np.load(os.path.join(cache_dir, "ts_l_float32.npy"))
            ts_r = np.load(os.path.join(cache_dir, "ts_r_float32.npy"))
            targets_ts = np.concatenate([ts_l, ts_r], axis=2).astype(np.float32, copy=False)  # (N,T,Vall)
        elif 'aseg' in targets:
            targets_ts = np.load(os.path.join(cache_dir, "ts_aseg_Subcortex_float32.npy"))

    # targets that are not dense voxel/vertex sets
    if 'onavg-ico8' in targets:
        targets_ts = compute_targets_ts_from_mapping(targets_ts, surface_space, 'onavg-ico8', mask=True)
    elif 'aseg-mean' in targets:
        targets_ts = get_target_ts(targets_ts, 'searchlight_mean', target_item=get_aseg_sls())
    elif any([tt in targets for tt in ['TianS1', 'TianS2', 'TianS3', 'TianS4']]):
        targets_ts = get_target_ts(targets_ts, 'searchlight_mean', target_item=get_Tian_sls(targets))
print("targets_ts:", targets_ts.shape)

CONTRA = args.targets_hemisphere == "contralateral"
if CONTRA:
    # Two target sets can be restricted, both laid out by the seed atlas's own ROIs:
    #   voxels     the seed atlas itself (CHA+aseg, or CHA+Tian_Subcortex which keeps the Tian
    #              targets voxelwise): keep every voxel of every opposite-hemisphere ROI
    #   ROI means  aseg-mean with aseg seeds: one target per seed ROI, in the same order
    #              (get_aseg_sls), so keep the means of the opposite-hemisphere ROIs
    UNIT_TARGETS = 'aseg-mean' in targets and atlas == 'aseg'
    if UNIT_TARGETS:
        assert targets_ts.shape[2] == len(rois), (
            f"aseg-mean targets should be one mean per seed ROI ({len(rois)}), got {targets_ts.shape[2]}")
    else:
        assert targets_ts.shape[2] == dms.shape[2], (
            "contralateral restriction indexes the TARGET by seed-atlas voxel, so the target set "
            "must be the seed atlas itself (--targets aseg / CHA+aseg / Tian_Subcortex / "
            "CHA+Tian_Subcortex), or aseg-mean with aseg seeds, not a cortical target")
    _hemis = [roi_hemisphere(_r) for _r in rois]
    _untagged = [_r for _r, _h in zip(rois, _hemis) if _h is None]
    assert not _untagged, (
        f"--targets-hemisphere contralateral needs a hemisphere per ROI to define "
        f"'opposite'; cannot parse {_untagged[:3]} in atlas={atlas}")
    contra_idx = {}
    for _i, _h in enumerate(_hemis):
        _keep = np.zeros(targets_ts.shape[2], bool)
        for _j, _hj in enumerate(_hemis):
            if _hj != _h:
                _keep[_j if UNIT_TARGETS else sls[_j]] = True
        contra_idx[_i] = np.where(_keep)[0]
    print(f"contralateral targets: {len(contra_idx[0])} / {len(contra_idx[1])} "
          f"{'ROI means' if UNIT_TARGETS else 'voxels'} "
          f"(seed 0 / seed 1), instead of {targets_ts.shape[2]}")

chunk_tag = f"sl{sl_start:04d}-{sl_stop:04d}"
outdir = os.path.join(RESULTS, dataset,
    out_root, args.out_subdir,
    f"memmap_ISC_loo_{atlas}_{task}_summary",
    chunk_tag)
os.makedirs(outdir, exist_ok=True)
print("Output dir:", outdir)
meta_fn = os.path.join(outdir, "meta.txt")
if not os.path.exists(meta_fn):
    with open(meta_fn, "w") as f:
        f.write(f"task={task}\n")
        f.write(f"atlas={atlas}\n")
        f.write(f"connectivity_source_space={source_space}\n")
        f.write(f"nSubject={N}\n")
        f.write(f"sl_start={sl_start}\n")
        f.write(f"sl_stop={sl_stop}\n")
        f.write(f"transformation_folder={transformation_folder}\n")

conn_dir = os.path.join(outdir, f"connectivity_targets-{targets}" + ("-contralateral" if CONTRA else ""))
shape = (sl_stop - sl_start, N)
all_mm = {
    "resp_coarse":     (os.path.join(outdir, "isc_resp_coarse.dat"),     shape),
    "resp_coarsefine": (os.path.join(outdir, "isc_resp_coarsefine.dat"), shape),
    "resp_fine":       (os.path.join(outdir, "isc_resp_fine.dat"),       shape),
    "conn_coarse":     (os.path.join(conn_dir, "isc_conn_coarse.dat"),     shape),
    "conn_coarsefine": (os.path.join(conn_dir, "isc_conn_coarsefine.dat"), shape),
    "conn_fine":       (os.path.join(conn_dir, "isc_conn_fine.dat"),       shape),
    "rdm_coarsefine":  (os.path.join(outdir, "isc_rdm_coarsefine.dat"),  shape),
}
METRICS = set(args.metrics)
mm = {k: ensure_memmap(path, shp) for k, (path, shp) in all_mm.items() if k in METRICS}
print(f"metrics: {sorted(METRICS)}")

# Voxelwise ISC is computed ONCE over every voxel, outside the region loop: the value at voxel v
# depends only on v's own time series (response) or connectivity, whichever region v belongs to.
# Only the undecomposed data has a region-free per-voxel definition ('coarse' is a region mean and
# 'fine' subtracts one), so there are no coarse/fine voxelwise variants.
V_TOTAL = dms.shape[2]

# conn_voxelwise is undefined under the contralateral restriction (the target set differs per nucleus).
if CONTRA and "conn_voxelwise" in METRICS:
    print("conn_voxelwise dropped: contralateral targets differ per nucleus")
    METRICS.discard("conn_voxelwise")
all_mm_vox = {
    "resp_voxelwise": (os.path.join(outdir, "isc_resp_voxelwise.dat"), (V_TOTAL, N)),
    "conn_voxelwise": (os.path.join(conn_dir, "isc_conn_voxelwise.dat"), (V_TOTAL, N)),
}
mm_vox = {k: ensure_memmap(path, shp) for k, (path, shp) in all_mm_vox.items() if k in METRICS}
_need_vox = {k for k, m in mm_vox.items()
             if not all(row_done(m[v]) for v in (0, V_TOTAL - 1))}
if not mm_vox:
    print("voxelwise ISC skipped (no voxelwise metric requested)")
elif _need_vox:
    print(f"Computing voxelwise ISC ({sorted(_need_vox)}) over all {V_TOTAL} voxels...")
    Xb = np.asarray(dms)  # (N, T, V_TOTAL)
    if "resp_voxelwise" in _need_vox:
        mm_vox["resp_voxelwise"][:] = loo_voxelwise_corrsim_dot(Xb, average=False).T
        mm_vox["resp_voxelwise"].flush()
    if "conn_voxelwise" in _need_vox:
        feats = conn_features_all_subjects(targets_ts, Xb, zscore_axis=0, flatten=False)
        mm_vox["conn_voxelwise"][:] = loo_voxelwise_corrsim_dot(feats, average=False).T
        mm_vox["conn_voxelwise"].flush()
    print("voxelwise ISC done")
else:
    print("voxelwise ISC already computed, skipping")

sids_txt = os.path.join(outdir, "sids.txt")
if not os.path.exists(sids_txt):
    with open(sids_txt, "w") as f:
        for sid in sids:
            f.write(f"{sid}\n")

# ---- compute seed region by seed region (checkpointed: finished rows are skipped) ----
print(f"Computing {len(mm)} metrics per seed region...")
for curr_row, isl in enumerate(range(sl_start, sl_stop)):
    sl_idx = sls[isl]

    X = dms[:, :, sl_idx]  # (N,T,Vsl)
    X_by_decomp = {"coarsefine": X}
    if METRICS & {"resp_coarse", "resp_fine", "conn_coarse", "conn_fine"}:
        X_by_decomp["coarse"], X_by_decomp["fine"] = parse_data_coarse_fine(X)
    # targets for THIS seed: the whole set, or only the opposite hemisphere
    tgt_ts = np.asarray(targets_ts[:, :, contra_idx[isl]]) if CONTRA else targets_ts

    for _d, _X in X_by_decomp.items():
        if f"resp_{_d}" in mm and not row_done(mm[f"resp_{_d}"][curr_row]):
            mm[f"resp_{_d}"][curr_row] = loo_corrsim_dot(_X); mm[f"resp_{_d}"].flush()
        if f"conn_{_d}" in mm and not row_done(mm[f"conn_{_d}"][curr_row]):
            feats = conn_features_all_subjects(tgt_ts, _X, zscore_axis=0)
            mm[f"conn_{_d}"][curr_row] = loo_corrsim_dot(feats); mm[f"conn_{_d}"].flush()

    # RDM (defined on the undecomposed data only)
    if "rdm_coarsefine" in mm and not row_done(mm["rdm_coarsefine"][curr_row]):
        rdmF = rdm_features_all_subjects(X)
        mm["rdm_coarsefine"][curr_row] = loo_corrsim_dot(rdmF); mm["rdm_coarsefine"].flush()

    if isl % 10 == 0:
        print(f"  isl {isl+1}/{nSL} done")

print("All done.")
