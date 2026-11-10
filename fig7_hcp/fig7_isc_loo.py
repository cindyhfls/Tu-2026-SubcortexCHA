"""Leave-one-out ISC of subcortical responses and connectivity in the HCP 7T movie data.

HCP 7T MOVIE runs 3-4 (34 test subjects), seeds = Tian parcels (7T Tian atlas, 1.6 mm). Same features
and conditions as fig2_3_isc/fig2_3_loo.py (resp_/conn_/rdm_ x coarse/coarsefine/fine,
leave-one-out ISC, optional contralateral targets), but for the HCP cache layout
(data/cache/HCP_MOVIE/run34/<space>/<transformation folder>/dms_*.npy).

Manuscript: Fig 7 (HCP 7T replication; AA vs ridgeCV transforms fit at Tian S1-S4, evaluated at the
TianS1 parcellation). Conditions: AA, and ridgeCV_TianS<s>_wholebrain for s = 1..4; targets are
contralateral Tian subcortex (--targets Tian_Subcortex / CHA+Tian_Subcortex with
--targets-hemisphere contralateral) and cortex (onavg-ico32, or CHA+onavg-ico32 for hyperaligned
cortex). See slurm/run_isc.sh.

Usage:  python fig7_isc_loo.py --atlas TianS1 --source-space mni-1pt6mm \
            --targets CHA+Tian_Subcortex --targets-hemisphere contralateral \
            --transformation-folder ridgeCV_TianS1_wholebrain/transformation_to_template_targets_onavg-ico8 \
            --targets-transformation-folder ridgeCV_TianS1_wholebrain/transformation_to_template_targets_onavg-ico8 \
            --metrics conn_coarse conn_coarsefine conn_fine
Input:  data/cache/HCP_MOVIE/run34/ (or --cache-dir)
Output: results/HCP/mni-1pt6mm/<transformation folder or AA>/memmap_ISC_loo_<atlas>_MOVIE_summary/
        sl<start>-<stop>/[connectivity_targets-<targets>[-contralateral]/]isc_*.dat
        (float32 memmaps, rows = seed parcels, columns = subjects)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import CACHE, RESULTS, resolve  # noqa: E402

import os
import argparse
import numpy as np
from utils.data_io_utils import ensure_memmap,row_done,return_Tian_labels,get_Tian_sls,roi_hemisphere
from utils.hyperalignment_utils import get_target_ts
from utils.isc_utils import loo_corrsim_dot,parse_data_coarse_fine,conn_features_all_subjects,rdm_features_all_subjects,compute_targets_ts_from_mapping

parser = argparse.ArgumentParser()
parser.add_argument("--sl-start", type=int, default=0)
parser.add_argument("--sl-stop", type=int, default=None)
parser.add_argument("--targets", type=str, default="onavg-ico32",
                    help="connectivity targets used for the ISC; labels as in fig2_3_loo.py")
parser.add_argument("--atlas", type=str, default="TianS1")
parser.add_argument("--source-space", type=str, default="mni-1pt6mm")
parser.add_argument("--transformation-folder", type=str, default="",
                    help="transform applied to the seeds (relative to mni-1pt6mm/); empty = AA")
parser.add_argument("--targets-transformation-folder", type=str, default="",
                    help="transform the CHA+ targets went through (relative to the target space folder)")
parser.add_argument("--targets-hemisphere", type=str, default="both",
                    choices=["both", "contralateral"],
                    help="contralateral: restrict each seed parcel's targets to the opposite hemisphere; "
                         "needs voxelwise targets on the seed atlas (Tian_Subcortex / CHA+Tian_Subcortex)")
_all_metrics = ["resp_coarse", "resp_coarsefine", "resp_fine",
                "conn_coarse", "conn_coarsefine", "conn_fine", "rdm_coarsefine"]
parser.add_argument("--metrics", type=str, nargs="+", default=_all_metrics, choices=_all_metrics,
                    help="Compute only these; a skipped metric is neither opened nor written.")
parser.add_argument("--cache-dir", type=str, default=None,
                    help="Cache root holding sids.npy, ts_*.npy and <space>/<transform>/dms_*.npy. "
                         "Default: data/cache/HCP_MOVIE/run34.")
args = parser.parse_args()
METRICS = set(args.metrics)

dataset = "HCP"
task = "MOVIE"
run = [3, 4]
cache_dir = args.cache_dir or f"{CACHE}/{dataset}_{task}/run{''.join(map(str, run))}"

volume_space = 'mni-1pt6mm'
surface_space = 'onavg-ico32'
transformation_folder = args.transformation_folder
source_space = args.source_space
atlas = args.atlas
targets = args.targets
sl_start = args.sl_start
sl_stop = args.sl_stop

# HCP caches have no resample suffix in their folder names
if 'mni' in source_space:
    source_folder = f"{volume_space}"
else:
    source_folder = f"{surface_space}"
if transformation_folder:
    out_root = f"{source_folder}/{transformation_folder}"
else:
    out_root = f"{source_folder}/AA/"
targets_transformation_folder = args.targets_transformation_folder
if 'onavg' in targets:
    targets_out_root = f"{surface_space}/{targets_transformation_folder}"
else:
    targets_out_root = f"{volume_space}/{targets_transformation_folder}"

sids = np.load(os.path.join(cache_dir, "sids.npy")).astype(str)
N = len(sids)
print("N total in file:", N)

# seed data (N, T, V): Tian seeds only (7T Tian atlas, 13974 voxels)
assert atlas.startswith("Tian"), f"only Tian seed atlases are supported for HCP, got {atlas}"
if transformation_folder:
    dms = np.load(os.path.join(cache_dir, out_root, "dms_Tian_Subcortex_float32.npy"), mmap_mode="r")
else:
    dms = np.load(os.path.join(cache_dir, "ts_Tian_Subcortex_float32.npy"), mmap_mode="r")

rois = return_Tian_labels(atlas, is_7T=True)
sls = get_Tian_sls(atlas, is_7T=True)
nSL = len(sls)
sl_stop = nSL if sl_stop is None else min(sl_stop, nSL)

# connectivity targets (N, T, nTargets)
if targets == atlas:
    targets_ts = dms
else:
    if targets.startswith('CHA+'):  # connectivity-hyperaligned targets
        if 'Tian' in targets:
            target_dms_fn = os.path.join(cache_dir, targets_out_root, "dms_Tian_Subcortex_float32.npy")
        elif 'onavg' in targets:
            target_dms_fn = os.path.join(cache_dir, targets_out_root, "dms_lr_float32.npy")
        else:
            raise ValueError(f"target not recognized: {targets}")
        targets_ts = np.load(target_dms_fn, mmap_mode="r")
    else:  # anatomically aligned targets
        if 'Tian' in targets:
            targets_ts = np.load(os.path.join(cache_dir, "ts_Tian_Subcortex_float32.npy"))
        elif 'onavg' in targets:
            ts_l = np.load(os.path.join(cache_dir, "ts_l_float32.npy"))
            ts_r = np.load(os.path.join(cache_dir, "ts_r_float32.npy"))
            targets_ts = np.concatenate([ts_l, ts_r], axis=2).astype(np.float32, copy=False)  # (N,T,Vall)
        else:
            raise ValueError(f"target not recognized: {targets}")

    # targets that are not dense voxel/vertex sets
    if 'onavg-ico8' in targets:
        targets_ts = compute_targets_ts_from_mapping(targets_ts, surface_space, 'onavg-ico8', mask=True)
    elif any([tt in targets for tt in ['TianS1', 'TianS2', 'TianS3', 'TianS4']]):
        sls_Tian = get_Tian_sls(targets.replace('CHA+', ''), is_7T=True)
        targets_ts = get_target_ts(targets_ts, 'searchlight_mean', target_item=sls_Tian)
print("targets_ts:", targets_ts.shape)

CONTRA = args.targets_hemisphere == "contralateral"
if CONTRA:
    assert targets_ts.shape[2] == dms.shape[2], (
        "contralateral restriction indexes the TARGET by seed-atlas voxel, so the target must be "
        "voxelwise on the seed atlas (Tian_Subcortex / CHA+Tian_Subcortex)")
    _hemis = [roi_hemisphere(_r) for _r in rois]
    _untagged = [_r for _r, _h in zip(rois, _hemis) if _h is None]
    assert not _untagged, f"cannot parse a hemisphere from {_untagged[:3]}"
    contra_idx = {}
    for _i, _h in enumerate(_hemis):
        _keep = np.zeros(targets_ts.shape[2], bool)
        for _j, _hj in enumerate(_hemis):
            if _hj != _h:
                _keep[sls[_j]] = True
        contra_idx[_i] = np.where(_keep)[0]
    print(f"contralateral targets: {len(contra_idx[0])} / {len(contra_idx[1])} voxels "
          f"(seed 0 / seed 1), instead of {targets_ts.shape[2]}")
CONN_SUBDIR = f"connectivity_targets-{targets}" + ("-contralateral" if CONTRA else "")

chunk_tag = f"sl{sl_start:04d}-{sl_stop:04d}"
outdir = os.path.join(RESULTS, dataset,
    out_root,
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

shape = (sl_stop - sl_start, N)
_mm_spec = {
    "resp_coarse":     (os.path.join(outdir, "isc_resp_coarse.dat"),     shape),
    "resp_coarsefine": (os.path.join(outdir, "isc_resp_coarsefine.dat"), shape),
    "resp_fine":       (os.path.join(outdir, "isc_resp_fine.dat"),       shape),
    "conn_coarse":     (os.path.join(outdir, CONN_SUBDIR, "isc_conn_coarse.dat"),     shape),
    "conn_coarsefine": (os.path.join(outdir, CONN_SUBDIR, "isc_conn_coarsefine.dat"), shape),
    "conn_fine":       (os.path.join(outdir, CONN_SUBDIR, "isc_conn_fine.dat"),       shape),
    "rdm_coarsefine":  (os.path.join(outdir, "isc_rdm_coarsefine.dat"),  shape),
}
mm = {k: ensure_memmap(*spec) for k, spec in _mm_spec.items() if k in METRICS}
print(f"metrics: {sorted(mm)}")

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
