"""Within-subject (WSC) vs between-subject (ISC) connectivity reliability across the two test runs.

The test data (Raiders runs 3 and 4, concatenated) are split into its two runs. For each seed nucleus
and subject i, connectivity features are computed separately in run 3 (run0) and run 4 (run1) and
  withinsubject               WSC_i = corr(subject i run 3, subject i run 4)
  acrosssubjectacrossruns     ISC_i = Fisher mean over j != i of corr(subject i run 3, subject j run 4)
  run0 / run1                 pairwise ISC within a single run, Fisher-averaged per subject.
Both quantities cross runs and differ only in whose data are paired. coarse / coarsefine / fine as in
fig2_3_loo.py.

Manuscript: Supp Fig 8 (WSC vs ISC noise ceiling of the connectivity; AA and CHA, connectivity to
hyperaligned cortex and to contralateral subcortex).

Usage:  python supp_isc_wsc_splitruns.py --atlas aseg --targets CHA+aseg \
            --transformation-folder ridgeCV/transformation_to_template_targets_onavg-ico8/ \
            --targets-transformation-folder ridgeCV/transformation_to_template_targets_onavg-ico8/ \
            --targets-hemisphere contralateral
        (labels as in fig2_3_loo.py; AA: empty --transformation-folder)
Input:  data/cache/Raiders_raiders/run34/ (written by cache_subcortical_raiders.py and cache_cortical_raiders.py)
Output: results/Raiders/<space>_<resample>/<transformation folder or AA>/
        memmap_ISC_pairwiseavg_splitruns_<atlas>_raiders_summary/sl<start>-<stop>/
        connectivity_targets-<targets>[-contralateral]/isc_conn_<variant>_<kind>.dat
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
from utils.data_io_utils import ensure_memmap,row_done,return_aseg_labels,get_aseg_sls,return_Tian_labels,get_Tian_sls,roi_hemisphere
from utils.hyperalignment_utils import get_target_ts
from utils.isc_utils import pairwise_corrsim_dot_avg,pairwise_corrsim_dot_xy_avg,parse_data_coarse_fine,conn_features_all_subjects,compute_targets_ts_from_mapping

parser = argparse.ArgumentParser()
parser.add_argument("--sl-start", type=int, default=0)
parser.add_argument("--sl-stop", type=int, default=None)
parser.add_argument("--targets", type=str, default="onavg-ico32")
parser.add_argument("--atlas", type=str, default="aseg")
parser.add_argument("--source-space", type=str, default="mni-2mm")
parser.add_argument("--transformation-folder", type=str, default="")
parser.add_argument("--targets-transformation-folder", type=str, default="")
parser.add_argument("--targets-hemisphere", type=str, default="both",
                    choices=["both", "contralateral"],
                    help="contralateral: restrict each seed's subcortical targets to the opposite "
                         "hemisphere, as in fig2_3_loo.py")
parser.add_argument("--run-ends", type=int, nargs=2, default=[840, 1680],
                    help="end frame of run 3 and of run 4 in the concatenated test data "
                         "(Raiders runs 3-4: 840 frames each)")
args = parser.parse_args()

dataset = "Raiders"
task = "raiders"
run = [3, 4]
run_ends = args.run_ends
cache_dir = f"{CACHE}/{dataset}_{task}/run{''.join(map(str, run))}"
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

sids = np.load(os.path.join(cache_dir, "sids.npy")).astype(str)
N = len(sids)
print("N total in file:", N)

# seed data (N, T, V)
if transformation_folder:
    if atlas.startswith("Tian"):
        dms_fn = os.path.join(cache_dir, out_root, "dms_Tian_Subcortex_float32.npy")
    else:
        dms_fn = os.path.join(cache_dir, out_root, f"dms_{atlas.replace('aseg', 'aseg_Subcortex')}_float32.npy")
    dms = np.load(dms_fn, mmap_mode="r")
elif atlas == 'lr':
    ts_l = np.load(os.path.join(cache_dir, "ts_l_float32.npy"), mmap_mode="r")
    ts_r = np.load(os.path.join(cache_dir, "ts_r_float32.npy"), mmap_mode="r")
    dms = np.concatenate([ts_l, ts_r], axis=2).astype(np.float32, copy=False)  # (N,T,Vall)
elif atlas.startswith("Tian"):
    dms = np.load(os.path.join(cache_dir, "ts_Tian_Subcortex_float32.npy"), mmap_mode="r")
else:
    dms = np.load(os.path.join(cache_dir, f"ts_{atlas.replace('aseg', 'aseg_Subcortex')}_float32.npy"), mmap_mode="r")

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
    # The target is the seed data itself, so with --transformation-folder it is the transformed
    # data (intended), while the output folder is named connectivity_targets-<atlas>.
    targets_ts = dms
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
    assert targets_ts.shape[2] == dms.shape[2], (
        "contralateral restriction indexes the TARGET by seed-atlas voxel, so the target set "
        "must be the seed atlas itself (e.g. --targets aseg or CHA+aseg), not a cortical target")
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
    print(f"contralateral targets: {len(contra_idx[0])} / {len(contra_idx[len(rois) // 2])} voxels "
          f"(first left / first right seed), instead of {targets_ts.shape[2]}")
CONN_SUBDIR = f"connectivity_targets-{targets}" + ("-contralateral" if CONTRA else "")

chunk_tag = f"sl{sl_start:04d}-{sl_stop:04d}"
outdir = os.path.join(RESULTS, dataset,
    out_root,
    f"memmap_ISC_pairwiseavg_splitruns_{atlas}_{task}_summary",
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

KINDS = ["run0", "run1", "withinsubject", "acrosssubjectacrossruns"]
VARIANTS = ["coarse", "coarsefine", "fine"]
mm = {f"conn_{v}_{k}": ensure_memmap(os.path.join(outdir, CONN_SUBDIR, f"isc_conn_{v}_{k}.dat"),
                                     (sl_stop - sl_start, N))
      for v in VARIANTS for k in KINDS}

sids_txt = os.path.join(outdir, "sids.txt")
if not os.path.exists(sids_txt):
    with open(sids_txt, "w") as f:
        for sid in sids:
            f.write(f"{sid}\n")

# ---- compute seed region by seed region (checkpointed: finished rows are skipped) ----
print("Computing 3 connectivity metrics per seed region...")
for curr_row, isl in enumerate(range(sl_start, sl_stop)):
    sl_idx = sls[isl]
    X = dms[:, :, sl_idx]  # (N,T,Vsl)
    X_coarse, X_fine = parse_data_coarse_fine(X)
    # targets for THIS seed: the whole set, or only the opposite hemisphere
    tgt_ts = np.asarray(targets_ts[:, :, contra_idx[isl]]) if CONTRA else targets_ts

    for v, _X in (("coarse", X_coarse), ("coarsefine", X), ("fine", X_fine)):
        feats0 = conn_features_all_subjects(tgt_ts[:, :run_ends[0], :], _X[:, :run_ends[0], :], zscore_axis=0)
        feats1 = conn_features_all_subjects(tgt_ts[:, run_ends[0]:run_ends[1], :], _X[:, run_ends[0]:run_ends[1], :], zscore_axis=0)
        for k, feats in (("run0", feats0), ("run1", feats1)):
            key = f"conn_{v}_{k}"
            if not row_done(mm[key][curr_row]):
                mm[key][curr_row] = pairwise_corrsim_dot_avg(feats); mm[key].flush()
        k_w, k_a = f"conn_{v}_withinsubject", f"conn_{v}_acrosssubjectacrossruns"
        if not row_done(mm[k_w][curr_row]) or not row_done(mm[k_a][curr_row]):
            across, within = pairwise_corrsim_dot_xy_avg(feats0, feats1)
            mm[k_w][curr_row] = within; mm[k_w].flush()
            mm[k_a][curr_row] = across; mm[k_a].flush()

    if isl % 10 == 0:
        print(f"  isl {isl+1}/{nSL} done")

print("All done.")
