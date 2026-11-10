"""Build the Budapest connectivity template for the subcortex.

Connectivity of every subcortical voxel to the cortical onavg-ico8 targets (1210 vertices) is computed in
the 10 template subjects (Budapest, runs 1-5) and the shared structure is extracted with PCA + Procrustes
(Methods, "Connectivity hyperalignment (CHA) procedure").

  --atlas aseg           one template per nucleus (16 aseg nuclei, Figs 1-5)         [compute_template]
  --atlas aseg_wholebrain ONE template over all aseg voxels (concatenated); each nucleus is a
                         searchlight inside it (pooled alpha, Supp Fig 18)               [searchlight_template]
  --atlas TianS{1..4}    ONE template over all Tian voxels (concatenated); each Tian parcel of the chosen
                         scale is a searchlight inside it (Fig 6)                       [searchlight_template]

Usage:  python template_budapest_subcortical.py --atlas aseg
        python template_budapest_subcortical.py --atlas aseg_wholebrain
        python template_budapest_subcortical.py --atlas TianS3
Output: results/Budapest/mni-2mm_1step_linear_overlap/templates_aseg/targets_onavg-ico8_<roi>.npy          (aseg)
        results/Budapest/mni-2mm_1step_linear_overlap/templates_TianS3_wholebrain/targets_onavg-ico8_Tian_Subcortex.npy
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import neuroboros as nb
import numpy as np
from hyperalignment import compute_template, searchlight_template
from scipy.spatial.distance import cdist
from scipy.stats import zscore

from utils.data_io_utils import apply_dm_transform, get_aseg_sls, get_dm, get_Tian_sls, return_aseg_labels
from utils.hyperalignment_utils import calculate_connectivity
from utils.paths import CONFOUND_BETAS, RESULTS, ROOT, resolve  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--atlas", required=True, help="aseg, aseg_wholebrain, or TianS1 / TianS2 / TianS3 / TianS4")
parser.add_argument("--outroot", default=str(RESULTS), help="root of the output tree (default: results/)")
parser.add_argument("--n-jobs", type=int, default=4, help="searchlight_template workers (Tian only)")
args = parser.parse_args()
if args.atlas not in ('aseg', 'aseg_wholebrain') and not args.atlas.startswith('Tian'):
    parser.error("--atlas must be aseg, aseg_wholebrain or TianS{1..4}")

DATASET, TASK = 'Budapest', 'budapest'
RUNS = [1, 2, 3, 4, 5]
SURFACE_SPACE, VOLUME_SPACE = 'onavg-ico32', 'mni-2mm'
SURFACE_RESAMPLE, VOLUME_RESAMPLE = '1step_pial_overlap', '1step_linear_overlap'

sids = list(np.loadtxt(ROOT / 'config' / 'budapest_subj_in_template.txt', dtype=str))  # the 10 template subjects
tag = args.atlas if args.atlas.startswith('aseg') else f'{args.atlas}_wholebrain'
template_dir = Path(args.outroot) / DATASET / f'{VOLUME_SPACE}_{VOLUME_RESAMPLE}' / f'templates_{tag}'
os.makedirs(template_dir, exist_ok=True)

dset = nb.Budapest(fp_version='20.2.7', space=[SURFACE_SPACE, VOLUME_SPACE],
                   resample=[SURFACE_RESAMPLE, VOLUME_RESAMPLE])
start = datetime.now()

if args.atlas == 'aseg':
    rois = return_aseg_labels(atlas_name=VOLUME_SPACE)
    ts_sub = [{roi: dset.get_data(sid, TASK, RUNS, roi, space=VOLUME_SPACE) for roi in rois} for sid in sids]
    ts_ctx = [{lr: dset.get_data(sid, TASK, RUNS, lr, space=SURFACE_SPACE) for lr in 'lr'} for sid in sids]
    mappings = {lr: nb.mapping(lr, SURFACE_SPACE, 'onavg-ico8', mask=True) for lr in 'lr'}
    print('data loaded', datetime.now() - start)
    for roi in rois:
        conn = []
        for ts_s, ts_c in zip(ts_sub, ts_ctx):
            targets = np.concatenate([ts_c['l'] @ mappings['l'], ts_c['r'] @ mappings['r']], axis=1)
            c = 1 - cdist(targets.T, ts_s[roi].T, 'correlation')
            conn.append(np.nan_to_num(zscore(c, nan_policy='omit', axis=0)))
        conn = np.stack(conn)
        out_fn = template_dir / f'targets_onavg-ico8_{roi}.npy'
        nb.record(str(out_fn), compute_template)(conn, kind='pca', max_npc=conn.shape[1],
                                                 common_topography=True)
        print(roi, 'template calculated', datetime.now() - start)
else:
    # one concatenated template over all voxels; the parcels (aseg nuclei, or Tian parcels of this scale)
    # are the searchlights
    saved_beta_path = str(CONFOUND_BETAS / DATASET)
    is_aseg = args.atlas.startswith('aseg')
    wb_atlas = args.atlas if is_aseg else f'{args.atlas}_wholebrain'
    wb_roi = 'aseg_Subcortex' if is_aseg else 'Tian_Subcortex'
    dms = get_dm(dset, sids, [wb_roi], TASK, RUNS, VOLUME_SPACE, atlas=wb_atlas,
                 prep='default', array_output=True)
    mappings = {lr: nb.mapping(lr, SURFACE_SPACE, 'onavg-ico8', mask=True) for lr in 'lr'}
    target_ts = [apply_dm_transform(get_dm(dset, sids, [lr], TASK, RUNS, SURFACE_SPACE, atlas='',
                                           saved_beta_path=saved_beta_path, array_output=True),
                                    mappings[lr]) for lr in 'lr']
    targets_ts = np.concatenate(target_ts, axis=2)
    dm = np.stack([calculate_connectivity(targets_ts[i], dms[i], zscore_axis=0) for i in range(len(sids))], axis=0)
    sls = get_aseg_sls() if is_aseg else get_Tian_sls(args.atlas)
    out_fn = template_dir / f'targets_onavg-ico8_{wb_roi}.npy'
    if out_fn.exists():
        print(f'{out_fn} exists, skipping')
    else:
        nb.record(str(out_fn), searchlight_template)(dm, sls, None, None, n_jobs=args.n_jobs, tpl_kind='pca')
        print('template calculated', datetime.now() - start)
