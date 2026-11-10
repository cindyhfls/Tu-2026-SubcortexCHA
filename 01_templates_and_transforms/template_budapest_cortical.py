"""Build the Budapest connectivity template for the cortex (searchlight CHA, onavg-ico32 vertices).

Connectivity of every cortical vertex to the 1210 cortical onavg-ico8 targets is computed in the 10
template subjects (Budapest, runs 1-5), and a template is built in every 20 mm searchlight by PCA +
Procrustes (`searchlight_template`); the searchlight templates are aggregated with distance weights
(the transformations, in contrast, are aggregated unweighted -- see transformation_to_template_raiders.py). Used for the hyperaligned cortical targets
(Methods, "Connectivity hyperalignment (CHA) procedure").

Usage:  python template_budapest_cortical.py [--outroot DIR]
Output: results/Budapest/onavg-ico32_1step_pial_overlap/templates_renamed/targets_onavg-ico8_{l,r}h.npy
        (the folder name is the one the cortical configs in config/ read)
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import neuroboros as nb
import numpy as np
from hyperalignment import searchlight_template
from scipy.spatial.distance import cdist
from scipy.stats import zscore

from utils.paths import RESULTS, ROOT, resolve  # noqa: E402

RADIUS = 20  # mm, searchlight radius
WEIGHTED = True  # distance-weighted aggregation of the searchlight templates (passing dists + radius)
TASK = 'budapest'
RUNS = [1, 2, 3, 4, 5]
CORTICAL_SPACE = 'onavg-ico32'
VOLUME_SPACE = 'mni-2mm'
VOLUME_RESAMPLE = '1step_linear_overlap'
SURFACE_RESAMPLE = '1step_pial_overlap'

parser = argparse.ArgumentParser()
parser.add_argument('--outroot', default=str(RESULTS), help='root of the output tree (default: results/)')
args = parser.parse_args()

sids = list(np.loadtxt(ROOT / 'config' / 'budapest_subj_in_template.txt', dtype=str))  # the 10 template subjects
save_path = Path(args.outroot) / 'Budapest' / f'{CORTICAL_SPACE}_{SURFACE_RESAMPLE}' / 'templates_renamed'
os.makedirs(save_path, exist_ok=True)

dset = nb.Budapest(fp_version='20.2.7', space=[CORTICAL_SPACE, VOLUME_SPACE],
                   resample=[SURFACE_RESAMPLE, VOLUME_RESAMPLE])
mappings = {lr: nb.mapping(lr, CORTICAL_SPACE, 'onavg-ico8', mask=True) for lr in 'lr'}

start = datetime.now()
ts = {sid: {lr: dset.get_data(sid, TASK, RUNS, lr, space=CORTICAL_SPACE) for lr in 'lr'} for sid in sids}
print('data loaded', datetime.now() - start)

# connectivity to the ico8 targets of both hemispheres, z-scored over the target axis
conn = {lr: [] for lr in 'lr'}
for sid in sids:
    targets = np.concatenate([ts[sid]['l'] @ mappings['l'], ts[sid]['r'] @ mappings['r']], axis=1)
    for lr in 'lr':
        c = 1 - cdist(targets.T, ts[sid][lr].T, 'correlation')
        conn[lr].append(np.nan_to_num(zscore(c, nan_policy='omit', axis=0)))
conn = {lr: np.stack(conn[lr]) for lr in 'lr'}
print('connectivity calculated', datetime.now() - start)

for lr in 'lr':
    sls, dists = nb.sls(lr, RADIUS, mask=True, return_dists=True)
    out_fn = save_path / f'targets_onavg-ico8_{lr}h.npy'
    # hyperalignment.searchlight_template weights by distance when dists and radius are given, uniformly when None
    d, r = (dists, RADIUS) if WEIGHTED else (None, None)
    nb.record(str(out_fn), searchlight_template)(conn[lr], sls, d, r, n_jobs=1)
    print(out_fn, 'template calculated', datetime.now() - start)
