"""Cache Raiders subcortical time series (and their transformed versions) as float32 arrays.

`ts_<atlas>_float32.npy` (N subjects x T x V) holds the confound-regressed, z-scored data (anatomical
alignment, AA). With --transformation-folder, each subject's transform from transformation_to_template_raiders.py
is applied voxelwise per nucleus and stored as `<space>/<transformation folder>/dms_<atlas>_float32.npy`.
The ISC / classification scripts read these caches.

Usage: python cache_subcortical_raiders.py --run 3 4                                   # AA
       python cache_subcortical_raiders.py --run 3 4 --transformation-folder ridgeCV/transformation_to_template_targets_onavg-ico8 --align-method ridgeCV
Output: data/cache/Raiders_raiders/run<runs>/...
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import CACHE, CONFOUND_BETAS, RESULTS, ROOT, resolve  # noqa: E402

import os
import argparse
import numpy as np
import pandas as pd
import neuroboros as nb
import sys
from utils.data_io_utils import get_dm,return_aseg_labels,return_Tian_labels,get_aseg_sls

parser = argparse.ArgumentParser()
parser.add_argument("--atlas", type=str, default="aseg")
parser.add_argument("--targets", type=str, default="onavg-ico8")
parser.add_argument("--transformation-folder", type=str, default="",
                     help="Empty string = AA (no transformation, just caches the raw ts). "
                          "e.g. 'procrustes_new/transformation_to_template_targets_onavg-ico8/', "
                          "'ridgeCV/transformation_to_template_targets_onavg-ico8/', "
                          "'self_shrinkage_alpha1e+01/transformation_to_self_targets_onavg-ico8/'.")
parser.add_argument("--align-method", type=str, default="",
                     help="e.g. 'procr', 'ridge', 'ridgeCV' -- must match the transform filenames "
                          "under --transformation-folder. Unused when --transformation-folder is empty.")
parser.add_argument("--run", type=int, nargs="+", default=[3, 4],
                    help="Runs to cache. The cache directory is run<runs>, e.g. run34 (the\n"
                         "default) or run12 -- the same run_tag fig2_3_loo.py derives from its --run.")
args = parser.parse_args()

dataset = 'Raiders'
surface_space = 'onavg-ico32'
volume_space = 'mni-2mm'
surface_resample = '1step_pial_overlap'
volume_resample = '1step_linear_overlap'

prep = 'default'
atlas = args.atlas
kwargs = { # for older datasets, there might be multiple versions/folders
    'space': [surface_space, volume_space],
    'resample': [surface_resample,volume_resample],
    'prep':prep
    }
task = "raiders"
run = args.run
source_space = "mni-2mm"
saved_beta_path = f"{CONFOUND_BETAS}/{dataset}"
cache_dir = f"{CACHE}/{dataset}_{task}/run{''.join(map(str, run))}"
xfm_targets = args.targets

transformation_folder = args.transformation_folder

transform_dir = RESULTS / dataset / f"{volume_space}_{volume_resample}" / transformation_folder
align_method = args.align_method

if transformation_folder:
    out_root = f"{volume_space}_{volume_resample}/{transformation_folder}"
else:
    out_root = f"{volume_space}_{volume_resample}/AA/"

os.makedirs(os.path.join(cache_dir,out_root), exist_ok=True)

with open(ROOT / "config" / "raiders_subj_not_in_budapest_template.txt") as f:
    sids = f.read().strip().split('\n') # use strip to get rid of the final new line
print("N total in file:", len(sids))
N = len(sids)
if 'mni' in source_space:
    if atlas == 'aseg':
        rois = return_aseg_labels(atlas_name=source_space)
        print(rois)
        sls = get_aseg_sls()
    elif atlas == 'aseg_Subcortex':
        rois = ['aseg_Subcortex']
        sls = [np.arange(10152)]
    elif atlas == 'Tian_Subcortex':
        rois = ['Tian_Subcortex']
        sls = [np.arange(8562)]

# 'aseg' means the per-ROI layout (16 separate transforms) but the cache it produces covers
# the whole subcortex, so both it and the single-transform 'aseg_Subcortex' write the same
# file. A blind .replace('aseg','aseg_Subcortex') turned the latter into
# 'aseg_Subcortex_Subcortex', a name fig2_3_loo.py never looks for -- hence the explicit map.
atlas_tag = 'aseg_Subcortex' if atlas in ('aseg', 'aseg_Subcortex') else atlas
dms_fn = os.path.join(cache_dir,out_root,f"dms_{atlas_tag}_float32.npy")
if os.path.exists(dms_fn):
    sys.exit(0)
else:
    if os.path.exists(os.path.join(cache_dir, f"ts_{atlas_tag}_float32.npy")):
        saved_sids = np.load(os.path.join(cache_dir, "sids.npy"))
        assert np.array_equal(sids, saved_sids.astype(str)), "Cached sids mismatch!"
    else:
        os.makedirs(os.path.join(cache_dir),exist_ok=True)
        dset=nb.datasets.datasets[dataset.lower()](fp_version="20.2.7",**kwargs)
        print(rois)
        print(atlas)
        ts = get_dm(dset, sids, rois, task, run, source_space, atlas=atlas,
                                saved_beta_path=saved_beta_path, array_output=True).astype(np.float32, copy=False)
        print(ts.shape)
        np.save(os.path.join(cache_dir, "sids.npy"), np.array(sids))
        np.save(os.path.join(cache_dir, f"ts_{atlas_tag}_float32.npy"), ts.astype(np.float32, copy=False))
        print('ts saved at' + os.path.join(cache_dir, f"ts_{atlas_tag}_float32.npy"))
        del ts

    ts = np.load(os.path.join(cache_dir, f"ts_{atlas_tag}_float32.npy"), mmap_mode="r")
    assert ts.ndim == 3 , f"Expected (N,T,V). Got {ts.shape}"
    N2, T, V = ts.shape
    assert N2 == N, "Subject count mismatch"

    if transformation_folder:
        dms = np.empty((N, T, V), dtype=np.float32)
        for si, sid in enumerate(sids):
            for roi,sl in zip(rois,sls):
                transform_fn = str(transform_dir / f"{sid}_targets-{xfm_targets}_{roi}_{align_method}.npy")
                try:
                    xfm = np.load(transform_fn)
                except FileNotFoundError:
                    xfm = nb.load(transform_fn.replace('.npy', '.npz'))
                dms[si, :, sl] = (np.asarray(ts[si,:,sl], np.float32).T @ xfm).T
        # ---- atomic save ----
        tmp = dms_fn + ".tmp.npy"   # ends with .npy
        np.save(tmp, dms)
        os.replace(tmp, dms_fn)
