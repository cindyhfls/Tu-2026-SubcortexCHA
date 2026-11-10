"""Cache Raiders cortical (onavg-ico32) time series, and their cortical-CHA transformed version, as float32.

Cortical counterpart of cache_subcortical_raiders.py: `ts_l/ts_r_float32.npy` hold the AA data;
with --transformation-folder the searchlight transforms from transformation_to_template_raiders.py are applied and
stored as `onavg-ico32_1step_pial_overlap/<transformation folder>/dms_lr_float32.npy`. The transformed
cortex provides the "hyperaligned cortical targets" for the connectivity ISC.

Usage: python cache_cortical_raiders.py --run 3 4 [--transformation-folder ridgeCV/transformation_to_template_targets_onavg-ico8 --align-method ridgeCV]
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

from utils.data_io_utils import get_dm

parser = argparse.ArgumentParser()
parser.add_argument("--targets", type=str, default="onavg-ico8")
parser.add_argument("--transformation-folder", type=str, default="",
                     help="Empty string = AA (no transformation, just caches the raw ts_l/ts_r). "
                          "e.g. 'ridgeCV/transformation_to_template_targets_onavg-ico8/', "
                          "'self_shrinkage_alpha1e+01/transformation_to_self_targets_onavg-ico8/'.")
parser.add_argument("--run", type=int, nargs="+", default=[3, 4],
                    help="Runs to cache. The cache directory is run<runs>, e.g. run34 (the\n"
                         "default) or run12 -- same convention as cache_subcortical_raiders.py and the run_tag fig2_3_loo.py derives\n"
                         "from its --run.")
parser.add_argument("--align-method", type=str, default="",
                     help="e.g. 'procr', 'ridge', 'ridgeCV' -- must match the transform filenames "
                          "under --transformation-folder. Unused when --transformation-folder is empty.")
args = parser.parse_args()

dataset = 'Raiders'
surface_space = 'onavg-ico32'
volume_space = 'mni-2mm'
surface_resample = '1step_pial_overlap'
volume_resample = '1step_linear_overlap'
prep = 'default'
kwargs = { # for older datasets, there might be multiple versions/folders
    'space': [surface_space, volume_space],
    'resample': [surface_resample,volume_resample],
    'prep':prep
    }
task = "raiders"
run = args.run
source_space = "onavg-ico32"
saved_beta_path = f"{CONFOUND_BETAS}/{dataset}"
cache_dir = f"{CACHE}/{dataset}_{task}/run{''.join(map(str, run))}"
targets = args.targets
transformation_folder = args.transformation_folder
align_method = args.align_method

transform_dir = RESULTS / dataset / f"{surface_space}_{surface_resample}" / transformation_folder

if transformation_folder:
    out_root = f"{surface_space}_{surface_resample}/{transformation_folder}"
else:
    out_root = f"{surface_space}_{surface_resample}/AA/"

os.makedirs(os.path.join(cache_dir,out_root), exist_ok=True)
print("saving to:", os.path.join(cache_dir,out_root))

t = pd.read_csv(
    ROOT / "config" / "raiders_subj_not_in_budapest_template.txt",
    header=None,
    names=["participant_id"]
)
sids = np.array(t.participant_id).astype(str)
print("N total in file:", len(sids))
N = len(sids)
dms_fn = os.path.join(cache_dir,out_root,"dms_lr_float32.npy")
if os.path.exists(dms_fn):
    sys.exit(0)
else:
    if os.path.exists(os.path.join(cache_dir, "ts_l_float32.npy")) and os.path.exists(os.path.join(cache_dir, "ts_r_float32.npy")):
        saved_sids = np.load(os.path.join(cache_dir, "sids.npy"))
        assert np.array_equal(sids.astype(str), saved_sids.astype(str)), "Cached sids mismatch!"
    else:
        os.makedirs(os.path.join(cache_dir),exist_ok=True)
        dset=nb.datasets.datasets[dataset.lower()](fp_version="20.2.7",**kwargs)
        ts_l = get_dm(dset, sids, "l", task, run, source_space,
                        saved_beta_path=saved_beta_path, array_output=True).astype(np.float32, copy=False)  # (N,T,Vl)
        np.save(os.path.join(cache_dir, "sids.npy"), sids.astype("U"))
        np.save(os.path.join(cache_dir, "ts_l_float32.npy"), ts_l.astype(np.float32, copy=False))
        print('ts_l saved')
        del ts_l
        ts_r = get_dm(dset, sids, "r", task, run, source_space,
                        saved_beta_path=saved_beta_path, array_output=True).astype(np.float32, copy=False)  # (N,T,Vr)
        np.save(os.path.join(cache_dir, "ts_r_float32.npy"), ts_r.astype(np.float32, copy=False))
        print('ts_r saved')
        del ts_r

    ts_l = np.load(os.path.join(cache_dir, "ts_l_float32.npy"), mmap_mode="r")
    ts_r = np.load(os.path.join(cache_dir, "ts_r_float32.npy"), mmap_mode="r")
    assert ts_l.ndim == 3 and ts_r.ndim == 3, f"Expected (N,T,V). Got {ts_l.shape} and {ts_r.shape}"
    N2, T, Vl = ts_l.shape
    assert N2 == N, "Subject count mismatch"
    Vr = ts_r.shape[2]
    print("Shapes:", ts_l.shape, ts_r.shape)
    
    if transformation_folder:
        dms_lr = np.empty((N, T, Vl + Vr), dtype=np.float32)
        for si, sid in enumerate(sids):
            xfm_l = nb.load(str(transform_dir / f"{sid}_targets-{targets}_lh_{align_method}.npz"))
            xfm_r = nb.load(str(transform_dir / f"{sid}_targets-{targets}_rh_{align_method}.npz"))
            dms_lr[si, :, :Vl] = np.asarray(ts_l[si], np.float32) @ xfm_l
            dms_lr[si, :, Vl:] = np.asarray(ts_r[si], np.float32) @ xfm_r
        # ---- atomic save ----
        tmp = dms_fn + ".tmp.npy"   # ends with .npy
        np.save(tmp, dms_lr)
        os.replace(tmp, dms_fn)