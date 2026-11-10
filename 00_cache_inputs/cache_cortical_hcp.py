"""Cache HCP 7T movie cortical time series (and their hyperaligned versions) as float32 arrays (Fig 7).

Cortical counterpart of cache_subcortical_hcp.py: `ts_l_float32.npy` / `ts_r_float32.npy`
(N x T x V, 34 test subjects, movie runs 3-4, onavg-ico32, anatomical alignment) and, with the
ridge (alpha 1e3) cortical transforms from transformation_to_template_hcp.py
(config_align_to_template_ridge_hcp_onavg-ico32.json) applied, `dms_lr_float32.npy`
(left and right hemisphere concatenated). The latter provides the hyperaligned cortical targets for
the subcortical connectivity ISC.

Usage:  python cache_cortical_hcp.py
Output: data/cache/HCP_MOVIE/run34/{ts_l,ts_r}_float32.npy,
        onavg-ico32/<transformation folder>/dms_lr_float32.npy
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import neuroboros as nb
import numpy as np
from utils.data_io_utils import get_HCP
from utils.paths import CACHE, DATA, RESULTS, resolve  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--prep-root", default=str(DATA / "HCP_YA" / "prep_oneregression"))
parser.add_argument("--sids-file", default="config/HCP_movie_v2_complete.txt")
parser.add_argument("--cache-dir", default=None, help="Cache root; default data/cache/HCP_MOVIE/run<runs>")
parser.add_argument("--transformation-folder",
                    default="ridge_alpha1e3_onavg-ico32/transformation_to_template_targets_onavg-ico8/",
                    help="Cortical transforms under results/HCP/onavg-ico32/. This is the ridge alpha=1e3 fit of "
                         "config_align_to_template_ridge_hcp_onavg-ico32.json (whose extra_str is "
                         "ridge_onavg-ico32; the output folder was renamed to this).")
args = parser.parse_args()

dataset = 'HCP'
task = "MOVIE"
run = [3, 4]  # test runs; the transforms were fit on runs 1-2
surface_space = 'onavg-ico32'
targets = "onavg-ico8"
align_method = "ridge"
prep_root = str(resolve(args.prep_root))
cache_dir = str(resolve(args.cache_dir)) if args.cache_dir else f"{CACHE}/{dataset}_{task}/run{''.join(map(str, run))}"
transformation_folder = args.transformation_folder
transform_dir = RESULTS / dataset / surface_space / transformation_folder
out_root = f"{surface_space}/{transformation_folder}"
os.makedirs(os.path.join(cache_dir, out_root), exist_ok=True)

with open(resolve(args.sids_file)) as f:
    sids = np.array(f.read().split()).astype(str)
N = len(sids)
print("N total in file:", N)

dms_fn = os.path.join(cache_dir, out_root, "dms_lr_float32.npy")
if os.path.exists(dms_fn):
    sys.exit(0)

ts_fns = {lr: os.path.join(cache_dir, f"ts_{lr}_float32.npy") for lr in 'lr'}
if all(os.path.exists(fn) for fn in ts_fns.values()):
    saved_sids = np.load(os.path.join(cache_dir, "sids.npy"))
    assert np.array_equal(sids, saved_sids.astype(str)), "Cached sids mismatch!"
else:
    np.save(os.path.join(cache_dir, "sids.npy"), sids.astype("U"))
    for lr, fn in ts_fns.items():
        ts = get_HCP(prep_root, sids, [lr], task, run, surface_space)
        np.save(fn, ts.astype(np.float32, copy=False))
        print(f'ts_{lr} saved')
        del ts

ts_l = np.load(ts_fns['l'], mmap_mode="r")
ts_r = np.load(ts_fns['r'], mmap_mode="r")
assert ts_l.ndim == 3 and ts_r.ndim == 3, f"Expected (N,T,V). Got {ts_l.shape} and {ts_r.shape}"
N2, T, Vl = ts_l.shape
assert N2 == N, "Subject count mismatch"
Vr = ts_r.shape[2]
print("Shapes:", ts_l.shape, ts_r.shape)

dms_lr = np.empty((N, T, Vl + Vr), dtype=np.float32)
for si, sid in enumerate(sids):
    xfm_l = nb.load(str(transform_dir / f"{sid}_targets-{targets}_lh_{align_method}.npz"))
    xfm_r = nb.load(str(transform_dir / f"{sid}_targets-{targets}_rh_{align_method}.npz"))
    dms_lr[si, :, :Vl] = np.asarray(ts_l[si], np.float32) @ xfm_l
    dms_lr[si, :, Vl:] = np.asarray(ts_r[si], np.float32) @ xfm_r
tmp = dms_fn + ".tmp.npy"  # atomic save
np.save(tmp, dms_lr)
os.replace(tmp, dms_fn)
