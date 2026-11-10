"""Cache HCP 7T movie subcortical time series (and their transformed versions) as float32 arrays (Fig 7).

HCP counterpart of 00_cache_inputs/cache_subcortical_raiders.py. `ts_Tian_Subcortex_float32.npy`
(N subjects x T x V; 34 test subjects, movie runs 3-4, 7T Tian atlas at 1.6 mm, anatomical alignment)
is built from data/HCP_YA/prep_oneregression, then each subject's ridgeCV transform from
transformation_to_template_hcp.py is applied voxelwise. The test-time ISC / classification
scripts read these caches.

Usage:  python cache_subcortical_hcp.py --scale 1      # transform ridgeCV_TianS1_wholebrain; also 2, 3, 4
Output: data/cache/HCP_MOVIE/run34/{sids.npy, ts_Tian_Subcortex_float32.npy,
        mni-1pt6mm/ridgeCV_TianS<k>_wholebrain/transformation_to_template_targets_onavg-ico8/dms_Tian_Subcortex_float32.npy}
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import neuroboros as nb
import numpy as np
from utils.data_io_utils import TianLUT_7T
from utils.data_io_utils import get_HCP
from utils.paths import CACHE, DATA, RESULTS, resolve  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--scale", type=int, required=True, choices=[1, 2, 3, 4],
                    help="Tian scale of the transform: ridgeCV_TianS<scale>_wholebrain")
parser.add_argument("--prep-root", default=str(DATA / "HCP_YA" / "prep_oneregression"),
                    help="Per-subject, per-run source .npy files (hcp_prep.py output)")
parser.add_argument("--sids-file", default="config/HCP_movie_v2_complete.txt",
                    help="The 34 test subjects")
parser.add_argument("--cache-dir", default=None,
                    help="Cache root; default data/cache/HCP_MOVIE/run<runs>")
args = parser.parse_args()

dataset = 'HCP'
task = "MOVIE"
run = [3, 4]  # test runs; the transforms were fit on runs 1-2
volume_space = 'mni-1pt6mm'
atlas = 'Tian_Subcortex'
targets = "onavg-ico8"
align_method = "ridgeCV"
prep_root = str(resolve(args.prep_root))
cache_dir = str(resolve(args.cache_dir)) if args.cache_dir else f"{CACHE}/{dataset}_{task}/run{''.join(map(str, run))}"
transformation_folder = f"ridgeCV_TianS{args.scale}_wholebrain/transformation_to_template_targets_{targets}/"
transform_dir = RESULTS / dataset / volume_space / transformation_folder
out_root = f"{volume_space}/{transformation_folder}"
os.makedirs(os.path.join(cache_dir, out_root), exist_ok=True)

with open(resolve(args.sids_file)) as f:
    sids = np.array(f.read().split()).astype(str)
N = len(sids)
print("N total in file:", N)
rois = [atlas]
sls = [np.arange(len(TianLUT_7T))]  # all 13974 voxels of the 7T Tian atlas, one block

dms_fn = os.path.join(cache_dir, out_root, f"dms_{atlas}_float32.npy")
if os.path.exists(dms_fn):
    sys.exit(0)

ts_fn = os.path.join(cache_dir, f"ts_{atlas}_float32.npy")
if os.path.exists(ts_fn):
    saved_sids = np.load(os.path.join(cache_dir, "sids.npy"))
    assert np.array_equal(sids, saved_sids.astype(str)), "Cached sids mismatch!"
else:
    ts = get_HCP(prep_root, sids, rois, task, run, volume_space)
    np.save(os.path.join(cache_dir, "sids.npy"), sids.astype("U"))
    np.save(ts_fn, ts.astype(np.float32, copy=False))
    print('ts saved')
    del ts

ts = np.load(ts_fn, mmap_mode="r")
assert ts.ndim == 3, f"Expected (N,T,V). Got {ts.shape}"
N2, T, V = ts.shape
assert N2 == N, "Subject count mismatch"

dms = np.empty((N, T, V), dtype=np.float32)
for si, sid in enumerate(sids):
    for roi, sl in zip(rois, sls):
        transform_fn = str(transform_dir / f"{sid}_targets-{targets}_{roi}_{align_method}.npy")
        try:
            xfm = np.load(transform_fn)
        except FileNotFoundError:
            xfm = nb.load(transform_fn.replace('.npy', '.npz'))
        dms[si, :, sl] = (np.asarray(ts[si, :, sl], np.float32).T @ xfm).T
tmp = dms_fn + ".tmp.npy"  # atomic save
np.save(tmp, dms)
os.replace(tmp, dms_fn)
