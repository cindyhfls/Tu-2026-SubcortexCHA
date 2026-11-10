"""Preprocess the HCP 7T movie runs for hyperalignment (Methods, "HCP preprocessing"; Fig 7).

For every subject and movie run (tfMRI_MOVIE1..4_7T), the resampled data (cortex: onavg-ico32
surface, masked; subcortex: Tian 7T voxels at 1.6 mm from resample_HCP_movie.py) are

  1. cut into the movie-clip blocks (rest periods dropped). The per-run [start, end) frame indices
     are hard-coded in BLOCKS (TR = 1 s); the 5 s delay of the Methods is assumed to be included
     in them -- the source does not document how these indices were derived,
  2. denoised with one regression over the concatenated blocks: HCP movement regressors (12),
     CSF and WM signals and Legendre polynomials up to order 2 (computed over the whole run),
  3. z-scored per run (per voxel/vertex, over the concatenated blocks); non-finite values -> 0.

Usage:  python hcp_prep.py --space mni-1pt6mm --rois Tian_Subcortex \\
            --resampled-dir <HCP_YA/resampled> --regressor-root <dir with <sid>/MNINonLinear/Results>
        python hcp_prep.py --space onavg-ico32 --rois l r ...
Output: data/HCP_YA/prep_oneregression/<space>/<roi>/sub-<sid>_task-MOVIE_run-<01..04>.npy,
        roi 'l'/'r' stored as 'l-cerebrum'/'r-cerebrum'
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import neuroboros as nb
import nibabel as nib
import numpy as np
from joblib import Parallel, delayed
from scipy.special import legendre
from scipy.stats import zscore
from utils.paths import DATA, resolve  # noqa: E402

TASKS = {1: 'tfMRI_MOVIE1_7T_AP', 2: 'tfMRI_MOVIE2_7T_PA', 3: 'tfMRI_MOVIE3_7T_PA', 4: 'tfMRI_MOVIE4_7T_AP'}

# [start, end) TR indices of the movie clips within each run (TR = 1 s)
BLOCKS = {
    'tfMRI_MOVIE1_7T_AP': [[25, 270], [289, 511], [530, 719], [738, 803], [822, 906]],
    'tfMRI_MOVIE2_7T_PA': [[25, 252], [271, 531], [550, 800], [819, 903]],
    'tfMRI_MOVIE3_7T_PA': [[25, 206], [225, 411], [430, 635], [654, 797], [816, 900]],
    'tfMRI_MOVIE4_7T_AP': [[25, 258], [277, 508], [527, 783], [802, 886]],
}


def legendre_regressors(polyord, nt):
    timepoints = np.linspace(-1, 1, nt)
    return np.array([legendre(i)(timepoints) for i in range(polyord + 1)]).T


def load_regressors(regressors_dir, task):
    d1 = np.loadtxt(f'{regressors_dir}/{task}/Movement_Regressors.txt')
    d2 = np.loadtxt(f'{regressors_dir}/{task}/{task}_CSF.txt')
    d3 = np.loadtxt(f'{regressors_dir}/{task}/{task}_WM.txt')
    return np.concatenate([d1, d2[:, np.newaxis], d3[:, np.newaxis]], axis=1)


def prep(sid, task, space, roi, resampled_dir, regressor_root, z=True, mask=True):
    """Preprocessed (T_blocks, V) data of one run."""
    nuisance = load_regressors(f'{regressor_root}/{sid}/MNINonLinear/Results/', task)
    if 'onavg' in space:
        gii = nib.load(f'{resampled_dir}/{space}/{sid}/{task}/{sid}_{roi}h_{space}_dilated.func.gii')
        orig = np.stack([_.data for _ in gii.darrays]).astype(np.float64)
        if mask:
            orig = orig[:, nb.mask(roi, space=space)]
    else:
        orig = np.load(f'{resampled_dir}/{space}/{roi}/{sid}/{task}/{task}.npy')

    regs = np.hstack([nuisance, legendre_regressors(polyord=2, nt=nuisance.shape[0])])
    # a single regression over the concatenated blocks (as neuroboros.get_data does)
    dm = np.concatenate([orig[start:end] for start, end in BLOCKS[task]], axis=0)
    regressors = np.concatenate([regs[start:end] for start, end in BLOCKS[task]], axis=0)
    beta = np.linalg.lstsq(regressors, dm, rcond=None)[0]
    finite_mask = np.all(np.isfinite(dm), axis=0)
    dm[:, finite_mask] = dm[:, finite_mask] - regressors @ beta
    if z:
        dm = np.nan_to_num(zscore(dm, axis=0, nan_policy='omit'))
    return dm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", required=True, choices=['onavg-ico32', 'mni-1pt6mm'])
    parser.add_argument("--rois", nargs="+", required=True,
                        help="'l' 'r' for onavg-ico32; 'Tian_Subcortex' for mni-1pt6mm")
    parser.add_argument("--sids-file", default="config/HCP_s178.txt")
    parser.add_argument("--resampled-dir", required=True,
                        help="Resampled HCP data: <dir>/onavg-ico32/<sid>/<task>/<sid>_{l,r}h_onavg-ico32_dilated.func.gii "
                             "and <dir>/mni-1pt6mm/Tian_Subcortex/<sid>/<task>/<task>.npy (resample_HCP_movie.py)")
    parser.add_argument("--regressor-root", required=True,
                        help="Dir holding <sid>/MNINonLinear/Results/<task>/{Movement_Regressors.txt,<task>_CSF.txt,<task>_WM.txt}")
    parser.add_argument("--out-dir", default=str(DATA / "HCP_YA" / "prep_oneregression"))
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()

    with open(resolve(args.sids_file)) as f:
        sids = f.read().splitlines()
    out_dir = resolve(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    jobs = []
    for sid in sids:
        for run, task in TASKS.items():
            for roi in args.rois:
                folder = f'{roi}-cerebrum' if roi in ('l', 'r') else roi
                out_fn = f"{out_dir}/{args.space}/{folder}/sub-{sid}_task-MOVIE_run-{run:02d}.npy"
                jobs.append(delayed(nb.record(out_fn, prep))(sid, task, args.space, roi,
                                                              args.resampled_dir, args.regressor_root))
    with Parallel(n_jobs=args.n_jobs, verbose=1) as parallel:
        parallel(jobs)
