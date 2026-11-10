"""Extract the Tian 7T subcortical voxels from the HCP 7T movie runs (Fig 7 preprocessing, step 1).

For each subject and each of the four 7T movie runs (tfMRI_MOVIE{1..4}_7T_*), the 4D NIfTI in MNI
space (1.6 mm; MNINonLinear/Results/<task>/<task>.nii.gz) is masked with the 7T Tian S1 atlas
(all Tian voxels at 1.6 mm, 13974 voxels) and saved as (T, V). No temporal processing happens here;
that is hcp_prep.py. Cortical data (onavg-ico32) were resampled from the same HCP runs to the
onavg-ico32 surface with neuroboros and are not produced by this repository.

Usage:  python resample_HCP_movie.py --hcp-dir <HCP1200 archive> --atlas-img Tian_Subcortex_S1_7T.nii \\
            --out-dir <resampled>/mni-1pt6mm/Tian_Subcortex --sid-index 0     # one subject per call
        (--sid-index selects the line of config/HCP_s178.txt, so that subjects can run as an array)
Output: <out-dir>/<sid>/<task>/<task>.npy
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import nibabel as nib
import numpy as np
from nilearn.image import new_img_like
from nilearn.masking import apply_mask
from utils.paths import resolve  # noqa: E402

TASKS = ["tfMRI_MOVIE1_7T_AP", "tfMRI_MOVIE2_7T_PA", "tfMRI_MOVIE3_7T_PA", "tfMRI_MOVIE4_7T_AP"]


def get_subcortex_volume(sid, tasks, hcp_dir, outdir, mask_img):
    for task in tasks:
        nii_file = os.path.join(hcp_dir, sid, 'MNINonLinear/Results', task, f'{task}.nii.gz')
        X = apply_mask(nib.load(nii_file), mask_img)
        out_fn = os.path.join(outdir, sid, task, f'{task}.npy')
        os.makedirs(os.path.dirname(out_fn), exist_ok=True)
        np.save(out_fn, X)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hcp-dir", required=True,
                        help="HCP1200 release root holding <sid>/MNINonLinear/Results/<task>/<task>.nii.gz")
    parser.add_argument("--atlas-img", required=True,
                        help="Tian_Subcortex_S1_7T.nii (Tian et al. 2020, 7T group parcellation)")
    parser.add_argument("--out-dir", required=True, help="<resampled>/mni-1pt6mm/Tian_Subcortex")
    parser.add_argument("--sids-file", default="config/HCP_s178.txt")
    parser.add_argument("--sid-index", type=int, required=True, help="Line of --sids-file to process")
    args = parser.parse_args()

    with open(resolve(args.sids_file)) as f:
        sid = f.read().splitlines()[args.sid_index]
    atlas = nib.load(args.atlas_img)
    mask_img = new_img_like(atlas, atlas.get_fdata() > 0)

    marker_dir = os.path.join(args.out_dir, "_check")
    finish_fn = os.path.join(marker_dir, f"{sid}_finish.txt")
    running_fn = os.path.join(marker_dir, f"{sid}_running.txt")
    if os.path.exists(finish_fn) or os.path.exists(running_fn):
        sys.exit(0)
    os.makedirs(marker_dir, exist_ok=True)
    open(running_fn, "w").close()
    print(sid)
    get_subcortex_volume(sid, TASKS, args.hcp_dir, args.out_dir, mask_img)
    open(finish_fn, "w").close()
    os.remove(running_fn)
