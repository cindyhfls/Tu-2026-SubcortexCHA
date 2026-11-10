"""Self-transform null for the alignment (Supp Fig 9 / Supp Note 4).

Same machinery as 01_templates_and_transforms/transformation_to_template_raiders.py, but a subject is
aligned to its OWN data instead of the group template, to separate the effect of the hyperalignment
from generic regularised filtering of the connectivity matrix. Only procrustes or a single-alpha
ridge are supported (ridgeCV has no meaning against a self target).

Modes
  default      self-to-self: fit X -> X on the run-averaged connectivity of --run.
  --cross-run  fit run[0]'s connectivity -> run[1]'s, same subject (the null used in the paper: the two
               runs share signal but have independent noise, so the transform cannot fit noise).
               Needs --run SRC TGT and --align-method ridge for the paper's null.
  --match-alpha-from DIR
               ridge alpha per (subject, nucleus) read from the ridgeCV `_bestalpha.txt` files of the
               real alignment in DIR, so the null is regularised exactly as hard as the alignment it
               controls for. Encode side: --run 1 2 with the transformation_to_template folder;
               decode side: --run 2 1 with the transformation_from_template folder.
  --match-alpha-from-pairs DIR
               matched null for the direct subject-to-subject maps of supp_transformation_to_other.py:
               one fit per alpha chosen by any pair with this subject as source (needs --cross-run
               --align-method ridge).

Usage (Supp Fig 9, matched cross-run null; --seeds aseg = 16 nuclei, Raiders runs 1-2):
    R=results/Raiders/mni-2mm_1step_linear_overlap/ridgeCV
    python supp_transformation_to_self.py --seeds aseg --run 1 2 --cross-run --align-method ridge \\
        --match-alpha-from $R/transformation_to_template_targets_onavg-ico8
    python supp_transformation_to_self.py --seeds aseg --run 2 1 --cross-run --align-method ridge \\
        --match-alpha-from $R/transformation_from_template_targets_onavg-ico8
    python supp_transformation_to_self.py --seeds aseg --run 1 2 --cross-run --align-method ridge \\
        --match-alpha-from-pairs $R_other/transformation_to_other_targets_onavg-ico8   # null for 43_

Output: results/Raiders/mni-2mm_1step_linear_overlap/<extra_str>/transformation_to_self_targets_<targets>/
        with extra_str = cross_run<a>to<b>_shrinkage_matched | self_shrinkage_{procrustes,alpha1e+03} | ...
        Cross-run transforms are stored once, alpha-tagged, in cross_run<a>to<b>/ and symlinked from
        extra_str, so a matrix fitted by one mode is never fitted again by another.
Procrustes self-to-self is the identity up to a rotation of the null space (X @ T == X), so that
mode carries no information; ridge self-to-self is a genuine spectral shrinkage.
"""
import argparse
import functools
import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import neuroboros as nb
import numpy as np
from hyperalignment import searchlight_hyperalignment, searchlight_weights
from hyperalignment.sparse import initialize_sparse_matrix
from joblib import Parallel, delayed
from utils.data_io_utils import get_aseg_sls, get_rois_and_space, get_Tian_sls
from utils.hyperalignment_utils import (calc_dm_across_runs, get_hyperalignment_func,
                                        reliability_weighting_hyperalignment)
from utils.paths import CONFOUND_BETAS, RESULTS, ROOT, resolve  # noqa: E402
from utils.resource_management_utils import print_memory_usage

parser = argparse.ArgumentParser()
parser.add_argument("--sids", type=str, nargs="+", default=None,
                    help="Defaults to config/raiders_subj_not_in_budapest_template.txt")
parser.add_argument("--seeds", type=str, default="aseg",
                    help="ROI atlas to self-align: 'aseg', 'lr' (cortical searchlights), 'TianS1', ...")
parser.add_argument("--targets", type=str, default="onavg-ico8",
                    help="Connectivity target the self-alignment is fit on (as in transformation_to_template_raiders.py)")
parser.add_argument("--align-method", type=str, default="procr", choices=["procr", "ridge"])
parser.add_argument("--ridge-alpha", type=float, default=1e3, help="Only used for --align-method ridge")
parser.add_argument("--searchlight-radius", type=float, default=20.0,
                    help="Only used when --seeds is a cortical ('lr') atlas")
parser.add_argument("--notweighted", action="store_true", default=True)  # always on: searchlights are not distance-weighted
parser.add_argument("--run", type=int, nargs="+", default=[1, 2],
                    help="Runs the connectivity is estimated on. Self-to-self averages them; --cross-run "
                         "uses them as (source, target).")
parser.add_argument("--match-alpha-from", type=str, default=None,
                    help="Folder of ridgeCV transforms whose '<sid>_targets-<targets>_<roi>_ridgeCV_bestalpha.txt' "
                         "files set the ridge alpha per (subject, ROI). Implies --align-method ridge.")
parser.add_argument("--match-alpha-from-pairs", type=str, default=None,
                    help="Folder of direct subject-to-subject ridgeCV transforms (supp_transformation_to_other.py); "
                         "fit once at every alpha this subject's pairs chose as source "
                         "(<sid>_targets-<targets>_<roi>_alpha<A>_ridge.npy). Use with --cross-run --align-method ridge.")
parser.add_argument("--cross-run", action="store_true", default=False,
                    help="Fit run[0]'s connectivity -> run[1]'s instead of X -> X (exactly two --run values)")
args = parser.parse_args()

if args.match_alpha_from:
    args.match_alpha_from = str(resolve(args.match_alpha_from))
if args.match_alpha_from_pairs:
    args.match_alpha_from_pairs = str(resolve(args.match_alpha_from_pairs))
    assert args.align_method == "ridge" and args.cross_run and not args.match_alpha_from, (
        "--match-alpha-from-pairs needs --cross-run --align-method ridge and no --match-alpha-from")
if args.match_alpha_from:
    assert args.align_method == "ridge", "--match-alpha-from sets the alpha, so use --align-method ridge"
if args.cross_run:
    assert len(args.run) == 2, "--cross-run needs exactly two --run values (source, target)"
    assert args.targets != "response", "--cross-run is only meaningful for connectivity targets"
targets = [args.targets]
targets_label = args.targets

dataset = "Raiders"
task = "raiders"
run = args.run
saved_beta_path = f"{CONFOUND_BETAS}/{dataset}"
surface_space = 'onavg-ico32'
volume_space = 'mni-2mm'
surface_resample = '1step_pial_overlap'
volume_resample = '1step_linear_overlap'
searchlight_center = 'onavg-ico8'  # only the cortical ('lr') branch; transformation_to_template_raiders.py uses onavg-ico32 from its config
outroot = str(RESULTS)

# extra_str: --cross-run gets its own prefix so the run pair is legible in the path and never
# collides with the self-to-self output
prefix = f"cross_run{run[0]}to{run[1]}_shrinkage" if args.cross_run else "self_shrinkage"
if args.match_alpha_from_pairs:
    extra_str = "ridge_cross_run_matchedalpha"
elif args.match_alpha_from:
    extra_str = f"{prefix}_matched"            # alpha varies per subject/ROI, so it is not in the name
elif args.align_method == "ridge":
    extra_str = f"{prefix}_alpha{args.ridge_alpha:.0e}"   # e.g. self_shrinkage_alpha1e+01
else:
    extra_str = f"{prefix}_procrustes"

# Every --cross-run transform lives ONCE, alpha-tagged, in cross_run<a>to<b>/ (the store); the
# extra_str folder holds relative symlinks under the names the downstream scripts read.
STORE = (os.path.join(outroot, dataset, f"{volume_space}_{volume_resample}", f"cross_run{run[0]}to{run[1]}",
                      f"transformation_to_self_targets_{args.targets}") if args.cross_run else None)


def matched_alpha(sid, roi):
    """The ridgeCV alpha chosen for this (subject, ROI) by the real alignment."""
    fn = os.path.join(args.match_alpha_from, f"{sid}_targets-{targets_label}_{roi}_ridgeCV_bestalpha.txt")
    with open(fn) as f:
        return float(f.read().strip())


def pair_alphas(sid, roi):
    """Every distinct alpha a direct pair chose with this subject as SOURCE, for this ROI."""
    fns = glob.glob(os.path.join(args.match_alpha_from_pairs,
                                 f"{sid}_to_*_targets-{targets_label}_{roi}_ridgeCV_bestalpha.txt"))
    assert fns, f"no pair alphas for {sid} {roi} in {args.match_alpha_from_pairs}"
    alphas = set()
    for fn in fns:
        with open(fn) as f:
            alphas.add(float(f.read().strip()))
    return sorted(alphas)


def store_and_link(link_fn, store_fn, X, Y, func):
    """Fit into the store unless that exact transform is already there, then point link_fn at it."""
    if not os.path.exists(store_fn):
        os.makedirs(os.path.dirname(store_fn), exist_ok=True)
        nb.record(store_fn, lambda X=X, Y=Y, f=func: reliability_weighting_hyperalignment(
            X, Y, f, voxel_reliability_scores=None))()
    if not os.path.lexists(link_fn):
        try:
            os.symlink(os.path.relpath(store_fn, os.path.dirname(link_fn)), link_fn)
        except FileExistsError:
            pass


if args.sids is None:
    with open(ROOT / "config" / "raiders_subj_not_in_budapest_template.txt") as f:
        sids = f.read().strip().split('\n')
else:
    sids = args.sids

seedrois, seedspace = get_rois_and_space(args.seeds, surface_space, volume_space)
kwargs = {'space': [surface_space, volume_space], 'resample': [surface_resample, volume_resample]}
dset = nb.datasets.datasets[dataset.lower()](fp_version="20.2.7", **kwargs)


def main(sid, dm_allruns, roi):
    if roi in ['l', 'r']:
        sls, dists = nb.sls(roi, args.searchlight_radius, space=surface_space, center_space=searchlight_center,
                            mask=True, return_dists=True)
    elif roi.startswith('Tian'):
        sls = get_Tian_sls(args.seeds)
    elif roi.startswith('aseg'):
        sls = get_aseg_sls()

    space_folder = f'{surface_space}_{surface_resample}' if seedspace == surface_space \
        else f'{volume_space}_{volume_resample}'
    xfm_dir = os.path.join(outroot, dataset, space_folder, extra_str, f'transformation_to_self_targets_{targets_label}')
    os.makedirs(xfm_dir, exist_ok=True)

    if roi in ['l', 'r']:
        nv = dm_allruns.shape[-1]
        mat0 = initialize_sparse_matrix(sls, nv)
        weights = searchlight_weights(sls, dists=None, radius=None) if args.notweighted \
            else searchlight_weights(sls, dists=dists, radius=args.searchlight_radius)
        out_fn = os.path.join(xfm_dir, f"{sid}_targets-{targets_label}_{roi}h_{args.align_method}.npz")
    elif roi.startswith('Tian') or roi.startswith('aseg'):
        nv = dm_allruns.shape[-1]
        mat0 = initialize_sparse_matrix(sls, nv)
        assert args.notweighted, "weighting by distance not supported for subcortical atlases"
        weights = searchlight_weights(sls, dists=None, radius=None)
        out_fn = os.path.join(xfm_dir, f"{sid}_targets-{targets_label}_{roi}_{args.align_method}.npz")
    else:
        out_fn = os.path.join(xfm_dir, f"{sid}_targets-{targets_label}_{roi}_{args.align_method}.npy")

    if args.match_alpha_from_pairs:
        # one fit per alpha this source's pairs use, on the subject's own run1 -> run2 connectivity, so
        # the target subject enters only through which alpha is chosen
        tail = f"_{args.align_method}.npy"
        assert out_fn.endswith(tail), "pair-alpha mode is for per-ROI .npy transforms"
        X, Y = dm_allruns[0], dm_allruns[1]
        written = []
        for alpha in pair_alphas(sid, roi):
            fn = out_fn[:-len(tail)] + f"_alpha{alpha:.0e}" + tail   # same format supp_isc_to_other.py rebuilds
            store_and_link(fn, os.path.join(STORE, os.path.basename(fn)), X, Y,
                           get_hyperalignment_func("ridge", alpha))
            written.append(fn)
        return written

    if os.path.exists(out_fn):
        print(f"{out_fn} exists, skipping")
        return

    # as in 02_: concatenate runs for raw-timeseries targets, average runs for connectivity targets
    if targets[0] == "response":
        X = np.concatenate(dm_allruns, axis=0)
    else:
        X = np.mean(dm_allruns, axis=0)
    Y = X
    if args.cross_run:  # source is run[0]'s connectivity, target is run[1]'s -- NOT the run average
        X, Y = dm_allruns[0], dm_allruns[1]

    if args.match_alpha_from:
        best_alpha = matched_alpha(sid, roi)
    else:
        best_alpha = args.ridge_alpha if args.align_method == "ridge" else None

    func = get_hyperalignment_func(args.align_method, best_alpha)
    if roi in ['l', 'r'] or roi.startswith('Tian') or roi.startswith('aseg'):
        func = functools.partial(searchlight_hyperalignment, sls=sls, mat0=mat0, sl_func=func, weights=weights)
    if args.cross_run and out_fn.endswith(".npy"):
        # per-ROI cross-run fit: store once under the alpha it was fitted at, link under out_fn
        tag = f"_alpha{best_alpha:.0e}" if args.align_method == "ridge" else ""
        store_and_link(out_fn, os.path.join(STORE, f"{sid}_targets-{targets_label}_{roi}{tag}_{args.align_method}.npy"),
                       X, Y, func)
        return out_fn
    xfm_final = reliability_weighting_hyperalignment(X, Y, func, voxel_reliability_scores=None)
    nb.record(out_fn, lambda: xfm_final)()
    return out_fn


jobs = []
for sid in sids:
    print(f"Processing subject {sid}...")
    for roi in seedrois:
        dm_allruns = calc_dm_across_runs(dset, sid, roi, seedspace, targets, args.seeds, task, run,
                                         saved_beta_path=saved_beta_path)
        jobs.append(delayed(main)(sid, dm_allruns, roi))

with Parallel(n_jobs=min(8, os.cpu_count() or 1)) as parallel:
    parallel(jobs)
print_memory_usage()
