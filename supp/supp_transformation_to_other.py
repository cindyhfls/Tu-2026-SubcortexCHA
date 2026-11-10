"""Direct subject-to-subject transforms (Supp Fig 9 / Supp Note 4: hyperalignment to a reference subject).

Same machinery as 01_templates_and_transforms/transformation_to_template_raiders.py, but the alignment
target is ANOTHER SUBJECT's connectivity instead of the group template:

    xfm[j -> i] = fit( conn_j -> conn_i )        for every ordered pair j != i

so subject j's data are carried into subject i's space in one step (ts_j @ xfm[j -> i]) instead of
through the template (ts_j @ xfm_to_template_j @ xfm_from_template_i). Each nucleus is fitted as one
plain ridge over the whole nucleus (per-label aseg ROIs; searchlight-aggregated seeds are not
supported). With --align-method ridgeCV the alpha is chosen per pair by leave-one-run-out CV over
the training runs, as in 02_; no alpha is inherited from the template fit.

Usage:  python supp_transformation_to_other.py --seeds aseg --target-sids sid000012   # one reference subject
        (--target-sids shards the work: one job per reference subject, the 12 incoming transforms each)
Output: results/Raiders/mni-2mm_1step_linear_overlap/ridgeCV_to_other/transformation_to_other_targets_onavg-ico8/
        <src>_to_<tgt>_targets-onavg-ico8_<roi>_ridgeCV.npy (+ _bestalpha.txt)
The matched cross-run null for these maps is supp_transformation_to_self.py --match-alpha-from-pairs.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import neuroboros as nb
import numpy as np
from joblib import Parallel, delayed
from utils.data_io_utils import get_rois_and_space
from utils.hyperalignment_utils import (calc_dm_across_runs, get_hyperalignment_func, loo_cv,
                                        reliability_weighting_hyperalignment)
from utils.paths import CONFOUND_BETAS, RESULTS, ROOT, resolve  # noqa: E402
from utils.resource_management_utils import print_memory_usage

parser = argparse.ArgumentParser()
parser.add_argument("--sids", type=str, nargs="+", default=None,
                    help="The full subject pool. Defaults to config/raiders_subj_not_in_budapest_template.txt.")
parser.add_argument("--target-sids", type=str, nargs="+", default=None,
                    help="Reference subjects i to fit incoming transforms for (j -> i for every j != i in "
                         "--sids). Defaults to all of --sids.")
parser.add_argument("--seeds", type=str, default="aseg",
                    help="'aseg' or 'Tian*' (per-label nuclei). Cortical ('lr') is not supported.")
parser.add_argument("--targets", type=str, default="onavg-ico8",
                    help="Connectivity target space the pairwise fit is computed on.")
parser.add_argument("--align-method", type=str, default="ridgeCV", choices=["ridgeCV", "ridge", "procr"])
parser.add_argument("--ridge-alpha", type=float, nargs="+",
                    default=[1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8],
                    help="ridgeCV grid (as in the transformation configs); a single value when --align-method ridge.")
parser.add_argument("--run", type=int, nargs="+", default=[1, 2],
                    help="Training runs, as in transformation_to_template_raiders.py. ridgeCV needs more than one.")
parser.add_argument("--extra-str", type=str, default="ridgeCV_to_other")
parser.add_argument("--n-jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
args = parser.parse_args()

dataset, task = "Raiders", "raiders"
surface_space, volume_space = 'onavg-ico32', 'mni-2mm'
surface_resample, volume_resample = '1step_pial_overlap', '1step_linear_overlap'
saved_beta_path = f"{CONFOUND_BETAS}/{dataset}"
outroot = str(RESULTS)

if args.sids is None:
    with open(ROOT / "config" / "raiders_subj_not_in_budapest_template.txt") as f:
        sids = f.read().strip().split('\n')
else:
    sids = args.sids
target_sids = args.target_sids or sids
assert set(target_sids) <= set(sids), "--target-sids must be a subset of --sids"
print(f"pool N={len(sids)}; fitting incoming transforms for {len(target_sids)}: {target_sids}")

seedrois, seedspace = get_rois_and_space(args.seeds, surface_space, volume_space)
assert seedspace == volume_space, "only subcortical seeds are supported"
space_folder = f'{volume_space}_{volume_resample}'
xfm_dir = os.path.join(outroot, dataset, space_folder, args.extra_str,
                       f'transformation_to_other_targets_{args.targets}')
os.makedirs(xfm_dir, exist_ok=True)
print("xfm_dir:", xfm_dir)

with open(os.path.join(xfm_dir, "args.json"), "w") as f:
    json.dump(dict(vars(args), dataset=dataset, task=task, sids=sids, rois=seedrois,
                   space_folder=space_folder, script=os.path.basename(__file__)),
              f, indent=2, default=str)

kwargs = {'space': [surface_space, volume_space], 'resample': [surface_resample, volume_resample]}
dset = nb.datasets.datasets[dataset.lower()](fp_version="20.2.7", **kwargs)


def out_fn(src, tgt, roi):
    return os.path.join(xfm_dir, f"{src}_to_{tgt}_targets-{args.targets}_{roi}_{args.align_method}.npy")


def fit_pair(src, tgt, roi, dm_src, dm_tgt):
    """dm_src/dm_tgt: (nruns, nTargets, nVox) connectivity for the training runs."""
    fn = out_fn(src, tgt, roi)
    if os.path.exists(fn):
        return
    assert roi not in ('l', 'r') and not roi.startswith(('aseg', 'Tian')), \
        f"{roi}: searchlight-aggregated seeds are not wired up here"

    if args.align_method == 'ridgeCV':
        assert len(args.run) > 1, "Need more than one run for leave-one-run-out cross validation"
        mse = []
        for alpha in args.ridge_alpha:
            func = get_hyperalignment_func('ridge', alpha)
            # X side carries the runs, Y is the fixed target-subject connectome (run-averaged),
            # mirroring 02_'s loo_cv(dm_allruns, tpl, func)
            mse.append(loo_cv(dm_src, dm_tgt.mean(axis=0), func)[1])
        best_alpha = args.ridge_alpha[int(np.argmin(mse))]
        with open(f"{fn[:-4]}_bestalpha.txt", "w") as f:
            f.write(str(best_alpha))
    elif args.align_method == 'ridge':
        assert len(args.ridge_alpha) == 1, "--align-method ridge takes a single --ridge-alpha"
        best_alpha = args.ridge_alpha[0]
    else:
        best_alpha = None

    func = get_hyperalignment_func(args.align_method, best_alpha)
    X, Y = dm_src.mean(axis=0), dm_tgt.mean(axis=0)   # transformation_to_template_raiders.py fits the final transform on the run average
    nb.record(fn, lambda: reliability_weighting_hyperalignment(X, Y, func, None))()


# ROI outer, subjects inner: only one ROI's connectomes are held in memory at a time
for roi in seedrois:
    print(f"--- {roi} ---", flush=True)
    dm = {sid: np.asarray(calc_dm_across_runs(dset, sid, roi, seedspace, [args.targets], args.seeds,
                                              task, args.run, saved_beta_path=saved_beta_path),
                          dtype=np.float32)
          for sid in sids}
    jobs = [delayed(fit_pair)(src, tgt, roi, dm[src], dm[tgt])
            for tgt in target_sids for src in sids if src != tgt]
    with Parallel(n_jobs=args.n_jobs) as parallel:
        parallel(jobs)
    del dm
print_memory_usage()
