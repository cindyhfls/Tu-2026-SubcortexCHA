"""Fit each HCP test subject's transformation into the HCP template (Fig 7).

HCP counterpart of 01_templates_and_transforms/transformation_to_template_raiders.py. The fitting is the
same (searchlight ridge / ridgeCV hyperalignment of seed-to-onavg-ico8 connectivity onto the template
from template_hcp_subcortical.py / template_hcp_cortical.py), but the data are read from the preprocessed per-run files
(data/HCP_YA/prep_oneregression, see 00_cache_inputs/hcp_resample/hcp_prep.py) instead of through a neuroboros
dataset, and outputs go to results/HCP/<space>/<extra_str>/ (no resample suffix). Transforms are fit on
the first two movie runs (config `run`: [1, 2]) of the 34 test subjects
(config/HCP_movie_v2_complete.txt); the Tian configs use ridgeCV (alpha chosen by leave-one-run-out
CV, pooled over nuclei), the cortical config a fixed ridge alpha of 1e3.

Usage:
    python transformation_to_template_hcp.py --load-json config/config_align_to_template_ridgeCV_hcp_TianS1_wholebrain.json \\
        --sids $(cat config/HCP_movie_v2_complete.txt)
    (also ..._TianS2/3/4_wholebrain.json, and config_align_to_template_ridge_hcp_onavg-ico32.json for cortex)
Output: results/HCP/mni-1pt6mm/ridgeCV_TianS<k>_wholebrain/transformation_to_template_targets_onavg-ico8/<sid>_targets-onavg-ico8_Tian_Subcortex_ridgeCV.npz
        (+ _bestalpha.txt); cortex: results/HCP/onavg-ico32/<extra_str>/.../<sid>_..._{l,r}h_ridge.npz
Not covered by 02_: HCP data loading (prep_root) and the output layout.
"""
import argparse
import functools
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import neuroboros as nb
import numpy as np
from hyperalignment import searchlight_hyperalignment, searchlight_weights
from hyperalignment.sparse import initialize_sparse_matrix
from joblib import Parallel, delayed
from neuroboros.stats import cronbach_alpha
from utils.data_io_utils import get_rois_and_space, get_Tian_sls
from utils.data_io_utils import get_HCP_acrossruns
from utils.hyperalignment_utils import (calculate_connectivity, get_hyperalignment_func, loo_cv,
                                        reliability_weighting_hyperalignment)
from utils.paths import resolve  # noqa: E402
from utils.resource_management_utils import (load_args_from_json, merge_args, print_memory_usage,
                                             save_args_to_json)


def main(sid, dm_allruns, tpl_fn, roi, calpha, args, seedspace):
    if roi in ['l', 'r']:
        tpl_suffix = 'h.npy'
        sls, dists = nb.sls(roi, args.searchlight_radius, space=args.surface_space,
                            center_space=args.searchlight_center, mask=True, return_dists=True)
    elif roi.startswith('Tian'):
        tpl_suffix = '.npy'
        sls = get_Tian_sls(args.seeds, is_7T=args.is_7T)
    else:
        raise ValueError(f"unsupported seed roi {roi}")
    tpl = np.load(f'{tpl_fn}_{roi}{tpl_suffix}')

    space = args.surface_space if seedspace == args.surface_space else args.volume_space
    xfm_dir = os.path.join(args.outroot, args.dataset, space, args.extra_str,
                           f'transformation_to_template_{os.path.basename(tpl_fn)}')
    print('xfm_dir:' + xfm_dir)
    os.makedirs(xfm_dir, exist_ok=True)

    nv = tpl.shape[1]
    mat0 = initialize_sparse_matrix(sls, nv)
    if roi in ['l', 'r']:
        if args.nodistweighted:
            weights = searchlight_weights(sls, dists=None, radius=None)
        else:
            weights = searchlight_weights(sls, dists=dists, radius=args.searchlight_radius)
        out_fn = os.path.join(xfm_dir, f"{sid}_targets-{args.targets[0]}_{roi}h_{args.alignfunc}.npz")
    else:
        assert args.nodistweighted, "weighting by distance not supported for subcortical atlases"
        assert args.searchlight_radius is None, "searchlight radius not supported for subcortical"
        weights = searchlight_weights(sls, dists=None, radius=None)
        out_fn = os.path.join(xfm_dir, f"{sid}_targets-{args.targets[0]}_{roi}_{args.alignfunc}.npz")

    if os.path.exists(out_fn):
        print(f"{out_fn} exists, skipping")
        return

    if args.alignfunc == 'ridgeCV':
        assert len(args.run) > 1, "Need more than one run for leave-one-run-out cross validation"
        avg_mse_val = np.zeros(len(args.ridge_alpha))
        for ii, alpha in enumerate(args.ridge_alpha):
            func = get_hyperalignment_func('ridge', alpha)
            func = functools.partial(searchlight_hyperalignment, sls=sls, mat0=mat0, sl_func=func, weights=weights)
            _, avg_mse_val[ii] = loo_cv(dm_allruns, tpl, func, voxel_reliability_scores=calpha)
        best_alpha = args.ridge_alpha[int(np.argmin(avg_mse_val))]
        with open(f"{out_fn.split('.np')[0]}_bestalpha.txt", "w") as f:
            f.write(str(best_alpha))
    elif args.alignfunc == 'ridge':
        assert not isinstance(args.ridge_alpha, list), "not using cross validation, cannot have multiple ridge alphas"
        best_alpha = args.ridge_alpha
    else:
        best_alpha = None

    func = get_hyperalignment_func(args.alignfunc, best_alpha)
    X = np.mean(dm_allruns, axis=0)  # final transform is fit on the average of the training runs
    func = functools.partial(searchlight_hyperalignment, sls=sls, mat0=mat0, sl_func=func, weights=weights)
    xfm_final = reliability_weighting_hyperalignment(X, tpl, func, calpha)
    nb.record(out_fn, lambda: xfm_final)()
    return out_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-json", type=str, required=True,
                        help="Configuration JSON (config/*.json); relative paths are taken from the repo root")
    parser.add_argument("--sids", type=str, nargs="+", required=True)
    parser.add_argument("--stage", type=str, default='align')
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Only save configuration JSON without running the main pipeline")
    parser.add_argument("--searchlight-radius", default=None, type=float)
    cli_args = parser.parse_args()

    load_json = resolve(cli_args.load_json)
    if not load_json.exists():
        parser.error(f"JSON file not found: {load_json}")
    cli_args.load_json = str(load_json)
    args = argparse.Namespace(**merge_args(load_args_from_json(cli_args.load_json), cli_args))

    args.outroot = str(resolve(args.outroot))
    args.template_dir = [str(resolve(d)) for d in args.template_dir]
    args.prep_root = str(resolve(args.prep_root))
    assert len(args.targets) == 1 and args.targets[0] == "onavg-ico8"

    seedrois, seedspace = get_rois_and_space(args.seeds, args.surface_space, args.volume_space)
    space = args.surface_space if seedspace == args.surface_space else args.volume_space
    args.outdir = os.path.join(args.outroot, args.dataset, space, args.extra_str)
    config_path = save_args_to_json(args)
    if args.dry_run:
        print("DRY RUN: Configuration saved. Exiting without running pipeline.")
        print(f"Config file: {config_path}")
        sys.exit(0)

    mappings = {lr: nb.mapping(lr, args.surface_space, args.targets[0], mask=True) for lr in 'lr'}
    jobs = []
    for sid in args.sids:
        print(f"Processing subject {sid}...")
        for roi in seedrois:
            seed_ts = get_HCP_acrossruns(args.prep_root, sid, roi, args.task, args.run, seedspace)
            target_ts = get_HCP_acrossruns(args.prep_root, sid, ['l', 'r'], args.task, args.run,
                                           args.surface_space, xfms=mappings)
            dm_allruns = np.stack([calculate_connectivity(target_ts[i], seed_ts[i])
                                   for i in range(len(args.run))], axis=0)
            calpha = cronbach_alpha(dm_allruns, rep_axis=0, var_axis=1) if args.reliability_weighting else None
            for tpl_fn in args.template_dir:
                jobs.append(delayed(main)(sid, dm_allruns, tpl_fn, roi, calpha, args, seedspace))

    with Parallel(n_jobs=min(8, os.cpu_count() or 1)) as parallel:
        parallel(jobs)
    print_memory_usage()
