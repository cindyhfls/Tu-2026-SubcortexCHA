"""Fit each subject's transformation into the common model space (template) with ridge/ridgeCV.

For every subject the connectivity of seed voxels (subcortical nuclei, or cortical searchlights) to the
cortical onavg-ico8 targets is computed in the training runs (Raiders runs 1-2) and mapped onto the
Budapest template. With `alignfunc=ridgeCV` the regularisation alpha is chosen by leave-one-run-out
cross-validation (pooled over all nuclei, or per nucleus with `alpha_per_roi`).

Usage:  python transformation_to_template_raiders.py --load-json config/<config>.json [--sids sid000005 ...]
Configs (config/): aseg ridgeCV (Figs 1-5), Tian S1-S4 (Fig 6), cortical ridgeCV (hyperaligned cortical
targets), Procrustes / fixed-alpha ridge (Supp Fig 4).
Output: <outroot>/<dataset>/<space>_<resample>/<extra_str>/transformation_to_template_<template>/<sid>_*.np[yz]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import resolve  # noqa: E402

import os,argparse,sys
import neuroboros as nb
import numpy as np
from hyperalignment.sparse import initialize_sparse_matrix
from joblib import Parallel, delayed
from utils.data_io_utils import get_rois_and_space,get_Tian_sls,get_aseg_sls
from utils.resource_management_utils import print_memory_usage,load_args_from_json,merge_args,save_args_to_json
from hyperalignment import searchlight_hyperalignment, searchlight_weights
from neuroboros.stats import cronbach_alpha
from utils.hyperalignment_utils import calc_dm_across_runs,loo_cv, reliability_weighting_hyperalignment, get_hyperalignment_func, loo_cv_per_searchlight, searchlight_hyperalignment_per_sl_alpha
import functools

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-json", type=str, required=True,
                       help="Configuration JSON (config/*.json); relative paths are taken from the repo root")
    parser.add_argument("--sids", type=str, nargs="+",)     
    parser.add_argument("--stage", type=str, default="align") 
    parser.add_argument("--dry-run", action="store_true", default=False, 
                    help="Only save configuration JSON without running the main pipeline")
    parser.add_argument("--searchlight-radius",default=None,type=float)

    cli_args = parser.parse_args()
    cli_args.load_json = str(resolve(cli_args.load_json))
    if cli_args.load_json:
        if not os.path.exists(cli_args.load_json):
            parser.error(f"JSON file not found: {cli_args.load_json}")
        
        print(f"Loading configuration from: {cli_args.load_json}")
        json_args = load_args_from_json(cli_args.load_json)
        
        # Merge JSON args with CLI args (CLI takes precedence for specific fields)
        merged_args = merge_args(json_args, cli_args)
        
        # Create a new Namespace object with merged arguments
        args = argparse.Namespace(**merged_args)
    else:
        args = cli_args
    
    args.outroot = str(resolve(args.outroot))
    args.template_dir = [str(resolve(d)) for d in args.template_dir]
    if getattr(args, 'saved_beta_path', None):
        args.saved_beta_path = str(resolve(args.saved_beta_path))
    seedrois,seedspace = get_rois_and_space(args.seeds,args.surface_space,args.volume_space)
    if seedspace == args.surface_space:
        args.outdir = os.path.join(args.outroot, args.dataset, f'{args.surface_space}_{args.surface_resample}', args.extra_str)
    else:
        args.outdir= os.path.join(args.outroot, args.dataset, f'{args.volume_space}_{args.volume_resample}', args.extra_str)
    config_path = save_args_to_json(args)
    if args.dry_run:
        print("DRY RUN: Configuration saved. Exiting without running pipeline.")
        print(f"Config file: {config_path}")
        sys.exit(0)    
    assert(len(args.targets)==1)
    kwargs = { # for older datasets, there might be multiple versions/folders
        'space': [args.surface_space, args.volume_space],
        'resample': [args.surface_resample,args.volume_resample],
        }   
    dset=nb.datasets.datasets[args.dataset.lower()](fp_version="20.2.7",**kwargs)
    sids = args.sids

    def main(sid,dm_allruns,tpl_fn,roi,calpha, args):
        tpl_suffix = '.npy'
        if roi in ['l','r']:
            tpl_suffix = 'h.npy'
            sls, dists = nb.sls(roi, args.searchlight_radius, space=args.surface_space, center_space=args.searchlight_center, mask=True, return_dists=True)
        elif roi.startswith('Tian'):
            sls = get_Tian_sls(args.seeds)
        elif roi.startswith('aseg'):
            sls = get_aseg_sls()
        tpl = np.load(f'{tpl_fn}_{roi}{tpl_suffix}')   
        if seedspace == args.surface_space:
            xfm_dir = os.path.join(args.outroot, args.dataset, f'{args.surface_space}_{args.surface_resample}', args.extra_str, f'transformation_to_template_{tpl_fn.split(os.sep)[-1]}')
        else:
            xfm_dir = os.path.join(args.outroot, args.dataset, f'{args.volume_space}_{args.volume_resample}', args.extra_str, f'transformation_to_template_{tpl_fn.split(os.sep)[-1]}')
        print('xfm_dir:'+ xfm_dir)
        os.makedirs(xfm_dir, exist_ok=True)  

        if roi in ['l','r']:
            nv = tpl.shape[1]
            mat0 = initialize_sparse_matrix(sls, nv)
            if args.notweighted:
                weights = searchlight_weights(sls, dists=None, radius=None)
            else:
                weights = searchlight_weights(sls, dists=dists, radius=args.searchlight_radius)
            out_fn = os.path.join(xfm_dir, f"{sid}_targets-{args.targets[0]}_{roi}h_{args.alignfunc}.npz")
        elif roi.startswith('Tian') or roi.startswith('aseg'):
            nv = tpl.shape[1]
            mat0 = initialize_sparse_matrix(sls, nv)
            assert args.notweighted,"weighting by distance not supported for subcortical atlases"
            assert args.searchlight_radius is None,"searchlight radius not supported for subcortical" # will do that for Tian later
            weights=searchlight_weights(sls, dists=None, radius=None)
            out_fn = os.path.join(xfm_dir, f"{sid}_targets-{args.targets[0]}_{roi}_{args.alignfunc}.npz")
        else:
            out_fn = os.path.join(xfm_dir, f"{sid}_targets-{args.targets[0]}_{roi}_{args.alignfunc}.npy")
        
        if os.path.exists(out_fn):
            print(f"{out_fn} exists, skipping")
            return
        
        alpha_per_roi = getattr(args, 'alpha_per_roi', False)
        best_alphas_per_sl = None
        if args.alignfunc == 'ridgeCV':
            assert len(args.run) > 1, "Need more than one run for leave-one-run-out cross validation"
            assert args.targets[0] != "response"
            if alpha_per_roi:
                # One alpha per atlas region instead of one pooled over the whole subcortex.
                # No extra fitting: the pooled search already fits every parcel at every alpha,
                # it just averages the errors before argmin sees them. Only defined where the
                # searchlights are the disjoint atlas parcels -- with overlapping cortical
                # searchlights a voxel's prediction superposes several blocks and the error
                # does not separate per searchlight.
                assert roi.startswith('Tian') or roi.startswith('aseg'), (
                    f"alpha_per_roi needs disjoint parcel searchlights, not defined for roi={roi}")
                mse_train_sl, mse_val_sl = loo_cv_per_searchlight(
                    dm_allruns, tpl, sls, args.ridge_alpha, voxel_reliability_scores=calpha)
                nvox_sl = np.array([len(sl) for sl in sls])
                # the voxel-count-weighted mean is exactly what the pooled branch minimises,
                # so this records what a single alpha would have been, for comparison
                pooled_alpha = args.ridge_alpha[int(np.argmin(
                    (mse_val_sl * nvox_sl[:, None]).sum(axis=0) / nvox_sl.sum()))]
                best_alphas_per_sl = [args.ridge_alpha[i] for i in np.argmin(mse_val_sl, axis=1)]
                best_alpha = None
                # deliberately NOT the scalar *_bestalpha.txt name: the self-transform scripts parse that with
                # float() and must fail loudly rather than silently read one region's alpha
                alpha_out_fn = f"{out_fn.split('.np')[0]}_bestalpha_per_sl.txt"
                with open(alpha_out_fn, "w") as f:
                    f.write(f"# pooled alpha this run would have used: {pooled_alpha}\n")
                    for isl, a in enumerate(best_alphas_per_sl):
                        f.write(f"{isl}\t{a}\n")
            else:
                avg_mse_train = np.zeros((len(args.ridge_alpha)))
                avg_mse_val = np.zeros((len(args.ridge_alpha)))
                for ii, alpha in enumerate(args.ridge_alpha):
                    func = get_hyperalignment_func('ridge', alpha)
                    if roi in ['l','r'] or roi.startswith('Tian') or roi.startswith('aseg'):
                        func = functools.partial(searchlight_hyperalignment, sls=sls, mat0=mat0, sl_func=func, weights=weights)
                    avg_mse_train[ii], avg_mse_val[ii] = loo_cv(dm_allruns, tpl, func, voxel_reliability_scores=calpha)
                best_idx = np.argmin(avg_mse_val)
                best_alpha = args.ridge_alpha[best_idx] 
                alpha_out_fn = f"{out_fn.split('.np')[0]}_bestalpha.txt"
                with open(alpha_out_fn, "w") as f:
                    f.write(str(best_alpha))
        elif args.alignfunc == 'ridge':
            assert not isinstance(args.ridge_alpha, list), "not using cross validation, cannot have multiple ridge alphas"  
            best_alpha = args.ridge_alpha
        else:
            best_alpha = None
            
        func = get_hyperalignment_func(args.alignfunc, best_alpha)
        if args.targets[0]=="response":
            X = np.concatenate(dm_allruns,axis=0)
        else:
            X = np.mean(dm_allruns, axis=0)  # for the final R we use the average of all training runs
        if roi in ['l','r'] or roi.startswith('Tian') or roi.startswith('aseg'):
            if best_alphas_per_sl is not None:
                func = functools.partial(searchlight_hyperalignment_per_sl_alpha, sls=sls,
                                         mat0=mat0, weights=weights, alphas=best_alphas_per_sl)
            else:
                func = functools.partial(searchlight_hyperalignment, sls=sls, mat0=mat0, sl_func=func, weights=weights)
            xfm_final = reliability_weighting_hyperalignment(
                X, tpl, func, calpha
            )
        else:
            xfm_final = reliability_weighting_hyperalignment(
                X, tpl, func, calpha
            )
        nb.record(out_fn, lambda: xfm_final)()  # wrap R into a callable for nb.record
        return out_fn

    # Loop through subjects first, then templates
    jobs = []

    for sid in sids:
        print(f"Processing subject {sid}...")
        for roi in seedrois:
            # Compute expensive operations once per subject
            if args.saved_beta_path:
                dm_allruns = calc_dm_across_runs(dset, sid, roi, seedspace, args.targets, args.seeds, args.task, args.run, saved_beta_path=args.saved_beta_path)
            else:
                dm_allruns = calc_dm_across_runs(dset, sid, roi, seedspace, args.targets, args.seeds, args.task, args.run, prep=args.prep)
            if args.reliability_weighting:
                calpha = cronbach_alpha(dm_allruns, rep_axis=0, var_axis=1)
            else:
                calpha = None       
            for tpl_fn in args.template_dir: 
                job = delayed(main)(sid, dm_allruns,tpl_fn,roi,calpha, args)
                jobs.append(job)

    with Parallel(n_jobs=min(8, os.cpu_count() or 1)) as parallel:
        parallel(jobs)      
    print_memory_usage()