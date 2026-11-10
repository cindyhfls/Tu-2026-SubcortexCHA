import os,functools
import neuroboros as nb
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import zscore
from .data_io_utils import get_dm,get_rois_and_space
from hyperalignment.ridge import ridge
from hyperalignment.procrustes import procrustes


def get_hyperalignment_func(alignfunc,alpha=None):
    # which transformation method?
    if alignfunc == "procr" or alignfunc == "procrustes":
        func = functools.partial(procrustes, reflection=True, scaling=False)
    elif alignfunc == "ridge" or alignfunc == "ridgeCV":
        func = functools.partial(ridge, alpha=alpha)
    else:
        ValueError("Unsupported alignment method provided! Choose from 'procr','ridge'")
    return func


def calc_slmeans(ts, sls):
    N_SLs = len(sls)
    sl_means = np.zeros((ts.shape[0], N_SLs))
    for sl in range(N_SLs):
        sl_means[:,sl] = np.mean(ts[:,sls[sl]], axis=1)
    return sl_means

def get_target_ts(dm, target_name,target_item=None):
    if isinstance(dm,np.ndarray): # I'll add the dictionary case later
        nS,nT,_ = dm.shape
        if target_name == 'mapping':
            mapping = target_item
            all_target_ts = np.zeros((nS,nT,mapping.shape[1]))
            for isubj in np.arange(nS):
                all_target_ts[isubj] = dm[isubj] @ mapping
        elif target_name == 'searchlight_mean':
            sls = target_item
            all_target_ts = np.zeros((nS,nT,len(sls)))
            for isubj in np.arange(nS):
                all_target_ts[isubj] = calc_slmeans(dm[isubj],sls)
    elif isinstance(dm,dict):
        all_target_ts = []
        for subj_ts in dm.values():
            target_ts = []
            for target_roi in subj_ts.keys():
                if target_name == 'mapping':
                    mapping = target_item
                    target_ts.append(subj_ts[target_roi] @ mapping[target_roi])
                elif target_name == 'searchlight_mean':
                    target_ts.append(np.mean(subj_ts[target_roi],axis=1)[:,np.newaxis])
            all_target_ts.append(np.concatenate(target_ts,axis=1))
    if len(all_target_ts)==1:
        return all_target_ts[0]
    else:
        return all_target_ts

def calc_target_ts_single_subj(dset,sid,target,task,run,**kwargs):
    targetrois,targetspace = get_rois_and_space(target,dset.surface_space,dset.volume_space)
    target_ts = get_dm(dset, sid, targetrois, task, run, targetspace,
            atlas=target,array_output=False,**kwargs)
    if target.startswith('onavg'):
        MAPPINGS = {roi:nb.mapping(roi, dset.surface_space, target, mask=True) for roi in targetrois}
        target_ts = get_target_ts(target_ts,'mapping',target_item=MAPPINGS)
    else:
        target_ts = get_target_ts(target_ts,'searchlight_mean')
    return target_ts

def calculate_connectivity(ts_target, ts_seed, zscore_axis=0, z=True, metric='dot_product_correlation',counts=None,skip_zscore=False,dtype=np.float64):
    """
    ts_target, ts_seed: (nT, nV)
    returns: (nV_target, nV_seed)
    """
    X = ts_target.astype(dtype)  # scipy used float64 as default so the dtype of float64 would match scipy result
    Y = ts_seed.astype(dtype)

    if metric=='dot_product_correlation':
        if not skip_zscore:
            if counts is None:
                X = X - np.mean(X, axis=0)
                std_x = np.std(X, axis=0)
                std_x[std_x == 0] = 1.0
                X /= std_x
                Y = Y - np.mean(Y, axis=0)
                std_y = np.std(Y, axis=0)
                std_y[std_y == 0] = 1.0
                Y /= std_y
            else:
                X = X - np.nanmean(X, axis=0)
                std_x = np.nanstd(X, axis=0)
                std_x[std_x == 0] = 1.0
                X /= std_x
                Y = Y - np.nanmean(Y, axis=0)
                std_y = np.nanstd(Y, axis=0)
                std_y[std_y == 0] = 1.0
                Y /= std_y
                X = np.nan_to_num(X)
                Y = np.nan_to_num(Y)
        T = counts if counts is not None else X.shape[0]
        conn = (X.T @ Y) / T
        conn[conn == 0] = np.nan
    else:
        assert counts is None and not skip_zscore,"only for data that has not been scrubbed"
        conn = 1 - cdist(X.T, Y.T, metric)
    if z:
        conn = conn - np.nanmean(conn, axis=zscore_axis, keepdims=True)
        std = np.nanstd(conn, axis=zscore_axis, keepdims=True)
        std[std==0]=1.0
        conn = np.nan_to_num(conn / std)
    return conn

def calc_dm_across_runs(dset,sid,roi,seedspace,targets,seeds,task,run,separate_zscore=True,zscore_axis=0,**kwargs):
    if not isinstance(targets,list):
        targets = [targets]
    if not isinstance(sid,str):
        ValueError("Expect single sid not an array or list")
    if type(run) is int:
        run = [run]
    dm_allruns = []
    for curr_run in run:
        seed_ts = get_dm(dset,sid,roi,task,curr_run, seedspace,atlas=seeds,array_output=True,**kwargs)
        dm = []
        for target in targets:
            if target == "None":
                raise ValueError("Transformations not needed for anatomical alignment. Exiting")
            else:
                if target == 'response':
                    dm_allruns.append(seed_ts)
                else:
                    target_ts = calc_target_ts_single_subj(dset,sid, target, task, curr_run,**kwargs)
                    if separate_zscore:
                        dm.append(calculate_connectivity(target_ts,seed_ts,zscore_axis=zscore_axis,z=True))
                    else:
                        dm.append(calculate_connectivity(target_ts,seed_ts,zscore_axis=zscore_axis,z=False))
                    dm = np.concatenate(dm,axis=0)
                    if not separate_zscore:
                        dm = np.nan_to_num(zscore(dm, axis=zscore_axis, nan_policy='omit'))
                    dm_allruns.append(dm)
    try:
        return np.stack(dm_allruns,axis=0)
    except:
        print("Not able to concatenate across runs, different lengths?")
        return dm_allruns

def reliability_weighting_hyperalignment(X, Y, func, voxel_reliability_scores=None, threshold=None):
    """
    Reliability-weighted hyperalignment.

    X, Y: (N, P)  # N timepoints/samples, P voxels/features
    func: callable (X, Y) -> transform
          - Procrustes: returns orthogonal Q (P,P) such that X @ Q ~ Y
          - Ridge: returns linear T_w (P,P) such that X_w @ T_w ~ Y_w

    voxel_reliability_scores: (P,) in [0,1] ideally (or will be clipped)
    threshold: if not None, binarize weights: w=1 if score>=threshold else 0

    Returns:
      - If func is procrustes-like: Q (P,P), orthogonal (up to numeric error)
      - If func is ridge-like:     T (P,P) mapped back to original space
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    if X.shape != Y.shape:
        raise ValueError(f"X and Y must have same shape, got {X.shape} vs {Y.shape}")
    if X.ndim != 2:
        raise ValueError(f"Expected 2D arrays (N,P); got X.ndim={X.ndim}")

    P = X.shape[1]

    # No weighting
    if voxel_reliability_scores is None:
        return func(X, Y)

    scores = np.asarray(voxel_reliability_scores).reshape(-1)
    if scores.shape[0] != P:
        raise ValueError(f"voxel_reliability_scores must have length P={P}, got {scores.shape[0]}")

    if threshold is None:
        w = np.clip(scores.astype(float), 0.0, 1.0)
    else:
        w = np.ones(P, dtype=float)
        w[scores < threshold] = 0.0

    # Construct W = diag(sqrt(w))
    wsqrt = np.sqrt(w)
    # Efficient column scaling instead of making a dense diag
    X_scaled = X * wsqrt[None, :]
    Y_scaled = Y * wsqrt[None, :]

    T_scaled = func(X_scaled, Y_scaled)

    # Decide whether we should map back (ridge) or not (procrustes).
    # Procrustes returns an orthogonal matrix in voxel space; keep it orthogonal.
    # Ridge returns an unconstrained linear map in the scaled space; map back.
    name = getattr(func, "__name__", None)
    is_partial = hasattr(func, "keywords")  # functools.partial
    if is_partial:
        base = getattr(func, "func", None)
        base_name = getattr(base, "__name__", "")
    else:
        base_name = name or ""

    procrustes_like = "procr" in base_name.lower()

    if procrustes_like:
        # This is already the orthogonal Q that best aligns X_scaled -> Y_scaled.
        # Applying it to unweighted X is what you want, and it preserves orthogonality.
        return T_scaled

    # Ridge-like: map back to original space: T = W * T_scaled * W^{-1}
    # Use pseudo-inverse on W if any zeros
    inv_wsqrt = np.zeros_like(wsqrt)
    nz = wsqrt > 0
    inv_wsqrt[nz] = 1.0 / wsqrt[nz]

    # Left-multiply scales rows, right-multiply scales columns
    T = (wsqrt[:, None] * T_scaled) * inv_wsqrt[None, :]
    return T

def _slice_features(A, sl):
    """Take a feature subset from either a 2D (nT, nV) or a 3D (nruns, nT, nV) array."""
    A = np.asarray(A)
    return A[:, :, sl] if A.ndim == 3 else A[:, sl]


def searchlight_hyperalignment_per_sl_alpha(X, Y, sls, alphas, mat0=None, weights=None):
    """searchlight_hyperalignment with a SEPARATE ridge alpha per searchlight.

    hyperalignment.searchlight_hyperalignment takes one ``sl_func`` for every searchlight,
    so a per-searchlight alpha has nowhere to enter. This writes the same blocks, in the
    same order and with the same weights, but regularises each searchlight with its own
    alpha. With one shared alpha it is identical to searchlight_hyperalignment.

    Only meaningful when the searchlights are disjoint atlas parcels (the *_wholebrain
    subcortical atlases). For overlapping cortical searchlights the blocks superpose, so
    a voxel's prediction would mix several alphas.
    """
    alphas = list(alphas)
    if len(alphas) != len(sls):
        raise ValueError(f"need one alpha per searchlight; got {len(alphas)} for {len(sls)} sls")
    mat = np.zeros((X.shape[1], Y.shape[1])) if mat0 is None else mat0.copy()
    if weights is None:
        weights = [np.ones(len(sl)) for sl in sls]
    for sl, w, alpha in zip(sls, weights, alphas):
        t = ridge(X[:, sl], Y[:, sl], alpha=alpha)
        mat[np.ix_(sl, sl)] += t * w[np.newaxis]
    return mat


def loo_cv_per_searchlight(X, Y, sls, alphas, voxel_reliability_scores=None):
    """loo_cv run separately inside each searchlight, across a grid of ridge alphas.

    Same leave-one-run-out split and the same fit direction as loo_cv (whichever of X, Y
    is 3D supplies the runs), just not pooled over searchlights before the error is read
    off. Calls loo_cv itself on each feature subset, so the direction handling and the
    reliability weighting cannot drift from the pooled path.

    Because the parcels are disjoint and cover every voxel, the pooled MSE is recoverable
    exactly as the voxel-count-weighted mean over searchlights -- so the caller can pick a
    single pooled alpha and per-searchlight alphas from this one result.

    Returns
    -------
    mse_train, mse_val : ndarray of shape (n_searchlights, n_alphas)
        Fold-averaged MSE within each searchlight.
    """
    alphas = list(alphas)
    mse_train = np.zeros((len(sls), len(alphas)))
    mse_val = np.zeros((len(sls), len(alphas)))
    scores = None if voxel_reliability_scores is None else np.asarray(voxel_reliability_scores)
    for p, sl in enumerate(sls):
        X_sl, Y_sl = _slice_features(X, sl), _slice_features(Y, sl)
        rel = None if scores is None else scores[sl]
        for a, alpha in enumerate(alphas):
            func = get_hyperalignment_func('ridge', alpha)
            mse_train[p, a], mse_val[p, a] = loo_cv(
                X_sl, Y_sl, func, voxel_reliability_scores=rel)
    return mse_train, mse_val


def evaluate_transformation(X_val, Y_val, xfm, compute_correlation=True):
    """
    Evaluate the transformation on validation set.

    compute_correlation=False skips the per-column correlation, which is a Python loop over
    every voxel and is pure overhead for callers that only read 'mse' -- loo_cv does exactly
    that, and loo_cv is inside every ridgeCV alpha search (every transformation script). The returned
    'correlation' is then None. Kept True by default so direct callers
    (hyperalignment_withweights.py) are unaffected.
    """
    # Predict on validation set
    Y_pred = X_val @ xfm  # (16,285) @ (285,285) = (16,285)

    # Calculate metrics
    mse = np.mean((Y_val - Y_pred) ** 2)
    if not compute_correlation:
        return {'mse': mse, 'correlation': None}
    # vectorised equivalent of the old per-column np.corrcoef loop
    A = Y_val - Y_val.mean(axis=0, keepdims=True)
    B = Y_pred - Y_pred.mean(axis=0, keepdims=True)
    denom = np.sqrt((A ** 2).sum(axis=0) * (B ** 2).sum(axis=0))
    denom[denom == 0] = np.nan
    correlation = np.nanmean((A * B).sum(axis=0) / denom)
    return {
        'mse': mse,
        'correlation': correlation
    }

def loo_cv(X, Y, func,voxel_reliability_scores=None):
    """
    Leave one run out cross-validation. Fits/evaluates func(X, Y) i.e. X -> Y,
    same order as passed in (so callers control alignment direction).
    Exactly one of X, Y must be 3D (nruns, nT, nV); that side supplies the runs
    for the leave-one-out loop, the other side is a fixed 2D (nT,nV) array.
    reliability: reliability [0-1] to be used for weighting
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    if X.ndim == 3 and Y.ndim == 2:
        runs_is_X = True
        runs_arr, fixed = X, Y
    elif Y.ndim == 3 and X.ndim == 2:
        runs_is_X = False
        runs_arr, fixed = Y, X
    else:
        raise ValueError(
            f"Exactly one of X, Y must be 3D (nruns, nT, nV); got X.ndim={X.ndim}, Y.ndim={Y.ndim}"
        )

    nruns = runs_arr.shape[0]
    mse_train_all = []
    mse_val_all   = []

    for i in range(nruns):
        runs_val = runs_arr[i]
        idx = np.arange(nruns) != i
        runs_train = np.mean(runs_arr[idx], axis=0)

        if runs_is_X:
            R = reliability_weighting_hyperalignment(runs_train, fixed, func, voxel_reliability_scores=voxel_reliability_scores,threshold=None)
            mt = evaluate_transformation(runs_train, fixed, R, compute_correlation=False)
            mv = evaluate_transformation(runs_val, fixed, R, compute_correlation=False)
        else:
            R = reliability_weighting_hyperalignment(fixed, runs_train, func, voxel_reliability_scores=voxel_reliability_scores,threshold=None)
            mt = evaluate_transformation(fixed, runs_train, R, compute_correlation=False)
            mv = evaluate_transformation(fixed, runs_val, R, compute_correlation=False)

        mse_train_all.append(mt['mse'])
        mse_val_all.append(mv['mse'])

    # average across LOO folds
    return np.mean(mse_train_all), np.mean(mse_val_all)
