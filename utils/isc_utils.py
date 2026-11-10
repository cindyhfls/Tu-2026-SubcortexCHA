import os
import numpy as np
import pandas as pd
import neuroboros as nb
from scipy.spatial.distance import squareform,cdist
# calculate_connectivity is imported lazily inside conn_features_all_subjects: importing it at
# module level drags in hyperalignment.ridge, which is absent from some envs (e.g. the "ann" env
# used for the encoding-model scripts), and that made isc_utils unimportable there even for
# callers that never touch connectivity.

# I did some tests and it looked like numpy.corrcoef and scipy.pearsonr and neuroboros.pearsonr are consistent with zscoring with ddof=0 (population standard deviation instead of sample standard deviation), which is what we have here

def FisherAvg(r, axis=0, has_nan=False, clip=0.999999):
    """Average correlations in Fisher-z space rather than as raw r.

    r is bounded and its sampling distribution is skewed, so a plain mean is biased toward
    zero and the bias grows with |r| -- exactly where conditions differ most.

    clip bounds |r| just inside 1 before arctanh. This matters more than it looks: pairwise
    ISC files carry the i==i diagonal at exactly 1.0, arctanh(1) is +inf, and the mean of
    anything with a +inf in it is +inf, so tanh() returns exactly 1.0 -- silently discarding
    every other value in the average rather than raising. Pass clip=None to disable.

    has_nan=False keeps np.mean, so NaNs propagate and a hole in the data is visible; pass
    True for nanmean. (This was the pre-existing default and is left alone.)
    """
    r = np.asarray(r, dtype=float)
    if clip is not None:
        r = np.clip(r, -clip, clip)
    avg = np.nanmean if has_nan else np.mean
    return np.tanh(avg(np.arctanh(r), axis=axis))


def pairwise_corrsim_dot(X, dtype=np.float32,skip_zscore=False, has_nan=False, valid=None):
    """Condensed pairwise Pearson correlation between the rows of X.

    valid : (N, T) bool or None
        None (default) reproduces the historical behaviour EXACTLY, bit for bit, for every
        existing caller.

        When given, each pair is correlated over only the timepoints BOTH subjects retain:
        the mean, the sd and the sum of products are all taken over that joint set. This is
        the estimator wanted for motion-censored data, where censored frames are stored as
        exact zeros -- those zeros are finite, so they are otherwise counted as real data and
        every correlation comes out attenuated by ~sqrt(f_i*f_j) (f = retained fraction).
        X must be (N, T, V) or (N, T); the mask applies along the time axis and is shared by
        all V features of a timepoint, which is what makes the exact computation cheap (see
        below). skip_zscore/has_nan are ignored in this branch.

    NOTE on has_nan=True: it does NOT exclude NaNs, despite appearances -- np.nan_to_num runs
    before np.isfinite below, so counts_mat is the full feature count for every pair. That
    behaviour is preserved here rather than fixed, so `valid=None` stays bit-identical for all
    existing callers; use `valid=` for the correct masked estimator.
    """
    X = np.asarray(X)
    N = X.shape[0]
    if valid is not None:
        return _pairwise_corrsim_dot_masked(X, np.asarray(valid), dtype=dtype)
    F = X.reshape(N, -1).astype(dtype, copy=False)
    if not skip_zscore:
        if has_nan:
            F -= np.nanmean(F,axis=1, keepdims=True)
            std = np.nanstd(F,axis=1, keepdims=True)
            std[std==0] = 1.0
            F /= std
            F = np.nan_to_num(F)
        else:
            F -= np.mean(F,axis=1, keepdims=True)
            std = np.std(F,axis=1, keepdims=True)
            std[std==0] = 1.0
            F /= std

    num = F @ F.T
    if has_nan:
        valid = np.isfinite(F)  # (N, T*V)
        counts_mat = valid.astype(np.float32) @ valid.astype(np.float32).T  # (N, N)
    else:
        counts_mat = F.shape[1]
    with np.errstate(invalid="ignore", divide="ignore"):
        C = num / counts_mat

    iu = np.triu_indices(N, k=1)
    return C[iu].astype(dtype, copy=False)


def _pairwise_corrsim_dot_masked(X, valid, dtype=np.float32):
    """Exact per-pair Pearson r over jointly-retained timepoints, without looping over pairs.

    X: (N, T, V) or (N, T);  valid: (N, T) bool.

    Because the frame mask is shared by all V features of a timepoint, the per-pair sums that
    a masked correlation needs factor into one big matmul (the same one the unmasked version
    already does) plus two cheap (N,T)@(T,N) ones -- so the correct estimator costs about the
    same as the wrong one, instead of the O(N^2) Python loop the mask naively implies:

        Sxy = Xm Xm^T                      sum over joint frames of x_i * x_j
        Sx  = (sum_v Xm) W^T               sum over joint frames of x_i
        Sxx = (sum_v Xm^2) W^T             sum over joint frames of x_i^2
        n   = V * (W W^T)                  number of jointly-valid features
        r   = (Sxy - Sx Sx^T / n) / sqrt((Sxx - Sx^2/n)(Sxx^T - (Sx^T)^2/n))

    Validated against an explicit per-pair loop: max abs diff 1.3e-7, ~35x faster.
    """
    if X.ndim == 2:
        X = X[:, :, None]
    N, T, V = X.shape
    if valid.shape != (N, T):
        raise ValueError(f"valid must be (N, T) = {(N, T)}, got {valid.shape}")
    W = valid.astype(dtype)
    Xm = X.astype(dtype, copy=False) * W[:, :, None]
    # Vertex dropout: some subjects have NaN at individual vertices even in UNSCRUBBED data.
    # A single NaN would propagate through the matmuls below and turn every pair involving
    # that subject into NaN, so it is zeroed here -- matching what np.nan_to_num did in the
    # unmasked path. Dropout is a spatial mask (per vertex, all timepoints) while censoring is
    # temporal (per timepoint, all vertices); the two do not share a factorisation, so those
    # entries are counted in n rather than excluded from it. That leaves the same small
    # attenuation on the affected subjects that the pipeline already had -- it is not made
    # worse here, but it is not fixed either. Excluding a dropout vertex from the searchlight
    # for ALL subjects would fix it exactly, at the cost of changing the feature set.
    n_bad = int((~np.isfinite(Xm)).sum())
    if n_bad:
        Xm = np.nan_to_num(Xm, copy=False)
    Sxy = Xm.reshape(N, -1) @ Xm.reshape(N, -1).T
    Sx = Xm.sum(axis=2) @ W.T
    Sxx = (Xm * Xm).sum(axis=2) @ W.T
    n = V * (W @ W.T)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = Sxy - Sx * Sx.T / n
        vx = Sxx - Sx * Sx / n
        C = cov / np.sqrt(vx * vx.T)
    iu = np.triu_indices(N, k=1)
    return C[iu].astype(dtype, copy=False)


def pairwise_corrsim_dot_xy(X, Y, dtype=np.float32, has_nan=False, nan=None):
    """
    Pairwise Pearson correlation similarity between rows of X and rows of Y.

    Parameters
    ----------
    X : array, shape (N, F) or (N, ...)
    Y : array, shape (M, F) or (M, ...)  # must match feature size after reshape
    dtype : np dtype
        Working dtype.
    has_nan : bool
        If True, z-score with nanmean/nanstd and normalize each pair by the
        number of jointly-valid columns instead of the full feature count.
    nan : float or None
        If not None, replace NaNs in output with this value.

    Returns
    -------
    C : array, shape (N, M)
        C[i, j] = corr( X[i], Y[j] ) computed across flattened features.
        Equivalent to: 1 - cdist(X, Y, metric="correlation") (up to fp precision).
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    if X.ndim < 2 or Y.ndim < 2:
        raise ValueError("Expected X and Y to have shape (N, ...) and (M, ...).")

    Fx = X.reshape(X.shape[0], -1).astype(dtype, copy=True)  # (N, F)
    Fy = Y.reshape(Y.shape[0], -1).astype(dtype, copy=True)  # (M, F)

    if Fx.shape[1] != Fy.shape[1]:
        raise ValueError(f"Feature mismatch: X has {Fx.shape[1]} features, Y has {Fy.shape[1]}.")

    if has_nan:
        valid_x = np.isfinite(Fx)
        valid_y = np.isfinite(Fy)
        Fx -= np.nanmean(Fx, axis=1, keepdims=True)
        Fy -= np.nanmean(Fy, axis=1, keepdims=True)
        sx = np.nanstd(Fx, axis=1, keepdims=True)
        sy = np.nanstd(Fy, axis=1, keepdims=True)
        sx[sx == 0] = 1.0
        sy[sy == 0] = 1.0
        Fx /= sx
        Fy /= sy
        Fx = np.nan_to_num(Fx)
        Fy = np.nan_to_num(Fy)
    else:
        Fx -= Fx.mean(axis=1, keepdims=True)
        Fy -= Fy.mean(axis=1, keepdims=True)

        sx = Fx.std(axis=1, keepdims=True)
        sy = Fy.std(axis=1, keepdims=True)
        sx[sx == 0] = 1.0
        sy[sy == 0] = 1.0

        Fx /= sx
        Fy /= sy

    num = Fx @ Fy.T
    if has_nan:
        counts_mat = valid_x.astype(np.float32) @ valid_y.astype(np.float32).T  # (N, M)
    else:
        counts_mat = Fx.shape[1]
    with np.errstate(invalid="ignore", divide="ignore"):
        C = num / counts_mat

    if nan is not None:
        C = np.nan_to_num(C, copy=False, nan=nan)

    return C.astype(dtype, copy=False)

def pairwise_corrsim_dot_avg(X,dtype=np.float32):
    """
    X: (N, F) or (N, ...)
    dtype: data precision
    Returns condensed pairwise Pearson correlation similarity, averaged across each subject.
    """
    r = pairwise_corrsim_dot(X,dtype=dtype)
    R = squareform(r)
    ravg = np.empty(R.shape[0])
    for curr_row in range(len(R)):
        lst = [i for i in range(len(R)) if i != curr_row]
        ravg[curr_row] = FisherAvg(R[curr_row,lst])
    return ravg.astype(dtype,copy=False)

def pairwise_corrsim_dot_xy_avg(X,Y,dtype=np.float32):
    """
    X: (N, F) or (N, ...)
    Y: (N, F) or (N, ...)
    dtype: data precision
    Returns condensed pairwise Pearson correlation similarity, averaged across each subject and the within-subject (diagonals).
    """
    R = pairwise_corrsim_dot_xy(X, Y, dtype)
    ravg = np.empty(len(R))
    for curr_row in range(len(R)):
        lst = [i for i in range(len(R)) if i != curr_row]
        ravg[curr_row] = FisherAvg(R[curr_row,lst])
    return ravg,np.diagonal(R)




def loo_corrsim_dot(X,dtype=np.float32):
    """
    This does not take into account the NaNs, we will think about it.
    X: (N, F) or (N, ...)
    dtype: data precision
    Returns:
        r: (N,) correlation of each row with the mean of all others
    """
    X = np.asarray(X)
    if X.ndim < 2:
        raise ValueError("Expected X to have shape (N, ...).")

    N = X.shape[0]
    F = X.reshape(N, -1).astype(dtype, copy=False)

    # z-score per subject
    F -= F.mean(axis=1, keepdims=True)
    std = F.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    F /= std

    # sum across subjects
    F_sum = F.sum(axis=0, keepdims=True)   # (1, F)

    # leave-one-out mean
    F_loo = (F_sum - F) / (N - 1)           # (N, F)

    # z-score LOO mean
    F_loo -= F_loo.mean(axis=1, keepdims=True)
    std = F_loo.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    F_loo /= std

    # correlation
    r = np.sum(F * F_loo, axis=1) / F.shape[1]
    return r.astype(dtype, copy=False)

def loo_corrsim_dot_xy(X,Y,dtype=np.float32):
    """
    X: (N, F) or (N, ...)
    Y: (N, F) or (N, ...)
    dtype: data precision
    Returns:
        r: (N,) correlation of each row in X with the mean of all others in Y
    """
    X = np.asarray(X)
    if X.ndim < 2 or Y.ndim <2:
        raise ValueError("Expected X to have shape (N, ...).")
    if (X.shape[0] !=Y.shape[0]) | (X.shape[1] !=Y.shape[1]):
        raise ValueError(f"X and Y does not have the same shape.")

    N = X.shape[0]
    Fx = X.reshape(X.shape[0], -1).astype(dtype, copy=True)  # (N, F)
    Fy = Y.reshape(Y.shape[0], -1).astype(dtype, copy=True)  # (N, F)

    # z-score each row (ddof=0 matches np.std default; aligns with many cdist impls)
    Fx -= Fx.mean(axis=1, keepdims=True)
    Fy -= Fy.mean(axis=1, keepdims=True)

    sx = Fx.std(axis=1, keepdims=True)
    sy = Fy.std(axis=1, keepdims=True)
    sx[sx == 0] = 1.0
    sy[sy == 0] = 1.0

    Fx /= sx
    Fy /= sy

    # sum across subjects
    Fy_sum = Fy.sum(axis=0, keepdims=True)   # (1, F)

    # leave-one-out mean
    Fy_loo = (Fy_sum - Fy) / (N - 1)           # (N, F)

    # z-score LOO mean
    Fy_loo -= Fy_loo.mean(axis=1, keepdims=True)
    std = Fy_loo.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    Fy_loo /= std

    # correlation
    r = np.sum(Fx * Fy_loo, axis=1) / Fx.shape[1]
    return r.astype(dtype, copy=False)

# ---------------------------------------------------------------------------
# Voxelwise (column-wise) ISC.
#
# pairwise_corrsim_dot_xy / loo_corrsim_dot_xy flatten (A, B) into one long vector and
# return a single SPATIOTEMPORAL correlation for the region. The two functions below
# instead correlate along axis A separately for each column b and Fisher-average the B
# resulting correlations -- i.e. "the ordinary per-voxel ISC, averaged over the voxels of
# the region". The two statistics answer different questions:
# the spatiotemporal one is sensitive to the region's spatial pattern, the voxelwise one
# is not (it cannot see whether voxels are in the right place relative to each other, only
# whether each voxel's own timecourse matches).
#
# The same pair of functions covers both domains, because "axis A" is whatever the
# features are indexed by:
#   response     A = timepoints,            B = voxels      -> voxelwise temporal ISC
#   connectivity A = connectivity targets,  B = seed voxels -> per-voxel connectivity ISC
# For connectivity this requires the UNFLATTENED (N, nTargets, nSeeds) features, i.e.
# conn_features_all_subjects(..., flatten=False).
# ---------------------------------------------------------------------------

def _zscore_axis1(X, dtype):
    """z-score along axis 1 (the feature axis being correlated over), per (subject, column)."""
    F = np.asarray(X).astype(dtype, copy=True)
    F -= F.mean(axis=1, keepdims=True)
    std = F.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    F /= std
    return F


def loo_voxelwise_corrsim_dot_xy(X, Y, dtype=np.float32, average=True):
    """
    Leave-one-out counterpart of voxelwise_corrsim_dot_xy, matching loo_corrsim_dot_xy's
    convention: each row of X against the mean of all OTHER rows of Y.

    X, Y: (N, A, B)
    Returns
    -------
    r : (N,) if average else (N, B)
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    if X.ndim != 3 or Y.ndim != 3:
        raise ValueError(f"Expected (N,A,B); got {X.shape} and {Y.shape}.")
    if X.shape != Y.shape:
        raise ValueError(f"X and Y do not have the same shape: {X.shape} vs {Y.shape}.")

    N, A, B = X.shape
    Fx = _zscore_axis1(X, dtype)
    Fy = _zscore_axis1(Y, dtype)

    Fy_loo = (Fy.sum(axis=0, keepdims=True) - Fy) / (N - 1)  # (N, A, B)
    Fy_loo = _zscore_axis1(Fy_loo, dtype)                    # re-z-score, as loo_corrsim_dot_xy does

    r = np.sum(Fx * Fy_loo, axis=1) / A  # (N, B)
    if not average:
        return r.astype(dtype, copy=False)
    return FisherAvg(r, axis=-1).astype(dtype, copy=False)


def loo_voxelwise_corrsim_dot(X, dtype=np.float32, average=True):
    """Within-X leave-one-out counterpart of loo_corrsim_dot, voxelwise.

    X : (N, A, B)   e.g. (subjects, timepoints, voxels) for response, or
                    (subjects, connectivity targets, seed voxels) for connectivity
    Returns (N,) -- each subject against the mean of all the others, correlated along axis A
    separately per column of B and Fisher-averaged across B. This is what fig2_3_loo.py
    needs; loo_voxelwise_corrsim_dot_xy is the two-array form supp_isc_to_other.py uses.
    """
    return loo_voxelwise_corrsim_dot_xy(X, X, dtype=dtype, average=average)


def psf_corrsim(X,dist,roi,voxel_pairs):
    all_r = np.empty(X.shape[0])
    for isubj in range(X.shape[0]):
        X_others_mean = np.mean(np.delete(X, isubj, axis=0), axis=0)
        if dist==0:
            r = nb.pearsonr(X[isubj,:,:],X_others_mean)
            all_r[isubj] = FisherAvg(r)
        elif voxel_pairs[roi][dist].size == 0:
            all_r[isubj] = np.nan
        else:
            r = nb.pearsonr(X[isubj,:,voxel_pairs[roi][dist][0]].T,X_others_mean[:,voxel_pairs[roi][dist][1]])
            all_r[isubj] = FisherAvg(r)
    return all_r

def psf_corrsim_withinsubj(X,dist,roi,voxel_pairs):
    all_r = np.empty(X.shape[0])
    for isubj in range(X.shape[0]):
        if dist==0:
            r = nb.pearsonr(X[isubj,:,:],X[isubj,:,:])
            all_r[isubj] = FisherAvg(r)
        elif voxel_pairs[roi][dist].size == 0:
            all_r[isubj] = np.nan
        else:
            r = nb.pearsonr(X[isubj,:,voxel_pairs[roi][dist][0]].T,X[isubj,:,voxel_pairs[roi][dist][1]].T)
            all_r[isubj] = FisherAvg(r)
    return all_r

def parse_data_coarse_fine(X):
    """
    X: (N, T, Vsl)
    Returns:
      coarse_X:     (N, T, 1)     # mean across vertices
      fine_X:       (N, T, Vsl)   # residuals
    """
    coarse_X = X.mean(axis=2, keepdims=True)   # (N,T,1)
    fine_X = X - coarse_X                      # (N,T,Vsl)
    return coarse_X, fine_X

def conn_features_all_subjects(targets_ts, seed_ts, zscore_axis=0, z=True, dtype=np.float32,flatten=True,counts=None,**kwargs):
    """
    targets_ts: (N, T, nTargets)
    seed_ts:    (N, T, nSeeds)
    dtype: data precision
    returns:    (N, nTargets*nSeeds) connectivity features (flattened)
    """
    N = seed_ts.shape[0]
    nTargets = targets_ts.shape[2]
    nSeeds = seed_ts.shape[2]
    if flatten:
        feats = np.empty((N, nTargets * nSeeds), dtype=dtype)
    else:
        feats = np.empty((N,nTargets,nSeeds),dtype=dtype)
    from utils.hyperalignment_utils import calculate_connectivity
    for s in range(N):
        c = counts[s] if counts is not None else None
        conn = calculate_connectivity(
            targets_ts[s], seed_ts[s],
            zscore_axis=zscore_axis, z=z, counts=c,**kwargs
        )
        if flatten:
            feats[s] = conn.reshape(-1).astype(dtype)
        else:
            feats[s] = conn.astype(dtype)
    return feats

def compute_targets_ts_from_mapping(dms_lr, source_space, target_space, mask=True, lr = "lr", dtype=np.float32):
    mapping = nb.mapping(lr, source_space, target_space, mask=mask)
    N, T, Vsrc = dms_lr.shape
    X2 = dms_lr.reshape(N*T, Vsrc)      # memmap view
    Y2 = X2 @ mapping                   # returns dense
    Y2 = np.asarray(Y2, dtype=dtype)
    return Y2.reshape(N, T, mapping.shape[1])  # (N,T,Vtgt)
