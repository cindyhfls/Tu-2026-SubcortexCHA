import numpy as np
import pandas as pd
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

TianLUT = pd.read_csv(os.path.join(HERE, "support_files", "atlas-TianS1S2S3S4_combined_keylabel.tsv"), sep="\t")
TianLUT_7T = pd.read_csv(os.path.join(HERE, "support_files", "atlas-TianS1S2S3S4_combined_keylabel_7T_space-MNI152NLin6Asym_res-1pt6mm.tsv"), sep="\t")

def roi_hemisphere(label):
    """'left', 'right', or None for a subcortical atlas ROI name.

    The two atlases tag the hemisphere differently: aseg uses an 'l-'/'r-' PREFIX
    ('l-amygdala'), Tian a '-lh'/'-rh' SUFFIX ('AMY-lh'). Anything that carries neither
    returns None, so callers can decide whether an untagged ROI is a problem.
    """
    if label.endswith('-lh'):
        return 'left'
    if label.endswith('-rh'):
        return 'right'
    if label.startswith('l-'):
        return 'left'
    if label.startswith('r-'):
        return 'right'
    return None


def return_Tian_labels(atlas_name,is_7T=False):
    if '_' in atlas_name:
        atlas_name = atlas_name.split('_')[0]
    if is_7T:
        labels = np.unique(TianLUT_7T[f'{atlas_name}_label'])
    else:
        labels = np.unique(TianLUT[f'{atlas_name}_label'])
    return list(labels)

def return_aseg_labels(atlas_name='aseg'):
    labels = ['l-accumbens', 'l-amygdala', 'l-caudate','l-hippocampus','l-pallidum','l-putamen','l-thalamus','l-ventral-diencephalon',
                    'r-accumbens', 'r-amygdala', 'r-caudate','r-hippocampus','r-pallidum','r-putamen','r-thalamus','r-ventral-diencephalon']
    if atlas_name.endswith('subj'):
        labels = [f'{label}_subj' for label in labels]
    return labels

def convert_dataarray_to_dictionary_byroi(data, atlas_name):
    '''
    data: Input is nTimepoints x nVoxels
    TianLUT: table specifying the keys for each ROI

    Returns:
    d: the dictionary containing data for each ROI
    labels: the labels based on the atlas given use this to check it matches with the ROIS you have

    '''
    d = {}
    labels = return_Tian_labels(atlas_name)
    for label in labels:
        idx = TianLUT[f'{atlas_name}_label'] == label
        d[label] = data[:,idx]
    return d,labels

def get_rois_and_space(atlas,surface_space,volume_space):
    if atlas.startswith('onavg') or atlas == 'lr':
        seedrois = ['l','r']
        space = surface_space
    elif atlas.startswith('Tian'):
        if 'wholebrain' in atlas:
            seedrois = ['Tian_Subcortex']
        else:
            seedrois = return_Tian_labels(atlas)
        space = volume_space
    elif atlas.startswith('aseg'):
        if 'wholebrain' in atlas:
            seedrois = ['aseg_Subcortex']
        else:
            seedrois = return_aseg_labels(atlas)
        space = volume_space
    return seedrois,space

def get_Tian_sls(atlas_name,is_7T=False):
    if '_' in atlas_name:
        atlas_name = atlas_name.split('_')[0]
    labels = return_Tian_labels(atlas_name,is_7T=is_7T)
    sls = []
    for label in labels:
        if is_7T:
            idx = np.where(TianLUT_7T[f'{atlas_name}_label'] == label)[0]
        else:
            idx = np.where(TianLUT[f'{atlas_name}_label'] == label)[0]
        sls.append(idx)
    return sls

def get_aseg_sls():
    '''
    Make a list of indices for the ROI identity of each voxel in the atlas.
    Make sure your subcortical_rois is loaded with the same function return_{atlas}_labels otherwise it might be wrong!
    '''
    subcortical_atlas = 'aseg'
    subcortical_rois = return_aseg_labels()
    subcortical_voxelcount = pd.read_csv(os.path.join(HERE, "support_files", f'atlas-{subcortical_atlas}_VoxelCount.txt'),sep='\t')

    sls = []
    offset = 0
    for roi in subcortical_rois:
        nvox = np.array(subcortical_voxelcount.VoxelCount[subcortical_voxelcount.Label==roi])[0]
        sls.append(np.arange(offset,offset+nvox))
        offset += nvox
    print('Please check this is the order of your ROIs \n')
    print(subcortical_rois)
    return sls

def unwrap_dm_dict(dm_dict):
    '''Reshape timeseries data from dictionary to array in shape nSubject x nTime x nV'''
    dm = []
    for sid in dm_dict.keys():
        allrois = dm_dict[sid].keys()
        dm.append(np.concatenate([dm_dict[sid][roi] for roi in allrois],axis=1))
    dm = np.stack(dm, axis=0)
    return dm

def get_dm(dset, sids, rois, task, run, space, atlas='',saved_beta_path='',array_output = False,**kwargs):
    '''
    Returns a nested dictionary by subject and roi if array_output = False, else a ndarray of shape (nSubject, nTime, nVoxels/nVertices)
    '''
    if isinstance(rois,str):
        rois = [rois]
    if isinstance(sids,str):
        sids = [sids]
    dm = {}
    for sid in sids:
        ts = {}
        if atlas.startswith('Tian'):
            if saved_beta_path:
                curr_out = dset.get_data(sid, task, run, 'Tian_Subcortex', space=space,prep='saved_beta',prep_kwargs={'saved_beta_path':saved_beta_path,'load_beta':True},**kwargs)
            else:
                curr_out = dset.get_data(sid, task, run, 'Tian_Subcortex', space=space,**kwargs)
            if isinstance(curr_out, tuple): # retained data, timemask
                nT = len(curr_out[1])
                nV = curr_out[0].shape[1]
                data = curr_out[0]
            else:
                data = curr_out
            if rois[0]=='Tian_Subcortex':
                ts['Tian_Subcortex'] = data
            elif rois[0].endswith('lh') or rois[0].endswith('rh'): # dirty check for Tian, we could prob improve that
                ts,labels = convert_dataarray_to_dictionary_byroi(data, atlas) # e.g. 'TianS1'
                assert(rois==labels)
        elif  atlas == 'aseg_Subcortex':
            if saved_beta_path:
                curr_out = dset.get_data(sid, task, run, 'aseg_Subcortex', space=space,prep='saved_beta',prep_kwargs={'saved_beta_path':saved_beta_path,'load_beta':True},**kwargs)
            else:
                curr_out = dset.get_data(sid, task, run, 'aseg_Subcortex', space=space,**kwargs)
            if rois[0]=='aseg_Subcortex':
                ts['aseg_Subcortex'] = curr_out
        else:
            for roi in rois:
                if saved_beta_path:
                    curr_out = dset.get_data(sid, task, run, roi, space=space,prep='saved_beta',prep_kwargs={'saved_beta_path':saved_beta_path,'load_beta':True},**kwargs)
                else:
                    curr_out = dset.get_data(sid, task, run, roi, space=space,**kwargs)
                if isinstance(curr_out, tuple): # retained data, timemask
                    nT = len(curr_out[1])
                    nV = curr_out[0].shape[1]
                    data = curr_out[0]
                else:
                    data = curr_out
                ts[roi] = data
        dm[sid] = ts
    if array_output:
        dm = unwrap_dm_dict(dm)
        return np.squeeze(dm)
    else:
        return dm

def apply_dm_transform(dms, xfm, dtype=np.float32):
    N, T, Vsrc = dms.shape
    X2 = dms.reshape(N*T, Vsrc)      # memmap view
    Y2 = X2 @ xfm                # returns dense
    Y2 = np.asarray(Y2, dtype=dtype)
    return Y2.reshape(N, T, xfm.shape[1])  # (N,T,Vtgt)

# For writing and loading isc from memmaps
def condensed_index(i, j, N):
    """Condensed index for unordered pair (i<j)."""
    return (i * (2 * N - i - 1)) // 2 + (j - i - 1)

def build_pair_indexer(idx_i, N_total, idx_j=None):
    """
    idx_j is None:
        → unique unordered pairs within idx_i
          order = upper triangle by position (no sorting)
          squareform-compatible

    idx_j is not None:
        → all cross pairs between idx_i and idx_j
          order = input order (pdist2-style)
          rectangular matrix mapping

    Returns
    -------
    If idx_j is None:
        pair_idx : (P,) int64 condensed indices
        pairs    : (P,2) int64 subject ids (i<j)

    If idx_j is not None:
        pair_idx_grid : (A,B) int64 condensed indices
                        (-1 where invalid if allow_self=False)
        self_mask     : (A,B) bool
    """
    idx_i = np.asarray(idx_i, dtype=np.int64)

    # ─────────────────────────────────────────────
    # Case 1: within-set (squareform-compatible)
    # ─────────────────────────────────────────────
    if idx_j is None:
        a, b = np.triu_indices(len(idx_i), k=1)   # by POSITION, no sorting
        I = idx_i[a]
        J = idx_i[b]

        i = np.minimum(I, J)
        j = np.maximum(I, J)

        pair_idx = condensed_index(i, j, N_total).astype(np.int64)
        pairs = np.stack([i, j], axis=1)

        return pair_idx, pairs

    # ─────────────────────────────────────────────
    # Case 2: cross-set (rectangular, pdist2-style)
    # ─────────────────────────────────────────────
    idx_j = np.asarray(idx_j, dtype=np.int64)

    I = idx_i[:, None]    # (A,1)
    J = idx_j[None, :]    # (1,B)

    self_mask = (I == J)

    i = np.minimum(I, J)
    j = np.maximum(I, J)

    grid = condensed_index(i, j, N_total).astype(np.int64)
    grid[self_mask] = -1

    return grid, self_mask

def load_isc(
    base_dir,
    metric,
    idx_i,
    N_total,
    nSL_total,
    pairwise=True,
    idx_j=None,
    sl_idx=None,
    reduce=None,          # None | "mean_pairs" | "mean_sl" | "fisher_pairs" | "fisher_sl"
    dtype=np.float32
):
    """
    sl_idx:
      None            -> all searchlights
      array-like      -> specific SL indices (global indices)
    """
    if pairwise:
        total_cols = N_total * (N_total - 1) // 2
        pair_obj0, pair_obj1 = build_pair_indexer(idx_i, N_total, idx_j=idx_j)
        if idx_j is None:
            pair_idx_1d = pair_obj0                      # (P,)
        else:
            grid = pair_obj0                             # (A,B)
            self_mask = pair_obj1                        # (A,B)
            pair_idx_1d = grid[~self_mask].ravel()       # (A*B,)
    else:
        total_cols = N_total
        pair_idx_1d = idx_i
        assert idx_j is None, "if not pairwise, idx_j should not be provided"

    # number of pair-columns we’ll pull from memmap
    nPairs_subset = pair_idx_1d.size

    # normalize

    if sl_idx is None:
        sl_idx = np.arange(nSL_total)
    else:
        sl_idx = np.asarray(sl_idx, dtype=int)

    sl_mask = np.zeros(nSL_total, dtype=bool)
    sl_mask[sl_idx] = True

    # output buffers
    if reduce is None:
        out = np.empty((len(sl_idx), nPairs_subset), dtype=dtype)
        out[:] = np.nan
        out_row = 0

    elif reduce == "mean_pairs":
        out = np.empty(len(sl_idx), dtype=dtype)
        out_row = 0

    elif reduce == "mean_sl":
        sum_pairs = np.zeros(nPairs_subset, dtype=np.float64)
        cnt_pairs = np.zeros(nPairs_subset, dtype=np.int64)

    elif reduce == "fisher_pairs":
        out = np.empty(len(sl_idx), dtype=dtype)
        out_row = 0

    elif reduce == "fisher_sl":
        sum_atanh = np.zeros(nPairs_subset, dtype=np.float64)
        cnt_pairs = np.zeros(nPairs_subset, dtype=np.int64)
    else:
        raise ValueError("Invalid reduce mode")

    # iterate chunks
    for d in sorted(os.listdir(base_dir)):
        if not d.startswith("sl"):
            continue

        sl_start, sl_stop = map(int, d[2:].split("-"))
        path = os.path.join(base_dir, d, metric)

        # which SLs from this chunk do we need?
        chunk_sl = np.arange(sl_start, sl_stop)
        use = sl_mask[chunk_sl]
        if not np.any(use):
            continue

        local_rows = np.where(use)[0]   # indices inside chunk

        mm = np.memmap(
            path,
            mode="r",
            dtype=dtype,
            shape=(sl_stop - sl_start, total_cols)
        )
        sub = mm[local_rows][:, pair_idx_1d]    # (nSL_sub, n_valid)

        if reduce is None:
            out[out_row:out_row + sub.shape[0]] = sub
            out_row += sub.shape[0]

        elif reduce == "mean_pairs":
            out[out_row:out_row + sub.shape[0]] = np.nanmean(sub, axis=1)
            out_row += sub.shape[0]

        elif reduce == "mean_sl":
            finite = np.isfinite(sub)
            sum_pairs += np.nansum(sub, axis=0)
            cnt_pairs += finite.sum(axis=0)

        elif reduce == "fisher_pairs":
            out[out_row:out_row + sub.shape[0]] = np.tanh(np.nanmean(np.atanh(sub), axis=1))
            out_row += sub.shape[0]

        elif reduce == "fisher_sl":
            finite = np.isfinite(sub)
            sum_atanh += np.nansum(np.atanh(sub), axis=0)
            cnt_pairs += finite.sum(axis=0)

        del mm, sub

    if reduce == "mean_sl":
        out = (sum_pairs / cnt_pairs).astype(dtype)

        if idx_j is not None:
            A, B = grid.shape
            tmp = np.full((A, B), np.nan, dtype=out.dtype)
            tmp[~self_mask] = out
            out = tmp

    elif reduce == "fisher_sl":
        out = np.tanh(sum_atanh / cnt_pairs).astype(dtype)

        if idx_j is not None:
            A, B = grid.shape
            tmp = np.full((A, B), np.nan, dtype=out.dtype)
            tmp[~self_mask] = out
            out = tmp

    elif idx_j is not None and reduce is None:
        A, B = grid.shape
        tmp = np.full((out.shape[0], A, B), np.nan, dtype=out.dtype)
        tmp[:, ~self_mask] = out
        out = tmp

    return out

def ensure_memmap(path, shape, dtype=np.float32, fill=np.nan):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        mm = np.memmap(path, mode="w+", dtype=dtype, shape=shape)
        mm[:] = fill
        mm.flush()
    return np.memmap(path, mode="r+", dtype=dtype, shape=shape)

def row_done(mm_row):
    return np.all(np.isfinite(mm_row))


def untransformed_target_label(targets_label):
    """The connectivity-target label for a condition that applies NO transform (AA).

    A target label is a chain: everything before the last '+' names transforms the target
    passed through, and the last segment is the atlas itself.

        'CHA+aseg'              common space, aligned   -> 'aseg'
        'CHA+aseg-mean'                                 -> 'aseg-mean'
        'onavg-ico8'            never transformed       -> 'onavg-ico8'  (unchanged)

    AA applies no transform, so its target carries no markers at all: everything before the
    last '+' is stripped. This is the ONE place that rule is written down -- import it rather
    than re-deriving it.
    """
    return targets_label.split('+')[-1]


# ---------------------------------------------------------------------------------------------
# Preprocessed HCP 7T movie data (written by 00_cache_inputs/hcp_resample/hcp_prep.py).
# Layout: <prep_root>/<space>/<roi>/sub-<sid>_task-<task>_run-<run:02d>.npy, each file (T, V).
# The cortex is stored under the folders 'l-cerebrum' / 'r-cerebrum' (addressed as 'l' / 'r' here).
# ---------------------------------------------------------------------------------------------
def get_HCP(prep_root, sids, rois, task, runs, space, force_array=True, xfms=None):
    """Runs concatenated in time. Returns (N, T, V) with the ROIs concatenated along V
    (or a list per ROI if `force_array` is False and there are several).

    xfms: optional dict {roi: matrix}, applied as ts @ xfm (used to map ico32 -> ico8 targets).
    """
    if not isinstance(rois, list):
        rois = [rois]
    dms = []
    for roi in rois:
        folder = f'{roi}-cerebrum' if roi in ('l', 'r') else roi
        dm = []
        for sid in sids:
            d = []
            for run in runs:
                ts = np.load(f'{prep_root}/{space}/{folder}/sub-{sid}_task-{task}_run-{run:02d}.npy')
                d.append(ts @ xfms[roi] if xfms is not None else ts)
            dm.append(np.concatenate(d, axis=0))
        dms.append(np.stack(dm, axis=0))
    if len(rois) == 1:
        return dms[0]
    return np.concatenate(dms, axis=2) if force_array else dms


def get_HCP_acrossruns(prep_root, sid, rois, task, runs, space, xfms=None):
    """One subject; returns a list with one (T, V) array per run (ROIs concatenated along V)."""
    if not isinstance(rois, list):
        rois = [rois]
    dm = []
    for run in runs:
        d = []
        for roi in rois:
            folder = f'{roi}-cerebrum' if roi in ('l', 'r') else roi
            ts = np.load(f'{prep_root}/{space}/{folder}/sub-{sid}_task-{task}_run-{run:02d}.npy')
            d.append(ts @ xfms[roi] if xfms is not None else ts)
        dm.append(np.concatenate(d, axis=1))
    return dm
