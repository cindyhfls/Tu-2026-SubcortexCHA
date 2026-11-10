import numpy as np
from scipy.spatial.distance import cdist

def argmin_with_nan_on_tie(d):
    """
    For each row of d:
      - return argmin index if the minimum is unique
      - return np.nan if there is a tie
    """
    mins = d.min(axis=1, keepdims=True)          # (nT, 1)
    is_min = (d == mins)                          # True where value == row-min
    counts = is_min.sum(axis=1)                   # number of minima per row

    # start with all NaN
    pred_labels = np.full(d.shape[0], np.nan)

    # fill in rows where there is exactly one minimum
    unique_rows = (counts == 1)
    pred_labels[unique_rows] = np.argmin(d[unique_rows], axis=1)

    return pred_labels

def classification_RDM(U,Uhat,size=None,npc=None,metric='correlation',reduce_func=np.mean):
    U, Uhat = U[:, :npc], Uhat[:, :npc]
    nt, nv = Uhat.shape
    if size is None:
        nc = nt
        Y, Yhat = U, Uhat
    else:
        nc = nt // size
        Y = U[: nc * size].reshape(nc, size, nv).reshape(nc, size * nv)
        Yhat = Uhat[: nc * size].reshape(nc, size, nv).reshape(nc, size * nv)
    d = cdist(Y , Yhat, metric)
    return d

def classification_accuracy(U,Uhat,size=None,npc=None,metric='correlation',reduce_func=np.mean):
    U, Uhat = U[:, :npc], Uhat[:, :npc]
    nt, nv = Uhat.shape
    if size is None:
        nc = nt
        Y, Yhat = U, Uhat
    else:
        nc = nt // size
        Y = U[: nc * size].reshape(nc, size, nv).reshape(nc, size * nv)
        Yhat = Uhat[: nc * size].reshape(nc, size, nv).reshape(nc, size * nv)
    d = cdist(Y , Yhat, metric)

    # multiclass
    pred_labels = argmin_with_nan_on_tie(d)
    valid = ~np.isnan(pred_labels)
    if np.sum(valid==0)>0:
        print(f"Tie occurence in {np.sum(valid==0):d} cases, marked as classification failed")
    matched = valid & (np.arange(nc) == pred_labels)
    acc_multiclass = matched
    distances = np.abs(pred_labels[valid] - np.arange(nc)[valid])

    # binary
    acc_binary = []
    for ii in np.arange(d.shape[0]):
        others = np.delete(np.arange(d.shape[0]), ii)
        acc_binary.append((d[ii,ii] < d[ii,others]).mean()) # N.B. in Feilong's tutorial he said ties should be 0.5 instead of 1 or 0 but I think ties suggest unable to classify
    acc_binary = np.array(acc_binary)

    if reduce_func is not None:
        acc_multiclass = reduce_func(acc_multiclass)
        acc_binary = reduce_func(acc_binary)
        distances = reduce_func(distances)

    return acc_multiclass, acc_binary, distances
