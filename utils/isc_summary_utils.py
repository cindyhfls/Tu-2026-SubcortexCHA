"""Loading of per-nucleus ISC summaries and the AA-vs-CHA linear mixed-effects model.

Shared by fig2_3_isc/fig2_3_plot.py and supp/supp_plot_denoising_null.py.
"""
import glob
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from utils.data_io_utils import load_isc
from utils.isc_utils import FisherAvg


def untransformed_metric(path):
    """AA applied no transform, so its connectivity_targets- folder carries no CHA+ marker.

    Same rule as utils.data_io_utils.untransformed_target_label, applied to the folder
    component of a metric path rather than to a bare label.
    """
    if '/' not in path:
        return path
    head, tail = path.split('/', 1)
    if head.startswith('connectivity_targets-'):
        head = 'connectivity_targets-' + head[len('connectivity_targets-'):].split('+')[-1]
    return f"{head}/{tail}"


def aa_metric_path(base, metric):
    """Which target folder AA is read from: the literal (CHA+) folder if AA was run against
    hyperaligned cortical targets (the like-for-like comparison against the CHA bars), else the
    untransformed label."""
    if glob.glob(os.path.join(base, 'sl*', metric + '.dat')):
        return metric
    return untransformed_metric(metric)


def roi_long(base, mpath, roi_names, n_subjects, allow_nan=False):
    """Subject x ROI ISC values (long format), or None if this cell was never computed.
    The LME needs the individual subject values; the Fisher averaging of the bars is done on
    top of this in roi_means(). Non-finite values raise unless allow_nan=True (the LME then drops
    those rows)."""
    if not glob.glob(os.path.join(base, 'sl*', mpath + '.dat')):
        return None
    M = load_isc(base, mpath + '.dat', np.arange(n_subjects), n_subjects, len(roi_names),
                 pairwise=False)
    if not allow_nan and not np.isfinite(M).all():
        raise ValueError(f"non-finite values in {base} / {mpath}")
    df = pd.DataFrame(M)
    df['ROI'] = roi_names
    return df.melt(id_vars='ROI', var_name='subject', value_name='ISC')


def roi_means(base, mpath, roi_names, n_subjects):
    """Fisher mean across subjects for each ROI, or None if this cell was never computed."""
    d = roi_long(base, mpath, roi_names, n_subjects)
    if d is None:
        return None
    out = d.groupby('ROI')['ISC'].apply(FisherAvg).reindex(roi_names)
    if out.isna().all():
        raise ValueError(
            f"all-NaN after reindexing {base}/{mpath} onto {len(roi_names)} ROI names -- the "
            f"labels do not match the data (e.g. Tian names against an aseg summary dir)")
    return out


def roi_mean_frame(means, rois):
    """Long frame (ROI, ISC, condition) from {condition: per-ROI Series}."""
    return pd.concat([pd.DataFrame({'ROI': rois, 'ISC': v.values, 'condition': c})
                      for c, v in means.items()], ignore_index=True)


def fit_lme(long_by_cond, keep, header, fh, rows=None):
    """Linear mixed-effects model for one pair of bars.

    zISC ~ condition with subject as the group and nuclei as a variance component (random
    intercepts for both), REML. Fitted on Fisher-z values, not raw r, for the same reason the
    bars are Fisher-averaged. A positive coefficient means `keep[1]` is higher than `keep[0]`.
    If `rows` is a list, the (header, ref, other, coef, p, converged) tuple is appended to it for a summary table.
    """
    a, b = keep
    if long_by_cond.get(a) is None or long_by_cond.get(b) is None:
        return
    df = pd.concat([long_by_cond[c].assign(condition=c) for c in keep], ignore_index=True)
    df = df.dropna(subset=["ISC"]).reset_index(drop=True)   # no-op unless roi_long(allow_nan=True)
    df["nuclei"] = df["ROI"].astype("category")
    df["subject"] = df["subject"].astype("category")
    df["condition"] = pd.Categorical(df["condition"], categories=keep, ordered=True)

    eps = 1e-6
    df["ISC_clip"] = df["ISC"].clip(-1 + eps, 1 - eps)
    df["zISC"] = np.arctanh(df["ISC_clip"])          # Fisher Z transform

    md = smf.mixedlm("zISC ~ condition", df, groups="subject",
                     vc_formula={"nuclei": "0 + C(nuclei)"})
    m = md.fit(reml=True, method="lbfgs")

    term = [t for t in m.params.index if t.startswith('condition')]
    coef = float(m.params[term[0]]) if term else float('nan')
    pval = float(m.pvalues[term[0]]) if term else float('nan')
    if rows is not None:
        rows.append((header, a, b, coef, pval, m.converged))

    fh.write("=" * 78 + "\n")
    fh.write(f"{header}\n")
    fh.write(f"{a} (reference)  vs  {b}\n")
    fh.write(f"{df.subject.nunique()} subjects x {df.nuclei.nunique()} ROIs x 2 conditions "
             f"= {len(df)} rows;  converged={m.converged}\n")
    fh.write("=" * 78 + "\n")
    fh.write(str(m.summary()) + "\n")
    if term:
        # r values are the inverse Fisher-z of the intercept (reference condition) and
        # intercept + condition term: model-estimated means across nuclei, back-transformed.
        lo, hi = (float(v) for v in m.conf_int().loc[term[0]])
        icpt = float(m.params['Intercept'])
        r_a, r_b = np.tanh(icpt), np.tanh(icpt + coef)
        p_txt = "p < 0.001" if pval < 0.001 else f"p = {pval:.3f}"
        fh.write(f"Δz = {coef:.3f}, 95% CI [{lo:.3f}, {hi:.3f}], {p_txt}, "
                 f"r ≈ {r_a:.3f} ({a}) to r ≈ {r_b:.3f} ({b}) across nuclei\n")
    fh.write("\n")
