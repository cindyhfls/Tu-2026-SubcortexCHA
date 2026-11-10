"""Supp Fig 9B: hyperalignment to the template vs directly to a reference subject, with matched nulls.

Leave-one-out ISC (spatiotemporal, aseg nuclei, Raiders runs 3-4), one bar per condition, one dot per nucleus,
thin lines joining each nucleus across conditions. Two panels per figure:

  reference subject transformed      every subject, the reference included, goes through one transform
                                     into the template space (fig2_3_loo.py output):
                                     AA | cross-run self transform (null) | CHA (ridgeCV)
  reference subject NOT transformed  all other subjects are carried directly into the reference
                                     subject's own space, one subject-to-subject transform
                                     (supp_isc_to_other.py output):
                                     AA | cross-run self transform (null, alpha matched per pair) | CHA (ridgeCV)

One figure per row of the supplementary figure; the x ticks are off and the legend is a separate file.
  conn_cortex_ico32               connectivity to anatomically aligned onavg-ico32 cortex
  conn_subcortex_contralateral    connectivity to the contralateral subcortex
  resp                            response

AA applies no transform: it has ONE output location (fig2_3_loo.py's) and both panels read it, so its bar is
the same number in both panels by construction.

Usage:  python supp_plot_isc_common_vs_subject.py
Input:  results/Raiders/mni-2mm_1step_linear_overlap/<condition folder>/memmap_ISC_loo[_projecttosubj]_aseg_raiders_summary
Output: results/figures/suppfig9B_{conn_cortex_ico32,conn_subcortex_contralateral,resp,legend}.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import FIGURES, RESULTS  # noqa: E402

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

from utils.data_io_utils import get_aseg_sls, return_aseg_labels
from utils.isc_utils import FisherAvg

parser = argparse.ArgumentParser()
parser.add_argument("--root", default=str(RESULTS / "Raiders" / "mni-2mm_1step_linear_overlap"))
parser.add_argument("--n-subjects", type=int, default=13, help="Raiders test subjects")
args = parser.parse_args()

ROOT, N = args.root, args.n_subjects
ROIS = return_aseg_labels(atlas_name="mni-2mm")
SLS = get_aseg_sls()

# condition name -> transformation folder (under ROOT). Within one panel the roles are the same:
# AA, the matched null, and CHA -- through the template in the common panel, direct in the subject panel.
CONDITIONS = {
    "common": {
        "AA": "AA",
        "cross-run shrinkage": "cross_run1to2_shrinkage_matched/transformation_to_self_targets_onavg-ico8",
        "ridgeCV": "ridgeCV/transformation_to_template_targets_onavg-ico8",
    },
    "subject": {
        "AA": "AA",
        "cross-run shrinkage (subj-to-subj)": "ridge_cross_run_matchedalpha/transformation_to_self_targets_onavg-ico8",
        "ridgeCV (subj-to-subj)": "ridgeCV_to_other/transformation_to_other_targets_onavg-ico8",
    },
}
PALETTE = {"AA": "#d62728", "cross-run shrinkage": "#2ca02c", "ridgeCV": "#c6dbef",
           "cross-run shrinkage (subj-to-subj)": "#2ca02c", "ridgeCV (subj-to-subj)": "#c6dbef"}
# connectivity-target folder per domain. AA applies no transform, so its target carries no 'CHA+' marker.
TARGET_DIR = {
    "conn_cortex_ico32": "connectivity_targets-onavg-ico32",
    "conn_subcortex_contralateral": "connectivity_targets-CHA+aseg-contralateral",
    "resp": "",
}
PANEL_TITLES = {"common": "reference subject\ntransformed", "subject": "reference subject\nNOT transformed"}


def from_fig2_3_loo(cond, space):
    """Written by fig2_3_loo.py (no 'projecttosubj' in the path): AA always, and the whole common panel."""
    return cond == "AA" or space == "common"


def summary_dir(cond, space):
    return ("memmap_ISC_loo_aseg_raiders_summary" if from_fig2_3_loo(cond, space)
            else "memmap_ISC_loo_projecttosubj_aseg_raiders_summary")


def one(pattern):
    """Non-recursive glob that must match at most one path (a recursive glob once picked up run12/)."""
    hits = sorted(glob.glob(pattern))
    if len(hits) > 1:
        raise AssertionError(f"{len(hits)} matches for a pattern that must have one: {pattern}")
    return hits[0] if hits else None


def chunk_dirs(base):
    """[(dir, a, b)] directly under the summary dir; a region may be claimed by at most one chunk."""
    out, claimed = [], {}
    for d in sorted(glob.glob(os.path.join(base, "sl*"))):
        m = re.match(r"sl(\d+)-(\d+)$", os.path.basename(d))
        if not m:
            continue
        a, b = int(m.group(1)), int(m.group(2))
        if b <= a or not os.listdir(d):
            continue
        for r in range(a, b):
            if r in claimed:
                raise AssertionError(f"region {r} claimed by two chunk dirs:\n  {claimed[r]}\n  {d}")
            claimed[r] = d
        out.append((d, a, b))
    return out


def load_isc(cond, space, domain):
    """(nROI, N) leave-one-out ISC for one condition and domain; NaN where nothing is on disk."""
    path = CONDITIONS[space][cond]
    sub = TARGET_DIR[domain]
    if cond == "AA":
        sub = sub.replace("CHA+", "")
    if from_fig2_3_loo(cond, space):
        stem = "isc_resp_coarsefine.dat" if domain == "resp" else "isc_conn_coarsefine.dat"
    else:   # supp_isc_to_other.py names the file by domain; the contralateral restriction is in the folder name
        file_domain = "conn_subcortex" if domain == "conn_subcortex_contralateral" else domain
        stem = f"isc_{file_domain}_coarsefine.dat"
    base = os.path.join(ROOT, path, summary_dir(cond, space))
    out = np.full((len(ROIS), N), np.nan, np.float32)
    for d, a, b in chunk_dirs(base):
        fn = one(os.path.join(d, sub, stem))
        if fn and os.path.getsize(fn) == (b - a) * N * 4:
            out[a:b] = np.memmap(fn, dtype=np.float32, mode="r").reshape(b - a, N)
    if not np.isfinite(out).any():
        print(f"[MISSING] {space} | {cond} | {domain}: nothing readable under {base} (target folder '{sub}', file {stem})")
    if np.isfinite(out).any() and (np.nanmin(out) < -1 or np.nanmax(out) > 1):
        raise ValueError(f"ISC outside [-1, 1] for {space}/{cond}/{domain}: a job may still be writing")
    return out


def draw_panel(ax, space, domain):
    vals = {c: load_isc(c, space, domain) for c in CONDITIONS[space]}
    conds = [c for c in vals if np.isfinite(vals[c]).any()]
    if not conds:
        ax.axis("off")
        return
    # Fisher mean across subjects within each nucleus
    df = pd.DataFrame([{"unit": i, "condition": c, "value": FisherAvg(vals[c][i])}
                       for c in conds for i in range(len(ROIS))])
    sns.barplot(data=df, x="condition", y="value", order=conds, hue="condition", palette=PALETTE,
                legend=False, estimator=FisherAvg, errorbar=None, capsize=0.1, ax=ax)
    sns.stripplot(data=df, x="condition", y="value", order=conds, marker=".", color="0.2", size=5,
                  alpha=0.6, jitter=False, ax=ax)
    wide = df.pivot(index="unit", columns="condition", values="value")
    for i in range(len(conds) - 1):
        for _, r in wide.iterrows():
            ax.plot([i, i + 1], [r[conds[i]], r[conds[i + 1]]], color="0.3", lw=0.6, alpha=0.4)
    ax.axhline(0, color="0.5", lw=0.8, ls="--")
    ax.set_xlabel("")
    ax.set_xticks([])          # conditions are identified by the separate legend file
    ax.set_title(PANEL_TITLES[space], fontsize=12)
    ax.set_ylabel("r", rotation=0, va="center", labelpad=12)


plt.rcParams.update({"font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12})
FIGURES.mkdir(parents=True, exist_ok=True)
for domain in TARGET_DIR:
    fig, axes = plt.subplots(1, 2, figsize=(5, 2), sharey=True)
    for ax, space in zip(axes, ("common", "subject")):
        draw_panel(ax, space, domain)
    axes[0].yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))   # same tick width in every figure
    fig.tight_layout()
    out_fn = FIGURES / f"suppfig9B_{domain}.png"
    fig.savefig(out_fn, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_fn)

# legend, saved on its own: both panels draw the same three roles in the same colours
legend_labels = {"AA": "AA", "cross-run shrinkage": "cross-run self transform", "ridgeCV": "CHA"}
fig = plt.figure(figsize=(3, 1.2))
# sns.barplot saturates at 0.75, so desaturate the swatches the same way
fig.legend(handles=[Patch(facecolor=sns.desaturate(PALETTE[c], 0.75), label=lab) for c, lab in legend_labels.items()],
           loc="center", frameon=False, ncol=1)
fig.savefig(FIGURES / "suppfig9B_legend.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("wrote", FIGURES / "suppfig9B_legend.png")
