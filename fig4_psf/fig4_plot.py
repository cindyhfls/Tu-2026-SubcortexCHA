"""Fig 4 / Supp Figs 13-15: intersubject PSF slopes, anatomical alignment (AA) vs CHA, with LME statistics.

For every nucleus and subject the Fisher-z residual ISC (fig4_psf.py) is regressed on
inter-voxel distance (2-8 mm); the slope (z per mm) quantifies how fast intersubject similarity decays.
Figures per metric: the l-thalamus example fits (Fig 4A/C/E), the nucleus-mean slope bars (Fig 4B/D/F) and the
fits of all nuclei (Supp Figs 13-15). Statistics: linear mixed model `slope ~ condition`, random intercepts for
subject and for nucleus (statsmodels MixedLM, REML).

Usage:  python fig4_plot.py [--metrics resp conn_cortex conn_contralateral]
Output: results/figures/{fig4A-F,suppfig13-15}_PSF_<metric>_*.png and results/stats/fig4_psf/PSF_<metric>_lme.txt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import FIGURES, RESULTS, STATS, resolve  # noqa: E402

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from utils.data_io_utils import load_isc, return_aseg_labels

plt.rcParams.update({"font.size": 12})

DISTANCES = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]  # mm
COLORS = {"AA": "#d62728", "CHA": "#c6dbef"}
# metric name -> memmap file of the CHA tree; the AA tree uses the same name without "CHA+"
METRICS = {
    "resp": "isc_resp_fine",
    "conn_cortex": "connectivity_targets-CHA+onavg-ico32/isc_conn_fine",
    "conn_contralateral": "connectivity_targets-CHA+aseg-contralateral/isc_conn_fine",
}
PSF_DIR = "memmap_ISC_loo_psf_euclidean_binned_aseg_raiders_summary"

parser = argparse.ArgumentParser()
parser.add_argument("--metrics", nargs="+", default=list(METRICS), choices=list(METRICS))
parser.add_argument("--n-subjects", type=int, default=13, help="Raiders test subjects (paper: 13)")
parser.add_argument("--cha-transformation-folder", default="ridgeCV/transformation_to_template_targets_onavg-ico8")
args = parser.parse_args()

OUT = FIGURES
# manuscript panels per metric: (l-thalamus example, all-nuclei slopes bar, per-nucleus supplementary figure)
PANEL = {"conn_cortex": ("fig4A", "fig4B", "suppfig13"), "conn_contralateral": ("fig4C", "fig4D", "suppfig14"),
         "resp": ("fig4E", "fig4F", "suppfig15")}
OUT.mkdir(parents=True, exist_ok=True)
STATS_OUT = STATS / "fig4_psf"
STATS_OUT.mkdir(parents=True, exist_ok=True)
ROOT_DIR = RESULTS / "Raiders" / "mni-2mm_1step_linear_overlap"
BASE = {"AA": ROOT_DIR / "AA" / PSF_DIR, "CHA": ROOT_DIR / args.cha_transformation_folder / PSF_DIR}
rois = return_aseg_labels()
x = np.asarray(DISTANCES)


def load_psf(tree, metric):
    """Fisher-z ISC, shape (nuclei, subjects, distances)."""
    name = metric if tree == "CHA" else metric.replace("CHA+", "")
    isc = [load_isc(str(BASE[tree]), f"{name}_{d}mm.dat", np.arange(args.n_subjects), args.n_subjects,
                    len(rois), pairwise=False) for d in DISTANCES]
    return np.atanh(np.stack(isc, axis=2))


def get_lines(Y):
    """Per-subject linear fit of ISC (subjects x distances) on distance; returns slopes, intercepts."""
    fits = np.array([np.polyfit(x, Y[i], 1) for i in range(Y.shape[0])])
    return fits[:, 0], fits[:, 1]


def to_long(slopes_by_cond):
    dfs = []
    for cond, S in slopes_by_cond.items():
        df = pd.DataFrame(S, index=rois).reset_index().melt(id_vars="index", var_name="subject", value_name="slope")
        df = df.rename(columns={"index": "ROI"})
        df["condition"] = cond
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def fig_slopes(df, conds, tag):
    """Fig 4B/D/F: nucleus-mean slope bars (one dot per nucleus, joined by lines) for two conditions."""
    palette = {c: COLORS[c] for c in conds}
    df = df[df.condition.isin(conds)]
    df_mean = df.groupby(["ROI", "condition"])["slope"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    sns.barplot(data=df_mean, x="condition", y="slope", hue="condition", order=conds, palette=palette,
                errorbar=None, legend=False, ax=ax)
    sns.stripplot(data=df_mean, x="condition", y="slope", order=conds, marker=".", color="0.2", size=5,
                  alpha=0.7, jitter=False, ax=ax)
    wide = df_mean.pivot(index="ROI", columns="condition", values="slope")
    for _, row in wide.iterrows():
        ax.plot([0, 1], [row[conds[0]], row[conds[1]]], color="0.3", lw=0.6, alpha=0.4)
    ax.axhline(0, color="0.5", lw=0.8, ls="--")
    ax.set_xlabel("")
    ax.set_ylabel("slope (z per mm)")
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.3f"))
    fig.tight_layout()
    fig.savefig(OUT / f"{PANEL[tag][1]}_PSF_{tag}_slopes_barplot_ROI_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fit_lme(df, conds):
    """slope ~ condition with random intercepts for subject (groups) and nucleus (variance component)."""
    d = df[df.condition.isin(conds)].copy()
    d["ROI"] = d["ROI"].astype("category")
    d["subject"] = d["subject"].astype("category")
    d["condition"] = pd.Categorical(d["condition"], categories=conds, ordered=True)
    md = smf.mixedlm("slope ~ condition", d, groups="subject", vc_formula={"ROI": "0 + C(ROI)"})
    return md.fit(reml=True, method="lbfgs")


for key in args.metrics:
    metric = METRICS[key]
    tag = key
    print(f"=== {key}: {metric}")
    isc = {"AA": load_psf("AA", metric), "CHA": load_psf("CHA", metric)}

    slopes, intercepts = {}, {}
    for cond, Z in isc.items():
        fits = [get_lines(Z[isl]) for isl in range(len(rois))]
        slopes[cond] = np.array([f[0] for f in fits])  # (nuclei, subjects)
        intercepts[cond] = np.array([f[1] for f in fits])
    df_long = to_long(slopes)

    conds = ["AA", "CHA"]
    fig_slopes(df_long, conds, tag)
    res = fit_lme(df_long, conds)
    lines = [f"##### {metric}: slope ~ condition ({' vs '.join(conds)}); random intercepts subject + nucleus\n"
             f"{res.summary()}\n"]
    thal = df_long[df_long.ROI == "l-thalamus"].groupby("condition")["slope"].mean()
    lines.append("l-thalamus mean slope (z/mm): " + ", ".join(f"{c}: {thal[c]:.3f}" for c in thal.index))
    (STATS_OUT / f"PSF_{tag}_lme.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # l-thalamus example: per-subject fitted lines (thin) and mean fit (bold)
    xfit = np.linspace(min(DISTANCES), max(DISTANCES), 100)
    isl = rois.index("l-thalamus")
    fig, ax = plt.subplots(figsize=(2.5, 2))
    for cond in ("AA", "CHA"):
        for s in range(args.n_subjects):
            ax.plot(xfit, slopes[cond][isl, s] * xfit + intercepts[cond][isl, s], color=COLORS[cond], lw=0.6, alpha=0.3)
        ax.plot(xfit, slopes[cond][isl].mean() * xfit + intercepts[cond][isl].mean(), color=COLORS[cond], lw=2, label=cond)
    ax.set_xlabel("Distance (mm)")
    ax.set_ylabel("ISC (z)")
    ax.set_title("l-thalamus")
    ax.set_xticks(DISTANCES)
    fig.tight_layout()
    fig.savefig(OUT / f"{PANEL[tag][0]}_PSF_{tag}_l-thalamus.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # all nuclei (Supp Figs 13-15): left nuclei in the left column, right nuclei in the right column
    fig, axes = plt.subplots(nrows=len(rois) // 2, ncols=2, figsize=(5, 12), sharex=True, sharey=True)
    axes = axes.ravel()
    panel = np.r_[np.arange(0, len(axes), 2), np.arange(1, len(axes), 2)]
    for isl in range(len(rois)):
        ax = axes[panel[isl]]
        for cond in ("AA", "CHA"):
            for s in range(args.n_subjects):
                ax.plot(xfit, slopes[cond][isl, s] * xfit + intercepts[cond][isl, s], color=COLORS[cond], lw=0.5, alpha=0.25)
            xf = np.array([min(DISTANCES), max(DISTANCES)])
            ax.plot(xf, slopes[cond][isl].mean() * xf + intercepts[cond][isl].mean(), color=COLORS[cond], linewidth=1)
        ax.set_ylabel(rois[isl][2:] if panel[isl] % 2 == 0 else "", fontsize=10)
        ax.set_xticks([0, 4, 8])
    fig.savefig(OUT / f"{PANEL[tag][2]}_PSF_{tag}_allROIs.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
