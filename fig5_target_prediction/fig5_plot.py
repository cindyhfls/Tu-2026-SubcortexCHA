"""Fig 5 / Supp Figs 16-17: cross-subject prediction of cortical connectivity targets, AA vs CHA.

Panel `stats` (default; Fig 5D-E, Supp Fig 17): reads the outputs of
fig5_classify.py (AA and CHA trees), plots the nucleus-mean binary / multiclass accuracy bars of AA vs CHA
(Fig 5D-E; one dot per nucleus) and the across-subject vs within-subject (ceiling) bars after CHA (Supp Fig 17A-B), and
fits the linear mixed model
`accuracy ~ condition` with random intercepts for subject and nucleus (statsmodels MixedLM, REML) for
across-subject AA vs CHA and for across- vs within-subject (ceiling) within each alignment.
Panel `examples` (Fig 1B/5A, Supp Fig 16; loads the full caches, memory/time heavy): for one nucleus and one
held-out subject, the leave-one-subject-out mean connectivity map of one cortical target in the nucleus,
the subject's own map, and the histogram / cortical map of the correlation distance between the subject's
pattern for that target and the other subjects' patterns for all targets (AA vs CHA).

Usage:  python fig5_plot.py [--panels stats examples] [--targets CHA+onavg-ico8]
Output: results/figures/{fig5A,fig5B,fig5C,fig5D,fig5E,suppfig16,suppfig17A,suppfig17B}_*.png; results/stats/fig5_target_prediction/accuracy_lme.txt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import CACHE, FIGURES, RESULTS, STATS, resolve  # noqa: E402

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from utils.data_io_utils import return_aseg_labels

plt.rcParams.update({"font.size": 12})

COLORS = {"across subject (AA)": "#d62728", "across subject (CHA)": "#c6dbef", "within subject (AA)": "#ad27d6",
          "within subject (CHA)": "#ad27d6"}
SOURCE_FOLDER = "mni-2mm_1step_linear_overlap"
N_TARGETS = 1210  # onavg-ico8 cortical targets; chance level of the multiclass accuracy is 1/N_TARGETS
# (the analysis notebook drew the chance line at 1/1200)

parser = argparse.ArgumentParser()
parser.add_argument("--panels", nargs="+", default=["stats"], choices=["stats", "examples"])
parser.add_argument("--targets", default="CHA+onavg-ico8",
                    help="targets of the CHA condition in 16_; the AA condition uses the same without 'CHA+'")
parser.add_argument("--cha-transformation-folder", default="ridgeCV/transformation_to_template_targets_onavg-ico8")
parser.add_argument("--targets-transformation-folder", default="",
                    help="examples only, for CHA+ targets: tree with dms_lr_float32.npy")
parser.add_argument("--run", type=int, nargs="+", default=[3, 4])
parser.add_argument("--run-ends", type=int, nargs=2, default=[840, 1680])
# example settings (hand-picked for the figures, not optimised)
parser.add_argument("--example-nucleus", default="l-thalamus")
parser.add_argument("--example-target", type=int, default=158,
                    help="cortical target (onavg-ico8 vertex) of the Supp Fig 16 example; 158 = left medial prefrontal")
parser.add_argument("--fig5a-targets", type=int, nargs=2, default=[158, 303],
                    help="the two cortical targets of Fig 5A: 158 = left medial prefrontal cortex, "
                         "303 = left lateral occipital cortex")
parser.add_argument("--example-subject", type=int, default=0, help="held-out subject index")
args = parser.parse_args()

OUT = FIGURES
OUT.mkdir(parents=True, exist_ok=True)
STATS_OUT = STATS / "fig5_target_prediction"
STATS_OUT.mkdir(parents=True, exist_ok=True)
rois = return_aseg_labels()


def load_acc(tree_folder, targets):
    d = RESULTS / "Raiders" / SOURCE_FOLDER / tree_folder / f"connectivity_target-{targets}_prediction_splitruns_residualfine"
    return {n: np.load(d / f"{n}.npy") for n in ("acc_multiclass", "acc_binary", "acc_multiclass_baseline", "acc_binary_baseline")}


def make_long(accs):
    """DataFrame(nuclei, subject, condition, accuracy) from {condition: (subjects, nuclei) accuracy array}."""
    dfs = []
    for cond, A in accs.items():
        df = pd.DataFrame(A, columns=rois)
        df["subject"] = np.arange(A.shape[0])
        dfs.append(df.melt(id_vars="subject", var_name="nuclei", value_name="accuracy").assign(condition=cond))
    return pd.concat(dfs, ignore_index=True)


def fig_bars(df_long, conds, acc_type, fn, labels=None):
    palette = {c: COLORS[c] for c in conds}
    df_mean = df_long[df_long.condition.isin(conds)].groupby(["nuclei", "condition"])["accuracy"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    sns.barplot(data=df_mean, x="condition", y="accuracy", hue="condition", order=conds, palette=palette,
                errorbar=None, legend=False, ax=ax)
    sns.stripplot(data=df_mean, x="condition", y="accuracy", order=conds, marker=".", color="0.2", size=5,
                  alpha=0.7, jitter=False, ax=ax)
    wide = df_mean.pivot(index="nuclei", columns="condition", values="accuracy")
    for _, row in wide.iterrows():
        ax.plot([0, 1], [row[conds[0]], row[conds[1]]], color="0.3", lw=0.6, alpha=0.4)
    ax.set_xlabel("")
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(labels or [c[c.index("(") + 1:-1] if "(" in c else c for c in conds])  # "across subject (AA)" -> "AA"
    if acc_type == "binary":
        ax.set_ylim([0.5, 1])
    else:
        ax.axhline(1 / N_TARGETS, color="0.5", lw=0.8, ls="--")
    ax.set_ylabel("accuracy")
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
    fig.tight_layout()
    fig.savefig(OUT / fn, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return df_mean


def fit_lme(df_long, conds):
    """accuracy ~ condition, random intercepts for subject (groups) and nucleus (variance component)."""
    d = df_long[df_long.condition.isin(conds)].copy()
    d["nuclei"] = d["nuclei"].astype("category")
    d["subject"] = d["subject"].astype("category")
    d["condition"] = pd.Categorical(d["condition"], categories=conds, ordered=True)
    md = smf.mixedlm("accuracy ~ condition", d, groups="subject", vc_formula={"nuclei": "0 + C(nuclei)"})
    return md.fit(reml=True, method="lbfgs")


def run_stats():
    # AA predicts the anatomically aligned targets, CHA the hyperaligned ones (as in the example panels)
    acc = {"AA": load_acc("AA", args.targets.replace("CHA+", "")),
           "CHA": load_acc(args.cha_transformation_folder, args.targets)}
    lines = []
    for acc_type in ("multiclass", "binary"):
        accs = {"across subject (AA)": acc["AA"][f"acc_{acc_type}"],
                "across subject (CHA)": acc["CHA"][f"acc_{acc_type}"],
                "within subject (AA)": acc["AA"][f"acc_{acc_type}_baseline"],
                "within subject (CHA)": acc["CHA"][f"acc_{acc_type}_baseline"]}
        df_long = make_long(accs)
        main = ["across subject (AA)", "across subject (CHA)"]
        df_mean = fig_bars(df_long, main, acc_type, f"fig5{'D' if acc_type == 'binary' else 'E'}_connectivity_target_prediction_{acc_type}_residualfine_barplot_ROI_summary.png")
        # across-subject accuracy vs the within-subject ceiling, after CHA (Supp Fig 17)
        fig_bars(df_long, ["across subject (CHA)", "within subject (CHA)"], acc_type,
                 f"suppfig17{'A' if acc_type == 'binary' else 'B'}_connectivity_target_prediction_{acc_type}_across_vs_within_CHA.png",
                 labels=["across", "within"])
        thal = df_mean[df_mean.nuclei == "l-thalamus"].set_index("condition")["accuracy"]
        lines.append(f"##### {acc_type}: l-thalamus " + ", ".join(f"{c}: {thal[c]:.3f}" for c in main))
        lines.append(f"{acc_type}: mean over nuclei and subjects " + ", ".join(
            f"{c}: {df_long[df_long.condition == c].accuracy.mean():.3f}" for c in accs))
        for conds in (main, ["across subject (AA)", "within subject (AA)"], ["across subject (CHA)", "within subject (CHA)"]):
            res = fit_lme(df_long, conds)
            lines.append(f"##### {acc_type}: accuracy ~ condition ({' vs '.join(conds)}); random intercepts subject + nucleus\n"
                         f"{res.summary()}\n")
    (STATS_OUT / "accuracy_lme.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def run_examples():
    import neuroboros as nb
    from utils.data_io_utils import get_aseg_sls, apply_dm_transform
    from utils.plotting_utils import map_subcortical_roi_to_img
    from utils.hyperalignment_utils import calculate_connectivity
    from utils.isc_utils import parse_data_coarse_fine
    from utils.classification_utils import classification_RDM

    cache_dir = CACHE / "Raiders_raiders" / ("run" + "".join(map(str, args.run)))
    run_ends = args.run_ends
    isl = rois.index(args.example_nucleus)
    sl = get_aseg_sls()[isl]
    target_id, isubj = args.example_target, args.example_subject

    def get_dms(folder):
        if folder:
            return np.load(os.path.join(cache_dir, SOURCE_FOLDER, folder, "dms_aseg_Subcortex_float32.npy"), mmap_mode="r")
        return np.load(cache_dir / "ts_aseg_Subcortex_float32.npy", mmap_mode="r")

    def get_targets(targets):
        if targets.startswith("CHA+"):
            assert args.targets_transformation_folder, "CHA+ targets need --targets-transformation-folder"
            ts = np.load(os.path.join(cache_dir, f"onavg-ico32_1step_pial_overlap/{args.targets_transformation_folder}",
                                      "dms_lr_float32.npy"), mmap_mode="r")
        else:
            ts = np.concatenate([np.load(cache_dir / "ts_l_float32.npy"), np.load(cache_dir / "ts_r_float32.npy")],
                                axis=2).astype(np.float32, copy=False)
        return apply_dm_transform(ts, nb.mapping("lr", "onavg-ico32", "onavg-ico8", mask=True))

    dms = {"AA": get_dms(""), "CHA": get_dms(args.cha_transformation_folder)}
    targets_ts = {"AA": get_targets(args.targets.replace("CHA+", "")), "CHA": get_targets(args.targets)}
    nS = dms["AA"].shape[0]

    cons = {}
    for cond in ("AA", "CHA"):
        _, X_fine = parse_data_coarse_fine(dms[cond][:, :, sl])
        c0 = [calculate_connectivity(targets_ts[cond][i, :run_ends[0]], X_fine[i, :run_ends[0]]) for i in range(nS)]
        c1 = [calculate_connectivity(targets_ts[cond][i, run_ends[0]:run_ends[1]], X_fine[i, run_ends[0]:run_ends[1]]) for i in range(nS)]
        cons[cond] = (np.stack(c0), np.stack(c1))  # (N, nTargets, nVoxels)
    avg_others = {c: np.mean(np.delete(cons[c][1], isubj, axis=0), axis=0) for c in cons}

    # where each example target is on the cortex (1 at the target vertex, grey on white)
    def plot_target_location(t, fn):
        show = np.full(targets_ts["AA"].shape[2], np.nan)
        show[t] = 1
        nb.plot(show, cmap="gray_r", vmax=1, vmin=0, colorbar=False, fn=str(fn))

    for t in args.fig5a_targets:
        plot_target_location(t, OUT / f"fig5A_target-{t}_location.png")
    plot_target_location(target_id, OUT / f"suppfig16_target-{target_id}_location.png")

    # Fig 5A: connectivity of the nucleus' voxels to each of the two targets, in the example subject's own
    # data (run 3) and as the average of all OTHER subjects (run 4), for AA and for CHA: 4 panels per target
    # (cut coordinates are hand-picked for l-thalamus)
    for t in args.fig5a_targets:
        for cond in ("AA", "CHA"):
            for name, data in (("subject_run3", cons[cond][0][isubj, t]), ("others_mean_run4", avg_others[cond][t])):
                fig = plt.figure(figsize=(3, 1))
                map_subcortical_roi_to_img(data, "aseg", rois[isl], figure=fig, annotate=False,
                                           cut_coords=(-12, -20, 7), vmin=-3, vmax=3, bg_img=None, black_bg=True,
                                           interactive=False, colorbar=False, draw_cross=False)
                fig.savefig(OUT / f"fig5A_target-{t}_{cond}_{name}.png", dpi=300, bbox_inches="tight")
                plt.close(fig)

    # correlation distance of the subject's pattern (all targets) to the others' pattern of every target
    RDM = {c: classification_RDM(cons[c][0][isubj], avg_others[c]) for c in cons}

    # Fig 5B (AA) and 5C (CHA): distances to the same target (diagonal) vs to all other targets (off-diagonal)
    for panel, cond in (("fig5B", "AA"), ("fig5C", "CHA")):
        D = RDM[cond]
        n = D.shape[0]
        offdiag = D.ravel()[:-1].reshape(n - 1, n + 1)[:, 1:].ravel()
        fig = plt.figure(figsize=(3, 3))
        plt.hist(offdiag, density=True, alpha=0.5, bins=50, label="different targets")
        plt.hist(np.diag(D), density=True, alpha=0.5, bins=50, label="same target")
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
        plt.xlabel("Correlational distance")
        plt.ylabel("density")
        plt.xlim([0, 2])
        fig.savefig(OUT / f"{panel}_{cond}_correlational_distance_hist.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    fig = plt.figure(figsize=(4, 3))
    plt.hist(RDM["AA"][target_id], density=True, alpha=0.5, label="AA", color=COLORS["across subject (AA)"])
    plt.hist(RDM["CHA"][target_id], density=True, alpha=0.5, label="CHA", color=COLORS["across subject (CHA)"])
    plt.xlim([0, 2])
    plt.ylabel("Density")
    plt.legend()
    fig.savefig(OUT / f"suppfig16_connectivitytargets-{target_id}_prediction_hist.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    for cond in ("AA", "CHA"):
        nb.plot(RDM[cond][target_id], cmap="hot_r", vmax=2, vmin=0, labelsize=20, colorbar=True,
                fn=str(OUT / f"suppfig16_connectivitytargets-{target_id}_prediction_{cond}_brainplot.png"))


if "stats" in args.panels:
    run_stats()
if "examples" in args.panels:
    run_examples()
