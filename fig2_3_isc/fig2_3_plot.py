"""ISC bar / box plots (AA vs CHA) and the linear mixed-effects statistics behind them.

Reads the leave-one-out ISC summaries written by fig2_3_loo.py / fig7_isc_loo.py and draws, for every metric, bars of the
Fisher-mean ISC per condition with one dot per nucleus (paired lines between adjacent conditions):

  tian     AA + CHA at Tian S1-S4 (ridgeCV, pooled alpha; also separate alpha per ROI)   Fig 6D-F, Supp Fig 18
           (the '<metric>_both_alpha_ISC_bar.png' panel pairs pooled vs separate alpha)
  aseg     AA vs CHA (ridgeCV, aseg nuclei): bars + per-nucleus subject boxplots        Figs 2, 3, Supp Figs 6, 7, 10-12
  methods  AA / Procrustes / ridge(alpha=1e3) / ridgeCV, runs 3-4 (test) and 1-2 (training)   Supp Fig 4
           plus the two-condition LMEs (lme_procrustes_vs_aa_<run>.txt, lme_ridgeCV_vs_ridge1e3_<run>.txt)
  wsc      between-subject (ISC) vs within-subject (WSC) connectivity, AA and CHA        Supp Fig 8
  hcp      HCP 7T MOVIE runs 3-4 (N=34), AA + CHA at Tian S1-S4                          Fig 7

Metrics: isc_resp_coarsefine (response), connectivity to contralateral subcortex, to hyperaligned
cortex (CHA+onavg-ico32) and to anatomically aligned cortex (onavg-ico32), plus the coarse / fine
splits for aseg (Fig 3). AA reads the untransformed target folder of each metric.

LME (statsmodels mixedlm, REML): Fisher-z ISC ~ condition, subject as the group and nuclei as a
variance component (random intercepts for subject and nucleus). One model per pair of bars; the
tian and aseg models are collected in results/stats/lme_results.txt.

Usage:  python fig2_3_plot.py                              # everything
        python fig2_3_plot.py --analyses aseg methods      # subset
        python fig2_3_plot.py --analyses tian --scales TianS1 TianS4 --schemes pooled_alpha
Output: results/figures/<panel>_*_ISC_{bar,box}.png (panel = fig2A, fig3D, suppfig6B, ...),
        results/stats/lme_results.txt,
        results/stats/lme_<tag>_<run12|run34>.txt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import FIGURES, RESULTS, STATS, resolve  # noqa: E402

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from utils.data_io_utils import return_aseg_labels, return_Tian_labels
from utils.isc_summary_utils import (aa_metric_path, fit_lme as _fit_lme, roi_long, roi_mean_frame,
                                      roi_means)
from utils.isc_utils import FisherAvg
from utils.plotting_utils import bar_panel, box_panel, paired_panel, ALPHA_PAIR_COLORS, AA_COLOR

XFM = "transformation_to_template_targets_onavg-ico8"
SCALES = ['TianS1', 'TianS2', 'TianS3', 'TianS4']
ALPHA_SCHEMES = {                      # label -> tree name pattern
    'pooled_alpha': "ridgeCV_{S}_wholebrain",
    'separate_alpha': "ridgeCV_{S}_wholebrain_alpha_per_ROI",
}
SCALE_COLORS = {'TianS1': "#c6dbef", 'TianS2': "#04d4f9", 'TianS3': "#0381f6", 'TianS4': "#041df8"}

# Tian ISC is always read at the TianS1 parcellation whatever the transform, so that the four
# scales stay comparable ROI for ROI. AA's connectivity_targets- folder has no CHA+ marker; the
# anatomical-cortex target is the one that does not move with the transform.
TIAN_METRICS = [
    "isc_resp_coarsefine",
    "connectivity_targets-CHA+Tian_Subcortex-contralateral/isc_conn_coarsefine",
    "connectivity_targets-CHA+onavg-ico32/isc_conn_coarsefine",
    "connectivity_targets-onavg-ico32/isc_conn_coarsefine",
]
ASEG_METRICS = [
    "isc_resp_coarsefine",
    "connectivity_targets-CHA+aseg-contralateral/isc_conn_coarsefine",
    # seed mean against the 8 opposite-hemisphere nucleus means (AA reads aseg-mean-contralateral)
    "connectivity_targets-CHA+aseg-mean-contralateral/isc_conn_coarse",
    "connectivity_targets-CHA+aseg-contralateral/isc_conn_fine",
    "connectivity_targets-CHA+onavg-ico32/isc_conn_coarsefine",
    "connectivity_targets-CHA+onavg-ico32/isc_conn_fine",
    "isc_resp_coarse",
    "isc_resp_fine",
    "connectivity_targets-CHA+onavg-ico8/isc_conn_coarse",
]
# metrics of the AA / Procrustes / ridge / ridgeCV LMEs (Supp Fig 4)
METHOD_LME_METRICS = [
    "connectivity_targets-CHA+onavg-ico32/isc_conn_coarsefine",
    "connectivity_targets-CHA+aseg-contralateral/isc_conn_coarsefine",
    "isc_resp_coarsefine",
]
HCP_METRICS = [m for m in TIAN_METRICS if m != "connectivity_targets-onavg-ico32/isc_conn_coarsefine"]  # Fig 7: hyperaligned cortex, contralateral subcortex, response

# Short names for the LME summary table
METRIC_LABEL = {
    "isc_resp_coarsefine": "resp",
    "connectivity_targets-CHA+Tian_Subcortex-contralateral/isc_conn_coarsefine": "conn_subcx_contra",
    "connectivity_targets-CHA+aseg-contralateral/isc_conn_coarsefine": "conn_subcx_contra",
    "connectivity_targets-CHA+aseg-mean-contralateral/isc_conn_coarse": "conn_subcx_mean_contra",
    "connectivity_targets-CHA+aseg-contralateral/isc_conn_fine": "conn_subcx_contra_fine",
    "connectivity_targets-CHA+onavg-ico32/isc_conn_coarsefine": "conn_ctx_hyper",
    "connectivity_targets-onavg-ico32/isc_conn_coarsefine": "conn_ctx_anat",
    "connectivity_targets-CHA+onavg-ico32/isc_conn_fine": "conn_ctx_hyper_fine",
    "isc_resp_coarse": "resp_coarse",
    "isc_resp_fine": "resp_fine",
    "connectivity_targets-CHA+onavg-ico8/isc_conn_coarse": "conn_ctx_ico8_hyper_coarse",
}

# --------------------------------------------------------------------------------------
# manuscript panel of every figure file (file names start with the panel, e.g. fig2A_..., suppfig6B_...);
# only panels of the manuscript are drawn
# --------------------------------------------------------------------------------------
_CTX_H = "connectivity_targets-CHA+onavg-ico32/isc_conn_coarsefine"          # hyperaligned cortex
_CTX_A = "connectivity_targets-onavg-ico32/isc_conn_coarsefine"              # anatomical cortex
_SUB_T = "connectivity_targets-CHA+Tian_Subcortex-contralateral/isc_conn_coarsefine"
_SUB_A = "connectivity_targets-CHA+aseg-contralateral/isc_conn_coarsefine"   # contralateral subcortex
_RESP = "isc_resp_coarsefine"
PANELS = {
    # (analysis, kind) -> {metric: panel}
    ('aseg', 'bar'): {_CTX_H: 'fig2A', _SUB_A: 'fig2B', _RESP: 'fig2C',
                      "connectivity_targets-CHA+onavg-ico8/isc_conn_coarse": 'fig3A',
                      "connectivity_targets-CHA+aseg-mean-contralateral/isc_conn_coarse": 'fig3B',
                      "isc_resp_coarse": 'fig3C',
                      "connectivity_targets-CHA+onavg-ico32/isc_conn_fine": 'fig3D',
                      "connectivity_targets-CHA+aseg-contralateral/isc_conn_fine": 'fig3E',
                      "isc_resp_fine": 'fig3F'},
    ('aseg', 'box'): {_CTX_H: 'suppfig6A', _SUB_A: 'suppfig6B', _RESP: 'suppfig7',
                      "connectivity_targets-CHA+onavg-ico8/isc_conn_coarse": 'suppfig10A',
                      "connectivity_targets-CHA+onavg-ico32/isc_conn_fine": 'suppfig10B',
                      "connectivity_targets-CHA+aseg-mean-contralateral/isc_conn_coarse": 'suppfig11A',
                      "connectivity_targets-CHA+aseg-contralateral/isc_conn_fine": 'suppfig11B',
                      "isc_resp_coarse": 'suppfig12A', "isc_resp_fine": 'suppfig12B'},
    ('tian', 'bar'): {_CTX_H: 'fig6D', _SUB_T: 'fig6E', _RESP: 'fig6F'},          # pooled alpha
    ('tian', 'both_alpha'): {_CTX_H: 'suppfig18B', _CTX_A: 'suppfig18C', _SUB_T: 'suppfig18D',
                             _RESP: 'suppfig18E'},
    ('hcp', 'bar'): {_CTX_H: 'fig7A', _SUB_T: 'fig7B', _RESP: 'fig7C'},
    ('methods', 'test'): {_CTX_H: 'suppfig4A', _SUB_A: 'suppfig4B', _RESP: 'suppfig4C'},     # runs 3-4
    ('methods', 'training'): {_CTX_H: 'suppfig4D', _SUB_A: 'suppfig4E', _RESP: 'suppfig4F'}, # runs 1-2
}


def panel_name(analysis, kind, metric, suffix):
    """'<panel>_<metric label>_<suffix>' for a panel of the manuscript, None if the manuscript has none."""
    panel = PANELS.get((analysis, kind), {}).get(metric)
    if panel is None:
        return None
    return f"{panel}_{METRIC_LABEL.get(metric, metric.replace('/', '-'))}_{suffix}"


LME_ROWS = []   # (header, reference, other, coef, p, converged) for the summary table


# --------------------------------------------------------------------------------------
# loading (shared loaders live in utils/isc_summary_utils.py)
# --------------------------------------------------------------------------------------
def hcp_roi_means(base, mpath, roi_names, n_subjects):
    """Fisher mean across subjects per parcel for the HCP 7T summary, or None if never computed."""
    fn = os.path.join(base, "sl0000-0016", mpath + ".dat")
    if not os.path.exists(fn) or os.path.getsize(fn) != len(roi_names) * n_subjects * 4:
        return None
    M = np.memmap(fn, dtype=np.float32, mode="r").reshape(len(roi_names), n_subjects)
    if not np.isfinite(M).any():
        return None                      # allocated but never computed
    if not np.isfinite(M).all():
        raise ValueError(f"partially non-finite: {fn}")
    return pd.Series([FisherAvg(M[i]) for i in range(len(roi_names))], index=roi_names)


def fit_lme(long_by_cond, keep, header, fh):
    """utils.isc_summary_utils.fit_lme, collecting the coefficients for the summary table."""
    _fit_lme(long_by_cond, keep, header, fh, rows=LME_ROWS)


def ylim_for(metric):
    return [0, 1] if "conn" in metric else [0, 0.3]


# --------------------------------------------------------------------------------------
# analyses
# --------------------------------------------------------------------------------------
def run_tian(args, lme_fh):
    """AA + CHA at Tian S1-S4: bars per alpha scheme, pooled-vs-separate panel, LMEs (pooled)."""
    rois = return_Tian_labels('TianS1')
    scale_cond = {S: f"CHA_S{S[-1]}" for S in args.scales}
    conditions = ['AA'] + list(scale_cond.values())
    palette = {'AA': AA_COLOR, **{scale_cond[S]: SCALE_COLORS[S] for S in args.scales}}
    n = args.n_subjects

    for metric in args.tian_metrics:
        aa_base = os.path.join(args.raiders_root, 'AA', args.tian_summary)
        aa_path = aa_metric_path(aa_base, metric)
        aa_long = roi_long(aa_base, aa_path, rois, n)
        aa = roi_means(aa_base, aa_path, rois, n)
        # subject-level frames for every (scheme, scale) cell, loaded once and reused
        cells_long, cells = {}, {}
        for scheme in args.schemes:
            for S in args.scales:
                base = os.path.join(args.raiders_root, ALPHA_SCHEMES[scheme].format(S=S), XFM,
                                    args.tian_summary)
                cells_long[(scheme, scale_cond[S])] = roi_long(base, metric, rois, n)
                cells[(scheme, scale_cond[S])] = roi_means(base, metric, rois, n)

        ylim = ylim_for(metric)
        for scheme in args.schemes:
            present = ['AA'] + [c for c in conditions[1:] if cells[(scheme, c)] is not None]
            if aa is None or len(present) == 1:
                print(f"skip {metric} [{scheme}]: nothing to plot")
                continue
            means = {c: (aa if c == 'AA' else cells[(scheme, c)]) for c in present}
            name = panel_name('tian', 'bar', metric, f"{scheme}_ISC_bar") if scheme == 'pooled_alpha' else None
            if name:   # Fig 6D-F: pooled alpha only
                bar_panel(roi_mean_frame(means, rois), present, palette, ylim, FIGURES / (name + ".png"))

            # LMEs for pooled alpha only: each adjacent pair (the pairs the paired lines join),
            # CHA_S1 vs CHA_S4 (end-to-end change across scales), and AA vs CHA_S4.
            if scheme != 'pooled_alpha':
                continue
            lbc = {c: (aa_long if c == 'AA' else cells_long[(scheme, c)]) for c in present}
            pairs = [[present[i], present[i + 1]] for i in range(len(present) - 1)]
            if 'CHA_S1' in present and 'CHA_S4' in present:
                pairs.append(['CHA_S1', 'CHA_S4'])
            if 'CHA_S4' in present:
                pairs.append(['AA', 'CHA_S4'])
            for pair in pairs:
                fit_lme(lbc, pair, f"{METRIC_LABEL.get(metric, metric)} [{scheme}]", lme_fh)

        # pooled vs separate alpha side by side, one tick per scale (Supp Fig 18)
        if len(args.schemes) == 2:
            sch = list(args.schemes)
            scales_present = [c for c in conditions[1:]
                              if all(cells[(s, c)] is not None for s in sch)]
            if aa is not None and scales_present and panel_name('tian', 'both_alpha', metric, 'x'):
                paired_panel(aa.values,
                             [(cells[(sch[0], c)].values, cells[(sch[1], c)].values)
                              for c in scales_present],
                             scales_present, ALPHA_PAIR_COLORS,
                             tuple(s.replace('_', ' ') for s in sch), ylim,
                             FIGURES / (panel_name('tian', 'both_alpha', metric, 'both_alpha_ISC_bar') + ".png"),
                             show_legend="conn" not in metric)


def aseg_dirs(args):
    root = args.raiders_root
    return {'AA': os.path.join(root, 'AA', args.aseg_summary),
            'CHA': os.path.join(root, 'ridgeCV', XFM, args.aseg_summary)}


def run_aseg(args, lme_fh):
    """AA vs CHA (ridgeCV, one alpha per nucleus): bars, per-nucleus boxplots and LMEs.
    The pooled-alpha ridgeCV_aseg_wholebrain tree is not used here."""
    rois = return_aseg_labels()
    dirs = aseg_dirs(args)
    n = args.n_subjects
    palette = {'AA': AA_COLOR, 'CHA': "#c6dbef"}
    for metric in args.aseg_metrics:
        long_by_cond, means = {}, {}
        for cond in ('AA', 'CHA'):
            mpath = aa_metric_path(dirs[cond], metric) if cond == 'AA' else metric
            long_by_cond[cond] = roi_long(dirs[cond], mpath, rois, n)
            if long_by_cond[cond] is None:
                print(f"skip aseg '{cond}' for {metric}: not computed")
                continue
            means[cond] = roi_means(dirs[cond], mpath, rois, n)
        if len(means) < 2:
            continue
        present = list(means)
        fit_lme(long_by_cond, present, f"aseg {METRIC_LABEL.get(metric, metric)}", lme_fh)
        bar_name = panel_name('aseg', 'bar', metric, 'ISC_bar')
        box_name = panel_name('aseg', 'box', metric, 'ISC_box')
        if bar_name:
            bar_panel(roi_mean_frame(means, rois), present, palette, ylim_for(metric),
                      FIGURES / (bar_name + ".png"), figsize=(2, 2), xtick_rotation=0)
        if not box_name:
            continue
        if "conn" in metric:
            # onavg-ico8 coarse has one subject's l-thalamus at -0.12 (AA)
            xlim = [-0.2, 1] if metric.endswith("onavg-ico8/isc_conn_coarse") else [0, 1]
        else:
            xlim = [-0.1, 0.4] if metric == "isc_resp_coarse" else [-0.05, 0.3]
        box_panel(pd.concat([long_by_cond[c].assign(condition=c) for c in present],
                            ignore_index=True), palette, xlim,
                  FIGURES / (box_name + ".png"))


def run_methods(args):
    """aseg AA / Procrustes / ridge (alpha=1e3) / ridgeCV (Supp Fig 4).

    Each set is drawn for runs 3-4 (summary dirs themselves) and for runs 1-2 (the run12/ folder
    each tree keeps beside them). LMEs: Procrustes vs AA and ridgeCV vs ridge(1e3), separately
    for each run set; hyperaligned cortical targets are used for both seed conditions."""
    rois = return_aseg_labels()
    n = args.n_subjects
    base_dirs = {
        'AA': aseg_dirs(args)['AA'],
        'Procrustes': os.path.join(args.raiders_root, 'procrustes_new', XFM, args.aseg_summary),
        'ridge (α=1e3)': os.path.join(args.raiders_root, 'ridge_alpha1e3_new', XFM,
                                           args.aseg_summary),
        'ridgeCV': aseg_dirs(args)['CHA'],
    }
    palette = {'AA': AA_COLOR, 'Procrustes': "#1f77b4",
               'ridge (α=1e3)': "#6baed6", 'ridgeCV': "#c6dbef"}

    def run_dirs(sub):
        """The same trees at another run: <tree>/<sub>/memmap_ISC_loo_* ('' = runs 3-4)."""
        return {c: (os.path.join(os.path.dirname(b), sub, os.path.basename(b)) if sub else b)
                for c, b in base_dirs.items()}

    def fit_pair_lme(run_sub, pair, output_tag):
        run_tag = run_sub or 'run34'
        label = 'Training' if run_sub else 'Test'
        runs = '1-2' if run_sub else '3-4'
        bases = run_dirs(run_sub)
        out_fn = STATS / f"lme_{output_tag}_{run_tag}.txt"
        with open(out_fn, "w", encoding="utf-8") as fh:
            fh.write(f"{label} runs {runs}: {pair[1]} minus {pair[0]}, separate two-condition "
                     "models.\nFisher-z ISC; subject-grouped ROI variance component; REML.\n"
                     "Hyperaligned cortical targets are used for BOTH seed conditions.\n\n")
            for metric in METHOD_LME_METRICS:
                long_by_cond = {}
                for cond in pair:
                    base = bases[cond]
                    # AA needs real hyperaligned cortical targets: no fallback for cortex
                    mpath = (aa_metric_path(base, metric)
                             if cond == 'AA' and 'CHA+aseg' in metric else metric)
                    values = roi_long(base, mpath, rois, n)
                    if values is None:
                        raise FileNotFoundError(f"Missing {cond} {run_tag} data: {base}/{mpath}")
                    long_by_cond[cond] = values
                    fh.write(f"Input {cond}: {base}/{mpath}\n")
                fit_lme(long_by_cond, pair, f"{label} run: {METRIC_LABEL[metric]}", fh)
        print(f"wrote {out_fn}")

    for run_sub in ('run12', ''):
        fit_pair_lme(run_sub, ['AA', 'Procrustes'], 'procrustes_vs_aa')
        fit_pair_lme(run_sub, ['ridge (α=1e3)', 'ridgeCV'], 'ridgeCV_vs_ridge1e3')

    for run_sub, metric in [(r, m) for r in ('', 'run12') for m in args.aseg_metrics]:
        name = panel_name('methods', 'training' if run_sub else 'test', metric, 'ISC_bar')
        if not name:
            continue
        means = {}
        for cond, base in run_dirs(run_sub).items():
            mpath = aa_metric_path(base, metric) if cond == 'AA' else metric
            v = roi_means(base, mpath, rois, n)
            if v is None:
                print(f"skip aseg methods {run_sub or 'run34'} '{cond}' for {metric}: not computed")
                continue
            means[cond] = v
        if len(means) < 2:
            continue
        # no x ticks -- the condition legend is added by hand -- so the figure can be short
        bar_panel(roi_mean_frame(means, rois), list(means), palette, ylim_for(metric),
                  FIGURES / (name + ".png"), figsize=(2.5, 2.3), xticks=False)


def run_wsc(args):
    """Between-subject (ISC) vs within-subject (WSC) connectivity, AA and CHA (Supp Fig 8).

    From supp_isc_wsc_splitruns.py on the held-out runs (run 3 vs run 4), so both
    quantities cross runs and differ only in whose data is paired:
      ISC = corr(subject i run 3, subject j run 4), Fisher-averaged over j != i
      WSC = corr(subject i run 3, subject i run 4)
    Subcortical targets are contralateral only (the seed's own nucleus is excluded)."""
    rois = return_aseg_labels()
    n = args.n_subjects
    dirs = {'AA': os.path.join(args.raiders_root, 'AA', args.split_summary),
            'CHA': os.path.join(args.raiders_root, 'ridgeCV', XFM, args.split_summary)}
    targets = {   # figure label -> (AA target folder, CHA target folder)
        'conn_ctx': ('connectivity_targets-onavg-ico32', 'connectivity_targets-CHA+onavg-ico32'),
        'conn_subcx_contra': ('connectivity_targets-aseg-contralateral',
                              'connectivity_targets-CHA+aseg-contralateral'),
    }
    isc_color = {'AA': AA_COLOR, 'CHA': "#c6dbef"}
    wsc_color = "#ad27d6"
    for label, by_cond in targets.items():
        for cond, tfolder in zip(('AA', 'CHA'), by_cond):
            means = {}
            for kind, suffix in (('ISC', 'acrosssubjectacrossruns'), ('WSC', 'withinsubject')):
                v = roi_means(dirs[cond], f"{tfolder}/isc_conn_coarsefine_{suffix}", rois, n)
                if v is None:
                    print(f"skip split-run {cond} {label} {kind}: not computed")
                    continue
                means[kind] = v
            if len(means) < 2:
                continue
            bar_panel(roi_mean_frame(means, rois), ['ISC', 'WSC'],
                      {'ISC': isc_color[cond], 'WSC': wsc_color}, [0, 1],
                      FIGURES / f"suppfig8_{label}_{cond}_ISC_vs_WSC_bar.png",
                      figsize=(2, 2), xtick_rotation=0)


def run_hcp(args):
    """HCP 7T MOVIE runs 3-4, seeds read at TianS1 (7T atlas: 16 TianS1 parcels, 13974 voxels).
    Cells that were allocated but never computed (all NaN) are skipped automatically."""
    rois = return_Tian_labels('TianS1', is_7T=True)
    root = os.path.join(args.hcp_root)
    dirs = {'AA': os.path.join(root, 'AA', args.hcp_summary)}
    dirs.update({f'CHA_S{s}': os.path.join(root, f'ridgeCV_TianS{s}_wholebrain', XFM,
                                           args.hcp_summary) for s in (1, 2, 3, 4)})
    palette = {'AA': AA_COLOR, **{f'CHA_S{s}': SCALE_COLORS[f'TianS{s}'] for s in (1, 2, 3, 4)}}
    for metric in HCP_METRICS:
        means = {}
        for cond, base in dirs.items():
            mpath = aa_metric_path(base, metric) if cond == 'AA' else metric
            v = hcp_roi_means(base, mpath, rois, args.hcp_n_subjects)
            if v is None:
                print(f"skip HCP '{cond}' for {metric}: not computed")
                continue
            means[cond] = v
        if len(means) < 2:
            continue
        bar_panel(roi_mean_frame(means, rois), list(means), palette, ylim_for(metric),
                  FIGURES / (panel_name('hcp', 'bar', metric, 'ISC_bar') + ".png"), figsize=(2.5, 2.3), xticks=False)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--analyses', nargs='+', default=['tian', 'aseg', 'methods', 'wsc', 'hcp'],
                   choices=['tian', 'aseg', 'methods', 'wsc', 'hcp'])
    p.add_argument('--scales', nargs='+', default=SCALES, choices=SCALES,
                   help="Tian scales for the CHA conditions (tian analysis)")
    p.add_argument('--schemes', nargs='+', default=list(ALPHA_SCHEMES), choices=list(ALPHA_SCHEMES),
                   help="ridgeCV alpha schemes (tian analysis); both are needed for the paired panel")
    p.add_argument('--tian-metrics', nargs='+', default=TIAN_METRICS)
    p.add_argument('--aseg-metrics', nargs='+', default=ASEG_METRICS)
    p.add_argument('--n-subjects', type=int, default=13, help="Raiders subjects (Budapest)")
    p.add_argument('--hcp-n-subjects', type=int, default=34)
    p.add_argument('--raiders-root', default=str(RESULTS / "Raiders" / "mni-2mm_1step_linear_overlap"))
    p.add_argument('--hcp-root', default=str(RESULTS / "HCP" / "mni-1pt6mm"))
    p.add_argument('--tian-summary', default="memmap_ISC_loo_TianS1_raiders_summary")
    p.add_argument('--aseg-summary', default="memmap_ISC_loo_aseg_raiders_summary")
    p.add_argument('--split-summary',
                   default="memmap_ISC_pairwiseavg_splitruns_aseg_raiders_summary")
    p.add_argument('--hcp-summary', default="memmap_ISC_loo_TianS1_MOVIE_summary")
    args = p.parse_args()

    FIGURES.mkdir(exist_ok=True)
    STATS.mkdir(exist_ok=True)
    lme_fh = None
    if set(args.analyses) & {'tian', 'aseg', 'methods'}:
        lme_fh = open(STATS / "lme_results.txt", "w", encoding="utf-8")  # uses delta and approx
        lme_fh.write("Linear mixed effects models for pairs of bars in the figures.\n"
                     "Model (statsmodels mixedlm, REML): zISC ~ condition, random intercept per\n"
                     "subject (groups) plus a variance component per nucleus. Fitted on Fisher-z.\n"
                     "A positive coefficient means the second condition is higher than the first.\n"
                     "The line under each table restates the condition term; its r values are the\n"
                     "inverse Fisher-z of Intercept and of Intercept + condition term.\n\n")
    if 'tian' in args.analyses:
        run_tian(args, lme_fh)
    if 'aseg' in args.analyses:
        run_aseg(args, lme_fh)
    if 'methods' in args.analyses:
        run_methods(args)   # its pair LMEs go to separate files but join the summary table below
    if 'wsc' in args.analyses:
        run_wsc(args)
    if 'hcp' in args.analyses:
        run_hcp(args)

    if lme_fh is not None:
        lme_fh.write("\n" + "#" * 78 + "\nSUMMARY: condition coefficient (Fisher-z) and p, per pair\n"
                     + "#" * 78 + "\n")
        lme_fh.write(f"{'comparison':<58}{'reference':<14}{'vs':<14}{'coef':>9}{'p':>12}  conv\n")
        for hdr, a, b, coef, pv, conv in LME_ROWS:
            lme_fh.write(f"{hdr:<42}{a:<15}{b:<15}{coef:>9.4f}{pv:>12.3g}  {conv}\n")
        lme_fh.close()
        print(f"wrote {STATS / 'lme_results.txt'}  ({len(LME_ROWS)} models)")


if __name__ == '__main__':
    main()
