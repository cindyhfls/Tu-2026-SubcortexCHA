"""Supp Fig 18A / Supp Note 7: which ridge alpha did cross-validation choose, per nucleus and subject?

For the aseg atlas and the four Tian scales (alpha selected separately for every nucleus / parcel), stacked
bars show the fraction of all subject x nucleus combinations that chose each alpha of the ridgeCV grid
(1e1 ... 1e8). Colours are fixed to the position in the alpha grid.

On-disk layouts of the chosen alphas (read from the Raiders transformations into the template):
  aseg   one '<sid>_targets-onavg-ico8_<nucleus>_ridgeCV_bestalpha.txt' per subject and nucleus
         (config_align_to_template_ridgeCV_raiders.json, one ridgeCV search per nucleus)
  Tian   one '<sid>_targets-onavg-ico8_Tian_Subcortex_ridgeCV_bestalpha_per_sl.txt' per subject, one alpha per
         parcel (the *_alpha_per_ROI configs of transformation_to_template_raiders.py)

Usage:  python supp_plot_ridgeCV_alpha_distribution.py
Output: results/figures/suppfig18A_ridgeCV_alpha_by_atlas.png
        results/stats/ridgeCV_alpha/ridgeCV_alpha_by_region.csv   (one row per subject x nucleus)
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import FIGURES, RESULTS, STATS, resolve  # noqa: E402

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.data_io_utils import return_aseg_labels, return_Tian_labels

plt.rcParams.update({'font.size': 12})

# transformation tree per atlas (seed ROI name of the Tian parcel-wise file in SEED_ROI)
XFM_DIR = {'aseg': 'ridgeCV',
           'TianS1': 'ridgeCV_TianS1_wholebrain_alpha_per_ROI', 'TianS2': 'ridgeCV_TianS2_wholebrain_alpha_per_ROI',
           'TianS3': 'ridgeCV_TianS3_wholebrain_alpha_per_ROI', 'TianS4': 'ridgeCV_TianS4_wholebrain_alpha_per_ROI'}
ALPHA_GRID = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]   # the ridge_alpha grid of the configs
ALPHA_INK = '#1a1a1a'


def read_alphas(base, atlas, sids, targets):
    """Every subject's alpha per nucleus / parcel: DataFrame(subject, roi, best_alpha)."""
    records = []
    if atlas == 'aseg':
        rois = return_aseg_labels()
        for sid in sids:
            for roi in rois:
                fn = os.path.join(base, f"{sid}_targets-{targets}_{roi}_ridgeCV_bestalpha.txt")
                records.append({'subject': sid, 'roi': roi, 'best_alpha': float(open(fn).read().strip())})
    else:
        rois = return_Tian_labels(atlas)
        for sid in sids:
            fn = os.path.join(base, f"{sid}_targets-{targets}_Tian_Subcortex_ridgeCV_bestalpha_per_sl.txt")
            lines = open(fn).read().strip().split('\n')[1:]     # first line: the alpha a pooled search would have chosen
            alphas = [float(line.split('\t')[1]) for line in lines]
            if len(alphas) != len(rois):
                raise ValueError(f"{fn}: {len(alphas)} alphas for {len(rois)} {atlas} labels")
            records += [{'subject': sid, 'roi': roi, 'best_alpha': a} for roi, a in zip(rois, alphas)]
    df = pd.DataFrame(records)
    df['atlas'] = atlas
    return df


def label_ink(rgba):
    """Black or white for a label on top of `rgba`, whichever has more contrast."""
    r, g, b = rgba[:3]
    return ALPHA_INK if (0.299 * r + 0.587 * g + 0.114 * b) > 0.55 else 'white'


def alpha_colors(alphas):
    """Single-hue (grey) ramp over the full alpha grid: alpha is ordinal, so light to dark."""
    lo, hi = 0.12, 0.80   # not to pure black: the in-bar labels need to stay legible
    n = len(ALPHA_GRID) - 1
    return {a: plt.cm.Greys(lo + (hi - lo) * ALPHA_GRID.index(a) / n) for a in sorted(alphas)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', default=str(RESULTS / 'Raiders' / 'mni-2mm_1step_linear_overlap'))
    p.add_argument('--targets', default='onavg-ico8')
    p.add_argument('--sids-file', default='config/raiders_subj_not_in_budapest_template.txt')
    args = p.parse_args()

    root = resolve(args.root)
    sids = [s.strip() for s in open(resolve(args.sids_file)) if s.strip()]
    frames = []
    for atlas, tree in XFM_DIR.items():
        base = os.path.join(root, tree, f"transformation_to_template_targets_{args.targets}")
        df = read_alphas(base, atlas, sids, args.targets)
        frames.append(df)
        print(f"{atlas:<7} {df.roi.nunique():>3} regions x {df.subject.nunique()} subjects, "
              f"alphas={[f'{a:g}' for a in sorted(df.best_alpha.unique())]}")
    all_df = pd.concat(frames, ignore_index=True)

    os.makedirs(STATS / 'ridgeCV_alpha', exist_ok=True)
    csv = STATS / 'ridgeCV_alpha' / 'ridgeCV_alpha_by_region.csv'
    all_df.to_csv(csv, index=False)

    alphas = sorted(all_df.best_alpha.unique())
    colors = alpha_colors(alphas)
    order = list(XFM_DIR)
    fig, ax = plt.subplots(figsize=(max(4.2, 1.25 * len(order)), 4), constrained_layout=True)
    for i, atlas in enumerate(order):
        frac = all_df[all_df.atlas == atlas].best_alpha.value_counts(normalize=True)
        bottom = 0.0
        for a in alphas:
            f = frac.get(a, 0.0)
            if f == 0:
                continue
            ax.bar(i, f, bottom=bottom, width=0.68, color=colors[a], edgecolor='white', linewidth=1.5, zorder=2)
            if f > 0.06:
                ax.text(i, bottom + f / 2, f"{100 * f:.0f}%", ha='center', va='center',
                        fontsize=plt.rcParams['font.size'], color=label_ink(colors[a]), zorder=3)
            bottom += f
    ax.set_xticks(range(len(order)), order)
    ax.set_ylim(0, 1)
    ax.set_ylabel('fraction')
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[a]) for a in alphas]
    fig.legend(handles, [f"{a:g}" for a in alphas], title='ridgeCV alpha', loc='outside lower center',
               ncols=len(alphas), frameon=False, fontsize=plt.rcParams['font.size'])
    FIGURES.mkdir(parents=True, exist_ok=True)
    fn = FIGURES / 'suppfig18A_ridgeCV_alpha_by_atlas.png'
    fig.savefig(fn, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"-> {fn}\n-> {csv}")


if __name__ == '__main__':
    main()
