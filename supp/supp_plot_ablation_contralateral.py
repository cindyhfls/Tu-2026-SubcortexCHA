"""Supp Fig 5: AA connectivity ISC under progressively stricter subcortical target sets.

Bars (Fisher mean over the 16 aseg nuclei, one dot and paired line per nucleus) for
  all targets                  every aseg voxel, incl. the seed's own nucleus (AA connectivity_targets-aseg,
                               from the fig2_3_loo.py leave-one-out summary)
  excl. same nucleus           supp_conn_subcortex_target_sets.py --targets-mode exclude_own
  contralateral                supp_conn_subcortex_target_sets.py --targets-mode contralateral
  contralateral, no homolog    supp_conn_subcortex_target_sets.py --targets-mode contralateral_no_homologue
plus a per-nucleus bar plot ordered by nucleus size.

Usage:  python supp_plot_ablation_contralateral.py
Input:  results/Raiders/mni-2mm_1step_linear_overlap/conn_subcortex_target_sets/results_<mode>.csv (supp_conn_subcortex_target_sets.py),
        results/Raiders/mni-2mm_1step_linear_overlap/AA/memmap_ISC_loo_aseg_raiders_summary (fig2_3_loo.py)
Output: results/figures/suppfig5A_conn_subcortex_target_sets_summary.png, suppfig5B_conn_subcortex_target_sets_by_nucleus.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import FIGURES, RESULTS, resolve  # noqa: E402

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

from utils.data_io_utils import return_aseg_labels
from utils.isc_utils import FisherAvg

parser = argparse.ArgumentParser()
parser.add_argument("--n-subjects", type=int, default=13)
parser.add_argument("--root", default=str(RESULTS / "Raiders" / "mni-2mm_1step_linear_overlap"))
args = parser.parse_args()

plt.rcParams.update({'font.size': 12, 'axes.titlesize': 12, 'axes.labelsize': 12,
                     'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12})
FIGURES.mkdir(exist_ok=True)
DPI = 300
N_ALL_TARGETS = 10152   # voxels of the aseg atlas
MODES = {'excl. same nucleus': 'exclude_own',
         'contralateral': 'contralateral',
         'contralateral, no homolog': 'contralateral_no_homologue'}
ROIS = return_aseg_labels(atlas_name='mni-2mm')
AA_SUMMARY = os.path.join(args.root, 'AA', 'memmap_ISC_loo_aseg_raiders_summary')
CSV_DIR = os.path.join(args.root, 'conn_subcortex_target_sets')

# Sequential purple ramp (the sets are nested), distinct from the condition colours elsewhere
PALETTE_TARGETS = {'all targets': '#3f007d', 'excl. same nucleus': '#6a51a3',
                   'contralateral': '#9e9ac8', 'contralateral, no homolog': '#dadaeb'}


def fisher_mean(r, axis=None):
    return float(FisherAvg(np.asarray(r, float), axis=axis, has_nan=True))


def load_aa_all_targets(metric='coarsefine'):
    """AA conn_subcortex ISC with every aseg voxel as a target, (nROI, N)."""
    out = np.full((len(ROIS), args.n_subjects), np.nan, np.float32)
    for d in sorted(glob.glob(os.path.join(AA_SUMMARY, 'sl*'))):
        m = re.match(r'sl(\d+)-(\d+)$', os.path.basename(d))
        if not m:
            continue
        a, b = int(m.group(1)), int(m.group(2))
        fn = os.path.join(d, 'connectivity_targets-aseg', f'isc_conn_{metric}.dat')
        if os.path.exists(fn) and os.path.getsize(fn) == (b - a) * args.n_subjects * 4:
            out[a:b] = np.memmap(fn, dtype=np.float32, mode='r').reshape(b - a, args.n_subjects)
    if not np.isfinite(out).any():
        raise FileNotFoundError(f'nothing readable under {AA_SUMMARY}')
    return out


def load_target_sets():
    full = load_aa_all_targets()
    rows = [{'roi': roi, 'target_set': 'all targets', 'isc': fisher_mean(full[i]),
             'nvox': np.nan, 'n_targets': N_ALL_TARGETS} for i, roi in enumerate(ROIS)]
    for label, mode in MODES.items():
        fn = os.path.join(CSV_DIR, f'results_{mode}.csv')
        if not os.path.exists(fn):
            print(f'  [MISSING] {fn} (run supp_conn_subcortex_target_sets.py --targets-mode {mode})')
            continue
        for _, r in pd.read_csv(fn).iterrows():
            rows.append({'roi': r.roi, 'target_set': label, 'isc': r.isc,
                         'nvox': int(r.nvox), 'n_targets': int(r.n_targets)})
    df = pd.DataFrame(rows)
    nvox = df.dropna(subset=['nvox']).groupby('roi').nvox.first()
    df['nvox'] = df.roi.map(nvox)
    return df


df = load_target_sets()
sets = ['all targets'] + [s for s in MODES if s in set(df.target_set)]
print(df.groupby('target_set', sort=False).isc.apply(fisher_mean).round(4))

# ---- summary: one bar per target set ----
fig, ax = plt.subplots(figsize=(6, 4.2))
sns.barplot(data=df, x='target_set', y='isc', order=sets, hue='target_set', palette=PALETTE_TARGETS,
            legend=False, estimator=fisher_mean, errorbar=None, capsize=0.1,
            edgecolor='0.4', linewidth=0.6, ax=ax)
sns.stripplot(data=df, x='target_set', y='isc', order=sets, marker='.', color='0.2', size=5,
              alpha=0.6, jitter=False, ax=ax)
wide = df.pivot(index='roi', columns='target_set', values='isc')
for i in range(len(sets) - 1):
    for _, r in wide.iterrows():
        ax.plot([i, i + 1], [r[sets[i]], r[sets[i + 1]]], color='0.3', lw=0.6, alpha=0.4)
ax.axhline(0, color='0.5', lw=0.8, ls='--')
ax.set_xlabel('')
ax.set_ylabel('r', rotation=0, va='center', labelpad=12)
ax.set_title('Subcortico-subcortical connectivity ISC', fontsize=12)
plt.setp(ax.get_xticklabels(), rotation=25, ha='right', fontsize=12)
fig.tight_layout()
fig.savefig(FIGURES / 'suppfig5A_conn_subcortex_target_sets_summary.png', dpi=DPI, bbox_inches='tight')
plt.close(fig)

# ---- per nucleus, ordered by size ----
order = df.dropna(subset=['nvox']).sort_values('nvox').roi.unique().tolist()
fig, ax = plt.subplots(figsize=(6, 4.2))
sns.barplot(data=df, x='roi', y='isc', hue='target_set', order=order, hue_order=sets,
            palette=PALETTE_TARGETS, estimator=fisher_mean, edgecolor='0.4', linewidth=0.6, ax=ax)
ax.set_xlabel('')
ax.set_ylabel('r', rotation=0, va='center', labelpad=12)
ax.set_title('Subcortico-subcortical connectivity ISC ordered by nuclei size')
plt.setp(ax.get_xticklabels(), rotation=40, ha='right')
h, l = ax.get_legend_handles_labels()
ax.get_legend().remove()
fig.legend(h, l, loc='lower center', ncol=2, frameon=False)   # outside the axes, below the bars
fig.subplots_adjust(bottom=0.55)
fig.savefig(FIGURES / 'suppfig5B_conn_subcortex_target_sets_by_nucleus.png', dpi=DPI, bbox_inches='tight')
plt.close(fig)
