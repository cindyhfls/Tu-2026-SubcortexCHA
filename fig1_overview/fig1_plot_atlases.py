"""Subcortical atlas plots: aseg atlas (Fig 1A) and Tian S1-S4 atlases (Fig 6A).

For each atlas draws (1) the parcels in MNI slices on the T1 template (nilearn; template from
TemplateFlow), (2) the 2D outline plot of the atlas (`subcortex_visualization`), coloured by
nucleus family (Set3 palette: accumbens, caudate, thalamus, hippocampus, amygdala, putamen,
pallidum, ventral diencephalon), and for aseg (3) a legend-only figure.

Requires the `subcortex_visualization` package (fork with the aseg atlas:
https://github.com/cindyhfls/subcortex_visualization) and the aseg/Tian lookup tables in utils/support_files.

Usage:  python fig1_plot_atlases.py [--atlases aseg TianS1 TianS2 TianS3 TianS4]
Output: results/figures/fig1A_aseg_atlas_MNI_wBrain.png, fig1A_3Dplot_aseg.png, fig1A_aseg_legend.png,
        results/figures/fig6A_<TianS#>_atlas_MNI_wBrain.png, fig6A_3Dplot_<TianS#>.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import FIGURES, resolve  # noqa: E402

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from subcortex_visualization.plotting import plot_subcortical_data

from utils.data_io_utils import get_aseg_sls, get_Tian_sls, return_aseg_labels, return_Tian_labels
from utils.plotting_utils import map_subcortical_roi_to_img

parser = argparse.ArgumentParser()
parser.add_argument("--atlases", nargs="+", default=["aseg", "TianS1", "TianS2", "TianS3", "TianS4"],
                    choices=["aseg", "TianS1", "TianS2", "TianS3", "TianS4"])
args = parser.parse_args()

plt.rcParams.update({'font.size': 12})
FIGURES.mkdir(exist_ok=True)

# nucleus family -> colour index in Set3
map_to_color_order = {'accumbens': 0, 'caudate': 1, 'thalamus': 2, 'hippocampus': 3,
                      'amygdala': 4, 'putamen': 5, 'pallidum': 6, 'ventral-diencephalon': 7}
# Tian region acronym -> nucleus family
map_from_acronym = {
    'NAc': 'accumbens',
    'CAU': 'caudate', 'aCAU': 'caudate', 'pCAU': 'caudate',
    'THA': 'thalamus', 'aTHA': 'thalamus', 'pTHA': 'thalamus',
    'aHIP': 'hippocampus', 'pHIP': 'hippocampus', 'HIP': 'hippocampus',
    'AMY': 'amygdala', 'lAMY': 'amygdala', 'mAMY': 'amygdala',
    'aPUT': 'putamen', 'pPUT': 'putamen', 'PUT': 'putamen',
    'GP': 'pallidum', 'aGP': 'pallidum', 'pGP': 'pallidum',
}
set3 = plt.get_cmap("Set3")
set3_8 = ListedColormap(set3.colors[:8], name="Set3_8")   # aseg: 8 families
set3_7 = ListedColormap(set3.colors[:7], name="Set3_7")   # Tian has no ventral diencephalon


def legend_only_from_dict(mydict, cmap, ncols=4, figsize=(4, 3), fontsize=8, title=None, frameon=False):
    """Legend-only figure: one patch per entry of {label: colour index}."""
    labels = list(mydict.keys())
    handles = [Patch(facecolor=cmap(idx), edgecolor="black", label=lab)
               for lab, idx in mydict.items()]
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    fig.legend(handles=handles, labels=labels, ncols=ncols, loc="center", frameon=frameon,
               fontsize=fontsize, title=title)
    return fig


def outline_table(regions, values):
    """Left and right hemisphere rows for plot_subcortical_data."""
    L = pd.DataFrame({"region": regions, "value": values}).assign(Hemisphere="L")
    R = pd.DataFrame({"region": regions, "value": values}).assign(Hemisphere="R")
    return pd.concat([L, R], axis=0)


def plot_aseg():
    sls = get_aseg_sls()
    rois = return_aseg_labels('aseg')

    # legend (Fig 1A)
    legfig = legend_only_from_dict(map_to_color_order, set3_8, ncols=2, figsize=(4, 2),
                                   fontsize=11, title="")
    legfig.savefig(FIGURES / "fig1A_aseg_legend.png", dpi=300)
    plt.close(legfig)

    # 2D outline plot, left hemisphere. NOTE: values are family index + 1 drawn with the 7-colour
    # map, as in the source notebook (the legend above uses the 8-colour map).
    atlas_labels = list(map_to_color_order)
    values = [map_to_color_order[a] + 1 for a in atlas_labels]
    f = plot_subcortical_data(subcortex_data=outline_table(atlas_labels, values), atlas='aseg',
                              hemisphere='L', fill_title="", line_color='black', line_thickness=0.5,
                              cmap=set3_7, show_legend=False, show_figure=False)
    f.set_size_inches((2, 1))
    f.savefig(FIGURES / "fig1A_3Dplot_aseg.png", dpi=300)
    plt.close(f)

    # MNI slices on the T1 template
    data = []
    for ii, sl in enumerate(sls):
        family = rois[ii].split('-', 1)[1]   # 'l-ventral-diencephalon' -> 'ventral-diencephalon'
        data.append(np.full(len(sl), fill_value=map_to_color_order[family] + 0.5))
    f = plt.figure(figsize=(4, 2))
    f = map_subcortical_roi_to_img(data, 'aseg', rois, black_bg=True, cut_coords=(18, 12, -6), figure=f,
                                   annotate=False, interactive=False, colorbar=False,
                                   draw_cross=False, cmap=set3_8, threshold=0)
    f.savefig(FIGURES / "fig1A_aseg_atlas_MNI_wBrain.png", dpi=300)
    plt.close('all')


def plot_tian(atlas):
    sls = get_Tian_sls(atlas)
    rois = return_Tian_labels(atlas)

    # MNI slices on the T1 template
    data = []
    for ii, sl in enumerate(sls):
        family = map_from_acronym[rois[ii].split('-')[0]]
        data.append(np.full(len(sl), fill_value=map_to_color_order[family] + 0.5))
    f = plt.figure(figsize=(4, 2))
    f = map_subcortical_roi_to_img(data, atlas, rois, black_bg=True, cut_coords=(18, 12, -6), figure=f,
                                   annotate=False, interactive=False, colorbar=False,
                                   draw_cross=False, cmap=set3_7)
    f.savefig(FIGURES / f"fig6A_{atlas}_atlas_MNI_wBrain.png", dpi=300)
    plt.close('all')

    # 2D outline plot: a first call without data to read the atlas' region names off its legend
    sub_atlas = atlas.replace('Tian', 'Melbourne_')
    fig = plot_subcortical_data(atlas=sub_atlas, hemisphere='both', line_color='black', line_thickness=2,
                                fill_title="", cmap=set3_7, show_legend=True, show_figure=False)
    leg = next(ax.get_legend() for ax in fig.axes if ax.get_legend() is not None)
    atlas_labels = [t.get_text() for t in leg.texts]
    plt.close(fig)
    order = dict(map_to_color_order, GP=map_to_color_order['pallidum'])
    values = [order[a.split('_')[0]] for a in atlas_labels]
    f = plot_subcortical_data(subcortex_data=outline_table(atlas_labels, values), atlas=sub_atlas,
                              hemisphere='both', fill_title="", line_color='black', line_thickness=0.5,
                              cmap=set3_7, show_legend=False, show_figure=False)
    f.set_size_inches((3, 1.5))
    f.savefig(FIGURES / f"fig6A_3Dplot_{atlas}.png", dpi=300)
    plt.close(f)


for atlas in args.atlases:
    plot_aseg() if atlas == "aseg" else plot_tian(atlas)
