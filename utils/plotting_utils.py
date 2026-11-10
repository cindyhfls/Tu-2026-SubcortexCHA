'''
Helper function to call nilearn to plot on subcortical structures
'''
import nilearn as nli
from nilearn import image, plotting
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SURFACES_DIR = os.path.join(HERE, "surfaces")

def get_T1_template(space="MNI152NLin2009cAsym", resolution=2):
    """Return the path of the T1w template, downloaded from TemplateFlow on first use."""
    from templateflow import api as tflow
    return str(tflow.get(space, resolution=resolution, suffix="T1w", desc=None, extension="nii.gz"))

def map_subcortical_roi_to_img(data, subcortical_atlas, rois, bg_img = 'T1_MNI2009cAsym',interactive=False, **kwargs):
   """
   Take the ROI data and map it back to a NIFTI image for visualization.
   """
   lut = pd.read_csv(os.path.join(HERE,"support_files",f"atlas-{subcortical_atlas}_space-MNI152NLin2009cAsym_res-02.lut.tsv"),sep="\t")
   img = image.load_img(get_T1_template("MNI152NLin2009cAsym"))
   if bg_img == 'T1_MNI152NLin6Asym':
      raise ValueError('The lookup table is for MNINLin2009cAsym so MNI152NLin6Asym is not supported')
   elif bg_img is not None:
      bg_img = img

   templatedata = img.get_fdata()
   stat_map = np.zeros_like(templatedata, dtype=np.float64)

   if isinstance(data,dict):
      l = []
      for roi in rois:
         l.append(data[roi])
      data = l
   elif not isinstance(rois,list):
      rois = [rois]
      data = [data]

   if len(data)==1 and len(rois)!=1:
      assert len(data[0])==len(lut),"if single data vector provided, it must match the full table length, i.e. all voxels in this specified atlas"
      stat_map = data
   else:
      assert len(data)==len(rois),"please provide data and rois in lists of the same size"
      for roi,d in zip(rois,data):
         stat_map[lut[lut.label==roi].i,lut[lut.label==roi].j,lut[lut.label==roi].k] = d

   stat_map_img = image.new_img_like(img, stat_map)
   if interactive:
      view = plotting.view_img(stat_map_img,bg_img=bg_img,**kwargs)
   else:
      view = plotting.plot_stat_map(stat_map_img,bg_img=bg_img,**kwargs)
   return view


def plot_box_with_strip(df_curr, curr_ax, palette_curr, x="ISC", y="ROI",
                        cond="condition", orient="h"):
    """Box plot with the individual observations overlaid as a strip."""
    import seaborn as sns
    sns.boxplot(
        data=df_curr, x=x, y=y, hue=cond, palette=palette_curr, orient=orient,
        fliersize=0,  # outlier markers; the strip already shows every point
        ax=curr_ax,
    )
    sns.stripplot(
        data=df_curr, x=x, y=y, hue=cond, dodge=True, palette='dark:.3',
        size=1.5, alpha=0.8, orient=orient, ax=curr_ax, legend=False,
    )


# ---------------------------------------------------------------------------------------
# ISC bar / box panels shared by fig2_3_isc/fig2_3_plot.py and supp/.
# Everything is saved at dpi=300 with bbox_inches='tight' and font size 12. The tight crop
# makes the SAVED image differ from figsize in either direction: it trims empty margins (the
# bar plots come out smaller) but grows to include anything drawn outside the axes (the box
# plots' legend sits below the figure, so they come out larger).
#
#   panel                               figsize      xticks   saved image
#   Tian AA + S1-S4 bars (bar_panel)    (3, 3)       45 deg   2.88 x 2.82 in
#   aseg AA vs CHA bars (bar_panel)     (2, 2)       0 deg    1.84 x 1.81 in
#   aseg AA vs CHA box (box_panel)      (8, 3.6)     --       8.11 x 3.96 in
# ---------------------------------------------------------------------------------------
AA_COLOR = "#d62728"                            # red
ALPHA_PAIR_COLORS = ("#c6dbef", "#fdbf6f")      # light blue pooled, light orange separate


def plot_box_with_lines(df_curr, curr_ax, palette_curr, x="ISC", y="ROI", sample="subject",
                        cond="condition", orient="h"):
    """Horizontal dodged boxes per ROI for exactly two conditions, with one line per sample
    (subject) joining its two conditions."""
    import seaborn as sns
    if orient != "h":
        raise NotImplementedError("This version assumes orient='h' (horizontal boxes).")

    sns.boxplot(data=df_curr, x=x, y=y, hue=cond, palette=palette_curr, orient=orient,
                fliersize=0, ax=curr_ax)

    hue_order = list(pd.unique(df_curr[cond].dropna()))
    if len(hue_order) != 2:
        raise ValueError(f"Expected exactly 2 conditions in '{cond}', got {hue_order}")

    roi_labels = [t.get_text() for t in curr_ax.get_yticklabels()]
    roi_ticks = np.asarray(curr_ax.get_yticks())

    # y-centre of each dodged box, from the drawn box patches, assigned to the nearest ROI tick
    roi_to_box_ys = {roi: [] for roi in roi_labels}
    for patch in curr_ax.patches:
        y_center = float(np.mean(patch.get_path().vertices[:, 1]))
        roi_to_box_ys[roi_labels[int(np.argmin(np.abs(roi_ticks - y_center)))]].append(y_center)

    # per ROI keep the two boxes closest to the tick (the two hue-dodged boxes)
    roi_to_y_pair = {}
    for roi, ys in roi_to_box_ys.items():
        if len(ys) < 2:
            continue
        tick = roi_ticks[roi_labels.index(roi)]
        ys = sorted(sorted(ys, key=lambda yy: abs(yy - tick))[:2])
        roi_to_y_pair[roi] = (ys[0], ys[1])

    wide = (df_curr.groupby([y, sample, cond], as_index=False)[x].mean()
            .pivot(index=[y, sample], columns=cond, values=x)
            .reindex(columns=hue_order).dropna())

    line_kwargs = dict(marker='.', markersize=6, linestyle='-', color="0.15", linewidth=0.4,
                       alpha=0.3, zorder=10)
    for (roi, _s), row in wide.iterrows():
        if roi not in roi_to_y_pair:
            continue
        y0, y1 = roi_to_y_pair[roi]
        curr_ax.plot([float(row.iloc[0]), float(row.iloc[1])], [y0, y1], **line_kwargs)
    return curr_ax


def bar_panel(df_roi_mean, present, palette, ylim, out_fn, figsize=(3, 3),
              xtick_rotation=45, xtick_ha='center', xticks=True, ylabel_rotation=0, bar_width=0.8):
    """Bar per condition (Fisher mean across ROIs), a dot per ROI, and paired lines between
    adjacent conditions. df_roi_mean has columns ROI, ISC, condition."""
    import seaborn as sns
    from utils.isc_utils import FisherAvg
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(figsize=figsize)
    # bar_width is the fraction of each condition's slot a bar fills (seaborn default 0.8)
    sns.barplot(data=df_roi_mean, x="condition", y="ISC", order=present, palette=palette,
                estimator=FisherAvg, errorbar=None, capsize=0.1, width=bar_width, ax=ax)
    sns.stripplot(data=df_roi_mean, x="condition", y="ISC", order=present, marker=".",
                  color="0.2", size=5, alpha=0.7, jitter=False, ax=ax)
    wide = df_roi_mean.pivot(index="ROI", columns="condition", values="ISC")
    for ii in range(len(present) - 1):
        for _, row in wide.iterrows():
            ax.plot([ii, ii + 1], [row[present[ii]], row[present[ii + 1]]],
                    color="0.3", lw=0.6, alpha=0.4)
    ax.axhline(0, color="0.5", lw=0.8, ls="--")
    ax.set_xlabel("")
    # ha='right' anchors a rotated label's END at its tick; 'center' keeps the default
    plt.xticks(rotation=xtick_rotation, ha=xtick_ha,
               rotation_mode='anchor' if xtick_ha != 'center' else 'default')
    if not xticks:
        ax.set_xticks([])   # no tick labels: the condition legend is added to the figure by hand
    if ylabel_rotation == 0:
        ax.set_ylabel("r", rotation=0, va="center", labelpad=10)
    else:
        ax.set_ylabel("r", rotation=ylabel_rotation)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    ax.set_ylim(ylim)
    plt.tight_layout()
    fig.savefig(out_fn, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {out_fn}  ({len(present)} conditions)")


def paired_panel(aa_vals, pair_vals, ticklabels, pair_colors, legend_labels, ylim, out_fn,
                 figsize=(4, 3), show_legend=True):
    """AA once on the left, then two bars sharing each subsequent tick (e.g. pooled vs separate
    ridge alpha for each Tian scale; AA has no alpha so it stays unpaired).

    aa_vals: per-ROI AA values; pair_vals: list of (left, right) per-ROI arrays, one per tick.
    """
    from utils.isc_utils import FisherAvg
    W, GAP = 0.38, 0.22
    xs, heights, colors, per_roi = [0.0], [FisherAvg(aa_vals)], [AA_COLOR], [aa_vals]
    centres = [0.0]
    for i, (left, right) in enumerate(pair_vals):
        c = 1.1 + i * 1.15
        centres.append(c)
        for j, v in enumerate((left, right)):
            xs.append(c + (j * 2 - 1) * GAP)
            heights.append(FisherAvg(v))
            colors.append(pair_colors[j])
            per_roi.append(v)

    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(xs, heights, width=W, color=colors, zorder=1)   # bars lowest, lines and dots over them
    per_roi = np.array(per_roi)
    for k in range(len(xs) - 1):
        for r in range(per_roi.shape[1]):
            ax.plot([xs[k], xs[k + 1]], [per_roi[k, r], per_roi[k + 1, r]],
                    color="0.3", lw=0.6, alpha=0.4, zorder=2)
    for k, x in enumerate(xs):
        ax.plot([x] * per_roi.shape[1], per_roi[k], marker='.', ls='none',
                color="0.2", ms=5, alpha=0.7, zorder=3)
    ax.axhline(0, color="0.5", lw=0.8, ls="--")
    ax.set_xticks(centres, labels=['AA'] + list(ticklabels))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_xlabel("")
    ax.set_ylabel("r", rotation=0, va="center", labelpad=10)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    ax.set_ylim(ylim)
    if show_legend:
        ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in pair_colors],
                  labels=list(legend_labels), loc='upper left', frameon=False,
                  fontsize=plt.rcParams['font.size'] * 0.75)
    plt.tight_layout()
    fig.savefig(out_fn, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {out_fn}  (paired)")


def box_panel(df_long, palette, xlim, out_fn, figsize=(8, 3.6), legend_y=-0.08):
    """Per-subject values as horizontal boxes, left | right hemisphere, one line per subject
    joining its two conditions. For aseg ROI names, whose hemisphere is an 'l-'/'r-' PREFIX.
    legend_y is how far below the axes the legend sits."""
    plt.rcParams.update({'font.size': 12})
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    halves = ((axes[0], df_long[df_long.ROI.str.startswith('l-')], "Left hemisphere"),
              (axes[1], df_long[df_long.ROI.str.startswith('r-')], "Right hemisphere"))
    for ax, d, title in halves:
        plot_box_with_lines(d, ax, palette, x="ISC", y="ROI", sample="subject",
                            cond="condition", orient="h")
        ax.set_title(title)
        ax.set_xlim(xlim)
        ax.set_xlabel("r")
    handles, labels = axes[1].get_legend_handles_labels()
    for ax in axes:
        if ax.legend_ is not None:
            ax.legend_.remove()
    # read the row names off the drawn axis, so the labels follow seaborn's category order
    axes[0].set_yticks(axes[0].get_yticks(),
                       labels=[t.get_text()[2:] for t in axes[0].get_yticklabels()])
    axes[0].set_ylabel("")
    axes[1].set_yticks([])
    axes[1].set_ylabel("")
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, legend_y),
               ncol=len(labels), frameon=False)
    fig.savefig(out_fn, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {out_fn}  (box)")
