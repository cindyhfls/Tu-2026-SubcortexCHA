# Tu-2026-SubcortexCHA

Code for the subcortical connectivity hyperalignment (CHA) manuscript: a common connectome space for the
subcortical nuclei, built from each voxel's connectivity to cortical targets (onavg-ico8 vertices) and used to
align fine-scale subcortical functional topography across individuals.

## Layout

```
00_cache_inputs/              cache_*.py: Raiders / HCP time series as float32 arrays (data/cache); hcp_resample/: HCP preprocessing
01_templates_and_transforms/  template_*.py: Budapest / HCP templates; transformation_to_template_*.py: each subject into the template
fig1_overview/ ... fig7_hcp/  the analysis and plotting scripts of Figs 1-7
supp/                         supplementary analyses and figures
utils/                        shared modules (data loading, ISC, hyperalignment helpers, plotting, paths)
config/                       subject lists and the JSON configs of the transformations
slurm/                        the command lines used on a SLURM cluster, in run order
data/                         preprocessed data and caches only (symlinks)
results/                      Raiders/, Budapest/, HCP/ (templates, transformations, ISC ...), figures/, stats/
```

Paths are resolved from the location of `utils/paths.py`, so scripts can be launched from any directory.

## Figures and the scripts that make them

| Figure | Computation | Plot |
|---|---|---|
| 1A, 6A | | `fig1_overview/fig1_plot_atlases.py` |
| 1B | | `fig1_overview/fig1_plot_example_connectivity.py` |
| 2, 3 | `fig2_3_isc/fig2_3_loo.py` (leave-one-out ISC, AA and CHA) | `fig2_3_isc/fig2_3_plot.py --analyses aseg` |
| 4 | `fig4_psf/fig4_calc_voxel_distances.py`, `fig4_psf.py` (`fig4_psf_withinsubj.py`: within-subject control) | `fig4_psf/fig4_plot.py` |
| 5 | `fig5_target_prediction/fig5_classify.py` | `fig5_target_prediction/fig5_plot.py` |
| 6B-C | | `fig6_tian_scales/fig6_plot_weights_by_distance.py` |
| 6D-F | `fig2_3_loo.py` with the Tian configs | `fig2_3_plot.py --analyses tian` |
| 7 | `fig7_hcp/fig7_isc_loo.py` (templates and transformations: `*_hcp_*.py` in `01_templates_and_transforms/`) | `fig2_3_plot.py --analyses hcp` |
| Supp 1-3 | | `supp/supp_plot_geometry_consistency.py` |
| Supp 4 | `fig2_3_loo.py` (Procrustes, ridge, ridgeCV; runs 1-2 and 3-4) | `fig2_3_plot.py --analyses methods` |
| Supp 5 | `supp/supp_conn_subcortex_target_sets.py` | `supp/supp_plot_ablation_contralateral.py` |
| Supp 6, 7, 10-12 | `fig2_3_loo.py` | `fig2_3_plot.py --analyses aseg` (boxplots) |
| Supp 8 | `supp/supp_isc_wsc_splitruns.py` | `fig2_3_plot.py --analyses wsc` |
| Supp 9B | `supp/supp_transformation_to_self.py`, `supp_transformation_to_other.py`, `supp_isc_to_other.py` | `supp/supp_plot_isc_common_vs_subject.py` |
| Supp 13-15 | `fig4_psf.py` | `fig4_psf/fig4_plot.py` |
| Supp 16, 17 | `fig5_classify.py` | `fig5_plot.py` |
| Supp 18A | | `supp/supp_plot_ridgeCV_alpha_distribution.py` |
| Supp 18B-E | `fig2_3_loo.py` with the Tian configs (pooled and per-nucleus alpha) | `fig2_3_plot.py --analyses tian` |

The schematics (Fig 1C, Supp 9A) were drawn by hand. Statistics (linear mixed-effects models) are written to
`results/stats/`.

## Order of the pipeline

1. `00_cache_inputs/` (caches) and, for HCP, `00_cache_inputs/hcp_resample/`
2. `01_templates_and_transforms/template_*.py` (Budapest template, 10 subjects), then
   `transformation_to_template_*.py --load-json config/<config>.json` (Raiders runs 1-2, 13 subjects)
3. `00_cache_inputs/cache_*.py --transformation-folder ...` (the transformed test runs 3-4)
4. ISC / classification / PSF scripts of the figure folders, then the plotting scripts

`slurm/run_pipeline_templates_and_transforms.sh` and `slurm/run_isc.sh` list the command lines for every condition.

## Environment

Python 3.13.5. The versions used:

| package | version |
|---|---|
| numpy / scipy / pandas | 2.2.5 / 1.16.3 / 2.2.3 |
| scikit-learn / joblib | 1.7.0 / 1.5.1 |
| statsmodels / seaborn / matplotlib | 0.14.6 / 0.13.2 / 3.10.1 |
| nilearn / nibabel / templateflow | 0.12.1 / 5.3.2 / 25.0.3 |
| [neuroboros](https://github.com/cindyhfls/neuroboros) | commit `27e4d5f` (main) |
| [hyperalignment](https://github.com/cindyhfls/hyperalignment) | commit `a5132e4` (dev) |
| [subcortex_visualization](https://github.com/cindyhfls/subcortex_visualization) (Fig 1A, 6A only) | commit `dd75618` (main) |

`hyperalignment` and `neuroboros` are installed from source. The pinned `hyperalignment` commit matters: `searchlight_template`
and `compute_template` changed signature in October 2025 (the temporary `weighted` argument was removed). The MNI T1
background images of the atlas plots are downloaded from TemplateFlow on first use.

## Data

* Budapest: OpenNeuro ds003017 (https://doi.org/10.18112/openneuro.ds003017.v1.0.2).
* Raiders: to be released on OpenNeuro.
* HCP 7T movie data: Human Connectome Project (https://www.humanconnectome.org); the preprocessing is in `00_cache_inputs/hcp_resample/`.

`data/` and `results/` hold symlinks to the storage locations used for this work.
