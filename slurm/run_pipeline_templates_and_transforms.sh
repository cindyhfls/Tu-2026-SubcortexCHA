#!/bin/bash
# Templates, subject transforms and data caches, as the command lines that were run.
#
#   bash slurm/run_pipeline_templates_and_transforms.sh <block>
#   block = templates | transforms | caches | hcp | supp | all
#
# Order matters: templates -> transforms -> caches. The ISC / classification scripts (slurm/run_isc.sh)
# need all three. All paths are relative to the repository root.
#
# Resources that were used on a SLURM cluster (1 job per command unless noted):
#   01_templates_and_transforms/template_*.py   8 CPUs, 40 GB, ~4 h   (--n-jobs 8)
#   transformation_to_template_raiders.py      1 CPU, 8 GB, ~3 h, ONE JOB PER SUBJECT (--sids $SID); each process
#                                         parallelises over ROIs/templates with joblib (up to 8 workers)
#   transformation_to_template_hcp.py   same, one job per HCP test subject
#   00_cache_inputs/cache_*.py   4 CPUs, 8-40 GB, ~1 h
#   supp/supp_transformation_to_self.py, supp/supp_transformation_to_other.py   8 CPUs, 32-64 GB; the latter is one job per reference subject
# Use one BLAS thread when a script parallelises with joblib (it oversubscribes the cores otherwise):
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PY=python        # an environment with neuroboros + hyperalignment (+ nilearn, templateflow for the plots)
TPL_SUB=01_templates_and_transforms/template_budapest_subcortical.py
TPL_CTX=01_templates_and_transforms/template_budapest_cortical.py
XFM=01_templates_and_transforms/transformation_to_template_raiders.py
T=transformation_to_template_targets_onavg-ico8
RAIDERS_SIDS=$(cat config/raiders_subj_not_in_budapest_template.txt)   # the 13 transform/test subjects
HCP_SIDS=$(cat config/HCP_movie_v2_complete.txt)                       # the 34 HCP transform/test subjects

# ---------------------------------------------------------------------------------------------
# 1. Templates: Budapest, 10 template subjects, runs 1-5, onavg-ico8 targets.
#    -> results/Budapest/<space>/templates_<atlas>/targets_onavg-ico8_*.npy
# ---------------------------------------------------------------------------------------------
block_templates() {
  $PY $TPL_SUB --atlas aseg --n-jobs 8          # per-nucleus aseg template (Procrustes / fixed-alpha ridge / ridgeCV, Figs 2-5, Supp 4)
  $PY $TPL_SUB --atlas aseg_wholebrain --n-jobs 8   # one concatenated aseg template (pooled alpha, Supp 18)
  for S in 1 2 3 4; do                          # one concatenated template per Tian scale (Fig 6, Supp 18)
    $PY $TPL_SUB --atlas TianS${S} --n-jobs 8
  done
  $PY $TPL_CTX                                  # cortical searchlight template (hyperaligned cortical targets)
}

# ---------------------------------------------------------------------------------------------
# 2. Transforms into the template, Raiders runs 1-2 (CLI --sids runs a subset; one job per subject
#    is the usual way to run this). -> results/Raiders/<space>/<extra_str>/$T/<sid>_*
# ---------------------------------------------------------------------------------------------
block_transforms() {
  # (sbatch: wrap the body of this loop as `$PY $XFM --sids $SID --load-json $CFG` in an array over $RAIDERS_SIDS)
  for CFG in \
      config_align_to_template_ridgeCV_raiders.json \
      config_align_to_template_procrustes_raiders.json \
      config_align_to_template_ridge_raiders.json \
      config_align_to_template_ridgeCV_raiders_aseg_wholebrain.json \
      config_align_to_template_ridgeCV_raiders_TianS1_wholebrain.json \
      config_align_to_template_ridgeCV_raiders_TianS2_wholebrain.json \
      config_align_to_template_ridgeCV_raiders_TianS3_wholebrain.json \
      config_align_to_template_ridgeCV_raiders_TianS4_wholebrain.json \
      config_align_to_template_ridgeCV_raiders_aseg_wholebrain_alpha_per_ROI.json \
      config_align_to_template_ridgeCV_raiders_TianS1_wholebrain_alpha_per_ROI.json \
      config_align_to_template_ridgeCV_raiders_TianS2_wholebrain_alpha_per_ROI.json \
      config_align_to_template_ridgeCV_raiders_TianS3_wholebrain_alpha_per_ROI.json \
      config_align_to_template_ridgeCV_raiders_TianS4_wholebrain_alpha_per_ROI.json \
      config_align_to_template_ridgeCV_raiders_cortical.json \
      config_align_to_template_procrustes_raiders_cortical.json \
      config_align_to_template_ridge_raiders_cortical.json ; do   # last: ridge alpha 1e3 = the hyperaligned cortical targets
    $PY $XFM --load-json config/$CFG --sids $RAIDERS_SIDS
  done
}

# ---------------------------------------------------------------------------------------------
# 3. Data caches (float32 arrays read by the ISC / classification scripts)
#    Test runs 3-4 (default) for the ISC; also --run 1 2 for the training-run ISC of Supp Fig 4.
# ---------------------------------------------------------------------------------------------
block_caches() {
  C3=00_cache_inputs/cache_subcortical_raiders.py
  C4=00_cache_inputs/cache_cortical_raiders.py
  for RUN in "3 4" "1 2"; do
    $PY $C3 --atlas aseg --run $RUN                                    # AA (anatomical alignment)
    $PY $C4 --run $RUN
    # per-nucleus aseg transforms: ridgeCV (CHA), Procrustes, ridge alpha 1e3
    $PY $C3 --atlas aseg --run $RUN --transformation-folder ridgeCV/$T/ --align-method ridgeCV
    $PY $C3 --atlas aseg --run $RUN --transformation-folder procrustes_new/$T/ --align-method procr
    $PY $C3 --atlas aseg --run $RUN --transformation-folder ridge_alpha1e3_new/$T/ --align-method ridge
    # hyperaligned cortex (ridge alpha 1e3) = targets of every subcortical condition
    $PY $C4 --run $RUN --transformation-folder ridge_alpha1e3/$T/ --align-method ridge
  done
  for S in 1 2 3 4; do   # Tian scales (Fig 6), pooled alpha
    $PY $C3 --atlas Tian_Subcortex --run 3 4 \
        --transformation-folder ridgeCV_TianS${S}_wholebrain/$T/ --align-method ridgeCV
  done
}

# ---------------------------------------------------------------------------------------------
# 4. HCP 7T movie replication (Fig 7): template from 144 subjects (v1 movie), transforms on runs 1-2 of
#    the 34 test subjects, test runs 3-4. Needs data/HCP_YA/prep_oneregression (00_cache_inputs/hcp_resample/).
# ---------------------------------------------------------------------------------------------
block_hcp() {
  $PY 01_templates_and_transforms/template_hcp_subcortical.py --n-jobs 8
  $PY 01_templates_and_transforms/template_hcp_cortical.py --n-jobs 8
  for CFG in \
      config_align_to_template_ridgeCV_hcp_TianS1_wholebrain.json \
      config_align_to_template_ridgeCV_hcp_TianS2_wholebrain.json \
      config_align_to_template_ridgeCV_hcp_TianS3_wholebrain.json \
      config_align_to_template_ridgeCV_hcp_TianS4_wholebrain.json \
      config_align_to_template_ridge_hcp_onavg-ico32.json ; do
    $PY 01_templates_and_transforms/transformation_to_template_hcp.py --load-json config/$CFG --sids $HCP_SIDS
  done
  # The cortical cache expects the folder ridge_alpha1e3_onavg-ico32 (default of cache_cortical_hcp.py); the last config
  # writes ridge_onavg-ico32, so rename it or pass --transformation-folder ridge_onavg-ico32/$T/ to cache_cortical_hcp.py.
  for S in 1 2 3 4; do $PY 00_cache_inputs/cache_subcortical_hcp.py --scale $S; done
  $PY 00_cache_inputs/cache_cortical_hcp.py
}

# ---------------------------------------------------------------------------------------------
# 5. Supp Fig 9 / Supp Note 4: cross-run self-transform null (matched alpha) and direct
#    subject-to-subject transforms. The null is fitted at the ridgeCV alpha of the real alignment.
# ---------------------------------------------------------------------------------------------
block_supp() {
  R=results/Raiders/mni-2mm_1step_linear_overlap/ridgeCV
  # encode side (run 1 -> 2, alpha of the to-template fit) and decode side (run 2 -> 1, from-template fit)
  $PY supp/supp_transformation_to_self.py --seeds aseg --run 1 2 --cross-run --align-method ridge \
      --match-alpha-from $R/transformation_to_template_targets_onavg-ico8
  $PY supp/supp_transformation_to_self.py --seeds aseg --run 2 1 --cross-run --align-method ridge \
      --match-alpha-from $R/transformation_from_template_targets_onavg-ico8
  # direct subject-to-subject ridgeCV maps (one job per reference subject), then their matched null
  for SID in $RAIDERS_SIDS; do
    $PY supp/supp_transformation_to_other.py --seeds aseg --target-sids $SID --align-method ridgeCV
  done
  $PY supp/supp_transformation_to_self.py --seeds aseg --run 1 2 --cross-run --align-method ridge \
      --match-alpha-from-pairs results/Raiders/mni-2mm_1step_linear_overlap/ridgeCV_to_other/transformation_to_other_targets_onavg-ico8
  # caches of the null: 00_cache_inputs/cache_subcortical_raiders.py --atlas aseg --transformation-folder
  #   cross_run1to2_shrinkage_matched/transformation_to_self_targets_onavg-ico8/ --align-method ridge
}

BLOCK=${1:-all}
case "$BLOCK" in
  templates)  block_templates ;;
  transforms) block_transforms ;;
  caches)     block_caches ;;
  hcp)        block_hcp ;;
  supp)       block_supp ;;
  all)        block_templates; block_transforms; block_caches; block_hcp; block_supp ;;
  *) echo "unknown block '$BLOCK' (templates|transforms|caches|hcp|supp|all)"; exit 1 ;;
esac
