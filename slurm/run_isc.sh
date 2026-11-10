#!/bin/bash
# Intersubject-correlation (ISC) conditions of the manuscript, as the command lines that were run.
#
#   bash slurm/run_isc.sh <block>        block = aa | cha | supp4 | tian | splitruns | hcp | supp9b | all
#   sbatch slurm/run_isc.sh cha          (any block can be submitted; one job runs the whole block)
#
# Prerequisites: the transforms (01_templates_and_transforms/02_*), the data caches
# (00_cache_inputs/cache_*.py; run the two Raiders cache scripts with `--run 1 2` for run12), and the
# transformed-cache files (dms_*.npy) of every transformation folder used below. All paths are relative
# to the repository root; outputs land in results/<dataset>/<space>_<resample>/<transformation folder or AA>/.
# fig2_3_loo.py, supp_isc_wsc_splitruns.py and fig7_isc_loo.py skip rows that are already computed, so a block can safely be re-run.
#
#SBATCH --job-name=isc
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# one BLAS thread per process (avoids oversubscribing the allocated cores)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

PY=python                       # the environment with neuroboros / hyperalignment installed
ISC=fig2_3_isc/fig2_3_loo.py
SPLIT=supp/supp_isc_wsc_splitruns.py
HCP=fig7_hcp/fig7_isc_loo.py
TO_OTHER=supp/supp_isc_to_other.py

T=transformation_to_template_targets_onavg-ico8
ASEG_CHA="ridgeCV/$T/"                  # aseg CHA (ridgeCV, alpha per nucleus), Figs 2-5
PROCR="procrustes_new/$T/"              # Supp Fig 4
RIDGE="ridge_alpha1e3_new/$T/"          # Supp Fig 4 (fixed alpha = 1e3)
CTX="ridge_alpha1e3/$T/"                # hyperaligned CORTICAL targets (ridge, alpha = 1e3), same for all conditions
MET_ALL="resp_coarse resp_coarsefine resp_fine conn_coarse conn_coarsefine conn_fine"
MET_CONN="conn_coarse conn_coarsefine conn_fine"
MET_TIAN="resp_coarsefine conn_coarsefine"
CONTRA="--targets-hemisphere contralateral"

# ---------------------------------------------------------------------------------------------
# AA = anatomical alignment (empty --transformation-folder). Fig 2/3 reference condition.
# ---------------------------------------------------------------------------------------------
block_aa() {
  # connectivity to hyperaligned cortex (ico32); response ISC comes from the same call
  $PY $ISC --atlas aseg --source-space mni-2mm --metrics $MET_ALL \
      --targets CHA+onavg-ico32 --targets-transformation-folder "$CTX"
  # connectivity to contralateral subcortical voxels (untransformed atlas)
  $PY $ISC --atlas aseg --source-space mni-2mm --metrics $MET_CONN \
      --targets aseg $CONTRA
  # hyperaligned cortex at ico8 (resolution of the Fig 5 target-prediction analysis)
  $PY $ISC --atlas aseg --source-space mni-2mm --metrics $MET_CONN \
      --targets CHA+onavg-ico8 --targets-transformation-folder "$CTX"
}

# ---------------------------------------------------------------------------------------------
# CHA with aseg seeds (ridgeCV transform), test runs 3-4. Fig 2/3.
# ---------------------------------------------------------------------------------------------
block_cha() {
  $PY $ISC --atlas aseg --source-space mni-2mm --metrics $MET_ALL \
      --transformation-folder "$ASEG_CHA" \
      --targets CHA+onavg-ico32 --targets-transformation-folder "$CTX"
  # the target is the seed atlas itself, so it went through the same transform
  $PY $ISC --atlas aseg --source-space mni-2mm --metrics $MET_CONN \
      --transformation-folder "$ASEG_CHA" \
      --targets CHA+aseg --targets-transformation-folder "$ASEG_CHA" $CONTRA
}

# ---------------------------------------------------------------------------------------------
# Supp Fig 4: Procrustes vs ridge vs ridgeCV. 'training' ISC = runs 1-2 (--out-subdir run12),
# 'test' ISC = runs 3-4. The AA reference is run for both run sets as well.
# ---------------------------------------------------------------------------------------------
block_supp4() {
  for RUNS in "3 4:" "1 2:run12"; do
    RUN=${RUNS%%:*}; SUB=${RUNS##*:}
    RUNARGS=(--run $RUN); [ -n "$SUB" ] && RUNARGS+=(--out-subdir "$SUB")
    # AA
    $PY $ISC "${RUNARGS[@]}" --atlas aseg --source-space mni-2mm --metrics $MET_CONN \
        --targets aseg $CONTRA
    $PY $ISC "${RUNARGS[@]}" --atlas aseg --source-space mni-2mm --metrics $MET_ALL \
        --targets CHA+onavg-ico32 --targets-transformation-folder "$CTX"
    for XFM in "$PROCR" "$RIDGE" "$ASEG_CHA"; do
      $PY $ISC "${RUNARGS[@]}" --atlas aseg --source-space mni-2mm --metrics $MET_ALL \
          --transformation-folder "$XFM" \
          --targets CHA+onavg-ico32 --targets-transformation-folder "$CTX"
      $PY $ISC "${RUNARGS[@]}" --atlas aseg --source-space mni-2mm --metrics $MET_CONN \
          --transformation-folder "$XFM" \
          --targets CHA+aseg --targets-transformation-folder "$XFM" $CONTRA
    done
  done
}

# ---------------------------------------------------------------------------------------------
# Fig 6 / Supp Fig 18: Tian S1-S4 transforms (pooled alpha = ridgeCV_TianS<s>_wholebrain, per-nucleus
# alpha = ..._alpha_per_ROI). Seeds are always evaluated at the TianS1 parcellation (16 regions) so the
# four scales are comparable row for row; only the transform changes.
# ---------------------------------------------------------------------------------------------
block_tian() {
  # AA reference
  $PY $ISC --atlas TianS1 --source-space mni-2mm --metrics $MET_ALL \
      --targets CHA+onavg-ico32 --targets-transformation-folder "$CTX"
  $PY $ISC --atlas TianS1 --source-space mni-2mm --metrics $MET_TIAN \
      --targets Tian_Subcortex $CONTRA
  for SUFFIX in "" "_alpha_per_ROI"; do           # pooled alpha / per-nucleus alpha
    for S in S1 S2 S3 S4; do
      XFM="ridgeCV_Tian${S}_wholebrain${SUFFIX}/$T/"
      # hyperaligned cortical targets (same ridge alpha=1e3 cortex for every scale)
      $PY $ISC --atlas TianS1 --source-space mni-2mm --metrics $MET_TIAN \
          --transformation-folder "$XFM" \
          --targets CHA+onavg-ico32 --targets-transformation-folder "$CTX"
      # contralateral subcortical targets (transformed with the seed's transform)
      $PY $ISC --atlas TianS1 --source-space mni-2mm --metrics $MET_TIAN \
          --transformation-folder "$XFM" \
          --targets CHA+Tian_Subcortex --targets-transformation-folder "$XFM" $CONTRA
    done
  done
}

# ---------------------------------------------------------------------------------------------
# Supp Fig 8: within-subject (WSC) vs between-subject (ISC) connectivity across run 3 / run 4
# (single-run pairwise correlations, supp_isc_wsc_splitruns.py), AA and CHA, cortical and contralateral subcortical targets.
# ---------------------------------------------------------------------------------------------
block_splitruns() {
  # AA (anatomical cortex; untransformed subcortex)
  $PY $SPLIT --atlas aseg --source-space mni-2mm --targets onavg-ico32
  $PY $SPLIT --atlas aseg --source-space mni-2mm --targets aseg $CONTRA
  # CHA
  $PY $SPLIT --atlas aseg --source-space mni-2mm --transformation-folder "$ASEG_CHA" \
      --targets CHA+onavg-ico32 --targets-transformation-folder "$CTX"
  $PY $SPLIT --atlas aseg --source-space mni-2mm --transformation-folder "$ASEG_CHA" \
      --targets CHA+aseg --targets-transformation-folder "$ASEG_CHA" $CONTRA
}

# ---------------------------------------------------------------------------------------------
# Fig 7: HCP 7T movie, runs 3-4, 34 test subjects, Tian transforms S1-S4, evaluated at TianS1.
# ---------------------------------------------------------------------------------------------
block_hcp() {
  H_T=transformation_to_template_targets_onavg-ico8
  H_CTX="ridge_alpha1e3_onavg-ico32/$H_T"     # hyperaligned cortex (ridge, alpha = 1e3)
  COMMON=(--atlas TianS1 --source-space mni-1pt6mm --metrics $MET_CONN)
  # AA: contralateral subcortex, and hyperaligned cortex
  $PY $HCP "${COMMON[@]}" --targets Tian_Subcortex $CONTRA
  $PY $HCP "${COMMON[@]}" --targets CHA+onavg-ico32 --targets-transformation-folder "$H_CTX"
  for S in 1 2 3 4; do
    XFM="ridgeCV_TianS${S}_wholebrain/$H_T"
    $PY $HCP "${COMMON[@]}" --transformation-folder "$XFM" \
        --targets CHA+Tian_Subcortex --targets-transformation-folder "$XFM" $CONTRA
    # cortical targets for the transformed conditions were the anatomically aligned vertices
    $PY $HCP "${COMMON[@]}" --transformation-folder "$XFM" --targets onavg-ico32
  done
}

# ---------------------------------------------------------------------------------------------
# Supp Fig 9B: reference subject NOT transformed. Subject j is carried directly into reference subject
# i's space (supp_isc_to_other.py), for real direct hyperalignment (supp_transformation_to_other.py transforms),
# the matched cross-run null (supp_transformation_to_self.py --match-alpha-from-pairs; alpha read from the
# real supp_transformation_to_other.py fits) and anatomical alignment (AA).
# Prerequisites: supp_transformation_to_other.py and supp_transformation_to_self.py (--cross-run
# --match-alpha-from-pairs) outputs. Run once with --sl-start 0 --sl-stop 0 first to fill the
# projection cache before chunked/parallel runs.
# ---------------------------------------------------------------------------------------------
block_supp9b() {
  # reference subject transformed (common space): CHA (ridgeCV) and the cross-run self-transform null
  # (supp_transformation_to_self.py --cross-run, alpha matched), both with the leave-one-out ISC of fig2_3_loo.py.
  # The panels use ANATOMICALLY aligned cortical targets and contralateral subcortical targets transformed with
  # the same transform; AA comes from block_aa.
  NULL_COMMON="cross_run1to2_shrinkage_matched/$T/"
  for F in "$ASEG_CHA" "$NULL_COMMON"; do
    $PY $ISC --atlas aseg --source-space mni-2mm --metrics resp_coarsefine conn_coarsefine \
        --transformation-folder "$F" --targets onavg-ico32
    $PY $ISC --atlas aseg --source-space mni-2mm --metrics conn_coarsefine \
        --transformation-folder "$F" --targets CHA+aseg --targets-transformation-folder "$F" $CONTRA
  done
  # reference subject NOT transformed: every other subject goes directly into the reference subject's space
  OTHER="ridgeCV_to_other/transformation_to_other_targets_onavg-ico8"
  NULL="ridge_cross_run_matchedalpha/transformation_to_self_targets_onavg-ico8"
  VOL="results/Raiders/mni-2mm_1step_linear_overlap"
  # real direct subject-to-subject transform (j -> i)
  $PY $TO_OTHER --to-other-transformation-folder "$OTHER" --align-method ridgeCV \
      --cortex-transformation-folder "$CTX"
  # matched cross-run null: j's own run1 -> run2 ridge transform at the alpha chosen for j -> i
  $PY $TO_OTHER --to-other-transformation-folder "$NULL" --align-method ridge \
      --direct-self-null --direct-alpha-dir "$VOL/$OTHER" --cortex-transformation-folder "$CTX"
  # anatomical alignment (same code path and domains)
  $PY $TO_OTHER --anatomical-alignment --cortex-transformation-folder "$CTX"
}

BLOCK=${1:-all}
case "$BLOCK" in
  aa)        block_aa ;;
  cha)       block_cha ;;
  supp4)     block_supp4 ;;
  tian)      block_tian ;;
  splitruns) block_splitruns ;;
  hcp)       block_hcp ;;
  supp9b)    block_supp9b ;;
  all)       block_aa; block_cha; block_supp4; block_tian; block_splitruns; block_hcp; block_supp9b ;;
  *) echo "unknown block '$BLOCK' (aa|cha|supp4|tian|splitruns|hcp|supp9b|all)"; exit 1 ;;
esac
