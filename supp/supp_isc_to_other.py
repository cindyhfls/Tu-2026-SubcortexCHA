"""ISC with subject j carried directly into reference subject i's space (Supp Fig 9B, reference NOT transformed).

For a reference subject i and every other subject j, the reference side ts_i is always the raw,
untransformed test-run data (full, unsplit runs 3-4) and the comparison side is subject j mapped into
subject i's space. Three conditions are computed by this one script, selected by the folder given:

  --to-other-transformation-folder <real>   ts_j @ xfm[j -> i]   direct subject-to-subject hyperalignment
                                            (transforms from supp_transformation_to_other.py)
  --to-other-transformation-folder <null> --direct-self-null --direct-alpha-dir <real>
                                            matched cross-run null: j's own run1->run2 ridge transform
                                            (supp_transformation_to_self.py --match-alpha-from-pairs),
                                            alpha taken from the real j -> i fit; i enters only via alpha
  --anatomical-alignment                    ts_j unchanged (no transform)

Seeds are the 16 aseg nuclei. Features (each as coarse = nucleus mean, fine = mean-subtracted,
coarsefine = all voxels) and ISC domains:
  resp                                response time series
  conn_cortex_ico32                   connectivity to cortical onavg-ico32 vertices
                                      (anatomical, or hyperaligned with --cortex-transformation-folder)
  conn_cortex_ico8                    connectivity to anatomical cortical onavg-ico8 targets
  conn_subcortex_contralateral        connectivity to opposite-hemisphere subcortical voxels, raw targets
  conn_subcortex_contralateral_xfmtarget
                                      same, with the targets mapped into subject i's space as well
                                      (not for --anatomical-alignment, where it equals the raw version)
Each is reported as pairwise (Fisher-averaged over j) and leave-one-out ISC.

Usage (see slurm/run_isc.sh, block supp9b, for the commands used):
    python supp_isc_to_other.py --to-other-transformation-folder \\
        ridgeCV_to_other/transformation_to_other_targets_onavg-ico8 --align-method ridgeCV \\
        --cortex-transformation-folder ridge_alpha1e3/transformation_to_template_targets_onavg-ico8/
    python supp_isc_to_other.py --anatomical-alignment \\
        --cortex-transformation-folder ridge_alpha1e3/transformation_to_template_targets_onavg-ico8/
Input:  data/cache/Raiders_raiders/run34/ (cache_subcortical_raiders.py, cache_cortical_raiders.py); transforms under
        results/Raiders/mni-2mm_1step_linear_overlap/<transformation folder>/
Output: results/Raiders/mni-2mm_1step_linear_overlap/<transformation folder or AA/projtosubj_AA>/
        memmap_ISC_{pairs,loo}_projecttosubj_aseg_raiders_summary/sl<start>-<stop>/
        [connectivity_targets-<target>/]isc_<domain>_<metric>[_pairwise].dat
        (float32 memmaps, rows = seed nuclei, columns = reference subjects i; _pairwise is nuclei x i x j)
Whole-atlas arrays projected into the reference subject's space are cached per reference subject under the same folder of
data/cache, so --sl-start/--sl-stop chunks reuse them; populate the cache from ONE task first
(--sl-start 0 --sl-stop 0), then run chunks in parallel.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import CACHE, RESULTS, ROOT, resolve  # noqa: E402

import os
import json
import argparse
import numpy as np
import neuroboros as nb
from utils.data_io_utils import (ensure_memmap, row_done, return_aseg_labels, get_aseg_sls,
                                 untransformed_target_label)
from utils.isc_utils import (pairwise_corrsim_dot_xy, loo_corrsim_dot_xy, FisherAvg, parse_data_coarse_fine,
                             conn_features_all_subjects, compute_targets_ts_from_mapping)

parser = argparse.ArgumentParser()
parser.add_argument("--sl-start", type=int, default=0, help="first seed nucleus of this chunk")
parser.add_argument("--sl-stop", type=int, default=None, help="end (exclusive) of this chunk; default all")
parser.add_argument("--to-other-transformation-folder", type=str, default=None,
                    help="Direct mode: folder (relative to mni-2mm_1step_linear_overlap/) of pairwise "
                         "transforms '<src>_to_<tgt>_targets-<label>_<roi>_<method>.npy' written by "
                         "supp_transformation_to_other.py. Subject j is carried into reference subject i's "
                         "space as ts_j @ xfm[j->i].")
parser.add_argument("--anatomical-alignment", action="store_true",
                    help="No transform: subject j enters the comparison raw. Writes to AA/projtosubj_AA/.")
parser.add_argument("--align-method", type=str, default=None,
                    help="Method suffix of the transform files (ridgeCV for the real maps, ridge for the null).")
parser.add_argument("--direct-self-null", action="store_true",
                    help="The --to-other folder holds SELF transforms keyed by the source subject alone "
                         "(<src>_targets-<label>_<roi>[_alpha<A>]_<method>.npy), as written by "
                         "supp_transformation_to_self.py (matched cross-run null).")
parser.add_argument("--direct-alpha-dir", type=str, default=None,
                    help="With --direct-self-null: folder of the real j->i fits; the ridgeCV alpha is read from "
                         "<dir>/<src>_to_<tgt>_targets-<label>_<roi>_ridgeCV_bestalpha.txt (relative paths are "
                         "taken from the repository root).")
parser.add_argument("--cortex-transformation-folder", type=str, default="",
                    help="Make the conn_cortex_ico32 target the HYPERALIGNED cortex, read from "
                         "dms_lr_float32.npy in this folder of the cortical cache (same cortex as "
                         "fig2_3_loo.py --targets-transformation-folder). Output goes to "
                         "connectivity_targets-CHA+onavg-ico32. Empty = anatomical cortex.")
parser.add_argument("--xfm-targets-label", type=str, default="onavg-ico8",
                    help="The --targets value supp_transformation_to_other.py / supp_transformation_to_self.py were run with; used in the transform file names and "
                         "for the conn_cortex_ico8 domain.")
args = parser.parse_args()

aa = args.anatomical_alignment
direct = args.to_other_transformation_folder is not None
if aa == direct:
    parser.error("pass exactly one of --to-other-transformation-folder or --anatomical-alignment")
if direct and args.align_method is None:
    parser.error("--align-method is required with --to-other-transformation-folder")
if args.direct_alpha_dir and not args.direct_self_null:
    parser.error("--direct-alpha-dir needs --direct-self-null")

dataset, task, run_tag = "Raiders", "raiders", "run34"
atlas = "aseg"
volume_space, volume_resample = 'mni-2mm', '1step_linear_overlap'
surface_space, surface_resample = 'onavg-ico32', '1step_pial_overlap'
xfm_targets_label = args.xfm_targets_label
align_method = args.align_method
cache_dir = f"{CACHE}/{dataset}_{task}/{run_tag}"
source_folder = f"{volume_space}_{volume_resample}"

transformation_folder = "AA/projtosubj_AA" if aa else args.to_other_transformation_folder
to_out_root = f"{source_folder}/{transformation_folder}"
xfm_dir = os.path.join(RESULTS, dataset, to_out_root)
if direct and not os.path.isdir(xfm_dir):
    raise SystemExit(f"transform folder does not exist: {xfm_dir}\n"
                     f"ISC results are written into the folder the transforms are read from.")
alpha_dir = str(Path(args.direct_alpha_dir)) if args.direct_alpha_dir else None
if alpha_dir and not os.path.isabs(alpha_dir):
    alpha_dir = os.path.join(ROOT, alpha_dir)
print(f"transformation folder: {transformation_folder}")

sids = np.load(os.path.join(cache_dir, "sids.npy")).astype(str)
N = len(sids)
print("N total in file:", N)

rois = return_aseg_labels(atlas_name=volume_space)
sls = get_aseg_sls()
# 'CHA+aseg': one transform (j -> i); AA applies none, so its target is the plain atlas 'aseg'
raw_targets_label = 'aseg'
targets_label = untransformed_target_label('CHA+aseg') if aa else 'CHA+aseg'
nSL = len(sls)
sl_start = args.sl_start
sl_stop = nSL if args.sl_stop is None else min(args.sl_stop, nSL)

# ---- raw whole-atlas and cortical data (the reference side is always raw) ----
raw_dms = np.load(os.path.join(cache_dir, f"ts_{atlas}_Subcortex_float32.npy"), mmap_mode="r")  # (N,T,Vfull)
Vfull, T = raw_dms.shape[2], raw_dms.shape[1]
_ts_l = np.load(os.path.join(cache_dir, "ts_l_float32.npy"), mmap_mode="r")
_ts_r = np.load(os.path.join(cache_dir, "ts_r_float32.npy"), mmap_mode="r")
cortex_raw_dms = np.concatenate([_ts_l, _ts_r], axis=2).astype(np.float32, copy=False)  # (N,T,Vcortex)
cortex_dms = cortex_raw_dms
if args.cortex_transformation_folder:
    cortex_dms = np.load(os.path.join(cache_dir, f"{surface_space}_{surface_resample}",
                                      args.cortex_transformation_folder, "dms_lr_float32.npy"),
                         mmap_mode="r")
    assert cortex_dms.shape == cortex_raw_dms.shape, (
        f"hyperaligned cortex {cortex_dms.shape} vs raw {cortex_raw_dms.shape}")
    print("conn_cortex_ico32 target: HYPERALIGNED", args.cortex_transformation_folder)
# the ico8 target is always the anatomically aligned cortex
cortex_ico8_dms = compute_targets_ts_from_mapping(cortex_raw_dms, surface_space, xfm_targets_label)  # (N,T,1210)


def load_xfm_pair(src, tgt, roi):
    """Transform carrying subject `src` into subject `tgt`'s space for one nucleus."""
    if args.direct_self_null:
        # fitted on src's own two runs; tgt enters only by selecting the alpha
        stem = f"{src}_targets-{xfm_targets_label}_{roi}"
        if alpha_dir:
            with open(os.path.join(alpha_dir, f"{src}_to_{tgt}_targets-{xfm_targets_label}_{roi}"
                                              f"_ridgeCV_bestalpha.txt")) as f:
                stem += f"_alpha{float(f.read()):.0e}"
        fn = os.path.join(xfm_dir, f"{stem}_{align_method}")
    else:
        fn = os.path.join(xfm_dir, f"{src}_to_{tgt}_targets-{xfm_targets_label}_{roi}_{align_method}")
    try:
        return np.asarray(np.load(fn + ".npy"))
    except FileNotFoundError:
        x = nb.load(fn + ".npz")
        return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


# disk cache of the whole atlas projected into each reference subject's space, one array per reference subject
projected_cache_dir = os.path.join(cache_dir, to_out_root, "projected_to_subject_cache")
os.makedirs(projected_cache_dir, exist_ok=True)

METRICS = ["coarse", "fine", "coarsefine"]
DOMAINS = ["resp", "conn_cortex_ico32", "conn_cortex_ico8", "conn_subcortex_contralateral"]
if not aa:
    DOMAINS.append("conn_subcortex_contralateral_xfmtarget")


def file_domain(domain):
    """On-disk stem; the two contralateral domains share the conn_subcortex name, told apart by directory."""
    return "conn_subcortex" if domain.startswith("conn_subcortex") else domain


def domain_subdir(root, domain):
    if domain == "resp":
        return root
    if domain == "conn_subcortex_contralateral":
        return os.path.join(root, f"connectivity_targets-{raw_targets_label}-contralateral")
    if domain == "conn_subcortex_contralateral_xfmtarget":
        return os.path.join(root, f"connectivity_targets-{targets_label}-contralateral")
    if domain == "conn_cortex_ico32":
        return os.path.join(root, "connectivity_targets-"
                            + ("CHA+" if args.cortex_transformation_folder else "") + "onavg-ico32")
    if domain == "conn_cortex_ico8":
        return os.path.join(root, f"connectivity_targets-{xfm_targets_label}")


chunk_tag = f"sl{sl_start:04d}-{sl_stop:04d}"
pairs_outdir = os.path.join(RESULTS, dataset, to_out_root, f"memmap_ISC_pairs_projecttosubj_{atlas}_{task}_summary", chunk_tag)
loo_outdir = os.path.join(RESULTS, dataset, to_out_root, f"memmap_ISC_loo_projecttosubj_{atlas}_{task}_summary", chunk_tag)
os.makedirs(pairs_outdir, exist_ok=True)
os.makedirs(loo_outdir, exist_ok=True)
print("Output dir (pairwise):", pairs_outdir)
print("Output dir (loo):     ", loo_outdir)

# Provenance: row_done() skips finite rows, so a directory must not be reused by another configuration.
_prov = {"dataset": dataset, "task": task, "run_tag": run_tag, "atlas": atlas, "nSubject": int(N),
         "anatomical_alignment": aa, "to_other_transformation_folder": args.to_other_transformation_folder,
         "align_method": align_method, "direct_self_null": args.direct_self_null,
         "direct_alpha_dir": args.direct_alpha_dir,
         "cortex_transformation_folder": args.cortex_transformation_folder,
         "xfm_targets_label": xfm_targets_label}
for _d in (os.path.dirname(pairs_outdir), os.path.dirname(loo_outdir)):
    _fn = os.path.join(_d, "args.json")
    if os.path.exists(_fn):
        _old = json.load(open(_fn))
        _diff = {k: (_old.get(k), v) for k, v in _prov.items() if k in _old and _old[k] != v}
        if _diff:
            raise SystemExit(f"provenance mismatch in {_d}\n"
                             + "\n".join(f"  {k}: on disk {o!r} -> now {n!r}" for k, (o, n) in _diff.items())
                             + "\nDelete this directory to recompute under the new configuration.")
    else:
        with open(_fn, "w") as _f:
            json.dump(_prov, _f, indent=2)

for d in (pairs_outdir, loo_outdir):
    with open(os.path.join(d, "sids.txt"), "w") as f:
        f.writelines(f"{sid}\n" for sid in sids)

TAGS = [f"{d}_{m}" for d in DOMAINS for m in METRICS]
mm_pairs, mm_loo = {}, {}
for domain in DOMAINS:
    for metric in METRICS:
        tag, ftag = f"{domain}_{metric}", f"{file_domain(domain)}_{metric}"
        mm_pairs[tag] = ensure_memmap(os.path.join(domain_subdir(pairs_outdir, domain), f"isc_{ftag}.dat"),
                                      (sl_stop - sl_start, N))
        mm_pairs[f"{tag}_pairwise"] = ensure_memmap(
            os.path.join(domain_subdir(pairs_outdir, domain), f"isc_{ftag}_pairwise.dat"),
            (sl_stop - sl_start, N, N))
        mm_loo[tag] = ensure_memmap(os.path.join(domain_subdir(loo_outdir, domain), f"isc_{ftag}.dat"),
                                    (sl_stop - sl_start, N))


def conn_feats(targets_ts, seed_dict):
    return {m: conn_features_all_subjects(targets_ts, seed_dict[m], zscore_axis=0, flatten=False)
            for m in METRICS}


# ---- reference side (raw, ts_i): independent of i, so precomputed for every nucleus up front ----
print(f"Precomputing raw reference features for {sl_stop - sl_start} nuclei...")
ref_resp, ref_conn_cortex, ref_conn_cortex_ico8, ref_conn_contra, contra_idx = {}, {}, {}, {}, {}
for isl in range(sl_start, sl_stop):
    X_raw_sl = np.asarray(raw_dms[:, :, sls[isl]])  # (N,T,Vsl)
    X_coarse, X_fine = parse_data_coarse_fine(X_raw_sl)
    ref_resp[isl] = {'coarse': X_coarse, 'fine': X_fine, 'coarsefine': X_raw_sl}
    # contralateral targets: every voxel of every ROI in the other hemisphere
    _other = 'r-' if rois[isl].startswith('l-') else 'l-'
    _keep = np.zeros(Vfull, bool)
    for _j, _rj in enumerate(rois):
        if _rj.startswith(_other):
            _keep[sls[_j]] = True
    contra_idx[isl] = np.where(_keep)[0]
    ref_conn_contra[isl] = conn_feats(np.asarray(raw_dms[:, :, contra_idx[isl]]), ref_resp[isl])
    ref_conn_cortex_ico8[isl] = conn_feats(cortex_ico8_dms, ref_resp[isl])
    ref_conn_cortex[isl] = conn_feats(cortex_dms, ref_resp[isl])

cross_matrix = {isl: {tag: np.full((N, N), np.nan, dtype=np.float32) for tag in TAGS}
                for isl in range(sl_start, sl_stop)}
loo_across = {isl: {tag: np.full(N, np.nan, dtype=np.float32) for tag in TAGS}
              for isl in range(sl_start, sl_stop)}

print(f"Mapping + computing ISC for {N} reference subjects x {sl_stop - sl_start} nuclei...")
for i, sid_i in enumerate(sids):
    projected_cache_fn = os.path.join(projected_cache_dir, f"{sid_i}.npy")
    if os.path.exists(projected_cache_fn):
        projected_dms = np.load(projected_cache_fn, mmap_mode="r")  # (N,T,Vfull)
    else:
        projected_dms = np.empty((N, T, Vfull), dtype=np.float32)
        if aa:
            projected_dms[:] = raw_dms
        else:
            # one transform per (source subject, nucleus), carrying j straight into i's space. j == i has
            # no transform (supp_transformation_to_other.py skips self-pairs) and stays raw: it never enters a reported number.
            for j, sid_j in enumerate(sids):
                if j == i:
                    projected_dms[j] = raw_dms[j]
                    continue
                for roi_name, idx in zip(rois, sls):
                    xfm_ji = load_xfm_pair(sid_j, sid_i, roi_name)
                    projected_dms[j, :, idx] = (np.asarray(raw_dms[j, :, idx], np.float32).T @ xfm_ji).T
        tmp_fn = projected_cache_fn + ".tmp.npy"
        np.save(tmp_fn, projected_dms)
        os.replace(tmp_fn, projected_cache_fn)
        projected_dms = np.load(projected_cache_fn, mmap_mode="r")

    for isl in range(sl_start, sl_stop):
        curr_row = isl - sl_start
        # checkpoint: a nucleus whose rows are all on disk is skipped
        if all(row_done(mm_pairs[tag][curr_row]) and row_done(mm_loo[tag][curr_row]) for tag in TAGS):
            continue

        X_bp_sl = projected_dms[:, :, sls[isl]]  # (N,T,Vsl)
        X_bp_coarse, X_bp_fine = parse_data_coarse_fine(X_bp_sl)
        bp_resp = {'coarse': X_bp_coarse, 'fine': X_bp_fine, 'coarsefine': X_bp_sl}
        # cortical targets are raw/AA (or hyperaligned) and never decoded; only the seed is mapped
        domain_triples = [
            ("resp", ref_resp[isl], bp_resp),
            ("conn_cortex_ico32", ref_conn_cortex[isl], conn_feats(cortex_dms, bp_resp)),
            ("conn_cortex_ico8", ref_conn_cortex_ico8[isl], conn_feats(cortex_ico8_dms, bp_resp)),
            ("conn_subcortex_contralateral", ref_conn_contra[isl],
             conn_feats(np.asarray(raw_dms[:, :, contra_idx[isl]]), bp_resp))]
        if not aa:
            # targets mapped into subject i's space as well as the seed
            domain_triples.append(("conn_subcortex_contralateral_xfmtarget", ref_conn_contra[isl],
                                   conn_feats(np.asarray(projected_dms[:, :, contra_idx[isl]]), bp_resp)))

        for domain, ref, bp in domain_triples:
            for metric in METRICS:
                tag = f"{domain}_{metric}"
                cross_matrix[isl][tag][i] = pairwise_corrsim_dot_xy(ref[metric][i:i + 1], bp[metric])[0]
                loo_across[isl][tag][i] = loo_corrsim_dot_xy(ref[metric], bp[metric])[i]

    if i % 5 == 0:
        print(f"  reference subject {i+1}/{N} done")

for isl in range(sl_start, sl_stop):
    curr_row = isl - sl_start
    for tag in TAGS:
        if not np.isfinite(cross_matrix[isl][tag]).any():
            continue      # not recomputed in this pass (already on disk)
        R = cross_matrix[isl][tag]
        isc_value = np.array([FisherAvg(R[k, [j for j in range(N) if j != k]]) for k in range(N)], dtype=np.float32)
        mm_pairs[tag][curr_row] = isc_value; mm_pairs[tag].flush()
        mm_pairs[f"{tag}_pairwise"][curr_row] = R; mm_pairs[f"{tag}_pairwise"].flush()
        mm_loo[tag][curr_row] = loo_across[isl][tag]; mm_loo[tag].flush()

print("All done.")
