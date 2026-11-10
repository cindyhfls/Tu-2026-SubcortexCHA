"""Shared data loading for the PSF scripts (fig4_psf.py across-subject, fig4_psf_withinsubj.py within-subject control)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import CACHE, RESULTS, resolve  # noqa: E402

import os
import numpy as np
from utils.data_io_utils import return_aseg_labels, get_aseg_sls
from utils.isc_utils import compute_targets_ts_from_mapping

DATASET = "Raiders"
TASK = "raiders"
RES = 2
VOLUME_SPACE = f"mni-{RES}mm"
VOLUME_RESAMPLE = "1step_linear_overlap"
SURFACE_SPACE = "onavg-ico32"
SURFACE_RESAMPLE = "1step_pial_overlap"
DISTANCE_METRIC = "euclidean"
VOXPAIR_FN = (f"voxel_pairs_subcortical_{DISTANCE_METRIC}_MNI152NLin2009cAsym_res-{RES:02d}"
              "_seg-aseg_dseg_binned.npy")


def add_common_args(parser):
    parser.add_argument("--sl-start", type=int, default=0)
    parser.add_argument("--sl-stop", type=int, default=None)
    parser.add_argument("--run", type=int, nargs="+", default=[3, 4],
                        help="Runs of the cached data (cache folder run<runs>); paper: test runs 3 4.")
    parser.add_argument("--targets", type=str, default="onavg-ico32",
                        help="Connectivity targets: 'onavg-ico32' (anatomically aligned cortex), 'CHA+onavg-ico32' "
                             "(hyperaligned cortex), 'aseg' (the seed atlas itself, taken from the same tree), "
                             "'CHA+aseg' (hyperaligned subcortex), optionally with onavg-ico8 instead of ico32.")
    parser.add_argument("--transformation-folder", type=str, default="",
                        help="Empty = anatomical alignment (AA). Otherwise the transformed caches under "
                             "<space>_<resample>/<folder>, e.g. ridgeCV/transformation_to_template_targets_onavg-ico8")
    parser.add_argument("--targets-transformation-folder", type=str, default="",
                        help="Only for CHA+ targets: tree holding the hyperaligned target caches "
                             "(dms_lr_float32.npy / dms_aseg_Subcortex_float32.npy).")
    parser.add_argument("--voxel-pairs-dir", type=str, default=str(RESULTS / "voxel_pairs"),
                        help="Folder written by fig4_calc_voxel_distances.py")


def load_inputs(args):
    """Load subcortical data, targets, nucleus indices and voxel pairs. Returns a dict."""
    run_tag = "run" + "".join(map(str, args.run))
    cache_dir = CACHE / f"{DATASET}_{TASK}" / run_tag
    source_folder = f"{VOLUME_SPACE}_{VOLUME_RESAMPLE}"
    transformation_folder = args.transformation_folder
    out_root = f"{source_folder}/{transformation_folder}" if transformation_folder else f"{source_folder}/AA/"

    sids = np.load(cache_dir / "sids.npy").astype(str)
    print("N total in file:", len(sids))

    if transformation_folder:
        dms = np.load(os.path.join(cache_dir, out_root, "dms_aseg_Subcortex_float32.npy"), mmap_mode="r")
    else:
        dms = np.load(cache_dir / "ts_aseg_Subcortex_float32.npy", mmap_mode="r")

    rois = return_aseg_labels(atlas_name=VOLUME_SPACE)
    sls = get_aseg_sls()
    nSL = len(sls)
    sl_stop = nSL if args.sl_stop is None else min(args.sl_stop, nSL)

    targets = args.targets
    if targets == "aseg":  # connectivity target = the seed atlas itself (not an alignment target)
        targets_ts = dms
    else:
        if targets.startswith("CHA+"):  # connectivity-hyperaligned targets
            assert args.targets_transformation_folder, "CHA+ targets need --targets-transformation-folder"
            if "onavg" in targets:
                targets_out_root = f"{SURFACE_SPACE}_{SURFACE_RESAMPLE}/{args.targets_transformation_folder}"
                fn = "dms_lr_float32.npy"
            elif "aseg" in targets:
                targets_out_root = f"{VOLUME_SPACE}_{VOLUME_RESAMPLE}/{args.targets_transformation_folder}"
                fn = "dms_aseg_Subcortex_float32.npy"
            else:
                raise ValueError(f"unsupported targets {targets}")
            targets_ts = np.load(os.path.join(cache_dir, targets_out_root, fn), mmap_mode="r")
        elif "onavg" in targets:  # anatomically aligned cortex
            ts_l = np.load(cache_dir / "ts_l_float32.npy")
            ts_r = np.load(cache_dir / "ts_r_float32.npy")
            targets_ts = np.concatenate([ts_l, ts_r], axis=2).astype(np.float32, copy=False)  # (N,T,Vall)
        else:
            raise ValueError(f"unsupported targets {targets}")
        # sparse connectivity targets: map the dense surface data onto ico8
        if "onavg-ico8" in targets:
            targets_ts = compute_targets_ts_from_mapping(targets_ts, SURFACE_SPACE, "onavg-ico8", mask=True)
    print("targets_ts:", targets_ts.shape)

    voxel_pairs = np.load(Path(args.voxel_pairs_dir) / VOXPAIR_FN, allow_pickle=True).item()
    distances = list(voxel_pairs[rois[0]].keys())  # bin centres in voxels
    return dict(sids=sids, dms=dms, rois=rois, sls=sls, nSL=nSL, sl_start=args.sl_start, sl_stop=sl_stop,
                targets_ts=targets_ts, voxel_pairs=voxel_pairs, distances=distances,
                out_root=out_root, transformation_folder=transformation_folder)


def make_outdir(inp):
    """Chunk output folder (results/Raiders/<space>_<resample>/<tree>/memmap_ISC_loo_psf_..._summary/sl<a>-<b>)
    plus meta.txt and sids.txt."""
    chunk_tag = f"sl{inp['sl_start']:04d}-{inp['sl_stop']:04d}"
    outdir = os.path.join(RESULTS, DATASET, inp["out_root"],
                          f"memmap_ISC_loo_psf_{DISTANCE_METRIC}_binned_aseg_{TASK}_summary", chunk_tag)
    os.makedirs(outdir, exist_ok=True)
    print("Output dir:", outdir)
    meta_fn = os.path.join(outdir, "meta.txt")
    if not os.path.exists(meta_fn):
        with open(meta_fn, "w") as f:
            f.write(f"task={TASK}\natlas=aseg\nconnectivity_source_space={VOLUME_SPACE}\n"
                    f"nSubject={len(inp['sids'])}\nsl_start={inp['sl_start']}\nsl_stop={inp['sl_stop']}\n"
                    f"transformation_folder={inp['transformation_folder']}\n")
    sids_txt = os.path.join(outdir, "sids.txt")
    if not os.path.exists(sids_txt):
        with open(sids_txt, "w") as f:
            for sid in inp["sids"]:
                f.write(f"{sid}\n")
    return outdir
