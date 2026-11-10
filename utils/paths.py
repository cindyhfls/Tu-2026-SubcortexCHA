"""Repository-relative paths.

Everything is resolved from the location of this file, so scripts work no matter
which directory they are launched from.

    data/     inputs and caches: cache/, confound_betas/, HCP/prep_oneregression, HCP/resampled
    results/  outputs of the pipeline: Budapest/, Raiders/, HCP/ (templates, transformations, ISC and
              prediction results), figures/ (plots) and stats/ (statistics tables and text)
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
STATS = RESULTS / "stats"
CACHE = DATA / "cache"
CONFOUND_BETAS = DATA / "confound_betas"


def resolve(path):
    """Return `path` as an absolute Path; relative paths are taken from the repo root."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path
