"""
setup.py — downloads SciQ and OpenBookQA into the resolved data root and makes sure the
three pretrained checkpoints are present in the Hugging Face cache. Plain script.

# ITEM 49: unchanged in behaviour.
Nothing in the run summaries or the critique implicates this file, so the dataset download,
the DATA_ROOT.txt pointer and the snapshot cache check stay exactly as they were. The only
additions are the resolved-snapshot sha (provenance: setup and main.py can now be compared
on the weights each one saw) and the explicit note on the two network-call handlers.

Design constraints (this script runs under a separate, short setup timeout):
  * never load model weights into memory here — that is main.py's job, and the import below
    is torch-free by construction, which is what keeps this script inside its timeout;
  * skip any Hub traffic for a model whose snapshot is already cached locally;
  * bound the remaining work with RC_SETUP_BUDGET_SEC (default 600 s) and, if a checkpoint is
    missing and the budget is spent, leave the download to main.py (which runs with network
    access and a much larger budget);
  * line-buffer stdout so progress is visible even if the process is killed.

This script computes no metric, no rate and no aggregate: it runs before any seed exists,
and every rate rule lives in analysis.py and methods.py so no second, diverging copy of one
can appear here. It makes no subprocess, shell, eval or exec call; the only network calls are
the two declared dataset loaders and snapshot_download.

DEVICE_FALLBACK_REASON is deliberately NOT imported: detect_device rebinds that module global
at call time, so a from-import would freeze a stale copy — and this script never probes the
device in any case.

Windows note: `datasets` names its lock file after the flattened absolute cache-dir path and
places it inside that directory, so a deep project folder overflows MAX_PATH (260). The shared
resolver (experiment_config.resolve_data_root) picks a short per-user root in that case; the
choice is recorded in <project>/data/DATA_ROOT.txt so main.py loads from the same place.
"""
import os
import sys
import time

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment_config import (MODEL_IDS, PROJECT_DATA_DIR, resolve_data_root,  # noqa: E402  (no torch import)
                               snapshot_cached, snapshot_revision)

T0 = time.time()
SETUP_BUDGET = float(os.environ.get("RC_SETUP_BUDGET_SEC", "600"))


def elapsed() -> float:
    """Seconds since the script started; a monotone difference of two clock reads, always finite.

    This is the only number the file produces, and it is never a metric and never written to
    the results payload."""
    return time.time() - T0


def log(msg: str) -> None:
    """One timestamped, line-buffered setup message; never raises."""
    print(f"[setup +{elapsed():.0f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------- data root
DATA_ROOT = resolve_data_root()
os.makedirs(DATA_ROOT, exist_ok=True)
os.makedirs(PROJECT_DATA_DIR, exist_ok=True)
# the pointer main.py reads back through the same resolver, so the two never load from
# different directories
with open(os.path.join(PROJECT_DATA_DIR, "DATA_ROOT.txt"), "w", encoding="utf-8") as f:
    f.write(DATA_ROOT)
if DATA_ROOT != PROJECT_DATA_DIR:
    log(f"project data path is {len(PROJECT_DATA_DIR)} chars (too long for Windows dataset locks); "
        f"using {DATA_ROOT}")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# ----------------------------------------------------------------------------- datasets
from datasets import load_dataset  # noqa: E402

sciq_dir = os.path.join(DATA_ROOT, "sciq")
log(f"loading allenai/sciq into {sciq_dir} ...")
try:
    sciq = load_dataset("allenai/sciq", cache_dir=sciq_dir)
except FileNotFoundError as e:
    # SciQ is the primary corpus: this is NOT a fallback, the script fails loudly and names the fix
    raise RuntimeError(
        f"SciQ download failed with FileNotFoundError; on Windows this means the cache path is "
        f"still too long ({len(sciq_dir)} chars). Set RC_DATA_ROOT to a short directory "
        f"(e.g. C:\\rc_data) and rerun. Original error: {e}"
    ) from e
log(f"sciq splits: { {k: len(v) for k, v in sciq.items()} }")

obqa_dir = os.path.join(DATA_ROOT, "openbookqa")
log(f"loading allenai/openbookqa (additional) into {obqa_dir} ...")
try:
    obqa = load_dataset("allenai/openbookqa", "additional", cache_dir=obqa_dir)
    log(f"openbookqa splits: { {k: len(v) for k, v in obqa.items()} }")
except Exception as e:
    # A network call whose concrete exception type moves between `datasets` versions, so the
    # handler is broad by necessity -- but the type and message are LOGGED, never swallowed, and
    # CorpusBuilder.ingest_openbookqa independently records openbookqa_error plus a flag in the
    # payload at run time. The secondary corpus missing is a recorded condition, not a silent one.
    log(f"WARNING: OpenBookQA download failed: {type(e).__name__}: {e}")

# ----------------------------------------------------------------------------- model cache check
for role, mid in MODEL_IDS.items():
    if snapshot_cached(mid):
        log(f"{role}: {mid} already in local HF cache (revision {snapshot_revision(mid)}); "
            f"skipping Hub traffic")
        continue
    remaining = SETUP_BUDGET - elapsed()
    if remaining < 60:
        log(f"WARNING: {role}: {mid} not cached and setup budget nearly spent "
            f"({remaining:.0f}s left); main.py will download it on first load")
        continue
    log(f"{role}: {mid} not cached; downloading snapshot (budget left {remaining:.0f}s) ...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(mid, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.spm"],
                          max_workers=4)
        log(f"{role}: {mid} downloaded (revision {snapshot_revision(mid)})")
    except Exception as e:
        # Same reasoning as above: a Hub call, a moving exception type, the cause logged in full.
        # main.py retries the load itself and records the outcome, so nothing is lost here.
        log(f"WARNING: {role}: {mid} download failed ({type(e).__name__}: {e}); "
            f"main.py will retry on first load")

log(f"data root: {DATA_ROOT}")
log(f"done in {elapsed():.0f}s")
sys.exit(0)