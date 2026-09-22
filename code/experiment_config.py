"""
experiment_config.py -- hyperparameters, model ids, prompt templates and the arm registry
for the corpus-supported-distractor (CSD) study.

(Named experiment_config rather than config so it cannot collide with the pip package
`config` or any other module of that name on sys.path.)

Imported by setup.py as well as by analysis.py, methods.py, models.py, main.py and the test
suite, so it must stay cheap to import: torch is touched ONLY inside the device probe, never
at module level, and no project module is imported here at all.
"""
import os
from typing import Any, Callable, Dict, Optional

# Both environment switches are read here with os.environ.get and the RAW strings are kept, so
# main.py can prove (through analysis.check_environment_switches) that it describes the same run
# this module recorded at import time. Comparing a raw '' against a parsed '43200.0' as text once
# aborted every run before the pre-flight self-tests, so both sides meet as the same kind of value.
ENV_SMOKE_AT_IMPORT = os.environ.get("RC_SMOKE_TEST", "0")
ENV_TIME_BUDGET_AT_IMPORT = os.environ.get("RC_TIME_BUDGET_SEC", "")
SMOKE = ENV_SMOKE_AT_IMPORT == "1"
TIME_BUDGET_SEC = float(ENV_TIME_BUDGET_AT_IMPORT) if str(ENV_TIME_BUDGET_AT_IMPORT).strip() else 43200.0

# No component is ever conditional on the clock or on another arm's result. This is a standing
# declaration main.py records in the payload; nothing reads it to decide whether something runs.
ARMS_ALWAYS_RUN = True

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DATA_DIR = os.path.join(PROJECT_DIR, "data")
# `datasets` writes a lock file INSIDE the cache dir whose NAME is the flattened absolute cache-dir
# path, so the effective path length is ~2x the cache-dir length (+ ~70 chars). Keep the root short
# enough that <root>/openbookqa/<flattened lock name> stays under Windows MAX_PATH (260).
MAX_SAFE_DATA_ROOT_LEN = 80

MODEL_IDS = {
    "generator": "Qwen/Qwen2.5-3B-Instruct",
    "nli": "cross-encoder/nli-deberta-v3-base",
    "embedder": "BAAI/bge-small-en-v1.5",
}
# The Hub revision each checkpoint is loaded at. None means the repo's default branch, resolved at
# load time; the commit sha of the snapshot ACTUALLY loaded is read back by snapshot_revision and
# recorded in the payload under backend.revisions either way, so a run always states which weights
# produced its numbers.
MODEL_REVISIONS: Dict[str, Optional[str]] = {"generator": None, "nli": None, "embedder": None}

# ITEM 4: why the device probe fell back, or None when it succeeded. A bare
# `except Exception: device = "cpu"` turned any torch failure into an unexplained CPU run: the
# device reached the payload but the CAUSE did not, so a run on the wrong hardware and a run with
# a broken install were indistinguishable.
DEVICE_FALLBACK_REASON: Optional[str] = None


def resolve_data_root() -> str:
    """Data root shared by setup.py and main.py; short enough for Windows dataset lock files."""
    env = os.environ.get("RC_DATA_ROOT")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    pointer = os.path.join(PROJECT_DATA_DIR, "DATA_ROOT.txt")
    pointed = ""
    if os.path.isfile(pointer):
        try:
            with open(pointer, encoding="utf-8") as f:
                pointed = f.read().strip()
        except OSError as e:
            print(f"[config] WARNING: could not read {pointer} ({type(e).__name__}: {e}); ignoring the pointer")
            pointed = ""
    if pointed:
        pointed = os.path.abspath(os.path.expanduser(pointed))
        if os.path.isdir(pointed):
            return pointed
        print(f"[config] WARNING: DATA_ROOT.txt points to a missing directory ({pointed}); ignoring the pointer")
    if os.name == "nt" and len(PROJECT_DATA_DIR) > MAX_SAFE_DATA_ROOT_LEN:
        return os.path.join(os.path.expanduser("~"), ".cache", "experiment_data")
    return PROJECT_DATA_DIR


def hf_hub_cache_root() -> str:
    """Absolute HF hub cache directory from the environment or the documented default."""
    env = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if env:
        return os.path.expanduser(env)
    home = os.environ.get("HF_HOME")
    if home:
        return os.path.join(os.path.expanduser(home), "hub")
    return os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")


def _snapshot_dir(repo_id: str) -> str:
    return os.path.join(_repo_cache_dir(repo_id), "snapshots")


def _complete_snapshots(repo_id: str):
    """Every cached snapshot directory of repo_id that holds config.json plus weight files.

    Returns a list of (commit_name, absolute_path) pairs, empty when nothing is cached. Never
    raises: a missing or unreadable cache directory simply yields no snapshot."""
    repo_dir = _snapshot_dir(repo_id)
    out = []
    if not os.path.isdir(repo_dir):
        return out
    try:
        names = sorted(os.listdir(repo_dir))
    except OSError as e:
        print(f"[config] WARNING: could not list {repo_dir} ({type(e).__name__}: {e}); treating it as uncached")
        return out
    for snap in names:
        d = os.path.join(repo_dir, snap)
        if not os.path.isdir(d):
            continue
        try:
            files = set(os.listdir(d))
        except OSError as e:
            print(f"[config] WARNING: could not list {d} ({type(e).__name__}: {e}); skipping that snapshot")
            continue
        if "config.json" in files and any(n.endswith((".safetensors", ".bin")) for n in files):
            out.append((snap, d))
    return out


def snapshot_cached(repo_id: str) -> bool:
    """True if a local HF cache snapshot for repo_id holds config.json plus weight files.

    models.py passes local_files_only=True when this is True, so a load never waits on a
    rate-limited Hub request."""
    return bool(_complete_snapshots(repo_id))


def snapshot_revision(repo_id: str) -> Optional[str]:
    """The commit sha of the snapshot this run loads, or None when nothing is cached.

    A thin reader over snapshot_revision_record, so there is exactly ONE resolution path and the
    scalar every caller already consumes cannot disagree with the record published beside it.
    Callers that need to know HOW the value was reached read snapshot_revision_record."""
    return snapshot_revision_record(repo_id)["resolved_revision"]


def _torch_device_probe() -> str:
    """The real probe: import torch and ask it whether a CUDA device is visible."""
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def detect_device(probe: Optional[Callable[[], str]] = None) -> str:
    """ITEM 4: probe the device and RECORD why the probe failed, instead of swallowing it.

    On success DEVICE_FALLBACK_REASON is None. On failure the device is "cpu" exactly as before,
    but the exception type and message are kept in DEVICE_FALLBACK_REASON and logged, so a CPU run
    caused by a broken torch install can be told apart from a CPU run on a machine with no GPU.
    The probe is injectable so the test suite can drive both branches with no torch of its own."""
    global DEVICE_FALLBACK_REASON
    fn = probe if probe is not None else _torch_device_probe
    try:
        device = str(fn())
        DEVICE_FALLBACK_REASON = None
    except Exception as e:                      # the cause is recorded, never discarded
        device = "cpu"
        DEVICE_FALLBACK_REASON = f"{type(e).__name__}: {e}"
        print(f"[config] WARNING: the device probe failed ({DEVICE_FALLBACK_REASON}); running on cpu. "
              f"The reason is recorded in the payload under backend.device_fallback_reason.")
    return device


# ITEM 1: the eight plan condition names, verbatim and in registry order. The plan's cap of eight
# conditions is on CONDITION_NAMES ONLY; the two mandated control arms live in their own list so
# neither rule has to bend. ALL_ARM_NAMES is what main.py hands the harness as expected_conditions.
CONDITION_NAMES = [
    "unfiltered_grounded_pool",
    "ungrounded_topic_only_control",
    "per_option_entailment_filter",
    "llm_judge_faithfulness_filter",
    "contradiction_polarity_item_filter",
    "not_entail_polarity_item_filter",
    "overgenerate_rerank_by_contradiction",
    "overgenerate_rerank_by_key_similarity",
]
CONTROL_ARM_NAMES = ["null_random_drop", "positive_control_planted"]
ALL_ARM_NAMES = CONDITION_NAMES + CONTROL_ARM_NAMES

# ----------------------------------------------------------------------------- prompts
PROMPT_GROUNDED = (
    "You are writing one multiple-choice quiz question from the passage below.\n"
    "Passage: {passage}\n"
    "Topic: {topic}\n"
    "Write ONE question that is answerable from the passage, with exactly one correct answer "
    "and three incorrect distractors. Return ONLY a JSON object of the form "
    '{{"stem": "...", "key": "...", "distractors": ["...", "...", "..."]}}'
)
PROMPT_TOPIC_ONLY = (
    "You are writing one multiple-choice quiz question about the course topic below.\n"
    "Topic: {topic}\n"
    "Write ONE question with exactly one correct answer and three incorrect distractors. "
    "Return ONLY a JSON object of the form "
    '{{"stem": "...", "key": "...", "distractors": ["...", "...", "..."]}}'
)
PROMPT_REPAIR = (
    "\n\nYour previous reply was not valid JSON with keys stem, key and distractors "
    "(a list of exactly 3 strings). Reply with the corrected JSON object only."
)
PROMPT_REWRITE = (
    "Passage: {passage}\n"
    "Question: {stem}\n"
    "Correct answer: {key}\n"
    "Rewrite the question so that, according to the passage, ONLY the correct answer above is correct. "
    "Add the qualifiers, scope or conditions the passage gives so no other answer could also be right. "
    "Return only the rewritten question text."
)
PROMPT_OVERGEN = (
    "Passage: {passage}\n"
    "Question: {stem}\n"
    "Correct answer: {key}\n"
    "Write 10 plausible but INCORRECT answer options for this question. "
    "Return ONLY a JSON list of 10 strings."
)
PROMPT_TOPIC_LABEL = (
    "Give a 3-6 word course-topic label for this passage. Return ONLY the label.\n"
    "Passage: {passage}"
)
PROMPT_LLM_JUDGE = (
    "Passage: {passage}\n"
    'Statement: The answer to "{stem}" is "{option}".\n'
    "Is this statement supported by the passage? Answer Yes or No."
)
PROMPT_ANSWER_WITH_PASSAGE = (
    "Passage: {passage}\nQuestion: {stem}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\n"
    "Answer with a single letter (A, B, C or D)."
)
PROMPT_ANSWER_NO_PASSAGE = (
    "Question: {stem}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\n"
    "Answer with a single letter (A, B, C or D)."
)
HYPOTHESIS_TEMPLATE = "The answer to {stem} is {option}."
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Config:
    """All tunable settings. Smoke mode (RC_SMOKE_TEST=1) shrinks EVERY stage and skips none."""

    def __init__(self) -> None:
        # ---- ITEM 2: the seed list, the deleted clock-conditional knobs and the new declared knobs.
        # DELETED here and never to return, because each one made a component conditional on the
        # clock or on another arm's result, which the No-Optional-Component rule forbids:
        #   h4_always_run, force_h4                  -> whether the two rerank arms ran at all
        #   proxy_reduce_elapsed_fraction            -> the proxy's sample count, by elapsed time
        #   functionality_samples_reduced            -> the reduced count that lever selected
        #   replication_elapsed_fraction_limit       -> whether the replication ran, by elapsed time
        #   retrieval_widen_k                        -> a per-seed widening that changed the evidence
        # models
        self.generator_id = MODEL_IDS["generator"]
        self.nli_id = MODEL_IDS["nli"]
        self.embedder_id = MODEL_IDS["embedder"]
        # the Hub revision each checkpoint loads at; None is the default branch and is recorded as
        # null, with the resolved snapshot sha published beside it in the payload
        self.generator_revision = MODEL_REVISIONS["generator"]
        self.nli_revision = MODEL_REVISIONS["nli"]
        self.embedder_revision = MODEL_REVISIONS["embedder"]
        self.load_4bit_nf4 = True          # opportunistic: used only if bitsandbytes is importable
        self.device = detect_device()
        self.data_root = resolve_data_root()
        # data / seeds  (BUG-183: exactly three seeds on a measured run, exactly one under smoke)
        self.seeds = [0, 1, 2]
        self.items_per_seed = 100
        self.fold_seed = 12345
        self.min_support_tokens = 40
        self.min_corpus_tokens = 8
        self.calibration_pairs = 1600
        # The bound the calibration set was ACTUALLY built under. None until
        # analysis.apply_uniform_reduction lowers calibration_pairs, which is its only writer:
        # the set is built before the pilot and the lever lowers the knob after it, so
        # executed_design_record must compare the built size against this value and not against
        # the reduced knob. DECLARED here rather than attached by setattr at run time, because a
        # design value the payload records must be visible to a reader of this class and to the
        # checker -- an attribute that exists only after a side effect is one no tool can audit.
        self.calibration_pairs_designed: Optional[int] = None
        self.n_overlap_deciles = 10
        self.passage_char_cap = 1500
        self.retrieval_top_k = 1
        self.min_cell_items = 30
        self.obqa_replication_items = 200
        # ITEM 2 / ITEM 31: the corpus cap. None on a measured run (the whole kept corpus is
        # embedded); 200 under smoke, applied by data.apply_corpus_cap BEFORE the index embeds.
        # ITEM 3 records it as null rather than dropping it.
        self.corpus_passage_cap: Optional[int] = None
        # ITEM 2: the declared knobs for the seed floor, the budget lever and the two control arms
        self.secondary_min_seeds = 2
        self.min_reduction_fraction = 0.25
        self.planted_slot = 1
        self.planted_fraction = 0.5
        self.null_retention_fraction = 0.5   # equal to retention_fraction by construction
        self.detection_ratio_gate = 0.8
        # generation
        self.gen_batch_size = 4
        self.judge_batch_size = 8
        self.gen_max_new_tokens = 160
        self.overgen_max_new_tokens = 200
        self.rewrite_max_new_tokens = 64
        self.topic_max_new_tokens = 16
        self.n_overgen_candidates = 10
        self.parse_failure_abort_rate = 0.25
        # judge
        self.nli_batch_size = 32
        self.nli_min_batch_size = 2
        # the fixed sequence budget behind data.nli_pair_token_plan: the hypothesis is capped and
        # the premise gets the FIXED remainder, so the specificity rewrite is judged against the
        # same passage prefix as the original stem
        self.nli_max_length = 512
        self.nli_hypothesis_token_budget = 64
        self.entail_threshold = 0.5
        self.judge_precision_gate = 0.6
        self.judge_min_top_decile_pairs = 5    # upper bound on the calibration top-decile floor
        self.low_vram_bytes = 1_073_741_824
        self.nli_gpu_min_free_bytes = 2_500_000_000
        self.embed_batch_size = 16
        self.judge_nan_abort_fraction = 0.05   # abort a scoring pass if > 5% of judge rows are NaN
        # filters
        self.retention_fraction = 0.5
        self.tau_sweep = [0.3, 0.5, 0.7, 0.9]
        self.lambda_div = 0.1
        self.tie_eps = 0.02
        self.quartile_ratio_smoothing = 0.01   # additive smoothing of the similarity-quartile ratio
        # functionality proxy (one sample count; no clock-conditional reduced variant)
        self.functionality_samples_per_context = 20
        self.functionality_temperature = 1.0
        self.functionality_top_p = 0.95
        # statistics
        self.bootstrap_resamples = 1000
        self.ci_level = 0.95
        # diagnostic gates. These are REPORTED verdicts only: no arm's execution reads one, so no
        # component is conditional on another arm's result.
        self.h1_base_rate_gate = 0.04
        self.h4_base_rate_gate = 0.08
        self.h4_distractor_share_gate = 0.4
        self.grounding_attribution_delta = 0.05
        # budget
        self.time_budget_s = TIME_BUDGET_SEC
        self.budget_margin = 0.20
        # Generation-equivalents per item across all ten arms, MEASURED rather than assumed.
        # 2026-09-05: 9.5 was the arm count, but most arms rescore cached candidates instead of
        # generating, so the real cost is far lower. The completed run of that date measured
        # 86.5s per item end to end against a 39.21s pilot generation call, which is 2.21
        # generation-equivalents. The old value overshot 4.3x, cut the design to a quarter, and
        # the run then finished in a quarter of its budget: a smaller study for no reason.
        self.pilot_cost_multiplier = 2.21
        self.results_path = "results.json"
        self.tensor_dir = "tensors"
        self.smoke = SMOKE
        if SMOKE:
            # ITEM 31: every stage shrinks, nothing is skipped. The 2026-09-04 smoke run embedded
            # 13,461 corpus passages and calibrated 120 pairs before one arm ran, and timed out.
            self.seeds = [0]
            self.items_per_seed = 3
            self.corpus_passage_cap = 200
            self.calibration_pairs = 20
            self.obqa_replication_items = 3
            self.functionality_samples_per_context = 2
            self.bootstrap_resamples = 50
            self.min_cell_items = 1
            self.parse_failure_abort_rate = 1.01

    def usable_budget_s(self) -> float:
        """Wall-clock seconds available after the margin; always a finite positive float."""
        return self.time_budget_s * (1.0 - self.budget_margin)

    def hyperparameters(self) -> Dict[str, Any]:
        """ITEM 3: every declared knob, with a None value recorded as null rather than dropped.

        The old filter silently omitted every None-valued knob, so corpus_passage_cap -- None on a
        measured run -- never reached the payload and could not be audited. calibration_pairs_designed
        is None on a run the budget lever did not touch, and that null is itself the record that no
        reduction happened."""
        out: Dict[str, Any] = {}
        for k, v in vars(self).items():
            if v is None or isinstance(v, (int, float, str, bool, list)):
                out[k] = v
        return out


def model_revision(key: str) -> Optional[str]:
    """The pinned Hub revision of ONE model, addressed by its ROLE or by its repo id.

    MODEL_REVISIONS is DECLARED by role ("generator", "nli", "embedder") because that is what
    Config binds; but models._local_kw holds only the repo id at the point of the lookup, and a
    plain MODEL_REVISIONS.get(repo_id) missed on every call and returned None. Every pin was
    therefore silently inert: the weights loaded from the repo's default branch while the payload
    recorded pinned_revision: null, so nothing in the run said the pin had been dropped and the
    numbers were attributed to weights that had not produced them.

    This resolver is the ONE place the two names meet, so there is still exactly one pin value per
    model and no second, repo-id-keyed copy that could drift from it. It RAISES on a name that is
    neither a role nor a repo id, because a typo must never be indistinguishable from "no pin set".
    """
    name = str(key)
    if name in MODEL_REVISIONS:
        return MODEL_REVISIONS[name]
    for role, repo_id in MODEL_IDS.items():
        if repo_id == name:
            return MODEL_REVISIONS[role]
    raise KeyError(f"model_revision: '{name}' is neither a model role {sorted(MODEL_REVISIONS)} "
                   f"nor a repo id {sorted(MODEL_IDS.values())}; a pin can only be read under a "
                   f"name the registry declares")


def _repo_cache_dir(repo_id: str) -> str:
    """The `models--org--name` directory of repo_id inside the HF hub cache.

    The refs and the snapshots both hang off this one directory, so both readers derive their
    paths from here and cannot drift onto different repos."""
    return os.path.join(hf_hub_cache_root(), "models--" + repo_id.replace("/", "--"))


def _read_ref_file(repo_id: str, ref: str):
    """The PAIR (commit sha or None, the absolute path that was consulted).

    This is the file a local_files_only load actually follows: the HF cache writes the commit a
    branch or tag currently points at into `models--<repo>/refs/<ref>`, and the loader reads it to
    choose a snapshot directory. Reading it is the difference between recording the weights that
    produced the numbers and guessing at them."""
    path = os.path.join(_repo_cache_dir(repo_id), "refs", str(ref))
    if not os.path.isfile(path):
        return None, path
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read().strip()
    except OSError as e:
        print(f"[config] WARNING: could not read the ref {path} ({type(e).__name__}: {e}); "
              f"the resolved revision falls back to the newest cached snapshot")
        return None, path
    return (text or None), path


def _newest_snapshot_by_mtime(snaps):
    """The name of the most recently written snapshot directory, or None for an empty list.

    The LAST resort only. A caller that reaches this is guessing, and every record built from it
    says so under resolution='mtime_fallback'; the getmtime failure that used to be swallowed to
    0.0 is logged here, because a silent 0.0 turns one unreadable directory into a wrong answer
    for the whole repo."""
    newest_name = None
    newest_mtime = None
    for name, path in snaps:
        try:
            mtime = os.path.getmtime(path)
        except OSError as e:
            print(f"[config] WARNING: could not stat the cached snapshot {path} "
                  f"({type(e).__name__}: {e}); it is ranked last for the mtime fallback")
            mtime = 0.0
        if newest_mtime is None or mtime > newest_mtime:
            newest_name, newest_mtime = name, mtime
    return newest_name


def snapshot_revision_record(repo_id: str) -> Dict[str, Any]:
    """How the loaded commit was determined, not just what it was.

    Resolved in the order the LOADER resolves it, so the payload names the weights that produced
    the numbers:
      1. pinned_revision -- MODEL_REVISIONS pins a commit that is cached, so that is what loads;
      2. ref_file        -- models--<repo>/refs/<pin or main>, the file a local_files_only load
                            follows when no commit is pinned;
      3. mtime_fallback  -- only when that file is absent or unreadable, and LABELLED as a guess.

    The old code went straight to (3): with two complete snapshots cached -- a re-pull after an
    upstream commit, then a cache scan touching the older directory -- it reported the commit the
    load did NOT follow, and nothing in the payload said the value was a guess. complete_snapshots
    lists every cached candidate so a reader can see when the question was ambiguous at all."""
    snaps = _complete_snapshots(repo_id)
    names = sorted(n for n, _ in snaps)
    rec: Dict[str, Any] = {"repo_id": repo_id, "resolved_revision": None,
                           "resolution": "uncached", "ref_read": None,
                           "ref_names_a_complete_snapshot": None,
                           "n_complete_snapshots": len(names), "complete_snapshots": names}
    if not snaps:
        return rec

    # model_revision RAISES on an undeclared name, which is right for a pin READ; this is cache
    # introspection, so a repo id the registry does not declare simply has no pin.
    pin = None
    if repo_id in MODEL_REVISIONS or repo_id in set(MODEL_IDS.values()):
        pin = model_revision(repo_id)

    if pin and str(pin) in names:
        rec["resolved_revision"] = str(pin)
        rec["resolution"] = "pinned_revision"
        rec["ref_names_a_complete_snapshot"] = True
        return rec

    ref = str(pin) if pin else "main"
    sha, ref_path = _read_ref_file(repo_id, ref)
    rec["ref_read"] = ref_path
    if sha:
        rec["resolved_revision"] = sha
        rec["resolution"] = "ref_file"
        rec["ref_names_a_complete_snapshot"] = bool(sha in names)
        if sha not in names:
            print(f"[config] WARNING: {repo_id} ref {ref_path} names {sha}, which is not among the "
                  f"complete cached snapshots {names}; the recorded revision is what the ref says "
                  f"and the mismatch is published beside it")
        return rec

    rec["resolved_revision"] = _newest_snapshot_by_mtime(snaps)
    rec["resolution"] = "mtime_fallback"
    if len(names) > 1:
        print(f"[config] WARNING: {repo_id} has {len(names)} complete cached snapshots {names} and "
              f"no readable ref at {ref_path}; the recorded revision "
              f"{rec['resolved_revision']} is the newest by mtime and is a GUESS")
    return rec
