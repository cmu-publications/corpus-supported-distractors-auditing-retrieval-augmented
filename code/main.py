"""
Corpus-supported distractors (CSD) in retrieval-augmented quiz generation.

(a) DATASET: SciQ (allenai/sciq) via the `datasets` library from the shared data root. The course
    corpus is every deduplicated SciQ support passage (train+validation+test) plus the OpenBookQA
    'additional' core facts. Evaluated items come from disjoint folds of SciQ TEST passages, one
    fold per seed (seeds 0/1/2; one seed in smoke). Judge calibration uses labelled pairs from SciQ
    VALIDATION gold items stratified by lexical-overlap decile, so no calibration item is ever an
    evaluated item. OpenBookQA test items replicate the H1 table.
(b) REGIMES: enumeration_density (median split of the list-like sentence fraction) x retrieval
    coverage (the top-1 retrieved passage is the gold passage).
(c) MODELS (frozen, nothing trained): Qwen/Qwen2.5-3B-Instruct as generator, stem rewriter, LLM
    judge and functionality-proxy answerer; cross-encoder/nli-deberta-v3-base as the NLI judge
    (3-way softmax, float32); BAAI/bge-small-en-v1.5 for dense retrieval and key similarity.
(d) PROTOCOL: greedy decoding with one repair retry; per seed ALL TEN arms run on the one grounded
    pool with NO gate -- the plan's eight conditions plus the two mandated control arms. Filters
    retain the top 50% by score and sweep tau in {0.3,0.5,0.7,0.9}; the rerankers select 3 of 10
    shared candidates; null_random_drop is a seeded uniform drop reading no judge tensor; and
    positive_control_planted overwrites one distractor slot with the item's own key and rescores
    with the real judge.
(e) EVALUATION: per-distractor flags from the validated judge against the retrieved passage.
    corpus-supported = P(entail) >= 0.5 under the original stem; distractor-caused = also >= 0.5
    under the specificity rewrite; else stem-caused. primary_metric is the MEAN OVER SEED IDS; the
    row-pooled rate is published separately as primary_metric_pooled. Intervals resample whole
    (seed, item_id) items, and every arm's comparison scheme is registered before the run.
(f) UNDEFINED CELLS: an empty group has no rate. Every such cell is None and serialises as JSON
    null; the token nan never appears on stdout or in the payload.

METRIC NAME: primary_metric (= distractor_caused_csd_rate)
DIRECTION: lower is better (minimize)
UNITS: fraction of distractor slots in the evaluated set, 0-1
FORMULA: #distractors flagged under BOTH stems / (3 x #items)
AGGREGATION: mean over seed ids; the pooled item-clustered bootstrap rate beside it
"""
import math
import os
import random
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # line-buffer so partial progress survives a harness kill
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# read BOTH switches with os.environ.get at the top, as the smoke contract requires
ENV_SMOKE = os.environ.get("RC_SMOKE_TEST", "0")
ENV_TIME_BUDGET = os.environ.get("RC_TIME_BUDGET_SEC", "")

import numpy as np
import torch

# experiment_config is imported as a MODULE as well as by name: detect_device rebinds
# DEVICE_FALLBACK_REASON at call time (Config() runs it below, after this import block), so a
# from-import of that name would freeze this module's copy at the None it held here and every
# fallback would be reported as "no cause". It is read as experiment_config.DEVICE_FALLBACK_REASON
# at each use instead -- the rule config.py already documents.
import experiment_config
from experiment_config import (ALL_ARM_NAMES, CONDITION_NAMES, CONTROL_ARM_NAMES,
                               ENV_SMOKE_AT_IMPORT, ENV_TIME_BUDGET_AT_IMPORT, MODEL_IDS, MODEL_REVISIONS, Config,
                               snapshot_revision)
import analysis
from analysis import (NLI_REFERENCE_RATES_KEY, NLI_REF_TABLE_KEY, PRIMARY, REFERENCE_ARM, SOFT_METRICS,
                      apply_uniform_reduction, arm_failure_record, assert_no_nonfinite, budget_insufficient_message,
                      check_environment_switches, check_payload_contract, check_seed_design, dumps,
                      enforce_rates_contract, executed_design_record, flag_safe, fmt, library_version_record,
                      plan_reduction, seed_metric_lines, summary_lines, truncate_folds)
from data import (CalibrationSetBuilder, CorpusBuilder, QuizItem, SeedAbort, SeedFolds, obqa_replication_indices,
                  obqa_replication_pool)
from methods import (ContradictionPolarityItemFilter, FlagDecomposer, FunctionalityProxy, GroundedQuizGenerator,
                     ItemScorer, LLMJudgeFaithfulnessFilter, NotEntailPolarityItemFilter, NullRandomDropFilter,
                     OvergenerateContradictionReranker, OvergenerateKeySimilarityReranker, PerOptionEntailmentFilter,
                     PositiveControlPlanted, TopicOnlyQuizGenerator, label_topics,
                     label_topics_with_fallback_mask, passage_free_fallback_label)
from models import JudgeCalibrator, RetrievalIndex, free_vram_bytes, load_all_models

CFG = Config()
SEEDS = list(CFG.seeds)
REWRITE_NOOP_FLAG_RATE = 0.20
TOPIC_FALLBACK_FLAG_RATE = 0.10
STOP_FRACTION = 0.8


# ============================================================== ITEM 41: the harness contract
class _StandInHarness:
    """Used only when experiment_harness is not importable.

    It enforces the SAME FOUR ABORTS as the real harness -- a repeated seed, an arm set differing
    from the expected one, a non-finite number anywhere, and a write no recorded seed stands behind.
    A permissive stand-in is how a contract breach reaches a GPU run: the local harness accepted a
    write between calibration and the seed loop, and only the real one rejected it hours later."""

    def __init__(self, time_budget: float) -> None:
        self.t0 = time.time()
        self.time_budget = float(time_budget)
        self.metrics: Dict[str, float] = {}
        self.recorded_seeds: List[int] = []
        self.payload: Optional[Dict[str, Any]] = None

    def should_stop(self) -> bool:
        return (time.time() - self.t0) > STOP_FRACTION * self.time_budget

    def check_value(self, value: Any, name: str) -> bool:
        del name
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    def report_metric(self, name: str, value: Any) -> None:
        if not self.check_value(value, name):
            raise ValueError(f"metric '{name}' is not finite: {value!r}")
        self.metrics[name] = float(value)

    def record_seed(self, seed: int, conditions: Dict[str, Any],
                    expected_conditions: Optional[Sequence[str]] = None) -> None:
        if int(seed) in self.recorded_seeds:
            raise ValueError(f"seed {seed} was already recorded; each seed is recorded exactly once")
        if expected_conditions is not None and set(conditions) != set(expected_conditions):
            missing = sorted(set(expected_conditions) - set(conditions))
            extra = sorted(set(conditions) - set(expected_conditions))
            raise ValueError(f"seed {seed} condition set differs from the plan: missing={missing} extra={extra}")
        assert_no_nonfinite(conditions, path=f"$.seed_{seed}")
        self.recorded_seeds.append(int(seed))

    def write_results(self, extra: Dict[str, Any]) -> None:
        analysis.enforce_seed_recorded_before_write(len(self.recorded_seeds))
        self.payload = dict(extra)

    def finalize(self) -> None:
        return None


def get_harness(time_budget: float) -> Tuple[Any, str, List[str]]:
    """The harness object, its source name and its member list.

    Falls back to the stand-in ONLY on ImportError; a module that exists but exposes no factory is
    a real problem and raises rather than silently degrading the contract."""
    try:
        import experiment_harness
    except ImportError as e:
        print(f"[harness] experiment_harness is not importable ({type(e).__name__}: {e}); "
              f"using the stand-in, which enforces the same four aborts")
        inner = _StandInHarness(time_budget)
        return inner, "stand-in", sorted(n for n in dir(inner) if not n.startswith("_"))
    factory = getattr(experiment_harness, "get_harness", None)
    if callable(factory):
        inner = factory(time_budget)
    else:
        cls = getattr(experiment_harness, "ExperimentHarness", None)
        if cls is None:
            raise RuntimeError("experiment_harness exposes neither get_harness nor ExperimentHarness; "
                               "the results contract cannot be honoured")
        inner = cls(time_budget=time_budget)
    return inner, "experiment_harness", sorted(n for n in dir(inner) if not n.startswith("_"))


class HarnessAdapter:
    """A uniform surface over whatever the factory returned.

    Every checking fallback is the STAND-IN's, never a weaker one: if the inner harness lacks
    record_seed or write_results, the same four aborts still apply."""

    def __init__(self, inner: Any, source: str, members: Sequence[str]) -> None:
        self.inner = inner
        self.source = source
        self.members = list(members)
        self.fallback = _StandInHarness(getattr(inner, "time_budget", 0.0) or 0.0)

    def should_stop(self) -> bool:
        fn = getattr(self.inner, "should_stop", None)
        return bool(fn()) if callable(fn) else False

    def check_value(self, value: Any, name: str) -> bool:
        if not self.fallback.check_value(value, name):
            return False
        fn = getattr(self.inner, "check_value", None)
        return bool(fn(value, name)) if callable(fn) else True

    def report_metric(self, name: str, value: Any) -> None:
        self.fallback.report_metric(name, value)          # raises on a non-finite value
        fn = getattr(self.inner, "report_metric", None)
        if callable(fn):
            fn(name, float(value))

    def record_seed(self, seed: int, conditions: Dict[str, Any],
                    expected_conditions: Optional[Sequence[str]] = None) -> None:
        self.fallback.record_seed(seed, conditions, expected_conditions)
        fn = getattr(self.inner, "record_seed", None)
        if callable(fn):
            fn(seed, conditions, expected_conditions=expected_conditions)

    def write_results(self, extra: Dict[str, Any]) -> None:
        self.fallback.write_results(extra)                 # raises the contract message when unbacked
        fn = getattr(self.inner, "write_results", None)
        if callable(fn):
            fn(extra)
        elif self.source != "stand-in":
            raise RuntimeError("the harness exposes no write_results; results may only be written "
                               "through the harness")

    def finalize(self) -> None:
        fn = getattr(self.inner, "finalize", None)
        if callable(fn):
            fn()


def write_payload(harness: HarnessAdapter, extra: Dict[str, Any]) -> None:
    """The SINGLE write path: the payload contract, then every number, then the harness.

    This module never calls json.dump on results and never opens the results file."""
    check_payload_contract(extra, len(harness.fallback.recorded_seeds))
    assert_no_nonfinite(extra, path="$")
    harness.write_results(extra)


# ============================================================== provenance and seeding
def set_all_seeds(seed: int) -> None:
    """Seed Python, numpy and torch from the run seed so a rerun reproduces the record."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_revision_record(cfg: Any) -> Dict[str, Any]:
    """Per-model repo id, pinned revision and resolved snapshot sha, so a run names the weights
    that produced its numbers -- and the PATH by which that sha was resolved.

    MODEL_REVISIONS is keyed by ROLE and the role is the loop variable here, so the table is read
    under its declared key and no resolution is needed. The old MODEL_REVISIONS.get(repo_id)
    missed on every model and wrote pinned_revision: null even when a pin was set, so a run that
    loaded the default branch instead of the pinned commit left no trace of the substitution.

    resolved_revision now comes from snapshot_revision_record, which follows the pin, then the
    refs file a local_files_only load reads, and only then the newest-by-mtime guess -- labelled
    as a guess. With MODEL_REVISIONS unpinned this field is the run's only artifact identity, and
    it used to be picked by mtime, which can name a commit the load did not follow whenever two
    complete snapshots are cached.

    device_fallback_reason is read as a MODULE ATTRIBUTE, never through a from-import: detect_device
    rebinds that global when the probe fails, long after this module was imported, so a captured
    copy would report null for every fallback -- a payload asserting a definite "no cause" beside
    device: cpu, contradicting the warning detect_device printed to stdout."""
    from experiment_config import snapshot_revision_record

    out: Dict[str, Any] = {}
    for role, repo_id in MODEL_IDS.items():
        rec = snapshot_revision_record(repo_id)
        out[role] = {"repo_id": repo_id, "pinned_revision": MODEL_REVISIONS.get(role),
                     "resolved_revision": rec["resolved_revision"],
                     "revision_resolution": rec["resolution"],
                     "revision_ref_read": rec["ref_read"],
                     "ref_names_a_complete_snapshot": rec["ref_names_a_complete_snapshot"],
                     "n_complete_snapshots": rec["n_complete_snapshots"],
                     "complete_snapshots": list(rec["complete_snapshots"])}
    out["device"] = cfg.device
    out["device_fallback_reason"] = experiment_config.DEVICE_FALLBACK_REASON
    return out


def transformers_version() -> str:
    """The installed transformers version; raises ImportError when absent, which
    library_version_record turns into a recorded reason rather than a bare 'unknown'."""
    import transformers
    return str(transformers.__version__)


class BudgetGuard:
    """Elapsed-time bookkeeping ONLY.

    It records checkpoints and never decides whether an arm, proxy, sweep, seed or replication
    runs -- the No-Optional-Component rule forbids a clock-conditional component."""

    def __init__(self, cfg: Any, harness: HarnessAdapter) -> None:
        self.cfg = cfg
        self.harness = harness
        self.t0 = time.time()
        self.usable = cfg.usable_budget_s()
        self.checkpoints: Dict[str, float] = {}

    def elapsed(self) -> float:
        return time.time() - self.t0

    def elapsed_fraction(self) -> float:
        return self.elapsed() / max(1.0, self.usable)

    def checkpoint(self, name: str) -> None:
        self.checkpoints[name] = round(self.elapsed(), 1)
        print(f"[BUDGET] {name} elapsed={self.elapsed():.0f}s "
              f"frac_of_usable={self.elapsed_fraction():.2f}")


# ============================================================== the per-seed pipeline
def closed_book_topics(corpus: Any, pids: Sequence[int], topics: Sequence[str],
                       used_fallback: Sequence[bool]) -> List[str]:
    """The closed-book label list DERIVED from the grounded one produced by a single generation
    pass: the same label wherever the model gave a usable one, and the passage-free placeholder at
    exactly the positions where the fallback fired.

    The grounded arm already has the passage in its prompt, so its passage-prefix fallback leaks
    nothing there; feeding that prefix into the closed-book prompt would leak the gold passage into
    the one control whose definition is that it has no passage. Regenerating the whole list to swap
    that one string cost a second generation per item and, if an OOM changed the batch size between
    the passes, could return a different label -- prompting the control with a topic other than the
    one retrieval used."""
    return [passage_free_fallback_label(corpus, p)[:80] if fb else t
            for t, fb, p in zip(topics, used_fallback, pids)]


def score_pools(scorer: Any, items_g: Sequence[QuizItem], items_t: Sequence[QuizItem],
                evaluator: str) -> None:
    """Rewrite the stems and score both pools ONCE with every judge the evaluator needs."""
    both = list(items_g) + list(items_t)
    scorer.rewrite_stems(both)
    scorer.score_nli(both)
    scorer.score_similarity_and_overlap(both)
    scorer.score_llm_judge(list(items_g), "orig")          # the LLM-judge filter always needs this
    if evaluator == "llm":
        scorer.score_llm_judge(list(items_g), "rewrite")
        scorer.score_llm_judge(list(items_t), "orig")
        scorer.score_llm_judge(list(items_t), "rewrite")


def save_tensors(cfg: Any, seed: int, arm: str, items: Sequence[QuizItem]) -> None:
    """Write the per-seed judge tensor dumps; skip with a NAMED reason when a tensor is absent
    rather than raising a shape error that would lose the seed."""
    if not items:
        return
    os.makedirs(cfg.tensor_dir, exist_ok=True)
    for field in ("nli_orig", "nli_rewrite", "llm_pyes"):
        vals = [getattr(it, field) for it in items]
        if any(v is None for v in vals):
            n_missing = sum(1 for v in vals if v is None)
            print(f"[tensors] seed={seed} arm={arm} field={field} not dumped: "
                  f"{n_missing}/{len(vals)} items carry no tensor")
            continue
        np.save(os.path.join(cfg.tensor_dir, f"seed{seed}_{arm}_{field}.npy"),
                np.stack([np.asarray(v) for v in vals]))


def ablation_checks(items_g: Sequence[QuizItem], cfg: Any, gen_g: Any, gen_t: Any) -> None:
    """One ABLATION_CHECK line per filter pair and one for the generators: the arms must rank the
    same pool differently, or the ablation the plan turns on is vacuous."""
    dec = FlagDecomposer(cfg, "nli")
    scored = {
        "per_option_entailment_filter": PerOptionEntailmentFilter(cfg, dec).score_pool(items_g),
        "contradiction_polarity_item_filter": ContradictionPolarityItemFilter(cfg, dec).score_pool(items_g),
        "not_entail_polarity_item_filter": NotEntailPolarityItemFilter(cfg, dec).score_pool(items_g),
    }
    if items_g and items_g[0].llm_pyes is not None:
        scored["llm_judge_faithfulness_filter"] = LLMJudgeFaithfulnessFilter(cfg, dec).score_pool(items_g)
    names = list(scored)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            differ = not np.allclose(scored[names[i]], scored[names[j]], atol=1e-6)
            print(f"ABLATION_CHECK: {names[i]} vs {names[j]} outputs_differ={differ}")
    print(f"ABLATION_CHECK: {REFERENCE_ARM} vs ungrounded_topic_only_control "
          f"prompt_uses_passage={gen_g.prompt_uses_passage()}/{gen_t.prompt_uses_passage()} "
          f"outputs_differ={gen_g.prompt_uses_passage() != gen_t.prompt_uses_passage()}")


def build_arm_objects(cfg: Any, llm: Any, nli: Any, embedder: Any, decomposer: Any,
                      decomposer_nli: Any, proxy: Any) -> Dict[str, Any]:
    """The six post-generation plan arms plus the TWO CONTROL ARMS, keyed by arm name.

    Every entry is constructed on every seed and no gate can omit one. The two rerank arms share
    the candidate cache on the items, so the ablation reranks exactly the same ten candidates."""
    return {
        "per_option_entailment_filter": PerOptionEntailmentFilter(cfg, decomposer, proxy),
        # the LLM-judge arm is always flagged by the NLI decomposer (one judge on both sides)
        "llm_judge_faithfulness_filter": LLMJudgeFaithfulnessFilter(cfg, decomposer_nli, proxy),
        "contradiction_polarity_item_filter": ContradictionPolarityItemFilter(cfg, decomposer, proxy),
        "not_entail_polarity_item_filter": NotEntailPolarityItemFilter(cfg, decomposer, proxy),
        "overgenerate_rerank_by_contradiction":
            OvergenerateContradictionReranker(cfg, llm, nli, embedder, decomposer, proxy),
        "overgenerate_rerank_by_key_similarity":
            OvergenerateKeySimilarityReranker(cfg, llm, nli, embedder, decomposer, proxy),
        "null_random_drop": NullRandomDropFilter(cfg, decomposer, proxy),
        "positive_control_planted": PositiveControlPlanted(cfg, llm, nli, decomposer, proxy),
    }


# ITEM 26: every arm runs, no gate, and a failure names its cause
def run_arm_sequence(arm_objects: Dict[str, Any], items_g: Sequence[QuizItem], seed: int,
                     res: Dict[str, Any], parse_failure_rate: float) -> None:
    """Run every arm in registry order with NO gate.

    A data-dependent failure is recorded with its type, message and traceback tail; every class in
    analysis.programming_errors() is RE-RAISED, because a renamed method, a missing symbol or a
    tuple consumed as a list is a defect in this file and filing it as an arm failure buries a bug
    that recurs identically on every seed.

    The tuple is read through the analysis MODULE, which main imports at the top: a definition
    placed in this file below the __main__ guard is not bound while main() runs, so the handler
    raised NameError on the first exception, lost the real cause and ended the run with no
    condition in the results at all."""
    for name in ALL_ARM_NAMES:
        obj = arm_objects.get(name)
        if obj is None:
            continue                                       # the two generator arms report elsewhere
        try:
            record = obj.report(list(items_g), seed)
            record["parse_failure_rate"] = parse_failure_rate
            res["conditions"][name] = record
            res["rows"][name] = obj.decomposer.flag_rows(obj.evaluated_items(), name)
        except analysis.programming_errors():
            raise                                          # a programming error, never an arm failure
        except Exception as e:                             # data-dependent: recorded with its cause
            traceback.print_exc()
            print(f"ARM_FAILED: {name} seed={seed} {type(e).__name__}: {e}")
            res["conditions"][name] = arm_failure_record(e)
            res["rows"][name] = analysis.empty_rows(name)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ITEM 32 (per-seed half): labelled lines over every arm, NA for an undefined value
def report_seed_metrics(res: Dict[str, Any], seed: int, harness: HarnessAdapter) -> None:
    """Print every labelled per-seed line over all ten arm names and report each FINITE value."""
    for line in seed_metric_lines(res.get("conditions", {}), seed, ALL_ARM_NAMES):
        print(line)
    for name in ALL_ARM_NAMES:
        c = res.get("conditions", {}).get(name, {}) or {}
        value = c.get(PRIMARY)
        if value is not None and harness.check_value(value, PRIMARY):
            harness.report_metric(f"{name}/seed_{seed}/{PRIMARY}", value)
        for soft in SOFT_METRICS:
            sv = c.get(soft)
            if sv is not None and harness.check_value(sv, soft):
                harness.report_metric(f"{name}/seed_{seed}/{soft}", sv)


# ITEM 27: one record_seed per seed, on the complete arm map, before that seed's write
def harness_seed_record(harness: HarnessAdapter, seed: int, res: Dict[str, Any]) -> Dict[str, Any]:
    """The complete ten-arm map after enforcing the rates contract on each arm.

    An absent arm is substituted with a status-only record, which the rates contract exempts, so
    the arm set the harness sees always equals ALL_ARM_NAMES."""
    conditions: Dict[str, Any] = {}
    for name in ALL_ARM_NAMES:
        record = res.get("conditions", {}).get(name)
        if record is None:
            record = {"status": f"absent: arm produced no record on seed {seed}"}
        enforce_rates_contract(seed, name, record)
        conditions[name] = record
    harness.record_seed(seed, conditions, expected_conditions=ALL_ARM_NAMES)
    return conditions


# ITEM 28: one pool per seed, the NLI reference under the SINGLE shared key, then all ten arms
def run_seed(cfg: Any, seed_index: int, seed: int, llm: Any, nli: Any, embedder: Any, corpus: Any,
             index: Any, folds: Any, evaluator: str, guard: BudgetGuard, harness: HarnessAdapter,
             flags: List[str], proxy: Any, scorer: Any, do_checks: bool = False) -> Dict[str, Any]:
    """One seed's record. Takes the SeedFolds object because it alone calls folds.fold()."""
    set_all_seeds(seed)
    res: Dict[str, Any] = {"seed": seed, "status": "ok", "conditions": {}, "rows": {}, "flags": []}
    entries = folds.fold(seed_index)
    gold_pids = [e["gold_pid"] for e in entries]

    # ITEM 39 consumer: ONE topic-labelling pass for BOTH generator arms. The grounded arm may use
    # the passage-prefix fallback; the closed-book arm gets a passage-free placeholder at exactly
    # the positions where the fallback fired, because the prefix would leak the gold passage into
    # the one control whose definition is that it has no passage. A second label_topics pass
    # repeated items_per_seed generations that the TIME_ESTIMATE arithmetic does not account for.
    topics, n_topic_fallback, used_fallback = label_topics_with_fallback_mask(llm, corpus, gold_pids, cfg)
    topics_cb = closed_book_topics(corpus, gold_pids, topics, used_fallback)
    topic_fallback_rate = n_topic_fallback / len(topics) if topics else None
    res["topic_label_fallback_count"] = n_topic_fallback
    # the closed-book list is derived from the SAME mask, so the two counts are equal by construction
    res["topic_label_fallback_count_closed_book"] = n_topic_fallback
    res["topic_label_fallback_rate"] = topic_fallback_rate
    print(f"[seed {seed}] topic_label_fallback_rate={fmt(topic_fallback_rate, 3)} "
          f"({n_topic_fallback}/{len(topics)} labels used the fallback)")
    if topic_fallback_rate is not None and topic_fallback_rate > TOPIC_FALLBACK_FLAG_RATE:
        msg = flag_safe(f"seed {seed}: {topic_fallback_rate:.2f} of topic labels used the fallback "
                        f"label, so retrieval coverage may be inflated")
        res["flags"].append(msg)
        flags.append(msg)
    guard.checkpoint(f"topics_seed{seed}")

    # ITEM 35 consumer: ONE fixed retrieval depth on every seed; there is no widening branch, so
    # the evidence never changes between seeds.
    pids = [index.retrieve_for_topic(t) for t in topics]
    covered = sum(1 for p, g in zip(pids, gold_pids) if int(p) == int(g))
    coverage = covered / len(pids) if pids else None
    res["retrieval_coverage_rate"] = coverage
    res["retrieval_top_k"] = int(cfg.retrieval_top_k)
    print(f"[seed {seed}] retrieval_coverage_rate={fmt(coverage, 3)} k={cfg.retrieval_top_k}")

    gen_g = GroundedQuizGenerator(cfg, llm, corpus)
    items_g = gen_g.generate_pool(seed, entries, topics, pids)
    guard.checkpoint(f"grounded_gen_seed{seed}")
    gen_t = TopicOnlyQuizGenerator(cfg, llm, corpus)
    items_t = gen_t.generate_pool(seed, entries, topics_cb, pids)
    guard.checkpoint(f"closed_book_gen_seed{seed}")
    if not items_g:
        raise SeedAbort("grounded pool is empty after parsing", stage="grounded_generation", rate=1.0)

    noops_before = scorer.rewrite_noops
    score_pools(scorer, items_g, items_t, evaluator)
    n_noop = scorer.rewrite_noops - noops_before
    n_rewritten = len(items_g) + len(items_t)
    rewrite_noop_rate = n_noop / n_rewritten if n_rewritten else None
    res["rewrite_noop_count"] = n_noop
    res["rewrite_noop_rate"] = rewrite_noop_rate
    print(f"[seed {seed}] rewrite_noop_rate={fmt(rewrite_noop_rate, 3)} ({n_noop}/{n_rewritten} stems "
          f"unchanged; for those items distractor-caused equals corpus-supported by construction)")
    if rewrite_noop_rate is not None and rewrite_noop_rate > REWRITE_NOOP_FLAG_RATE:
        msg = flag_safe(f"seed {seed}: the stem rewrite was a no-op for {rewrite_noop_rate:.2f} of "
                        f"items; the distractor-caused / stem-caused split is degenerate there")
        res["flags"].append(msg)
        flags.append(msg)
    save_tensors(cfg, seed, "grounded", items_g)
    save_tensors(cfg, seed, "closed_book", items_t)
    guard.checkpoint(f"scoring_seed{seed}")
    if do_checks:
        ablation_checks(items_g, cfg, gen_g, gen_t)

    decomposer = FlagDecomposer(cfg, evaluator)
    decomposer_nli = FlagDecomposer(cfg, "nli")

    # the two generator arms. The proxy's exclusion counts are read IMMEDIATELY after that arm's
    # own functional_rate call, because the counters describe the most recent call: a rate
    # published without them cannot say whether an imputed answerer row was excluded or counted.
    r_g = decomposer.rates(items_g)
    r_g.update({"item_yield": 1.0, "retrieval_coverage_rate": coverage,
                "parse_failure_rate": gen_g.last_parse_failure_rate,
                "rewrite_noop_rate": rewrite_noop_rate,
                "topic_label_fallback_rate": topic_fallback_rate,
                "functional_distractor_rate": proxy.functional_rate(items_g, seed)})
    r_g.update(proxy.exclusion_record())
    res["conditions"][REFERENCE_ARM] = r_g
    res["rows"][REFERENCE_ARM] = decomposer.flag_rows(items_g, REFERENCE_ARM)

    # the NLI-flagged view of the same pool, written under the ONE shared constant the aggregate
    # reads; when writer and reader drifted on this name the independent-judge arm lost its
    # seed-level reference and its enrichment was null on every run
    res[NLI_REFERENCE_RATES_KEY] = decomposer_nli.rates(items_g)
    res["rows"][NLI_REF_TABLE_KEY] = decomposer_nli.flag_rows(items_g, NLI_REF_TABLE_KEY)

    r_t = decomposer.rates(items_t)
    r_t.update({"item_yield": 1.0, "retrieval_coverage_rate": coverage,
                "parse_failure_rate": gen_t.last_parse_failure_rate,
                "key_lexically_in_retrieved_passage_rate": gen_t.key_in_passage_rate,
                "rewrite_noop_rate": rewrite_noop_rate,
                "topic_label_fallback_rate": topic_fallback_rate,
                "functional_distractor_rate": proxy.functional_rate(items_t, seed)})
    r_t.update(proxy.exclusion_record())
    res["conditions"]["ungrounded_topic_only_control"] = r_t
    res["rows"]["ungrounded_topic_only_control"] = decomposer.flag_rows(items_t,
                                                                       "ungrounded_topic_only_control")

    arm_objects = build_arm_objects(cfg, llm, nli, embedder, decomposer, decomposer_nli, proxy)
    run_arm_sequence(arm_objects, items_g, seed, res, gen_g.last_parse_failure_rate)
    guard.checkpoint(f"arms_seed{seed}")

    report_seed_metrics(res, seed, harness)
    res["items_sample"] = [it.to_record() for it in items_g[:10]]
    return res


# ============================================================== ITEM 29: the replication always runs
def run_replication(cfg: Any, llm: Any, corpus: Any, index: Any, evaluator: str, scorer: Any,
                    guard: BudgetGuard) -> Dict[str, Any]:
    """The OpenBookQA H1 replication for both generator arms."""
    pool = obqa_replication_pool(cfg, corpus)
    if not pool:
        return {"corpus": "openbookqa", "status": "pool_empty",
                "reason": "OpenBookQA was unavailable, so no replication item could be built",
                "n_items": 0}
    seed = 100
    set_all_seeds(seed)
    gold_pids = [e["gold_pid"] for e in pool]
    # ONE labelling pass here too; the closed-book list is derived from its fallback mask
    topics, n_fb, used_fallback = label_topics_with_fallback_mask(llm, corpus, gold_pids, cfg)
    topics_cb = closed_book_topics(corpus, gold_pids, topics, used_fallback)
    pids = [index.retrieve_for_topic(t) for t in topics]
    coverage = sum(1 for p, g in zip(pids, gold_pids) if int(p) == int(g)) / len(pids)
    gen_g = GroundedQuizGenerator(cfg, llm, corpus)
    gen_t = TopicOnlyQuizGenerator(cfg, llm, corpus)
    items_g = gen_g.generate_pool(seed, pool, topics, pids)
    items_t = gen_t.generate_pool(seed, pool, topics_cb, pids)
    noops_before = scorer.rewrite_noops
    score_pools(scorer, items_g, items_t, evaluator)
    n_rewritten = len(items_g) + len(items_t)
    dec = FlagDecomposer(cfg, evaluator)
    out: Dict[str, Any] = {
        "corpus": "openbookqa", "status": "ok", "n_items": len(pool),
        "retrieval_coverage_rate": coverage,
        "topic_label_fallback_rate": (n_fb / len(topics)) if topics else None,
        "rewrite_noop_rate": ((scorer.rewrite_noops - noops_before) / n_rewritten) if n_rewritten else None,
        REFERENCE_ARM: dec.rates(items_g),
        "ungrounded_topic_only_control": dec.rates(items_t),
    }
    out[REFERENCE_ARM]["parse_failure_rate"] = gen_g.last_parse_failure_rate
    out["ungrounded_topic_only_control"]["parse_failure_rate"] = gen_t.last_parse_failure_rate
    for arm in (REFERENCE_ARM, "ungrounded_topic_only_control"):
        print(f"REPLICATION openbookqa condition={arm} {PRIMARY}: {fmt(out[arm][PRIMARY])}")
    guard.checkpoint("replication_openbookqa")
    return out


def guarded_replication(cfg: Any, llm: Any, corpus: Any, index: Any, evaluator: str, scorer: Any,
                        guard: BudgetGuard) -> Dict[str, Any]:
    """The replication result, or a failure record naming the cause.

    It runs on EVERY run at the possibly reduced item count and reads no clock:
    replication_elapsed_fraction_limit made it conditional on remaining time, which the
    No-Optional-Component rule forbids. It never contributes to skipped_components.

    analysis.programming_errors() propagates for the same reason it does in run_arm_sequence: a
    missing symbol or a renamed method here would otherwise be recorded as "the replication failed
    on this data" and repeat on every run. It is read through the analysis MODULE so the tuple is
    bound before main() starts.

    arm_failure_record returns a str-valued map, so the extra keys (one of which is an int) are
    merged into a COPY declared Dict[str, Any]: updating the returned map in place both fails the
    checker and would mutate a record other readers treat as status-only."""
    try:
        return run_replication(cfg, llm, corpus, index, evaluator, scorer, guard)
    except analysis.programming_errors():
        raise
    except Exception as e:
        traceback.print_exc()
        print(f"REPLICATION_FAILED: openbookqa {type(e).__name__}: {e}")
        record: Dict[str, Any] = dict(arm_failure_record(e))
        record.update({"corpus": "openbookqa", "n_items": int(cfg.obqa_replication_items)})
        return record


# ============================================================== reporting
def report_paired(name: str, S: Dict[str, Any], agg: Dict[str, Any]) -> None:
    """One PAIRED line for an arm against its REGISTERED reference, taking every number from the
    aggregate record and recomputing none."""
    cmp_all = agg.get("comparisons_vs_unfiltered", {}).get(name)
    if not cmp_all:
        return
    boot = cmp_all.get("paired_bootstrap_items", {}) or {}
    wil = cmp_all.get("wilcoxon_seeds", {}) or {}
    common = cmp_all.get("aligned_seeds", []) or []
    ref_by_seed = (agg.get("unfiltered_pool_nli_reference", {}).get("per_seed_by_seed", {})
                   if name in analysis.NLI_REFERENCE_CONDITIONS
                   else (S.get(REFERENCE_ARM, {}) or {}).get("per_seed_by_seed", {}))
    x, ref, _ = analysis.aligned_by_seed((S.get(name, {}) or {}).get("per_seed_by_seed", {}), ref_by_seed)
    t_stat, p_val, cohen = analysis.paired_t_and_cohen(x, ref)
    mean_diff = (float(np.mean(np.asarray(x) - np.asarray(ref))) if len(x) >= 1 else None)
    ci = list(boot.get("ci", [None, None]))
    print(f"PAIRED: {name} vs {cmp_all.get('reference', REFERENCE_ARM)} seeds={common} "
          f"mean_diff={fmt(mean_diff)} t_stat={fmt(t_stat, 3)} p_value={fmt(p_val)} "
          f"cohen_d={fmt(cohen, 3)} item_bootstrap_scheme={boot.get('scheme')} "
          f"degraded={boot.get('degraded')} item_bootstrap_diff={fmt(boot.get('diff'))} "
          f"item_bootstrap_ci=[{fmt(ci[0])},{fmt(ci[1])}] "
          f"wilcoxon_p={fmt(wil.get('p'))} underpowered={wil.get('underpowered')}")


# ITEM 32: one labelled line per arm plus exactly one SUMMARY and one SUMMARY_SOFT
def report_summary(cfg: Any, agg: Dict[str, Any], harness: HarnessAdapter) -> Dict[str, float]:
    """Print the per-arm lines over ALL TEN names through fmt, then the two summary lines.

    No printed line can contain the token nan: every undefined value prints NA. The success_rate
    line is emitted by summary_lines and by NOTHING ELSE: this function used to print a second one
    per arm from len(cfg.seeds), so every arm carried two lines for one quantity that could
    disagree once the design changed under it."""
    S = agg.get("conditions", {}) or {}
    collected: Dict[str, float] = {}
    for line in summary_lines(S, ALL_ARM_NAMES):           # ends with SUMMARY: and SUMMARY_SOFT:
        print(line)
    for name in ALL_ARM_NAMES:
        c = S.get(name, {}) or {}
        value = c.get("primary_metric")
        if value is not None and harness.check_value(value, PRIMARY):
            harness.report_metric(f"{name}/{PRIMARY}_mean", value)
            collected[f"{name}_{PRIMARY}_mean"] = float(value)
        std = c.get("std")
        if std is not None and harness.check_value(std, "std"):
            collected[f"{name}_{PRIMARY}_std"] = float(std)
        for soft in SOFT_METRICS:
            sm = (c.get("secondary", {}) or {}).get(soft, {}) or {}
            if sm.get("mean") is not None and harness.check_value(sm["mean"], soft):
                harness.report_metric(f"{name}/{soft}_mean", sm["mean"])
        for key, cell in (c.get("regime_cells", {}) or {}).items():
            if cell.get("below_floor"):
                print(f"condition={name} regime={key} below_floor: n_items={cell.get('n_items', 0)} "
                      f"< {cfg.min_cell_items}")
            else:
                print(f"condition={name} regime={key} primary_metric: {fmt(cell.get(PRIMARY))} "
                      f"(n_items={cell.get('n_items', 0)})")
    return collected


def setup_corpus_and_index(cfg: Any, embedder: Any, flags: List[str]) -> Tuple[Any, Any, Any, Dict[str, Any]]:
    """The corpus, the SeedFolds object, the retrieval index and the composition record.

    The cap is applied EXACTLY ONCE, with the protected ids in hand, and always before the index
    embeds anything: the cap remaps every passage id, so indexing first would point every fold at
    the wrong passage. finalize_density fills the enumeration density without capping, so the folds
    and the replication ids are chosen from the WHOLE corpus; capping first (to get the density)
    pruned 13,461 passages to 200 with nothing protected, drew the folds from the survivors, and
    left the protected-id call with nothing to do -- which then rewrote cap_record with a no-op
    'capped: False' contradicting the flag the first call had appended.

    The folds are NOT rebuilt after the cap: a fold row holds SciQ TEST indices, not passage ids,
    and fold() resolves each one through the remapped gold_support_id, so the rows stay valid.
    Rebuilding drew a different fold from the pruned eligible list, using ids nothing had
    protected."""
    corpus = CorpusBuilder(cfg).build(cfg.data_root)
    corpus.finalize_density()                              # density only; no id moves
    folds = SeedFolds(cfg, corpus)                         # chosen from the UNCAPPED corpus
    replication_ids = obqa_replication_indices(cfg, corpus)
    protected = corpus.protected_passage_ids(folds.folds, replication_ids)
    cap_record = corpus.finalize_corpus(protected)         # the ONE cap, before any embedding
    if cap_record.get("n_protected_over_cap", 0) > 0:
        flags.append(flag_safe(f"corpus cap {cap_record['cap_requested']} is below the "
                               f"{cap_record['n_protected']} protected passages; kept "
                               f"{cap_record['cap_effective']} so no fold lost its gold passage"))
    flags.extend(flag_safe(f) for f in corpus.flags)
    flags.extend(flag_safe(f) for f in folds.flags)
    index = RetrievalIndex(cfg, corpus, embedder)
    return corpus, folds, index, corpus.composition()


def executed_design_or_error(cfg: Any, fold_rows: Any, calibration_pairs: Optional[int],
                             flags: List[str]) -> Dict[str, Any]:
    """The executed-design record, or a record naming why it could not be built.

    Calls analysis.executed_design_record with the interface sheet's three parameters. The bound
    the calibration set was built under is NOT passed here: apply_uniform_reduction recorded it on
    cfg when it lowered cfg.calibration_pairs, and executed_design_record reads it from there, so
    the built size is never compared against a knob the lever has since reduced.

    A ValueError here used to escape main() after the entire seed loop and before write_payload,
    so a design mismatch discovered at the very end discarded hours of generation and judging. The
    seeds are real measurements and must reach the payload, so the cause is RECORDED and flagged
    instead. A TypeError still propagates: a wrong fold shape is a programming error, not a data
    condition, and filing it as a record would bury a bug that recurs on every run."""
    try:
        return executed_design_record(cfg, fold_rows, calibration_pairs)
    except ValueError as e:
        print(f"EXECUTED_DESIGN_UNRECORDED: {type(e).__name__}: {e}")
        flags.append(flag_safe(f"the executed design could not be recorded "
                               f"({type(e).__name__}: {e}); every seed that ran is still written "
                               f"and this record names the cause"))
        designed = getattr(cfg, "calibration_pairs_designed", None)
        return {"error_type": type(e).__name__, "error_message": str(e),
                "n_seeds": len(cfg.seeds), "items_per_seed": int(cfg.items_per_seed),
                "obqa_replication_items": int(cfg.obqa_replication_items),
                "calibration_pairs": (None if calibration_pairs is None else int(calibration_pairs)),
                "calibration_pairs_designed": (int(cfg.calibration_pairs) if designed is None
                                               else int(designed)),
                "functionality_samples_per_context": int(cfg.functionality_samples_per_context)}


def minimal_payload(cfg: Any, harness: HarnessAdapter, flags: Sequence[str],
                    per_seed_public: Sequence[Dict[str, Any]],
                    reduced_components: Dict[str, Any], exc: BaseException) -> Dict[str, Any]:
    """Every REQUIRED_EXTRA_KEYS key plus the per-seed records, for the case where the full payload
    could not be assembled.

    skipped_components is EMPTY because nothing was skipped: the seeds ran, and this record exists
    so their measurements reach a results file instead of dying with the assembly error."""
    return {
        "metric_def": {"name": "primary_metric", "alias": PRIMARY, "direction": "minimize",
                       "estimator": analysis.PRIMARY_ESTIMATOR,
                       "units": "fraction of distractor slots (0-1)"},
        "hyperparameters": cfg.hyperparameters(),
        "seeds": SEEDS,
        "condition_names": ALL_ARM_NAMES,
        "arm_names": ALL_ARM_NAMES,
        "control_arms": CONTROL_ARM_NAMES,
        "comparison_modes": analysis.COMPARISON_MODE,
        "run_metadata": {"smoke": cfg.smoke, "harness": harness.source,
                         "payload_build_error": f"{type(exc).__name__}: {exc}"},
        "skipped_components": [],
        "reduced_components": dict(reduced_components),
        "undefined_values": "an empty group has no rate and is written as null, never NaN",
        "corpus": {},
        "backend": {"device": cfg.device, "data_root": cfg.data_root, "harness": harness.source},
        "calibration": {},
        "scientific_validity": "incomplete_payload",
        "flags": list(flags) + [flag_safe(f"the full payload could not be assembled "
                                          f"({type(exc).__name__}: {exc}); the seeds already "
                                          f"measured are written with this reduced record")],
        "per_seed": list(per_seed_public),
    }


# ============================================================== main
def main() -> None:
    cfg = CFG
    inner, source, members = get_harness(cfg.time_budget_s)
    harness = HarnessAdapter(inner, source, members)
    guard = BudgetGuard(cfg, harness)

    print("METRIC_DEF: primary_metric | direction=lower | desc=distractor_caused_csd_rate: the "
          "fraction of distractor slots the judge supports under BOTH the original stem and its "
          "specificity rewrite | estimator=mean over seed ids")
    print("REGISTERED_CONDITIONS: " + ", ".join(CONDITION_NAMES))
    print("CONTROL_ARMS: " + ", ".join(CONTROL_ARM_NAMES))
    print("REGISTERED_ARMS: " + ", ".join(ALL_ARM_NAMES))
    print(f"SEEDS: {SEEDS}")
    print(f"SMOKE_TEST: {cfg.smoke}")
    print(f"HARNESS: {harness.source}")
    print(f"DATA_ROOT: {cfg.data_root}")
    free = free_vram_bytes()
    # read through the module, never a captured copy: detect_device rebinds this global when the
    # probe fails, and Config() ran it after this module was imported
    print(f"DEVICE: {cfg.device} free_vram={'n/a' if free is None else f'{free / 1e9:.2f} GB'} "
          f"fallback_reason={experiment_config.DEVICE_FALLBACK_REASON}")

    # the two switches main read must describe the same run experiment_config recorded
    check_environment_switches(ENV_SMOKE, ENV_TIME_BUDGET, ENV_SMOKE_AT_IMPORT, ENV_TIME_BUDGET_AT_IMPORT)
    check_seed_design(SEEDS, cfg.smoke)

    # ITEM 43: four deterministic pre-flight checks BEFORE any weight load
    for line in analysis.self_tests(cfg):
        print(line)

    flags: List[str] = []
    skipped_components: List[str] = []                     # ITEM 42: empty on every valid run
    reduced_components: Dict[str, Any] = {}

    llm, nli, embedder = load_all_models(cfg)
    versions: Dict[str, str] = {}
    versions.update(library_version_record("torch", lambda: str(torch.__version__)))
    versions.update(library_version_record("numpy", lambda: str(np.__version__)))
    versions.update(library_version_record("transformers", transformers_version))
    if llm.precision != "nf4":
        flags.append(flag_safe(f"the generator loaded in {llm.precision} rather than nf4 "
                               f"({llm.quantization_error})"))

    corpus, folds, index, composition = setup_corpus_and_index(cfg, embedder, flags)
    guard.checkpoint("setup")

    calib = CalibrationSetBuilder(cfg, corpus).build()
    # the size ACTUALLY built; the bound it was built under is recorded on cfg by
    # apply_uniform_reduction below, so executed_design_record never compares this count against a
    # knob the lever has already lowered
    n_calib_built = len(calib)
    evaluator, r_nli, r_llm = JudgeCalibrator(cfg).select_evaluator(nli, llm, calib)
    print(f"judge_precision_top_overlap_decile: {fmt(r_nli.get('judge_precision_top_overlap_decile'))}")
    print(f"judge_r2_entail_on_overlap: {fmt(r_nli.get('judge_r2_entail_on_overlap'))}")
    if evaluator == "none":
        flags.append(flag_safe("no grounding judge passed calibration; every rate is UNVALIDATED "
                               "and the NLI judge is used for reporting"))
        print("WARNING: no judge reached the required precision; the rates are UNVALIDATED")
    elif evaluator == "llm":
        flags.append(flag_safe("the NLI judge failed calibration and the LLM judge was promoted; "
                               "llm_judge_faithfulness_filter is compared against the NLI-flagged pool"))
    guard.checkpoint("calibration")

    # ITEM 30: the pilot, ONE uniform fraction, and the budget refusal
    pilot_entry = folds.fold(0)[0]
    pilot_topics, _ = label_topics(llm, corpus, [pilot_entry["gold_pid"]], cfg)
    pilot_prompt = GroundedQuizGenerator(cfg, llm, corpus).build_prompt(
        pilot_topics[0], corpus.passage_text(pilot_entry["gold_pid"]))
    t0 = time.time()
    llm.generate_batch([pilot_prompt], cfg.gen_max_new_tokens)
    t_gen = time.time() - t0
    n_items = len(SEEDS) * cfg.items_per_seed
    est = (guard.elapsed() + n_items * t_gen * cfg.pilot_cost_multiplier
           + cfg.obqa_replication_items * t_gen * 3.0)
    usable = cfg.usable_budget_s()
    print(f"TIME_ESTIMATE: {est:.0f}s = elapsed {guard.elapsed():.0f}s + {n_items} items x "
          f"{t_gen:.2f}s x {cfg.pilot_cost_multiplier} arms-equivalent + "
          f"{cfg.obqa_replication_items} replication items x {t_gen:.2f}s x 3.0 "
          f"(usable budget {usable:.0f}s of {cfg.time_budget_s:.0f}s)")

    fraction = plan_reduction(est, usable, cfg.min_reduction_fraction, cfg.smoke)   # ONE value
    if fraction is None:
        print(budget_insufficient_message(est, usable, cfg.min_reduction_fraction, cfg))
        sys.exit(2)                                        # never a partial design
    reduction = apply_uniform_reduction(cfg, fraction)
    if reduction:
        reduced_components = reduction["components"]
        flags.append(flag_safe(f"every item count was lowered uniformly by fraction "
                               f"{reduction['fraction']:.3f}; no component was dropped"))
        print(f"UNIFORM_REDUCTION: fraction={reduction['fraction']:.3f} "
              f"components={sorted(reduced_components)}")
        folds.folds = truncate_folds(folds.folds, cfg.items_per_seed)

    proxy = FunctionalityProxy(cfg, llm)
    scorer = ItemScorer(cfg, llm, nli, embedder)
    st = analysis.Statistics(cfg)

    seed_runs: List[Dict[str, Any]] = []
    per_seed_public: List[Dict[str, Any]] = []
    # declared once for both branches: the failure literal alone infers a value type that excludes
    # the strings arm_failure_record adds, so the merge below would not type-check
    s: Dict[str, Any]
    for si, seed in enumerate(SEEDS):
        try:
            s = run_seed(cfg, si, seed, llm, nli, embedder, corpus, index, folds, evaluator, guard,
                         harness, flags, proxy, scorer, do_checks=(si == 0))
        except analysis.programming_errors():
            raise                                          # a defect here recurs on every seed
        except Exception as e:                             # one seed never aborts the run
            traceback.print_exc()
            print(f"SEED_FAILED: seed={seed} {type(e).__name__}: {e}")
            s = {"seed": seed, "conditions": {}, "rows": {}, "flags": []}
            s.update(arm_failure_record(e))
            if isinstance(e, SeedAbort):
                s["abort"] = e.to_record()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        seed_runs.append(s)
        harness_seed_record(harness, seed, s)              # exactly once, BEFORE the write
        per_seed_public.append({k: v for k, v in s.items() if k != "rows"})
        guard.checkpoint(f"seed{seed}_done")

    agg = analysis.aggregate(cfg, seed_runs, st, ALL_ARM_NAMES)
    gates = analysis.decide_gates(cfg, agg, evaluator)
    for w in agg.get("bootstrap_warnings", []):
        flags.append(flag_safe(f"bootstrap: {w}"))
        print(f"WARNING: bootstrap {w}")
    for w in agg.get("seed_alignment_warnings", []):
        flags.append(flag_safe(f"seed alignment: {w}"))
        print(f"WARNING: seed alignment {w}")

    replication = guarded_replication(cfg, llm, corpus, index, evaluator, scorer, guard)

    collected = report_summary(cfg, agg, harness)
    S = agg.get("conditions", {}) or {}
    for name in ALL_ARM_NAMES:
        if name != REFERENCE_ARM:
            report_paired(name, S, agg)
    nli_ref = agg.get("unfiltered_pool_nli_reference", {}) or {}
    print(f"unfiltered_grounded_pool_nli_flags {PRIMARY}_mean: {fmt(nli_ref.get('mean'))} "
          f"n_distractors={nli_ref.get('n_distractors', 0)}")
    print(f"grounding_attribution_posterior: {fmt(agg.get('grounding_attribution_posterior'))}")
    print(f"rewrite_noop_rate_mean: {fmt(agg.get('rewrite_noop_rate_mean'))}")
    print(f"topic_label_fallback_rate_mean: {fmt(agg.get('topic_label_fallback_rate_mean'))}")
    print("PER_OPTION_ENRICHMENT: " + dumps(agg.get("per_option_enrichment", {})))
    print("GATES: " + dumps(gates))

    headline = (S.get(REFERENCE_ARM, {}) or {}).get("primary_metric")
    if headline is not None and harness.check_value(headline, "primary_metric"):
        harness.report_metric("primary_metric", headline)
        collected["primary_metric"] = float(headline)
        print(f"primary_metric: {fmt(headline)}")
    else:
        print("primary_metric: NA (the reference arm measured no seed)")

    health = {"generator": llm.health(), "nli": nli.health(), "embedder": embedder.health()}
    print("NUMERIC_HEALTH: " + dumps(health))
    # ONE flag per substitution KIND that fired, each naming its own count and the exact value
    # substituted. The single line this replaces reported both kinds under one total and called
    # the judge's 0.0 "a neutral value" -- it is chosen to sit BELOW every entail threshold
    # precisely so it can never read as support, which is the opposite of neutral.
    for line in analysis.substitution_flag_lines(llm.nan_substitutions_by_kind,
                                                 llm.NAN_P_YES_SUBSTITUTE,
                                                 llm.NAN_LETTER_PROB_SUBSTITUTE):
        flags.append(flag_safe(line))
    if llm.nucleus_uniform_fallbacks > 0:
        flags.append(flag_safe(f"{llm.nucleus_uniform_fallbacks} functionality-proxy letter "
                               f"distributions fell back to uniform"))
    n_oom = llm.oom_fallbacks + nli.oom_fallbacks + embedder.oom_fallbacks
    if n_oom > 0:
        flags.append(flag_safe(f"{n_oom} OOM batch-size fallbacks occurred; the final batch sizes "
                               f"are in numeric_health"))

    executed_design = executed_design_or_error(cfg, folds.folds, n_calib_built, flags)
    extra: Dict[str, Any]
    try:
        extra = {
            "metric_def": {"name": "primary_metric", "alias": PRIMARY, "direction": "minimize",
                           "estimator": analysis.PRIMARY_ESTIMATOR,
                           "units": "fraction of distractor slots (0-1)"},
            "hyperparameters": cfg.hyperparameters(),
            "seeds": SEEDS,
            "condition_names": ALL_ARM_NAMES,
            "arm_names": ALL_ARM_NAMES,
            "control_arms": CONTROL_ARM_NAMES,
            "comparison_modes": analysis.COMPARISON_MODE,
            "run_metadata": {"smoke": cfg.smoke, "harness": harness.source,
                             "harness_members": harness.members,
                             "models": model_revision_record(cfg), "versions": versions,
                             "executed_design": executed_design,
                             "time_estimate_s": round(est, 1), "usable_budget_s": round(usable, 1),
                             "reduction_fraction": fraction,
                             "functionality_proxy_is_not_examinee_data": True,
                             "required_followup": "validation with real examinees"},
            "skipped_components": skipped_components,      # empty on every valid run
            "reduced_components": reduced_components,
            "undefined_values": "an empty group has no rate and is written as null, never NaN",
            "corpus": composition,
            "backend": {"device": cfg.device, "generator_precision": llm.precision,
                        "nli_device": str(nli.device), "versions": versions,
                        "data_root": cfg.data_root, "harness": harness.source},
            "calibration": {"nli": r_nli, "llm": r_llm, "evaluator": evaluator,
                            "n_pairs": n_calib_built},
            "scientific_validity": "measured" if evaluator != "none" else "unvalidated_judge",
            "flags": flags,
            "summary": agg,
            "gates": gates,
            "per_seed": per_seed_public,
            "replication_openbookqa": replication,
            "numeric_health": health,
            "metrics": collected,
            "budget": dict(guard.checkpoints, total_elapsed_s=round(guard.elapsed(), 1)),
        }
    except Exception as e:            # nothing assembled here may cost the run its measured seeds
        traceback.print_exc()
        print(f"PAYLOAD_BUILD_FAILED: {type(e).__name__}: {e}")
        extra = minimal_payload(cfg, harness, flags, per_seed_public, reduced_components, e)
    write_payload(harness, extra)                          # the ONE write path
    harness.finalize()
    print(f"[done] elapsed={guard.elapsed():.0f}s arms={len(ALL_ARM_NAMES)} seeds={len(SEEDS)}")
    if cfg.smoke:
        print("SMOKE_RUN_COMPLETE")


# ITEM 51: catch Exception only; SystemExit propagates so the budget exit code survives
if __name__ == "__main__":
    try:
        main()
    except Exception as e:                                 # SystemExit is NOT an Exception subclass
        traceback.print_exc()
        print(f"FAIL: {type(e).__name__}: {e}")
        sys.exit(1)


def programming_errors() -> Tuple[type, ...]:
    """The exception classes that are ALWAYS a defect in this code, DELEGATED to analysis.

    The declaration itself moved to analysis.programming_errors, and the three re-raise sites in
    this file call it through the analysis MODULE. This definition sits below the
    `if __name__ == "__main__":` guard, so the name it binds does not exist yet while main() is
    running: when the handlers read THIS name, the first exception of the run raised
    NameError instead of the real one, escaped the `except Exception` recovery branch on the same
    try, and ended the run with no payload and not one condition recorded -- both mandated control
    arms among them. Reading the rule from a module imported at the top of this file makes the
    tuple bound before main() starts, whatever a future edit does with the order down here.

    The name is kept, and still answers with the same three classes, so any caller that reaches
    for it after the module has finished importing gets the one declaration."""
    return analysis.programming_errors()
