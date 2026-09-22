"""
test_experiment.py — the offline pytest suite.

# ITEM 44: the whole suite.
Imports analysis, config, data, experiment_config and methods only -- never main and never
models, so no test can pull in torch, transformers or a checkpoint. No network, no GPU, no
model load; every model call is an injected fake passed as a constructor argument. Every
expected value below is hand-computed in the derivation from a two-to-four-item toy pool.

The suite inspects no project source: it reads no .py file, calls no inspect.getsource,
parses no module and asserts nothing about sys.modules or the environment. Every assertion
is on a value a function under test returned.
"""
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest

import analysis
import config
import data
import experiment_config
import methods

TOL = 1e-5


# ============================================================ toy pools (# ITEM 44)
def make_item(item_id: int, seed: int, p_orig: Any, p_rewrite: Any, enum_dense: bool = False,
              covered: bool = True, stem: Optional[str] = None) -> data.QuizItem:
    """A REAL data.QuizItem, because plant and the rerank rebuild copy an item with
    dataclasses.replace; a stand-in namespace would not survive that."""
    po = np.asarray(p_orig, dtype=np.float32)
    pr = np.asarray(p_rewrite, dtype=np.float32)
    nli_o = np.zeros((4, 3), dtype=np.float32)
    nli_r = np.zeros((4, 3), dtype=np.float32)
    # Column order is contradiction, entailment, neutral. The off-entailment mass is SPLIT 60/40
    # between contradiction and neutral so each row sums to one AND the two polarity filters see a
    # real distribution: with neutral pinned to zero, P_contradict == 1 - P_entail exactly, the
    # contradiction and not-entail arms produce identical score vectors, and the one offline check
    # that would catch the two H3 arms collapsing into one condition cannot tell them apart.
    nli_o[:, 1] = po
    nli_o[:, 0] = (1.0 - po) * 0.6
    nli_o[:, 2] = (1.0 - po) * 0.4
    nli_r[:, 1] = pr
    nli_r[:, 0] = (1.0 - pr) * 0.6
    nli_r[:, 2] = (1.0 - pr) * 0.4
    it = data.QuizItem(
        item_id=item_id, seed=seed, gold_idx=item_id, topic=f"topic {item_id}",
        passage=f"passage {item_id} about photosynthesis and chlorophyll",
        passage_id=item_id, retrieval_covered=covered,
        stem=stem if stem is not None else f"what does item {item_id} describe?",
        options=[f"key{item_id}", f"d{item_id}a", f"d{item_id}b", f"d{item_id}c"],
        enumeration_dense=enum_dense, arm="unfiltered_grounded_pool",
    )
    it.stem_rewritten = it.stem + " under the passage's stated conditions"
    it.nli_orig = nli_o
    it.nli_rewrite = nli_r
    it.llm_pyes = po.copy()
    it.llm_pyes_rewrite = pr.copy()
    it.key_sim = np.array([1.0, 0.8, 0.5, 0.2], dtype=np.float32)
    it.overlap = np.array([0.9, 0.4, 0.2, 0.1], dtype=np.float32)
    return it


def toy_pool(seed: int = 0) -> List[data.QuizItem]:
    """Three items; the hand-computed rates are in the derivation.

    slots are [key, d1, d2, d3] and the threshold is 0.5:
      item 0  d1 supported under BOTH stems  -> distractor-caused
      item 1  d1 supported under orig only   -> stem-caused
      item 2  nothing supported
    The seed argument only moves the (seed, item_id) keys, so a disjoint pool can be asked
    for without changing a single rate."""
    return [
        make_item(0, seed, [0.9, 0.8, 0.1, 0.1], [0.9, 0.7, 0.1, 0.1]),
        make_item(1, seed, [0.9, 0.6, 0.3, 0.1], [0.9, 0.2, 0.1, 0.1]),
        make_item(2, seed, [0.9, 0.2, 0.1, 0.1], [0.9, 0.2, 0.1, 0.1]),
    ]


def unsupported_pool(n: int = 2) -> List[data.QuizItem]:
    """No distractor is corpus-supported, so the share denominator is zero and the share
    must be None while corpus_supported_rate is a measured 0.0."""
    return [make_item(i, 0, [0.9, 0.1, 0.1, 0.1], [0.9, 0.1, 0.1, 0.1]) for i in range(n)]


def identical_pool(n: int = 4) -> List[data.QuizItem]:
    """Every item carries identical flags, so EVERY subset has exactly the pool's rates and
    the null arm's random drop cannot move one of them."""
    return [make_item(i, 0, [0.9, 0.8, 0.1, 0.1], [0.9, 0.7, 0.1, 0.1]) for i in range(n)]


def clustered_pool() -> List[data.QuizItem]:
    """Four items whose three rows agree inside the item: two all-dc, two all-clean. Row
    resampling would treat 12 independent draws where there are only 4."""
    hot = [0.9, 0.9, 0.9, 0.9]
    cold = [0.9, 0.1, 0.1, 0.1]
    return [make_item(0, 0, hot, hot), make_item(1, 0, hot, hot),
            make_item(2, 0, cold, cold), make_item(3, 0, cold, cold)]


def candidate_item(item_id: int, entail_flags: Sequence[bool], sims: Sequence[float]) -> data.QuizItem:
    """An item carrying an already-scored candidate cache, for the quartile and selection
    statistics of the two rerank arms."""
    it = make_item(item_id, 0, [0.9, 0.1, 0.1, 0.1], [0.9, 0.1, 0.1, 0.1])
    n = len(entail_flags)
    it.candidates = [f"cand{item_id}_{j}" for j in range(n)]
    cand_nli = np.zeros((n, 3), dtype=np.float32)
    for j, flag in enumerate(entail_flags):
        e = 0.9 if flag else 0.1
        cand_nli[j] = [1.0 - e, e, 0.0]
    it.cand_nli = cand_nli
    it.cand_sim = np.asarray(sims, dtype=np.float32)
    it.cand_emb = np.eye(n, 8, dtype=np.float32)
    return it


def seed_record(seed: int, value: float, arms: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """A per-seed record shaped the way run_seed builds it, carrying the same primary value
    for every named arm."""
    names = list(arms) if arms is not None else list(experiment_config.ALL_ARM_NAMES)
    conditions: Dict[str, Any] = {}
    for name in names:
        conditions[name] = {
            "n_items": 3,
            "distractor_caused_csd_rate": value,
            "stem_caused_csd_rate": 0.0,
            "corpus_supported_rate": value,
            "stem_caused_share": 0.0,
            "key_hallucination_rate": 0.0,
            "per_item_multi_key_rate": value,
            "mean_distractor_p_support_orig": value,
            "soft_distractor_caused_score": value,
            "mean_key_p_support_orig": 0.9,
        }
    return {"seed": seed, "status": "ok", "conditions": conditions, "rows": {}}


def nli_reference_seed_record(key: str) -> Dict[str, Any]:
    """A one-seed record whose NLI-flagged pool sits under the GIVEN key, so the shared-key
    contract can be driven from both sides: pass the constant and the reference is found,
    pass anything else and it is not."""
    rec = seed_record(0, 0.25)
    rec[key] = {
        "n_items": 3,
        "distractor_caused_csd_rate": 0.25,
        "stem_caused_csd_rate": 0.0,
        "corpus_supported_rate": 0.25,
        "stem_caused_share": 0.0,
        "key_hallucination_rate": 0.125,
        "per_item_multi_key_rate": 0.25,
        "mean_distractor_p_support_orig": 0.25,
        "soft_distractor_caused_score": 0.25,
        "mean_key_p_support_orig": 0.9,
    }
    return rec


def complete_extra() -> Dict[str, Any]:
    """A payload carrying every REQUIRED_EXTRA_KEYS key with an empty skipped_components."""
    extra: Dict[str, Any] = {}
    for key in analysis.REQUIRED_EXTRA_KEYS:
        extra[key] = {}
    extra["skipped_components"] = []
    extra["reduced_components"] = {}
    extra["condition_names"] = list(experiment_config.ALL_ARM_NAMES)
    extra["scientific_validity"] = "measured"
    extra["flags"] = []
    return extra


def ci_bounds(record: Dict[str, Any]) -> Tuple[float, float]:
    """The two bounds read from the record's single "ci" list. Asserts neither is None
    rather than hunting for scalar lo/hi keys, which must not exist."""
    ci = record["ci"]
    assert isinstance(ci, list) and len(ci) == 2, f"ci must be one two-element list, got {ci!r}"
    assert ci[0] is not None and ci[1] is not None
    return float(ci[0]), float(ci[1])


# ============================================================ injected fakes (# ITEM 44)
class FakeLLM:
    """A generator stand-in returning the replies the test hands it, cycling within a batch."""

    def __init__(self, replies: Sequence[str]) -> None:
        self.replies = list(replies)
        self.calls = 0
        self.prompts: List[str] = []

    def generate_batch(self, prompts: Sequence[str], max_new_tokens: int) -> List[str]:
        self.calls += 1
        self.prompts.extend(prompts)
        if not self.replies:
            return ["" for _ in prompts]
        return [self.replies[i % len(self.replies)] for i in range(len(prompts))]


class ScriptedLLM:
    """A generator stand-in returning one reply per CALL, so a first pass and its repair
    retry can differ -- which is what the longer-list rule needs."""

    def __init__(self, per_call: Sequence[str]) -> None:
        self.per_call = list(per_call)
        self.calls = 0

    def generate_batch(self, prompts: Sequence[str], max_new_tokens: int) -> List[str]:
        idx = min(self.calls, len(self.per_call) - 1) if self.per_call else 0
        reply = self.per_call[idx] if self.per_call else ""
        self.calls += 1
        return [reply for _ in prompts]


class FakeNLI:
    """A judge stand-in whose 3-way output the test fixes, injected where the cross-encoder
    would go. Order is contradiction, entailment, neutral."""

    def __init__(self, probs: Sequence[float] = (0.05, 0.9, 0.05)) -> None:
        self.probs = np.asarray(probs, dtype=np.float32)
        self.n_pairs_scored = 0

    def score_pairs(self, premises: Sequence[str], hypotheses: Sequence[str]) -> np.ndarray:
        n = len(premises)
        self.n_pairs_scored += n
        return np.tile(self.probs, (n, 1)).astype(np.float32)

    def score_options(self, passage: str, stem: str, options: Sequence[str]) -> np.ndarray:
        return self.score_pairs([passage] * len(options), list(options))


class FakeEmbedder:
    """Deterministic unit vectors keyed by the text itself -- never a random draw, so a
    rerun reproduces every similarity."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    def encode(self, texts: Sequence[str], batch_size: Optional[int] = None) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for j, ch in enumerate(str(t)):
                out[i, j % self.dim] += (ord(ch) % 17) + 1.0
            norm = float(np.linalg.norm(out[i]))
            out[i] = out[i] / norm if norm > 0 else np.eye(1, self.dim, dtype=np.float32)[0]
        return out


class FakeProxyLLM:
    """A judge and answerer whose letter probabilities and P(Yes) are fixed.

    The constructor takes the sheet's one argument. Per-row INVALIDITY -- the mask a non-finite
    forward pass produces -- is configured AFTER construction through impute_pyes_rows and
    impute_letter_rows, so a test can drive the imputation path without widening the signature
    every other file was written against."""

    def __init__(self, probs: Sequence[float] = (0.7, 0.1, 0.1, 0.1)) -> None:
        self.probs = np.array(list(probs), dtype=np.float32)
        self.pyes = 0.9
        self.invalid_letter_rows: set = set()
        self.invalid_pyes_rows: set = set()
        self.pyes_calls = 0
        self.letter_calls = 0

    def impute_pyes_rows(self, rows: Sequence[int]) -> "FakeProxyLLM":
        """Mark judge rows the forward pass could not measure; returns self so a call site stays
        one expression."""
        self.invalid_pyes_rows = {int(i) for i in rows}
        return self

    def impute_letter_rows(self, rows: Sequence[int]) -> "FakeProxyLLM":
        """Mark answerer rows the forward pass could not measure; returns self."""
        self.invalid_letter_rows = {int(i) for i in rows}
        return self

    def generate_batch(self, prompts: Sequence[str], max_new_tokens: int) -> List[str]:
        del max_new_tokens
        return ["candidate" for _ in prompts]

    def letter_probs_batch(self, prompts: Sequence[str], temperature: float, top_p: float):
        """The PAIR (probs, ok). An invalid row carries the UNIFORM 0.25 substitute the real
        letter_probs_batch writes for a non-finite forward pass, so the test exercises exactly
        the value the pipeline would see, not a sentinel."""
        del temperature, top_p
        self.letter_calls += 1
        out = np.tile(self.probs, (len(prompts), 1))
        ok = np.ones(len(prompts), dtype=bool)
        for i in self.invalid_letter_rows:
            if 0 <= i < len(prompts):
                out[i] = np.full(4, 0.25, dtype=np.float32)
                ok[i] = False
        return out, ok

    def p_yes_batch_with_validity(self, prompts: Sequence[str]):
        """The PAIR (p, ok), with 0.0 substituted on an invalid row exactly as the real judge's
        NAN_P_YES_SUBSTITUTE does."""
        self.pyes_calls += 1
        p = np.full(len(prompts), float(self.pyes), dtype=np.float32)
        ok = np.ones(len(prompts), dtype=bool)
        for i in self.invalid_pyes_rows:
            if 0 <= i < len(prompts):
                p[i] = 0.0
                ok[i] = False
        return p, ok

    def p_yes_batch(self, prompts: Sequence[str]) -> np.ndarray:
        return self.p_yes_batch_with_validity(prompts)[0]

    @staticmethod
    def sample_letters(probs: np.ndarray, n: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(int(seed))
        return rng.choice(4, size=int(n), p=np.asarray(probs, dtype=np.float64))


class FakeCorpus:
    """The three members label_topics, the fallback labels and a generator need."""

    def __init__(self, texts: Sequence[str]) -> None:
        self.passages = list(texts)

    def passage_text(self, pid: int) -> str:
        return self.passages[pid]

    def is_enumeration_dense(self, pid: int) -> bool:
        return False


class FoldsHolder:
    """A SeedFolds look-alike carrying the row lists on a .folds attribute -- the exact
    shape executed_design_record must refuse rather than iterate."""

    def __init__(self, rows: Sequence[Sequence[int]]) -> None:
        self.folds = [list(r) for r in rows]


def _cfg() -> Any:
    return experiment_config.Config()


def _decomposer() -> methods.FlagDecomposer:
    return methods.FlagDecomposer(_cfg(), "nli")


# ============================================================ ITEM 44: the suite runs on fakes alone
def test_the_toy_item_is_the_real_quiz_item_dataclass():
    import dataclasses
    it = make_item(0, 0, [0.9, 0.8, 0.1, 0.1], [0.9, 0.7, 0.1, 0.1])
    assert isinstance(it, data.QuizItem)
    twin = dataclasses.replace(it, arm="positive_control_planted")
    assert twin.arm == "positive_control_planted"
    assert twin.item_id == it.item_id and it.arm == "unfiltered_grounded_pool"
    # every provenance marker is False on a fresh item, so an arm reports how its input was
    # produced instead of reconstructing it by comparison
    assert it.planted is False and it.cand_padded is False and it.cand_line_split is False
    assert it.cand_repair_improved is False
    assert "cand_repair_improved" in it.to_record()
    # an unset judge validity mask means "not scored by that judge", never "measured"
    assert it.llm_pyes_valid is None and it.llm_pyes_rewrite_valid is None
    assert "llm_pyes_valid" in it.to_record()


def test_the_flag_pipeline_runs_on_injected_fakes_alone():
    scorer = methods.ItemScorer(_cfg(), FakeLLM(["rewritten and far more specific stem"]),
                                FakeNLI(), FakeEmbedder())
    items = toy_pool()
    scorer.score_nli(items)
    scorer.score_similarity_and_overlap(items)
    rates = _decomposer().rates(items)
    for key in analysis.RATE_KEYS:
        v = rates[key]
        assert v is None or (isinstance(v, float) and math.isfinite(v)), f"{key} -> {v!r}"


def test_the_config_shim_re_exports_every_name():
    assert config.ALL_ARM_NAMES is experiment_config.ALL_ARM_NAMES
    assert config.CONDITION_NAMES == experiment_config.CONDITION_NAMES
    assert config.CONTROL_ARM_NAMES == experiment_config.CONTROL_ARM_NAMES
    assert config.PROMPT_GROUNDED == experiment_config.PROMPT_GROUNDED
    assert config.MODEL_IDS == experiment_config.MODEL_IDS
    # the pin table and the resolver that reads it must arrive through the shim together, or a
    # caller reaching the shim would be back to looking the table up under a name it does not carry
    assert config.MODEL_REVISIONS is experiment_config.MODEL_REVISIONS
    assert config.model_revision is experiment_config.model_revision


# ============================================================ ITEM 1/2/3/4: the config
def test_the_arm_registry_keeps_eight_conditions_and_two_controls():
    assert len(experiment_config.CONDITION_NAMES) == 8
    assert experiment_config.CONDITION_NAMES[0] == "unfiltered_grounded_pool"
    assert "ungrounded_topic_only_control" in experiment_config.CONDITION_NAMES
    assert experiment_config.CONTROL_ARM_NAMES == ["null_random_drop", "positive_control_planted"]
    all_names = experiment_config.ALL_ARM_NAMES
    assert len(all_names) == 10 and len(set(all_names)) == 10
    assert all_names[:8] == experiment_config.CONDITION_NAMES
    assert any(n.startswith("null_") for n in all_names)
    assert any(n.startswith("positive_control") for n in all_names)


def test_config_drops_every_clock_conditional_knob():
    cfg = _cfg()
    for knob in ("h4_always_run", "force_h4", "proxy_reduce_elapsed_fraction",
                 "functionality_samples_reduced", "replication_elapsed_fraction_limit",
                 "retrieval_widen_k"):
        assert not hasattr(cfg, knob), f"{knob} makes a component conditional on the clock"
    assert cfg.seeds == [0, 1, 2]
    for knob in ("secondary_min_seeds", "min_reduction_fraction", "planted_slot",
                 "planted_fraction", "null_retention_fraction", "detection_ratio_gate",
                 # DECLARED on Config, not attached by a side effect: a design value the payload
                 # records must be visible before anything has run
                 "calibration_pairs_designed"):
        assert hasattr(cfg, knob)


def test_hyperparameters_records_an_unset_knob_as_null():
    cfg = _cfg()
    hp = cfg.hyperparameters()
    assert "corpus_passage_cap" in hp
    assert hp["corpus_passage_cap"] is None
    assert hp["seeds"] == [0, 1, 2]
    # the designed calibration bound is null until the budget lever lowers the knob, and that
    # null IS the record that no reduction happened on this run
    assert "calibration_pairs_designed" in hp
    assert hp["calibration_pairs_designed"] is None
    assert hp["calibration_pairs"] == cfg.calibration_pairs


def test_a_device_fallback_records_its_reason():
    def failing_probe() -> str:
        raise RuntimeError("libcuda missing")

    assert experiment_config.detect_device(probe=failing_probe) == "cpu"
    reason = experiment_config.DEVICE_FALLBACK_REASON
    assert reason is not None and "RuntimeError" in reason and "libcuda missing" in reason
    assert experiment_config.detect_device(probe=lambda: "cpu") == "cpu"
    assert experiment_config.DEVICE_FALLBACK_REASON is None


def test_the_measured_config_is_not_capped_below_the_full_corpus():
    cfg = _cfg()
    assert cfg.corpus_passage_cap is None
    assert cfg.retrieval_top_k >= 1


def test_smoke_config_shrinks_every_stage():
    cfg = _cfg()
    assert cfg.items_per_seed >= 3
    assert cfg.calibration_pairs > 20
    assert cfg.obqa_replication_items >= 3
    assert cfg.functionality_samples_per_context >= 2


# ============================================================ ITEM 5/48: the JSON contract
def test_dumps_never_emits_a_nan_token():
    text = analysis.dumps({"a": float("nan"), "b": [1.0, float("inf")],
                           "c": {"d": np.float32("nan")}, "e": None})
    assert "NaN" not in text and "Infinity" not in text and "nan" not in text
    assert "null" in text


def test_fmt_prints_na_for_an_undefined_value():
    assert analysis.fmt(None) == "NA"
    assert analysis.fmt(float("nan")) == "NA"
    assert analysis.fmt(float("inf")) == "NA"
    assert analysis.fmt(0.5) == "0.5000"
    assert analysis.fmt(0.5, 2) == "0.50"


def test_a_nonfinite_value_names_its_json_path():
    with pytest.raises(ValueError) as exc:
        analysis.assert_no_nonfinite({"summary": {"arms": [{"rate": float("nan")}]}})
    msg = str(exc.value)
    assert "summary" in msg and "arms" in msg and "rate" in msg
    assert analysis.assert_no_nonfinite({"a": 1.0, "b": None, "c": [0.5]}) is None


# ============================================================ ITEM 6/7/8: the record contracts
def test_the_rates_contract_rejects_nan_and_accepts_none():
    good = {k: 0.1 for k in analysis.RATE_KEYS}
    assert analysis.enforce_rates_contract(0, "unfiltered_grounded_pool", good) is None
    with_none = dict(good)
    with_none["stem_caused_share"] = None
    assert analysis.enforce_rates_contract(0, "unfiltered_grounded_pool", with_none) is None
    with pytest.raises(ValueError):
        bad = dict(good)
        bad["distractor_caused_csd_rate"] = float("nan")
        analysis.enforce_rates_contract(0, "unfiltered_grounded_pool", bad)
    with pytest.raises(ValueError) as exc:
        partial = dict(good)
        del partial["corpus_supported_rate"]
        analysis.enforce_rates_contract(0, "unfiltered_grounded_pool", partial)
    assert "corpus_supported_rate" in str(exc.value)
    status_only = {"status": "failed: ValueError: boom"}
    assert analysis.carries_rate_keys(status_only) is False
    assert analysis.enforce_rates_contract(0, "overgenerate_rerank_by_contradiction", status_only) is None


def test_the_payload_guard_lists_every_required_key():
    extra = complete_extra()
    assert analysis.check_payload_contract(extra, 1) is None
    for key in analysis.REQUIRED_EXTRA_KEYS:
        missing = {k: v for k, v in extra.items() if k != key}
        with pytest.raises(ValueError) as exc:
            analysis.check_payload_contract(missing, 1)
        assert key in str(exc.value)


def test_the_smoke_run_skips_no_component():
    extra = complete_extra()
    extra["skipped_components"] = ["replication"]
    with pytest.raises(ValueError) as exc:
        analysis.check_payload_contract(extra, 1)
    assert "skipped_components" in str(exc.value)


def test_a_write_before_any_recorded_seed_is_refused():
    with pytest.raises(ValueError) as exc:
        analysis.enforce_seed_recorded_before_write(0)
    assert str(exc.value) == analysis.RESULTS_CONTRACT_NO_SEED
    assert analysis.enforce_seed_recorded_before_write(1) is None
    with pytest.raises(ValueError) as exc2:
        analysis.check_payload_contract(complete_extra(), 0)
    assert analysis.RESULTS_CONTRACT_NO_SEED in str(exc2.value)


def test_the_payload_write_order_requires_a_recorded_seed():
    extra = complete_extra()
    with pytest.raises(ValueError):
        analysis.check_payload_contract(extra, 0)
    assert analysis.check_payload_contract(extra, 3) is None


def test_an_unknown_library_version_carries_its_reason():
    ok = analysis.library_version_record("numpy", lambda: "2.1.0")
    assert ok == {"numpy": "2.1.0"}

    def failing() -> str:
        raise ImportError("no module named transformers")

    rec = analysis.library_version_record("transformers", failing)
    assert rec["transformers"] == "unknown"
    assert "transformers_error" in rec
    assert "ImportError" in rec["transformers_error"]
    assert "no module named transformers" in rec["transformers_error"]


def test_a_failed_arm_records_its_error_type_and_message():
    # rec is bound BEFORE the try: a name assigned only inside an except handler is unbound on the
    # path where the body does not raise, which the checker reports as used-before-assignment. The
    # record is still built inside a live handler, because arm_failure_record calls
    # traceback.format_exc(), which outside one returns "NoneType: None" and would make the
    # traceback_tail assertion below vacuous.
    rec: Dict[str, str] = {}
    try:
        raise ValueError("candidate cache was empty")
    except ValueError as e:
        rec = analysis.arm_failure_record(e)
    assert rec, "the except handler must have produced the record under test"
    assert rec["error_type"] == "ValueError"
    assert "candidate cache was empty" in rec["error_message"]
    assert rec["traceback_tail"]
    assert "status" in rec
    assert analysis.carries_rate_keys(rec) is False


# ============================================================ ITEM 18/45: the nine rates
def test_the_primary_rate_denominator_is_three_times_the_item_count():
    r = _decomposer().rates(toy_pool())
    assert r["n_items"] == 3
    assert r["distractor_caused_csd_rate"] == pytest.approx(1.0 / 9.0, abs=TOL)


def test_the_stem_caused_rate_counts_the_rewrite_only_slot():
    r = _decomposer().rates(toy_pool())
    assert r["stem_caused_csd_rate"] == pytest.approx(1.0 / 9.0, abs=TOL)


def test_the_corpus_supported_rate_is_the_sum_of_its_two_parts():
    r = _decomposer().rates(toy_pool())
    assert r["corpus_supported_rate"] == pytest.approx(2.0 / 9.0, abs=TOL)
    assert (r["distractor_caused_csd_rate"] + r["stem_caused_csd_rate"]
            == pytest.approx(r["corpus_supported_rate"], abs=TOL))


def test_the_stem_caused_share_is_the_split_of_the_supported_slots():
    r = _decomposer().rates(toy_pool())
    assert r["stem_caused_share"] == pytest.approx(0.5, abs=TOL)


def test_the_key_hallucination_rate_is_over_items_not_slots():
    r = _decomposer().rates(toy_pool())
    assert r["key_hallucination_rate"] == pytest.approx(0.0, abs=TOL)
    halluc = [make_item(0, 0, [0.2, 0.1, 0.1, 0.1], [0.2, 0.1, 0.1, 0.1]),
              make_item(1, 0, [0.9, 0.1, 0.1, 0.1], [0.9, 0.1, 0.1, 0.1])]
    assert _decomposer().rates(halluc)["key_hallucination_rate"] == pytest.approx(0.5, abs=TOL)


def test_the_multi_key_rate_counts_items_with_any_supported_distractor():
    r = _decomposer().rates(toy_pool())
    assert r["per_item_multi_key_rate"] == pytest.approx(2.0 / 3.0, abs=TOL)


def test_the_mean_distractor_support_averages_all_nine_slots():
    r = _decomposer().rates(toy_pool())
    assert r["mean_distractor_p_support_orig"] == pytest.approx(2.4 / 9.0, abs=TOL)


def test_the_soft_score_is_the_minimum_over_both_stems():
    r = _decomposer().rates(toy_pool())
    assert r["soft_distractor_caused_score"] == pytest.approx(1.7 / 9.0, abs=TOL)


def test_the_mean_key_support_averages_the_key_slot_only():
    r = _decomposer().rates(toy_pool())
    assert r["mean_key_p_support_orig"] == pytest.approx(0.9, abs=TOL)


def test_rates_on_an_empty_group_are_none_and_share_needs_a_denominator():
    empty = _decomposer().rates([])
    for key in analysis.RATE_KEYS:
        assert empty[key] is None, f"{key} on an empty group must be None, got {empty[key]!r}"
    unsupported = _decomposer().rates(unsupported_pool(2))
    assert unsupported["corpus_supported_rate"] == pytest.approx(0.0, abs=TOL)
    assert unsupported["distractor_caused_csd_rate"] == pytest.approx(0.0, abs=TOL)
    assert unsupported["stem_caused_share"] is None


# ============================================================ ITEM 19: rows keyed by id
def test_flag_rows_key_every_distractor_by_seed_and_item_id():
    rows = _decomposer().flag_rows(toy_pool(seed=2), "unfiltered_grounded_pool")
    assert analysis.n_rows(rows) == 9
    keys = rows["keys"]                       # the published name; "key" is not a column
    assert "key" not in rows
    assert len(keys) == 9
    assert len(set(keys)) == 3
    assert set(keys) == {(2, 0), (2, 1), (2, 2)}
    assert sum(1 for k in keys if k == (2, 0)) == 3
    assert sum(bool(v) for v in rows["dc"]) == 1
    assert sum(bool(v) for v in rows["sc"]) == 1
    assert sum(bool(v) for v in rows["cs"]) == 2


# ============================================================ ITEM 9: the clustered interval
def test_the_ci_record_publishes_its_bounds_as_one_list():
    st = analysis.Statistics(_cfg())
    rec = st.bootstrap_rate_ci(_decomposer().flag_rows(toy_pool(), "x"))
    assert set(rec) == {"rate", "ci", "n_items", "n_rows"}
    assert "lo" not in rec and "hi" not in rec
    lo, hi = ci_bounds(rec)
    assert lo <= rec["rate"] <= hi
    assert rec["n_items"] == 3 and rec["n_rows"] == 9
    empty = st.bootstrap_rate_ci(analysis.empty_rows("x"))
    assert empty["rate"] is None and empty["ci"] == [None, None]


def test_the_pooled_ci_resamples_items_not_distractor_rows():
    st = analysis.Statistics(_cfg())
    rows = _decomposer().flag_rows(clustered_pool(), "x")
    clustered = st.bootstrap_rate_ci(rows)
    row_level = st.bootstrap_rate_ci_unclustered(rows)
    c_lo, c_hi = ci_bounds(clustered)
    r_lo, r_hi = ci_bounds(row_level)
    assert clustered["rate"] == pytest.approx(row_level["rate"], abs=TOL)
    assert (c_hi - c_lo) >= (r_hi - r_lo) - TOL


# ============================================================ ITEM 10/11: registered schemes
def test_the_comparison_scheme_of_every_arm_is_the_registered_one():
    mode = analysis.COMPARISON_MODE
    for name in experiment_config.ALL_ARM_NAMES:
        if name == analysis.REFERENCE_ARM:
            continue
        assert name in mode, f"{name} has no registered comparison scheme"
        assert mode[name] in ("nested", "paired", "independent")
    assert mode["per_option_entailment_filter"] == "nested"
    assert mode["null_random_drop"] == "nested"
    assert mode["overgenerate_rerank_by_contradiction"] == "paired"
    assert mode["positive_control_planted"] == "paired"
    assert mode["ungrounded_topic_only_control"] == "independent"

    st = analysis.Statistics(_cfg())
    pool = _decomposer().flag_rows(toy_pool(), "pool")
    disjoint = _decomposer().flag_rows(toy_pool(seed=7), "other")
    rec = st.paired_bootstrap_diff(disjoint, pool, mode="nested")
    assert rec["requested_scheme"] == "nested"
    assert rec["degraded"] is True
    assert rec["scheme"] != "nested"
    assert "nested" in str(rec.get("warning", ""))
    # the degradation is the ONLY diagnostic here, so it is the only entry
    assert rec["warnings"] == [rec["warning"]]

    # a clean comparison raises no diagnostic and carries NEITHER key: aggregate() gates on
    # `"warning" in cmp_items`, so an always-present key would flag every arm
    clean = st.paired_bootstrap_diff(_decomposer().flag_rows(toy_pool()[:2], "sub"), pool,
                                     mode="nested")
    assert "warning" not in clean
    assert "warnings" not in clean


# ============================================================ ITEM 12: the H2 floor
def test_the_h2_enrichment_floor_scales_with_the_design():
    assert analysis.h2_min_sweep_items(3, 100) == 30
    assert analysis.h2_min_sweep_items(1, 3) == 5
    assert analysis.h2_min_sweep_items(3, 3) == 5

    # per_option_enrichment reads each cell's "rate"; a cell keyed by the metric name alone is
    # invisible to it, so both cells would be inadmissible and no tau could ever be selected
    cells = {
        "0.3": {"n_items": 100, "n_seeds": 3, "rate": 0.02},
        "0.9": {"n_items": 4, "n_seeds": 1, "rate": 0.33},
    }
    out = analysis.per_option_enrichment(cells, 0.0133, min_items=30, min_seeds=2,
                                         taus=[0.3, 0.9], plan_tau=0.9)
    # the admission flag and the rate are published as two flat maps, not as a "cells" map
    assert "cells" not in out
    assert out["admitted"]["0.9"] is False       # 4 pooled items is below the floor of 30
    assert out["rates"]["0.9"] is None           # an inadmissible cell reports NO rate
    assert out["admitted"]["0.3"] is True
    assert out["rates"]["0.3"] == pytest.approx(0.02, abs=TOL)
    assert out["selected_tau"] == pytest.approx(0.3, abs=TOL)
    assert out["selected_n_items"] == 100
    assert out["enrichment_at_selected_tau"] == pytest.approx(0.02 - 0.0133, abs=TOL)
    assert out["enrichment_at_plan_tau"] is None  # the plan's tau=0.9 cell was not admitted
    assert out["min_items"] == 30 and out["min_seeds"] == 2
    assert out["spearman_rho_tau_vs_rate"] is None   # one defined cell cannot carry a correlation
    assert out["spearman_n_taus"] == 1


# ============================================================ ITEM 13/14: seed-backed verdicts
def test_a_verdict_backed_by_one_seed_is_untested():
    summary = {"arm": {"secondary": {"key_hallucination_rate": {"mean": 0.4, "n_seeds": 1}}}}
    value, n = analysis.secondary_value(summary, "arm", "key_hallucination_rate", min_seeds=2)
    assert isinstance((value, n), tuple)
    assert value is None
    assert n == 1
    summary["arm"]["secondary"]["key_hallucination_rate"]["n_seeds"] = 3
    value3, n3 = analysis.secondary_value(summary, "arm", "key_hallucination_rate", min_seeds=2)
    assert value3 == pytest.approx(0.4, abs=TOL)
    assert n3 == 3
    missing, n0 = analysis.secondary_value(summary, "arm", "absent_key", min_seeds=2)
    assert missing is None and n0 == 0


def test_the_gates_report_both_control_arm_verdicts():
    cfg = _cfg()
    st = analysis.Statistics(cfg)
    per_seed = [seed_record(s, 0.1 * (s + 1)) for s in (0, 1, 2)]
    agg = analysis.aggregate(cfg, per_seed, st)
    gates = analysis.decide_gates(cfg, agg, "nli")
    assert "gate_secondary_seed_counts" in gates
    text = analysis.dumps(gates)
    assert "null_random_drop" in text
    assert "positive_control_planted" in text
    assert "nan" not in text and "NaN" not in text
    # no verdict may carry a non-finite number; checked by the guard the payload write uses
    assert analysis.assert_no_nonfinite(gates) is None

    # hand-computed from the fixture: every arm carries 0.1, 0.2, 0.3 over seeds 0, 1, 2
    assert gates["h1_base_rate"] == pytest.approx(0.2, abs=TOL)
    assert gates["h1_powered"] is True
    assert gates["h1_grounded_minus_ungrounded"] == pytest.approx(0.0, abs=TOL)
    assert gates["h1_position"] == "indeterminate"

    # the null arm carries the same rate as the pool, so its effect is exactly zero
    assert gates["null_arm_effect"] == pytest.approx(0.0, abs=TOL)
    assert gates["null_arm_tolerance"] == analysis.NULL_EFFECT_TOLERANCE
    # the fixture ships no row tables, so the item bootstrap has no interval to report ...
    assert gates["null_arm_ci"] == [None, None]
    # ... and a verdict whose input is undefined is None, NEVER False
    assert gates["null_arm_near_zero"] is None
    assert gates["positive_control_detection_ratio"] is None
    assert gates["positive_control_measured_increase"] is None
    assert gates["positive_control_detected"] is None
    assert gates["gate_secondary_seed_counts"]["positive_control_planted.detection_ratio"] == 0
    assert gates["gate_secondary_min_seeds"] == cfg.secondary_min_seeds
    assert gates["gates_are_diagnostics_only"] is True


# ============================================================ ITEM 16/47: pairing by id
def test_values_pair_by_id_with_list_order_differing_from_id_order():
    # built in DESCENDING id order, and the two maps disagree on which seed is missing
    a = {2: 0.30, 1: 0.20, 0: 0.10}
    b = {2: 0.03, 0: 0.01, 3: 0.99}
    a_vals, b_vals, common = analysis.aligned_by_seed(a, b)
    assert common == [0, 2]
    assert a_vals == [0.10, 0.30]
    assert b_vals == [0.01, 0.03]
    # the positional answer zips the insertion orders and is WRONG; assert we differ from it
    positional_b = list(b.values())[:2]
    assert positional_b == [0.03, 0.01]
    assert b_vals != positional_b
    with_none = analysis.aligned_by_seed({0: 0.1, 1: None}, {0: 0.2, 1: 0.3})
    assert with_none[2] == [0]


def test_the_aggregate_over_seeds_is_the_mean_and_not_the_last_value():
    cfg = _cfg()
    st = analysis.Statistics(cfg)
    values = {0: 0.10, 1: 0.20, 2: 0.60}
    per_seed = [seed_record(s, v) for s, v in values.items()]
    agg = analysis.aggregate(cfg, per_seed, st)
    arm = agg["conditions"]["unfiltered_grounded_pool"]
    assert arm["mean"] == pytest.approx(0.30, abs=TOL)
    assert arm["mean"] != pytest.approx(0.60, abs=TOL)   # not the last value
    assert arm["mean"] != pytest.approx(0.10, abs=TOL)   # not the first value
    assert arm["per_seed_by_seed"] == {0: 0.10, 1: 0.20, 2: 0.60}
    assert "per_seed" not in arm


def test_the_headline_metric_is_the_declared_estimator_not_the_pooled_rate():
    cfg = _cfg()
    st = analysis.Statistics(cfg)
    per_seed = [seed_record(s, v) for s, v in {0: 0.10, 1: 0.20, 2: 0.60}.items()]
    agg = analysis.aggregate(cfg, per_seed, st)
    head = analysis.headline_metric(agg["conditions"]["unfiltered_grounded_pool"])
    assert head["primary_metric"] == pytest.approx(0.30, abs=TOL)
    assert head["primary_metric_estimator"] == analysis.PRIMARY_ESTIMATOR
    assert head["primary_metric_n_seeds"] == 3
    # the estimator and the seed count are published under those names and no others
    assert "estimator" not in head and "n_seeds" not in head
    assert "primary_metric_pooled" in head
    empty = analysis.headline_metric({"per_seed_by_seed": {}, "n_seeds": 0, "rows": None})
    assert empty["primary_metric"] is None
    assert empty["primary_metric_n_seeds"] == 0


def test_a_failed_seed_is_counted_never_imputed_with_a_rate_of_one():
    cfg = _cfg()
    st = analysis.Statistics(cfg)
    per_seed = [seed_record(0, 0.10), seed_record(1, 0.20),
                {"seed": 2, "status": "failed: SeedAbort: empty pool", "conditions": {}, "rows": {}}]
    agg = analysis.aggregate(cfg, per_seed, st)
    arm = agg["conditions"]["unfiltered_grounded_pool"]
    assert 2 not in arm["per_seed_by_seed"]
    assert arm["n_failed"] == 1
    assert arm["n_seeds"] == 2
    assert arm["mean"] == pytest.approx(0.15, abs=TOL)   # mean of 0.10 and 0.20, not of 1.0
    assert 1.0 not in arm["per_seed_by_seed"].values()


def test_the_aggregate_reads_the_nli_reference_rates_under_the_shared_key():
    cfg = _cfg()
    st = analysis.Statistics(cfg)
    right = analysis.aggregate(cfg, [nli_reference_seed_record(analysis.NLI_REFERENCE_RATES_KEY)], st)
    ref = right["unfiltered_pool_nli_reference"]
    assert ref["mean"] == pytest.approx(0.25, abs=TOL)
    wrong = analysis.aggregate(cfg, [nli_reference_seed_record("some_other_key")], st)
    assert wrong["unfiltered_pool_nli_reference"]["mean"] is None


# ============================================================ ITEM 17: the fold-row shape
def test_the_executed_design_record_reads_the_fold_rows():
    cfg = _cfg()
    cfg.items_per_seed = 3
    cfg.seeds = [0, 1]
    rows = [[10, 11, 12], [13, 14, 15]]
    rec = analysis.executed_design_record(cfg, rows)
    assert rec["n_seeds"] == 2
    assert rec["items_per_seed"] == 3
    with pytest.raises(TypeError) as exc:
        analysis.executed_design_record(cfg, FoldsHolder(rows))
    assert ".folds" in str(exc.value)
    with pytest.raises(TypeError):
        analysis.executed_design_record(cfg, 17)
    with pytest.raises(ValueError):
        analysis.executed_design_record(cfg, [[10, 11], [13, 14]])


# ============================================================ ITEM 20: the four scorers differ
def test_the_filters_rank_the_same_pool_differently():
    cfg = _cfg()
    dec = _decomposer()
    pool = toy_pool()
    scorers = {
        "per_option": methods.PerOptionEntailmentFilter(cfg, dec),
        "llm": methods.LLMJudgeFaithfulnessFilter(cfg, dec),
        "contradiction": methods.ContradictionPolarityItemFilter(cfg, dec),
        "not_entail": methods.NotEntailPolarityItemFilter(cfg, dec),
    }
    vectors = {k: v.score_pool(pool) for k, v in scorers.items()}
    labels = {k: v.report(pool, 0)["filter_scorer"] for k, v in scorers.items()}
    assert len(set(labels.values())) == 4
    assert not np.allclose(vectors["per_option"], vectors["contradiction"], atol=1e-6)
    assert not np.allclose(vectors["contradiction"], vectors["not_entail"], atol=1e-6)
    assert not np.allclose(vectors["per_option"], vectors["not_entail"], atol=1e-6)
    # the two polarity arms differ by EXACTLY the neutral share the fixture leaves: the
    # contradiction term counts 0.6 of the off-entailment mass, the not-entail term counts all of
    # it, so a fixture with no neutral mass would make the ablation vacuous
    assert np.allclose(vectors["contradiction"], 0.6 * np.asarray(vectors["not_entail"]),
                       atol=1e-5)
    # hand-computed for item 0: key 0.9, distractors 1 - [0.8, 0.1, 0.1] -> mean 2/3
    assert float(vectors["not_entail"][0]) == pytest.approx(0.9 * 2.0 / 3.0, abs=1e-5)
    assert float(vectors["contradiction"][0]) == pytest.approx(0.9 * 0.6 * 2.0 / 3.0, abs=1e-5)


def test_the_not_entail_scorer_differs_from_contradiction_only_in_the_polarity_term():
    cfg = _cfg()
    dec = _decomposer()
    contra = methods.ContradictionPolarityItemFilter(cfg, dec)
    not_ent = methods.NotEntailPolarityItemFilter(cfg, dec)
    it = toy_pool()[0]
    assert contra.key_term(it) == pytest.approx(not_ent.key_term(it), abs=TOL)
    # 1 - P_E >= P_C always, and STRICTLY above wherever the passage leaves neutral mass
    assert not_ent.distractor_polarity_term(it) > contra.distractor_polarity_term(it)
    assert not_ent.distractor_polarity_term(it) == pytest.approx(2.0 / 3.0, abs=1e-5)
    assert contra.distractor_polarity_term(it) == pytest.approx(0.6 * 2.0 / 3.0, abs=1e-5)
    share = not_ent.neutral_share_of_polarity_term(it)
    assert 0.0 <= share <= 1.0 and math.isfinite(share)
    assert share == pytest.approx(0.4, abs=1e-5)   # the neutral share the fixture puts there


# ============================================================ ITEM 21/46: the null control arm
def test_the_null_arm_returns_its_input_statistics_unchanged():
    cfg = _cfg()
    dec = _decomposer()
    pool = identical_pool(4)
    pool_rates = dec.rates(pool)
    null_arm = methods.NullRandomDropFilter(cfg, dec)
    assert null_arm.reads_judge_tensor is False
    report = null_arm.report(pool, seed=0)
    for key in analysis.RATE_KEYS:
        if key == "n_items":
            continue
        assert report[key] == pytest.approx(pool_rates[key], abs=TOL), key
    assert report["filter_scorer"] == "seeded_uniform_random"
    assert report["threshold_sweep"] == {}


def test_the_null_arm_retains_the_same_ids_on_a_repeat_call():
    cfg = _cfg()
    dec = _decomposer()
    pool = toy_pool()
    arm = methods.NullRandomDropFilter(cfg, dec)
    first = list(arm.report(pool, seed=1)["retained_item_ids"])
    second = list(arm.report(pool, seed=1)["retained_item_ids"])
    assert first == second
    other = list(methods.NullRandomDropFilter(cfg, dec).report(pool, seed=1)["retained_item_ids"])
    assert other == first


# ============================================================ ITEM 22/23/46: the positive control
def test_the_planted_control_does_not_mutate_the_input_pool():
    cfg = _cfg()
    dec = _decomposer()
    pool = toy_pool()
    before_options = [list(it.options) for it in pool]
    before_nli = [it.nli_orig.copy() for it in pool]
    arm = methods.PositiveControlPlanted(cfg, FakeProxyLLM(), FakeNLI(), dec)
    ids = arm.choose_ids(pool, seed=0)
    copies = arm.plant(pool, ids)
    arm.rescore(copies)
    for i, it in enumerate(pool):
        assert it.options == before_options[i]
        assert np.allclose(it.nli_orig, before_nli[i])
        assert it.planted is False
    assert any(c.planted for c in copies)


def test_the_planted_slot_carries_the_items_own_key():
    cfg = _cfg()
    arm = methods.PositiveControlPlanted(cfg, FakeProxyLLM(), FakeNLI(), _decomposer())
    pool = toy_pool()
    copies = arm.plant(pool, [it.item_id for it in pool])
    slot = cfg.planted_slot
    for original, copy in zip(pool, copies):
        assert copy.options[slot] == original.options[0]
        assert copy.planted is True


def test_the_planted_control_raises_the_rate_by_the_planted_fraction():
    cfg = _cfg()
    dec = _decomposer()
    pool = unsupported_pool(4)          # baseline rate 0.0, keys all supported at 0.9
    baseline = dec.rates(pool)["distractor_caused_csd_rate"]
    arm = methods.PositiveControlPlanted(cfg, FakeProxyLLM(), FakeNLI(probs=(0.05, 0.9, 0.05)), dec)
    report = arm.report(pool, seed=0)
    assert report["distractor_caused_csd_rate"] > baseline
    assert report["planted_fraction"] == pytest.approx(cfg.planted_fraction, abs=TOL)
    assert report["measured_rate_increase"] > 0.0
    assert "expected_rate_increase_naive" in report


def test_the_planted_expectation_corrects_for_the_overwritten_slot():
    cfg = _cfg()
    arm = methods.PositiveControlPlanted(cfg, FakeProxyLLM(), FakeNLI(), _decomposer())
    # slot 1 is ALREADY distractor-caused on every item, so overwriting it adds nothing
    already = [make_item(i, 0, [0.9, 0.9, 0.1, 0.1], [0.9, 0.9, 0.1, 0.1]) for i in range(2)]
    corrected = arm.expected_rate_increase(already, [0, 1])
    assert corrected == pytest.approx(0.0, abs=TOL)
    naive = cfg.planted_fraction / 3.0
    assert naive > 0.0                      # the naive constant would have claimed a rise
    assert arm.expected_rate_increase([], []) is None
    assert arm.expected_rate_increase(toy_pool(), []) is None


# ============================================================ ITEM 24/46: candidate provenance
def test_both_rerank_arms_report_the_same_candidate_provenance():
    cfg = _cfg()
    dec = _decomposer()
    reply = '["alpha", "beta", "gamma", "delta"]'
    items = [make_item(0, 0, [0.9, 0.1, 0.1, 0.1], [0.9, 0.1, 0.1, 0.1])]
    contra = methods.OvergenerateContradictionReranker(cfg, FakeLLM([reply]), FakeNLI(),
                                                       FakeEmbedder(), dec)
    sim = methods.OvergenerateKeySimilarityReranker(cfg, FakeLLM([reply]), FakeNLI(),
                                                    FakeEmbedder(), dec)
    r1 = contra.report(items, seed=0)
    r2 = sim.report(items, seed=0)          # same cached candidates, no second generation
    assert r1["n_padded_candidate_sets"] == r2["n_padded_candidate_sets"] == 0
    assert r1["n_line_split_candidate_sets"] == r2["n_line_split_candidate_sets"] == 0
    assert r1["n_repair_improved_candidate_sets"] == r2["n_repair_improved_candidate_sets"] == 0
    assert r1["selection_criterion"] == "descending_p_contradict"
    assert r2["selection_criterion"] == "descending_cosine_to_key"


def test_the_candidate_padding_fallback_marks_the_item_and_counts():
    cfg = _cfg()
    items = [make_item(0, 0, [0.9, 0.1, 0.1, 0.1], [0.9, 0.1, 0.1, 0.1])]
    assert items[0].cand_padded is False
    arm = methods.OvergenerateContradictionReranker(cfg, FakeLLM(['["only one"]']), FakeNLI(),
                                                    FakeEmbedder(), _decomposer())
    arm.generate_candidates(items)
    assert items[0].cand_padded is True
    assert len(items[0].candidates) >= 3
    clean = [make_item(1, 0, [0.9, 0.1, 0.1, 0.1], [0.9, 0.1, 0.1, 0.1])]
    arm2 = methods.OvergenerateContradictionReranker(
        cfg, FakeLLM(['["a", "b", "c", "d"]']), FakeNLI(), FakeEmbedder(), _decomposer())
    arm2.generate_candidates(clean)
    assert clean[0].cand_padded is False


def test_the_repair_retry_keeps_the_longer_candidate_list():
    cfg = _cfg()
    # first call parses four candidates; the repair reply would give only two
    llm = ScriptedLLM(['["a", "b", "c", "d"]', '["x", "y"]'])
    items = [make_item(0, 0, [0.9, 0.1, 0.1, 0.1], [0.9, 0.1, 0.1, 0.1])]
    arm = methods.OvergenerateContradictionReranker(cfg, llm, FakeNLI(), FakeEmbedder(),
                                                    _decomposer())
    arm.generate_candidates(items)
    assert len(items[0].candidates) == 4
    assert "a" in items[0].candidates


def test_parse_candidate_list_reports_the_line_split_fallback():
    clean, flag = data.parse_candidate_list('["alpha", "beta", "gamma"]', "key")
    assert clean == ["alpha", "beta", "gamma"]
    assert flag is False
    lines, flag2 = data.parse_candidate_list("- alpha\n- beta\n- gamma", "key")
    assert lines == ["alpha", "beta", "gamma"]
    assert flag2 is True
    empty, flag3 = data.parse_candidate_list("", "key")
    assert empty == [] and flag3 is False


# ============================================================ ITEM 25: the H4 quartile floor
def test_the_h4_quartile_ratio_needs_support_in_both_quartiles():
    cfg = _cfg()
    arm = methods.OvergenerateContradictionReranker(cfg, FakeLLM([""]), FakeNLI(),
                                                    FakeEmbedder(), _decomposer())
    assert methods.OvergenerateContradictionReranker.quartile_floor(200) == 5
    assert methods.OvergenerateContradictionReranker.quartile_floor(400) == 10
    thin = [candidate_item(0, [True, False, False, False], [0.9, 0.7, 0.5, 0.3])]
    out = arm.similarity_quartiles(thin)
    assert out["similarity_quartile_csd_ratio"] is None
    assert out["similarity_quartile_csd_ratio_raw"] is None


def test_an_item_with_too_few_candidates_is_not_counted_as_support():
    cfg = _cfg()
    arm = methods.OvergenerateContradictionReranker(cfg, FakeLLM([""]), FakeNLI(),
                                                    FakeEmbedder(), _decomposer())
    tiny = [candidate_item(0, [True, True, True], [0.9, 0.6, 0.2])]
    out = arm.similarity_quartiles(tiny)
    assert out["n_candidates_scored"] == 0
    assert out["similarity_quartile_csd_ratio"] is None


# ============================================================ ITEM 39/46: topic labels
def test_topic_label_fallbacks_are_counted():
    cfg = _cfg()
    corpus = FakeCorpus(["photosynthesis converts light energy into chemical energy in plants",
                         "mitochondria release energy stored in glucose molecules"])
    labels, n_fallback = methods.label_topics(FakeLLM(["Cellular Respiration"]), corpus, [0, 1], cfg)
    assert len(labels) == 2
    assert n_fallback == 0
    _, n_fb = methods.label_topics(FakeLLM([""]), corpus, [0], cfg)
    assert n_fb == 1
    _, n_mixed = methods.label_topics(FakeLLM(["Good Label", ""]), corpus, [0, 1], cfg)
    assert n_mixed == 1


def test_the_closed_book_control_prompt_never_carries_the_passage():
    cfg = _cfg()
    passage = "ZQXPROBE the gold passage text that must never leak"
    corpus = FakeCorpus([passage])
    gen = methods.TopicOnlyQuizGenerator(cfg, FakeLLM([""]), corpus)
    assert gen.prompt_uses_passage() is False
    prompt = gen.build_prompt("some topic", passage)
    assert "ZQXPROBE" not in prompt
    placeholder = methods.passage_free_fallback_label(corpus, 0)
    assert "ZQXPROBE" not in placeholder
    assert placeholder
    assert "ZQXPROBE" in methods.topic_fallback_label(corpus, 0)   # grounded arm may use it


# ============================================================ ITEM 40: the proxy streams
def test_the_functionality_proxy_gives_every_context_its_own_stream():
    seen = set()
    for seed in range(4):
        for item_id in range(6):
            a, b = methods.proxy_sampling_seeds(seed, item_id)
            assert a != b
            assert a not in seen and b not in seen
            seen.add(a)
            seen.add(b)


def test_the_functionality_proxy_does_not_mutate_shared_items():
    cfg = _cfg()
    proxy = methods.FunctionalityProxy(cfg, FakeProxyLLM())
    proxy.set_samples(2)
    pool = toy_pool()
    before = [(list(it.options), it.stem, it.nli_orig.copy(), it.functional) for it in pool]
    rate = proxy.functional_rate(pool, seed=0)
    assert rate is None or (0.0 <= rate <= 1.0 and math.isfinite(rate))
    for it, (opts, stem, nli, func) in zip(pool, before):
        assert it.options == opts
        assert it.stem == stem
        assert np.allclose(it.nli_orig, nli)
        assert it.functional is func
    assert proxy.functional_rate([], seed=0) is None


# ============================================================ ITEM 46: the rewrite-noop counter
def test_the_rewrite_noop_counter_increments_on_an_unchanged_stem():
    cfg = _cfg()
    item = make_item(0, 0, [0.9, 0.1, 0.1, 0.1], [0.9, 0.1, 0.1, 0.1], stem="what is photosynthesis?")
    echo = methods.ItemScorer(cfg, FakeLLM(["what is photosynthesis?"]), FakeNLI(), FakeEmbedder())
    assert echo.rewrite_noops == 0
    echo.rewrite_stems([item])
    assert echo.rewrite_noops == 1
    assert item.stem_rewritten is not None

    fresh = make_item(1, 0, [0.9, 0.1, 0.1, 0.1], [0.9, 0.1, 0.1, 0.1], stem="what is photosynthesis?")
    real = methods.ItemScorer(cfg, FakeLLM(["which process in plants converts light to sugar?"]),
                              FakeNLI(), FakeEmbedder())
    real.rewrite_stems([fresh])
    assert real.rewrite_noops == 0
    assert fresh.stem_rewritten != fresh.stem


# ============================================================ ITEM 37: SeedAbort
def test_seed_abort_records_an_undefined_rate_as_none():
    unset = data.SeedAbort("grounded pool is empty", stage="generation")
    assert unset.rate is None
    assert unset.to_record()["rate"] is None
    assert "rate" not in str(unset)
    nan_rate = data.SeedAbort("bad", stage="generation", rate=float("nan"))
    assert nan_rate.to_record()["rate"] is None
    measured = data.SeedAbort("parse failure", stage="generation", rate=0.4)
    assert measured.to_record()["rate"] == pytest.approx(0.4, abs=TOL)
    assert "0.4" in str(measured)


# ============================================================ ITEM 34: the NLI token plan
def test_the_nli_premise_budget_does_not_depend_on_the_hypothesis():
    short = data.nli_pair_token_plan(600, 10)
    long = data.nli_pair_token_plan(600, 60)
    assert short["premise_allowance"] == long["premise_allowance"]
    assert short["premise_keep"] == long["premise_keep"]
    assert short["removed_tokens"] == long["removed_tokens"] > 0
    assert short["truncated"] is True
    fits = data.nli_pair_token_plan(100, 10)
    assert fits["truncated"] is False and fits["removed_tokens"] == 0


def test_a_pass_always_starts_at_the_home_batch_size():
    assert data.pass_start_batch_size(32, 2) == 32
    assert data.pass_start_batch_size(8, 8) == 8


# ============================================================ ITEM 36: the corpus cap
def test_the_corpus_cap_keeps_protected_ids_and_remaps_every_passage_id():
    passages = [f"passage number {i}" for i in range(20)]
    sources = ["sciq_support"] * 20
    gold = {("sciq", "test", i): i for i in range(20)}
    protected = [3, 11, 17]
    kept_p, kept_s, remapped, kept_idx, record = data.apply_corpus_cap(
        passages, sources, gold, protected, cap=8, seed=0)
    assert len(kept_p) == 8 and len(kept_s) == 8
    assert record["kept"] == 8 and record["total"] == 20 and record["capped"] is True
    assert record["n_protected"] == 3
    for pid in protected:
        new_id = remapped[("sciq", "test", pid)]
        assert kept_p[new_id] == passages[pid]     # same TEXT, remapped id
    for old_key, new_id in remapped.items():
        assert kept_p[new_id] == passages[old_key[2]]


def test_a_cap_below_the_protected_count_is_reported_not_silently_widened():
    passages = [f"p{i}" for i in range(10)]
    sources = ["sciq_support"] * 10
    gold = {("sciq", "test", i): i for i in range(10)}
    _, _, _, _, record = data.apply_corpus_cap(passages, sources, gold, [0, 1, 2, 3, 4],
                                               cap=2, seed=0)
    assert record["cap_requested"] == 2
    assert record["cap_effective"] >= 5
    assert record["n_protected_over_cap"] == 3


def test_the_replication_pool_size_follows_the_config():
    cfg = _cfg()
    assert cfg.obqa_replication_items >= 3


# ============================================================ ITEM 30/51: the budget lever
def test_plan_reduction_returns_one_value_not_a_pair():
    fits = analysis.plan_reduction(100.0, 1000.0, 0.25)
    assert not isinstance(fits, tuple)
    assert fits == pytest.approx(1.0, abs=TOL)
    tight = analysis.plan_reduction(2000.0, 1000.0, 0.25)
    assert not isinstance(tight, tuple)
    assert tight is not None and 0.25 <= tight < 1.0
    impossible = analysis.plan_reduction(100000.0, 1000.0, 0.25)
    assert impossible is None
    assert analysis.plan_reduction(100000.0, 1000.0, 0.25, smoke=True) == pytest.approx(1.0, abs=TOL)


def test_uniform_reduction_scales_every_component_by_one_fraction():
    cfg = _cfg()
    assert analysis.apply_uniform_reduction(cfg, 1.0) == {}
    assert analysis.apply_uniform_reduction(cfg, None) == {}
    cfg2 = _cfg()
    before = {name: getattr(cfg2, name) for name, _ in analysis.REDUCIBLE_COMPONENTS}
    rec = analysis.apply_uniform_reduction(cfg2, 0.5)
    assert rec["fraction"] == pytest.approx(0.5, abs=TOL)
    for name, floor in analysis.REDUCIBLE_COMPONENTS:
        after = getattr(cfg2, name)
        assert after >= floor
        assert after <= before[name]
        assert name in rec["components"]
    assert cfg2.seeds == cfg.seeds          # a seed is never a reducible component


def test_budget_insufficient_message_states_the_arithmetic():
    cfg = _cfg()
    msg = analysis.budget_insufficient_message(90000.0, 1000.0, 0.25, cfg)
    assert msg.startswith("BUDGET_INSUFFICIENT")
    assert "90000" in msg
    assert "1000" in msg
    assert "0.25" in msg


def test_truncate_folds_cuts_without_reordering():
    rows = [[9, 8, 7, 6], [5, 4, 3, 2]]
    out = analysis.truncate_folds(rows, 2)
    assert out == [[9, 8], [5, 4]]
    assert rows == [[9, 8, 7, 6], [5, 4, 3, 2]]     # input untouched


# ============================================================ ITEM 32: the printed lines
def test_seed_line_and_summary_line_cover_every_arm_and_never_print_nan():
    conditions = {name: {"distractor_caused_csd_rate": 0.02, "n_items": 3, "item_yield": 1.0}
                  for name in experiment_config.ALL_ARM_NAMES}
    conditions["overgenerate_rerank_by_contradiction"] = {"status": "failed: ValueError: boom"}
    lines = analysis.seed_metric_lines(conditions, seed=0)
    joined = "\n".join(lines)
    for name in experiment_config.ALL_ARM_NAMES:
        assert f"condition={name} seed=0" in joined
    assert "nan" not in joined.lower().replace("unchanged", "")
    assert "NA" in joined

    cfg = _cfg()
    st = analysis.Statistics(cfg)
    agg = analysis.aggregate(cfg, [seed_record(s, 0.02) for s in (0, 1, 2)], st)
    out = analysis.summary_lines(agg["conditions"])
    text = "\n".join(out)
    assert sum(1 for ln in out if ln.startswith("SUMMARY:")) == 1
    assert sum(1 for ln in out if ln.startswith("SUMMARY_SOFT:")) == 1
    summary_line = [ln for ln in out if ln.startswith("SUMMARY:")][0]
    for name in experiment_config.ALL_ARM_NAMES:
        assert name in summary_line
    assert "nan" not in text.lower().replace("unchanged", "")


# ============================================================ ITEM 43: the pre-flight checks
def test_the_self_tests_pass_on_the_real_config():
    lines = analysis.self_tests(_cfg())
    assert len(lines) == 4
    for line in lines:
        assert line.startswith("SELF_TEST:")


def test_a_clock_conditional_flag_is_rewritten():
    rewritten = analysis.flag_safe("OpenBookQA replication skipped (budget)")
    assert "skipped" not in rewritten.lower()
    plain = "generator loaded in float16 instead of nf4"
    assert analysis.flag_safe(plain) == plain


def test_the_environment_switches_must_agree():
    assert analysis.check_environment_switches(True, 1500.0, True, 1500.0) is None
    with pytest.raises(ValueError):
        analysis.check_environment_switches(True, 1500.0, False, 1500.0)
    with pytest.raises(ValueError):
        analysis.check_environment_switches(False, 900.0, False, 1500.0)


def test_the_seed_design_check_names_the_seed_list():
    assert analysis.check_seed_design([0, 1, 2], smoke=False) is None
    assert analysis.check_seed_design([0], smoke=True) is None
    with pytest.raises(ValueError) as exc:
        analysis.check_seed_design([0, 1], smoke=False)
    assert "[0, 1]" in str(exc.value)


# ============================================================ paired statistics
def test_the_paired_test_is_none_below_two_pairs_or_at_zero_variance():
    t, p, d = analysis.paired_t_and_cohen([0.1], [0.2])
    assert (t, p, d) == (None, None, None)
    t2, p2, d2 = analysis.paired_t_and_cohen([0.1, 0.2], [0.1, 0.2])
    assert (t2, p2, d2) == (None, None, None)
    t3, p3, d3 = analysis.paired_t_and_cohen([0.3, 0.4, 0.9], [0.1, 0.1, 0.1])
    for v in (t3, p3, d3):
        assert v is not None and math.isfinite(v)


def test_the_seed_level_test_names_itself_underpowered():
    st = analysis.Statistics(_cfg())
    rec = st.wilcoxon_seed_level({0: 0.3, 1: 0.4, 2: 0.5}, {0: 0.1, 1: 0.1, 2: 0.1})
    assert rec["n_seeds"] == 3
    assert rec["underpowered"] is True
    assert rec["seeds"] == [0, 1, 2]          # the published name for the id-aligned seed list
    assert "aligned_ids" not in rec
    assert rec["mean_diff"] == pytest.approx(0.3, abs=TOL)


def test_a_regime_cell_below_the_floor_carries_no_rate():
    cfg = _cfg()
    cfg.min_cell_items = 50
    st = analysis.Statistics(cfg)
    cells = st.regime_cells(_decomposer().flag_rows(toy_pool(), "x"))
    for cell in cells.values():
        if cell.get("below_floor"):
            assert cell.get(analysis.PRIMARY) is None


def test_row_helpers_never_share_a_list_between_calls():
    a = analysis.empty_rows("one")
    b = analysis.empty_rows("two")
    # ONE WHOLE ROW: n_rows counts the "keys" column, and assert_rows_rectangular requires every
    # ROW_FIELDS column to advance together, so a probe that appended to "dc" alone measured
    # nothing and left the table ragged -- the shape the row contract exists to forbid.
    a["keys"].append((0, 0))
    a["dc"].append(True)
    assert b["keys"] == [] and b["dc"] == []
    assert analysis.n_rows(None) == 0
    assert analysis.n_rows(a) == 1
    rows = _decomposer().flag_rows(toy_pool(), "x")
    subset = analysis.subset_rows(rows, [True, False, False] * 3)
    assert analysis.n_rows(subset) == 3
    assert analysis.n_rows(rows) == 9      # input untouched
    joined = analysis.concat_rows([rows, subset], "x")
    assert analysis.n_rows(joined) == 12


# ============================================================ text utilities
def test_lexical_overlap_is_zero_not_nan_without_content_tokens():
    assert data.lexical_overlap("", "some passage") == 0.0
    assert data.lexical_overlap("the of and", "some passage") == 0.0
    assert data.lexical_overlap("chlorophyll", "chlorophyll absorbs light") == pytest.approx(1.0, abs=TOL)


def test_enumeration_density_is_zero_not_nan_on_an_empty_passage():
    assert data.enumeration_density("") == 0.0
    assert 0.0 <= data.enumeration_density("Plants use water, light, and carbon dioxide.") <= 1.0


def test_parse_item_json_refuses_a_partial_item():
    ok = data.parse_item_json('{"stem": "q?", "key": "a", "distractors": ["b", "c", "d"]}')
    assert ok is not None and len(ok["distractors"]) == 3
    assert data.parse_item_json('{"stem": "q?", "key": "a", "distractors": ["b", "b"]}') is None
    assert data.parse_item_json("not json at all") is None
    assert data.parse_item_json("") is None


def test_a_pinned_revision_is_reachable_by_role_and_by_repo_id():
    """The pin table is declared by ROLE, but models._local_kw holds only the repo id when it
    decides whether to pass `revision=`. A role-keyed dict read with a repo id returned None on
    every lookup, so a set pin loaded the default branch and the payload still said null."""
    table = experiment_config.MODEL_REVISIONS
    ids = experiment_config.MODEL_IDS
    # every declared model has exactly one pin entry, so a new model cannot arrive unpinnable
    assert set(table) == set(ids)

    original = dict(table)
    try:
        for role, repo_id in ids.items():
            assert experiment_config.model_revision(role) is None
            assert experiment_config.model_revision(repo_id) is None
            table[role] = f"sha-for-{role}"
            # the value the payload records (read by role) ...
            assert experiment_config.model_revision(role) == f"sha-for-{role}"
            # ... is the value the loader requests (read by repo id), for the SAME one entry
            assert experiment_config.model_revision(repo_id) == f"sha-for-{role}"
            assert experiment_config.model_revision(repo_id) == table[role]
            # the other two models are untouched by this model's pin
            for other_role, other_repo in ids.items():
                if other_role == role:
                    continue
                assert experiment_config.model_revision(other_repo) is None
            table[role] = None
            assert experiment_config.model_revision(repo_id) is None
    finally:
        table.clear()
        table.update(original)
    assert dict(table) == original

    # a name that is neither a role nor a repo id must raise: an unreadable pin is not "no pin"
    with pytest.raises(KeyError) as exc:
        experiment_config.model_revision("nli-deberta-v3-base")
    assert "nli-deberta-v3-base" in str(exc.value)


def toy_corpus_builder(cfg: Any, n: int = 20) -> Any:
    """A CorpusBuilder holding n plain passages with one SciQ-test gold entry each, built with no
    dataset load: the cap, the density split and the protected ids are pure list work."""
    corpus = data.CorpusBuilder(cfg)
    corpus.passages = [f"passage number {i} about chlorophyll, light and water" for i in range(n)]
    corpus.sources = ["sciq_support"] * n
    corpus.gold_support_id = {("sciq", "test", i): i for i in range(n)}
    return corpus


def test_the_corpus_is_capped_exactly_once_with_the_protected_ids_in_hand():
    """finalize_density fills the density without capping, so the folds can be chosen from the
    UNCAPPED corpus; finalize_corpus then applies the cap ONCE with those ids protected. Capping
    to get the density pruned the corpus with an empty protected list and the second call then
    overwrote cap_record with a no-op 'capped: False'."""
    cfg = _cfg()
    cfg.corpus_passage_cap = 8
    corpus = toy_corpus_builder(cfg, 20)

    corpus.finalize_density()
    assert corpus.density_finalized is True
    assert corpus.finalized is False                  # nothing capped, no id moved
    assert corpus.enum_density is not None and len(corpus.enum_density) == 20
    assert len(corpus.passages) == 20
    assert corpus.composition()["cap"] == {}

    protected = corpus.protected_passage_ids([[3, 11, 17]], [])
    assert protected == [3, 11, 17]
    record = corpus.finalize_corpus(protected)
    assert record["capped"] is True
    assert record["kept"] == 8 and record["total"] == 20
    assert record["n_protected"] == 3
    # the record and the flag describe the SAME cut
    assert corpus.composition()["cap"]["capped"] is True
    assert corpus.composition()["cap"]["kept"] == 8
    assert any("corpus capped to 8 of 20" in f for f in corpus.flags)
    # every protected id survived and kept a remapped gold entry pointing at its own text
    for gi in (3, 11, 17):
        new_id = corpus.gold_support_id[("sciq", "test", gi)]
        assert corpus.passages[new_id] == f"passage number {gi} about chlorophyll, light and water"

    # a second call would re-cap the already-capped corpus and rewrite the record; it must refuse
    with pytest.raises(RuntimeError) as exc:
        corpus.finalize_corpus(protected)
    assert "already" in str(exc.value)
    assert corpus.composition()["cap"]["capped"] is True   # the real record survived the refusal


def test_an_uncapped_corpus_records_no_cut():
    cfg = _cfg()
    cfg.corpus_passage_cap = None                    # the measured design embeds the whole corpus
    corpus = toy_corpus_builder(cfg, 12)
    corpus.finalize_density()
    record = corpus.finalize_corpus(corpus.protected_passage_ids([[0, 1]], []))
    assert record["capped"] is False
    assert record["kept"] == 12 and record["cap_requested"] is None
    assert not any("corpus capped" in f for f in corpus.flags)


def test_the_topic_labels_are_generated_once_for_both_arms():
    """ONE generation pass yields the grounded labels and the mask; the closed-book list is derived
    from that mask, so it cannot cost a second pass and cannot decode a different label."""
    cfg = _cfg()
    corpus = FakeCorpus(["photosynthesis converts light energy into chemical energy in plants",
                         "mitochondria release energy stored in glucose molecules"])
    llm = FakeLLM(["Cellular Respiration", ""])      # item 0 usable, item 1 falls back
    topics, n_fb, used_fallback = methods.label_topics_with_fallback_mask(llm, corpus, [0, 1], cfg)
    assert llm.calls == 1                            # one pass for BOTH arms
    assert n_fb == 1
    assert list(used_fallback) == [False, True]
    assert topics[0] == "Cellular Respiration"
    assert topics[1] == methods.topic_fallback_label(corpus, 1)

    topics_cb = [methods.passage_free_fallback_label(corpus, p)[:80] if fb else t
                 for t, fb, p in zip(topics, used_fallback, [0, 1])]
    assert topics_cb[0] == topics[0]                 # identical where the model gave a label
    assert topics_cb[1] == methods.PASSAGE_FREE_TOPIC
    assert "mitochondria" not in topics_cb[1]        # the control never receives passage text

    # the pair form is the same parsing and counting path, so the two cannot drift
    labels, n_fb2 = methods.label_topics(FakeLLM(["Cellular Respiration", ""]), corpus, [0, 1], cfg)
    assert labels == topics and n_fb2 == n_fb


def test_the_executed_design_record_compares_against_the_design_it_was_built_under():
    """The calibration set is built before the budget lever lowers calibration_pairs, so the built
    size must be checked against the bound in force when it was built. Comparing it against the
    reduced knob raised after the whole seed loop and discarded the run.

    The record takes the interface sheet's three parameters; the designed bound reaches it through
    cfg.calibration_pairs_designed, which Config declares as None and apply_uniform_reduction is
    the only writer of."""
    cfg = _cfg()
    cfg.items_per_seed = 3
    cfg.seeds = [0, 1]
    rows = [[10, 11, 12], [13, 14, 15]]
    n_built = 1600
    cfg.calibration_pairs = n_built

    # before any reduction the attribute EXISTS and is None, so the record falls back to the
    # config value; it is not an attribute that appears only after the lever has run
    assert cfg.calibration_pairs_designed is None
    rec0 = analysis.executed_design_record(cfg, rows, n_built)
    assert rec0["calibration_pairs"] == n_built
    assert rec0["calibration_pairs_designed"] == n_built
    assert rec0["n_seeds"] == 2 and rec0["items_per_seed"] == 3

    # the lever lowers the knob and records the bound the set was actually built under
    analysis.apply_uniform_reduction(cfg, 0.6)
    assert cfg.calibration_pairs == 960                  # what the reduction left behind
    assert cfg.calibration_pairs_designed == n_built
    assert cfg.items_per_seed == 3                       # at its floor, so the rows still match

    rec = analysis.executed_design_record(cfg, rows, n_built)
    assert rec["calibration_pairs"] == n_built           # the old comparison raised here
    assert rec["calibration_pairs_designed"] == n_built

    with pytest.raises(ValueError):                      # more pairs built than were ever designed
        analysis.executed_design_record(cfg, rows, n_built + 1)

    # a run the lever never touched still records the bound, as the config value
    untouched = _cfg()
    assert analysis.apply_uniform_reduction(untouched, 1.0) == {}
    assert untouched.calibration_pairs_designed is None


def test_the_row_table_columns_must_advance_together():
    rows = _decomposer().flag_rows(toy_pool(), "x")
    assert methods.assert_rows_rectangular(rows, "x") is None
    assert set(len(rows[f]) for f in analysis.ROW_FIELDS) == {9}
    ragged = analysis.empty_rows("x")
    ragged["dc"].append(True)                        # one column advanced without the others
    with pytest.raises(ValueError) as exc:
        methods.assert_rows_rectangular(ragged, "x")
    assert "ragged" in str(exc.value)
    assert "x" in str(exc.value)


def test_exactly_one_success_rate_line_is_emitted_per_arm():
    """summary_lines is the ONLY emitter of the success_rate line; main used to print a second one
    per arm from len(cfg.seeds), so one quantity carried two lines that could disagree."""
    cfg = _cfg()
    st = analysis.Statistics(cfg)
    agg = analysis.aggregate(cfg, [seed_record(s, 0.02) for s in (0, 1, 2)], st)
    lines = analysis.summary_lines(agg["conditions"], experiment_config.ALL_ARM_NAMES)
    for name in experiment_config.ALL_ARM_NAMES:
        matches = [ln for ln in lines if ln.startswith(f"condition={name} success_rate:")]
        assert len(matches) == 1, matches
        assert matches[0] == f"condition={name} success_rate: 3/3"


def test_the_device_fallback_reason_must_be_read_at_use_time():
    """detect_device rebinds DEVICE_FALLBACK_REASON at CALL time, so a name bound by a from-import
    freezes at whatever the global held during import. main.py holds only the module and reads
    experiment_config.DEVICE_FALLBACK_REASON at each use; a captured copy reported null for every
    fallback, which is a false 'no cause' beside device: cpu rather than a missing one."""
    def failing_probe() -> str:
        raise RuntimeError("libcuda missing")

    assert experiment_config.detect_device(probe=lambda: "cpu") == "cpu"
    assert experiment_config.DEVICE_FALLBACK_REASON is None
    captured_at_import = experiment_config.DEVICE_FALLBACK_REASON      # what a from-import binds

    assert experiment_config.detect_device(probe=failing_probe) == "cpu"
    live = experiment_config.DEVICE_FALLBACK_REASON                    # what a module read gives
    assert live is not None and "RuntimeError" in live and "libcuda missing" in live
    assert captured_at_import is None                                  # the stale copy never moves
    assert captured_at_import != live

    experiment_config.detect_device(probe=lambda: "cpu")               # leave the global clean
    assert experiment_config.DEVICE_FALLBACK_REASON is None


def test_a_failure_record_is_copied_before_extra_keys_are_merged():
    """guarded_replication merges an int-valued key into the str-valued map arm_failure_record
    returns, so it merges into a COPY: updating the record in place would both fail the checker
    and mutate a record other readers treat as status-only."""
    # bound before the try for the same reason as above
    base: Dict[str, str] = {}
    try:
        raise ValueError("openbookqa pool exploded")
    except ValueError as e:
        base = analysis.arm_failure_record(e)
    assert base, "the except handler must have produced the record under test"
    merged: Dict[str, Any] = dict(base)
    merged.update({"corpus": "openbookqa", "n_items": 3})
    assert "corpus" not in base and "n_items" not in base
    assert merged["n_items"] == 3
    assert merged["error_type"] == "ValueError"
    assert base["error_type"] == "ValueError"
    # both stay status-only, so the rates contract still exempts them
    assert analysis.carries_rate_keys(base) is False
    assert analysis.carries_rate_keys(merged) is False


def test_a_repaired_candidate_set_is_reported_by_both_rerank_arms():
    """The repair count is a fact about the shared cache, so it lives on the ITEM. Held on the
    arm instead, the arm that finds the cache already filled reported 0 for a pool that had been
    repaired, and one candidate pool carried two contradictory numbers in one payload."""
    cfg = _cfg()
    dec = _decomposer()
    items = [make_item(0, 0, [0.9, 0.1, 0.1, 0.1], [0.9, 0.1, 0.1, 0.1])]
    assert items[0].cand_repair_improved is False
    # the first reply parses two candidates (below the floor of three); the repair recovers four
    llm = ScriptedLLM(['["alpha", "beta"]', '["alpha", "beta", "gamma", "delta"]'])
    contra = methods.OvergenerateContradictionReranker(cfg, llm, FakeNLI(), FakeEmbedder(), dec)
    r1 = contra.report(items, seed=0)
    assert llm.calls == 2                                  # one generation pass plus its repair
    assert items[0].cand_repair_improved is True           # marked on the item, not the arm
    assert items[0].candidates is not None and len(items[0].candidates) == 4
    assert items[0].cand_padded is False                   # the repair made padding unnecessary
    assert r1["n_repair_improved_candidate_sets"] == 1

    # the ablation arm reranks the SAME cache and generates nothing; its count must still be 1
    second_llm = FakeLLM([""])
    sim = methods.OvergenerateKeySimilarityReranker(cfg, second_llm, FakeNLI(), FakeEmbedder(), dec)
    r2 = sim.report(items, seed=0)
    assert second_llm.calls == 0                           # no second generation pass
    assert r2["n_repair_improved_candidate_sets"] == 1
    assert r2["n_repair_improved_candidate_sets"] == r1["n_repair_improved_candidate_sets"]
    assert r2["n_padded_candidate_sets"] == r1["n_padded_candidate_sets"]
    assert r2["n_line_split_candidate_sets"] == r1["n_line_split_candidate_sets"]


def test_the_similarity_reranker_declares_the_reports_the_sheet_names():
    """Behavioural: the ablation arm's report answers on a toy pool with the cosine criterion, the
    full rate contract and the ids it evaluated, reranking the SAME cached candidates.

    Asserting on the class dictionary instead would pass on a broken body and fail on a rename."""
    cfg = _cfg()
    dec = _decomposer()
    reply = '["alpha", "beta", "gamma", "delta"]'
    items = [make_item(0, 0, [0.9, 0.1, 0.1, 0.1], [0.9, 0.1, 0.1, 0.1])]
    sim = methods.OvergenerateKeySimilarityReranker(cfg, FakeLLM([reply]), FakeNLI(),
                                                    FakeEmbedder(), dec)
    r = sim.report(items, seed=0)
    assert isinstance(r, dict)
    assert r["selection_criterion"] == "descending_cosine_to_key"
    assert r["reads_judge_tensor"] is True
    for key in analysis.RATE_KEYS:
        assert key in r
    assert list(items[0].candidates) == ["alpha", "beta", "gamma", "delta"]
    assert len(sim.evaluated_items()) == 1                       # 3 of 4 candidates selected
    assert r["selected_item_ids"] == [it.item_id for it in sim.evaluated_items()]
    assert r["item_yield"] == pytest.approx(1.0, abs=TOL)


def test_the_nli_evaluator_measures_every_option_slot():
    """NLIJudge.score_pairs raises on a non-finite tensor instead of substituting one, so the NLI
    decomposition has nothing to exclude and its denominators stay exactly 3n and n."""
    r = _decomposer().rates(toy_pool())
    assert r["n_distractor_slots_measured"] == 9
    assert r["n_distractor_slots_excluded"] == 0
    assert r["n_key_slots_excluded"] == 0
    assert r["distractor_caused_csd_rate"] == pytest.approx(1.0 / 9.0, abs=TOL)
    assert r["key_hallucination_rate"] == pytest.approx(0.0, abs=TOL)
    assert analysis.n_rows(_decomposer().flag_rows(toy_pool(), "x")) == 9


def test_an_unmeasured_judge_slot_is_excluded_not_counted_as_supported():
    """A substituted judge row is an UNDEFINED cell. It used to be imputed with 0.5 -- exactly the
    support threshold, against a >= rule -- so a broken forward pass read as a corpus-supported
    distractor and, on the key slot, as a measured key hallucination."""
    cfg = _cfg()
    dec = methods.FlagDecomposer(cfg, "llm")

    measured = make_item(1, 0, [0.9, 0.9, 0.1, 0.1], [0.9, 0.9, 0.1, 0.1])
    r_all = dec.rates([measured])
    assert r_all["n_distractor_slots_measured"] == 3
    assert r_all["distractor_caused_csd_rate"] == pytest.approx(1.0 / 3.0, abs=TOL)
    assert r_all["key_hallucination_rate"] == pytest.approx(0.0, abs=TOL)

    masked = make_item(0, 0, [0.9, 0.9, 0.1, 0.1], [0.9, 0.9, 0.1, 0.1])
    # the judge's forward pass was not finite for the key slot and for distractor 1
    masked.llm_pyes_valid = np.array([False, False, True, True])
    masked.llm_pyes_rewrite_valid = np.array([False, False, True, True])
    r = dec.rates([masked])
    assert r["n_distractor_slots_measured"] == 2
    assert r["n_distractor_slots_excluded"] == 1
    assert r["n_key_slots_excluded"] == 1
    # the only slot that could have flagged was excluded, so the rate is over the two measured
    # slots and neither is supported
    assert r["distractor_caused_csd_rate"] == pytest.approx(0.0, abs=TOL)
    assert r["corpus_supported_rate"] == pytest.approx(0.0, abs=TOL)
    # no measured key slot means the key rates are undefined, never a fabricated hallucination
    assert r["key_hallucination_rate"] is None
    assert r["mean_key_p_support_orig"] is None
    assert r["stem_caused_share"] is None
    for key in analysis.RATE_KEYS:
        v = r[key]
        assert v is None or (isinstance(v, float) and math.isfinite(v)), f"{key} -> {v!r}"

    # an unmeasured slot emits no row either, so the item bootstrap cannot resample it
    rows = dec.flag_rows([masked], "x")
    assert analysis.n_rows(rows) == 2
    assert set(rows["keys"]) == {(0, 0)}
    assert methods.assert_rows_rectangular(rows, "x") is None


def test_every_slot_excluded_leaves_the_rates_undefined_not_zero():
    cfg = _cfg()
    dec = methods.FlagDecomposer(cfg, "llm")
    it = make_item(0, 0, [0.9, 0.9, 0.9, 0.9], [0.9, 0.9, 0.9, 0.9])
    it.llm_pyes_valid = np.zeros(4, dtype=bool)
    it.llm_pyes_rewrite_valid = np.zeros(4, dtype=bool)
    r = dec.rates([it])
    assert r["n_items"] == 1
    assert r["n_distractor_slots_measured"] == 0
    assert r["n_distractor_slots_excluded"] == 3
    for key in analysis.RATE_KEYS:
        assert r[key] is None, f"{key} over an unmeasured pool must be None, got {r[key]!r}"
    assert analysis.n_rows(dec.flag_rows([it], "x")) == 0


def test_a_calibration_cell_below_the_support_floor_is_recorded_not_raised():
    """ITEM 33: a calibration cell with an empty denominator is None, never a measured 0.0 -- that
    number decides the evaluator and so relabels every rate in the results.

    The floor and the verdict live in analysis so this suite can drive them: the suite may not
    import models.py, and models.JudgeCalibrator calls these same two functions."""
    # the floor comes from the set ACTUALLY built, so a 20-pair smoke run is not judged against a
    # floor it structurally cannot meet, and a 1600-pair run is held to the configured cap
    assert analysis.calibration_support_floor(1600, 10, 5) == 5
    assert analysis.calibration_support_floor(20, 10, 5) == 2
    assert analysis.calibration_support_floor(5, 10, 5) == 1       # never below one
    assert analysis.calibration_support_floor(None, 10, 5) == 5    # never above the cap

    below = analysis.calibration_precision_verdict(n_top_decile_pairs=2, min_top_decile_pairs=5,
                                                   precision=0.0, gate=0.6, n_pairs=20,
                                                   n_pred_pos=0)
    assert below["top_decile_below_floor"] is True
    assert below["judge_precision_top_overlap_decile"] is None    # recorded undefined, not 0.0
    assert below["passes"] is False
    assert "no verdict is claimed" in below["passes_reason"]
    assert "2 pairs" in below["passes_reason"] and "floor of 5" in below["passes_reason"]

    undefined = analysis.calibration_precision_verdict(n_top_decile_pairs=30,
                                                       min_top_decile_pairs=5, precision=None,
                                                       gate=0.6, n_pairs=1600, n_pred_pos=0)
    assert undefined["top_decile_below_floor"] is False
    assert undefined["judge_precision_top_overlap_decile"] is None
    assert undefined["passes"] is False
    assert "not zero" in undefined["passes_reason"]

    passing = analysis.calibration_precision_verdict(n_top_decile_pairs=30,
                                                     min_top_decile_pairs=5, precision=0.7,
                                                     gate=0.6, n_pairs=1600, n_pred_pos=25)
    assert passing["passes"] is True
    assert passing["judge_precision_top_overlap_decile"] == pytest.approx(0.7, abs=TOL)
    assert ">=" in passing["passes_reason"]

    failing = analysis.calibration_precision_verdict(n_top_decile_pairs=30,
                                                     min_top_decile_pairs=5, precision=0.5,
                                                     gate=0.6, n_pairs=1600, n_pred_pos=25)
    assert failing["passes"] is False
    assert failing["judge_precision_top_overlap_decile"] == pytest.approx(0.5, abs=TOL)
    assert ">=" not in failing["passes_reason"]

    # every branch RETURNS: an under-filled decile is a diagnostic that must reach the payload,
    # not an exception that discards the models, corpus, index and pilot already paid for
    for verdict in (below, undefined, passing, failing):
        assert set(verdict) == {"judge_precision_top_overlap_decile", "top_decile_below_floor",
                                "passes", "passes_reason"}
        assert isinstance(verdict["passes"], bool)


def test_the_pooled_sweep_counts_only_the_seeds_whose_cell_was_admitted():
    """The pooled tau cell's n_items must be the support behind its rate. Summing every seed's
    retained count while averaging only the ADMITTED seeds published a larger number than the
    rate stood on, and decide_gates republishes it as h2_enrichment_n_items."""
    cfg = _cfg()
    st = analysis.Statistics(cfg)
    per_seed: List[Dict[str, Any]] = []
    for seed, (kept, admitted) in enumerate([(30, True), (30, True), (9, False)]):
        rec = seed_record(seed, 0.02)
        cell: Dict[str, Any] = {k: (0.05 if admitted else None) for k in analysis.RATE_KEYS}
        cell["n_items"] = kept
        cell["admitted"] = admitted
        cell["min_items"] = 10
        rec["conditions"]["per_option_entailment_filter"]["threshold_sweep"] = {"0.9": cell}
        per_seed.append(rec)

    agg = analysis.aggregate(cfg, per_seed, st)
    pooled = agg["per_option_threshold_sweep"]["0.9"]
    assert pooled["n_seeds"] == 2                       # only the two admitted cells carry a rate
    assert pooled["n_items"] == 60                      # 69 while the breach is present
    assert pooled["n_items_all_seeds"] == 69            # the suppressed cell is still reported
    assert pooled["rate"] == pytest.approx(0.05, abs=TOL)

    # the floor and the published support both read the admitted-only count
    enr = agg["per_option_enrichment"]
    assert enr["min_items"] == analysis.h2_min_sweep_items(len(cfg.seeds), cfg.items_per_seed)
    assert enr["admitted"]["0.9"] is True
    assert enr["selected_tau"] == pytest.approx(0.9, abs=TOL)
    assert enr["selected_n_items"] == 60
    assert enr["enrichment_at_selected_tau"] == pytest.approx(0.05 - 0.02, abs=TOL)

    gates = analysis.decide_gates(cfg, agg, "nli")
    assert gates["h2_enrichment_n_items"] == 60


def _write_fake_snapshot(hub_root: Any, repo_id: str, sha: str,
                         mtime: Optional[float] = None) -> str:
    """One COMPLETE cached snapshot (config.json plus a weight file) under a fake hub root.

    Hermetic and write-only: it creates files inside the pytest tmp_path the caller owns, touches
    no network, loads no model and READS no file at all -- least of all a project source file.
    Written through pathlib rather than the builtin open so the suite contains no file-reading
    idiom for a reader to mistake for source introspection. os is here only for utime, which is
    what makes the mtime-versus-ref distinction testable; both imports are function-local so the
    helper stays a self-contained top-level definition."""
    import os
    from pathlib import Path

    d = Path(str(hub_root)) / ("models--" + repo_id.replace("/", "--")) / "snapshots" / sha
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text("{}", encoding="utf-8")
    (d / "model.safetensors").write_text("not real weights; this test never loads a checkpoint",
                                         encoding="utf-8")
    if mtime is not None:
        os.utime(str(d), (float(mtime), float(mtime)))
    return str(d)


def _write_fake_ref(hub_root: Any, repo_id: str, ref: str, sha: str) -> str:
    """The refs file a local_files_only load follows, written under the fake hub root."""
    from pathlib import Path

    d = Path(str(hub_root)) / ("models--" + repo_id.replace("/", "--")) / "refs"
    d.mkdir(parents=True, exist_ok=True)
    path = d / ref
    path.write_text(sha + "\n", encoding="utf-8")
    return str(path)


def test_the_resolved_revision_is_read_from_the_ref_not_guessed_from_mtime(tmp_path, monkeypatch):
    """resolved_revision is the run's only artifact identity while MODEL_REVISIONS is unpinned.

    Two complete snapshots are cached and refs/main names the OLDER one -- a re-pull after an
    upstream commit, then anything that touches the older directory. The newest-by-mtime guess
    names the commit the load does NOT follow, so the payload would attribute the numbers to
    weights that did not produce them."""
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)

    loaded = "a" * 40          # what refs/main points at: the commit a load resolves
    newer = "b" * 40           # cached too, and written later, so the mtime guess prefers it
    repo = experiment_config.MODEL_IDS["embedder"]
    _write_fake_snapshot(tmp_path, repo, loaded, mtime=1_000_000.0)
    _write_fake_snapshot(tmp_path, repo, newer, mtime=2_000_000.0)
    _write_fake_ref(tmp_path, repo, "main", loaded)

    assert experiment_config.snapshot_cached(repo) is True
    rec = experiment_config.snapshot_revision_record(repo)
    assert rec["resolution"] == "ref_file"
    assert rec["resolved_revision"] == loaded          # the mtime guess would answer `newer`
    assert rec["ref_read"].endswith("main")
    assert rec["ref_names_a_complete_snapshot"] is True
    assert rec["n_complete_snapshots"] == 2
    assert rec["complete_snapshots"] == sorted([loaded, newer])
    # the scalar every caller consumes is the SAME value, never a second resolution path
    assert experiment_config.snapshot_revision(repo) == loaded

    # a pin that names a cached snapshot IS what loads, ahead of the ref
    table = experiment_config.MODEL_REVISIONS
    original = dict(table)
    try:
        table["embedder"] = newer
        pinned = experiment_config.snapshot_revision_record(repo)
        assert pinned["resolution"] == "pinned_revision"
        assert pinned["resolved_revision"] == newer
        assert experiment_config.snapshot_revision(repo) == newer
    finally:
        table.clear()
        table.update(original)
    assert experiment_config.snapshot_revision(repo) == loaded   # the pin is gone again

    # a repo with no refs directory at all: the guess is still taken, and LABELLED as a guess
    other = experiment_config.MODEL_IDS["nli"]
    _write_fake_snapshot(tmp_path, other, "c" * 40, mtime=1_000_000.0)
    _write_fake_snapshot(tmp_path, other, "d" * 40, mtime=2_000_000.0)
    guessed = experiment_config.snapshot_revision_record(other)
    assert guessed["resolution"] == "mtime_fallback"
    assert guessed["resolved_revision"] == "d" * 40
    assert guessed["complete_snapshots"] == sorted(["c" * 40, "d" * 40])

    # nothing cached at all stays None, and says so
    empty = experiment_config.snapshot_revision_record(experiment_config.MODEL_IDS["generator"])
    assert empty["resolved_revision"] is None
    assert empty["resolution"] == "uncached"
    assert empty["n_complete_snapshots"] == 0


def test_the_rerank_arm_refreshes_the_judge_mask_for_its_new_options():
    """The rebuilt item's judge mask must describe the options it actually holds.

    build_selected_items replaces three of four options, and dataclasses.replace copies
    llm_pyes_valid from the pre-rerank item; a mask carried across describes text the item no
    longer contains. The module-level _rescore_selected that refreshed it was never reachable --
    it took a self argument no caller supplied -- so the live method left the stale mask in place.

    Here slot 3 is imputed BEFORE the rerank and slot 1 is imputed by the judge that scores the
    new options. A stale mask would exclude slot 3 and admit the imputed slot 1, which is the
    exact inversion the defect produced."""
    cfg = _cfg()
    cfg.n_overgen_candidates = 4
    dec = methods.FlagDecomposer(cfg, "llm")
    # the post-rerank judge marks slot 1 of every item invalid; the pre-rerank mask marks slot 3
    llm = FakeProxyLLM().impute_pyes_rows([1])
    rr = methods.OvergenerateContradictionReranker(cfg, llm, FakeNLI(), FakeEmbedder(), dec)

    items = [candidate_item(0, [False, False, False, False], [0.9, 0.7, 0.5, 0.3])]
    it = items[0]
    it.llm_pyes = np.array([0.9, 0.9, 0.9, 0.0], dtype=np.float32)
    it.llm_pyes_rewrite = np.array([0.9, 0.9, 0.9, 0.0], dtype=np.float32)
    it.llm_pyes_valid = np.array([True, True, True, False])       # slot 3 was imputed pre-rerank
    it.llm_pyes_rewrite_valid = np.array([True, True, True, False])

    sel = rr.build_selected_items(items)
    assert len(sel) == 1
    new = sel[0]
    # the mask now describes the judge call that scored THESE options, not the previous ones
    assert list(new.llm_pyes_valid) == [True, False, True, True]
    assert list(new.llm_pyes_rewrite_valid) == [True, False, True, True]
    # and the decomposer excludes the slot the judge imputed, not the one it inherited
    v = dec.both_stages_valid(sel)
    assert list(v[0]) == [True, False, True, True]
    r = dec.rates(sel)
    assert r["n_distractor_slots_measured"] == 2
    assert r["n_distractor_slots_excluded"] == 1
    assert r["n_key_slots_excluded"] == 0

    # the module-level duplicate is gone: the name must not be a silently working function
    with pytest.raises(RuntimeError):
        methods._rescore_selected([])


def test_an_imputed_letter_row_is_excluded_from_the_functional_rate():
    """A uniform 0.25 row is a substitution, never four equiprobable observations.

    letter_probs_batch writes that row for a non-finite forward pass and reports it beside the
    probabilities; discarding the mask let the with/without comparison manufacture a functional
    flag out of noise, which feeds functional_distractor_rate and then h3_supported. A context is
    one item's WITH-passage / NO-passage pair and the statistic is the comparison between them,
    so either row imputed voids the whole context."""
    cfg = _cfg()
    cfg.functionality_samples_per_context = 8
    # rows 0 and 1 are item 0's with-passage and no-passage prompts; row 1 is imputed
    llm = FakeProxyLLM((0.1, 0.7, 0.1, 0.1)).impute_letter_rows([1])
    proxy = methods.FunctionalityProxy(cfg, llm)

    items = toy_pool(seed=0)[:2]
    for i, it in enumerate(items):
        it.item_id = i
    rate = proxy.functional_rate(items, seed=0)
    assert proxy.n_contexts_measured == 1        # item 1 only: item 0's no-passage row is imputed
    assert proxy.n_contexts_excluded == 1
    assert rate is not None
    rec = proxy.exclusion_record()
    assert rec["functional_contexts_measured"] == 1
    assert rec["functional_contexts_excluded"] == 1

    # every context imputed leaves the rate UNDEFINED, never 0.0
    llm_all_bad = FakeProxyLLM((0.1, 0.7, 0.1, 0.1)).impute_letter_rows([0, 1, 2, 3])
    proxy_all_bad = methods.FunctionalityProxy(cfg, llm_all_bad)
    assert proxy_all_bad.functional_rate(items, seed=0) is None
    assert proxy_all_bad.n_contexts_measured == 0
    assert proxy_all_bad.n_contexts_excluded == 2
    # the two counters and the mapping they back agree: every arm and functional_record read the
    # RECORD, never the raw attributes, so a counter that moved without the record following it
    # would leave the payload describing a rate that no longer stands on those contexts
    assert proxy_all_bad.exclusion_record() == {"functional_contexts_measured": 0,
                                                "functional_contexts_excluded": 2}

    # An arm publishes the counts for ITS OWN retained set, not for the pool. retention_fraction
    # is 0.5, so this filter keeps one item of two -- and the stable sort keeps item 0, whose
    # no-passage row is the imputed one. Its whole retained set is therefore unmeasured, so the
    # arm reports NO rate and the two counts say why: a 0.0 here would be a fabricated reading
    # of a comparison that never happened.
    f = methods.PerOptionEntailmentFilter(cfg, _decomposer(), proxy)
    r = f.report(items, seed=0)
    assert r["retained_item_ids"] == [0]
    assert r["functional_contexts_measured"] == 0
    assert r["functional_contexts_excluded"] == 1
    assert r["functional_distractor_rate"] is None
    # the counters are per CALL, so the pool-level 1/1 above did not leak into this arm's record
    assert proxy.n_contexts_measured == 0
    assert proxy.exclusion_record()["functional_contexts_measured"] == 0

    # an arm with no proxy at all publishes all three keys as null, never omits them
    no_proxy = methods.PerOptionEntailmentFilter(cfg, _decomposer())
    r2 = no_proxy.report(items, seed=0)
    assert r2["functional_distractor_rate"] is None
    assert r2["functional_contexts_measured"] is None
    assert r2["functional_contexts_excluded"] is None


def test_every_arm_report_carries_the_exclusion_counts_under_both_evaluators():
    """ITEM 44: the record the gate reads must satisfy its contract for EVERY arm, not just the
    three the spot tests cover.

    The exclusion counts were declared on the plan and produced by models.py but published by no
    consumer, so three shipped tests raised KeyError and the offline gate blocked before any GPU
    work. This drives all ten ALL_ARM_NAMES arms under both evaluators on a fresh three-item toy
    pool with injected fakes only -- no checkpoint, no network -- and asserts the counts are
    present, are integers, and cover exactly the three slots of every item.

    Nothing here is imputed, so the excluded counts must be zero: an arm that quietly dropped a
    measured slot would fail this as surely as one that never published the counts."""
    cfg = _cfg()
    for evaluator in ("nli", "llm"):
        dec = methods.FlagDecomposer(cfg, evaluator)
        records: Dict[str, Any] = {}

        # the two generator arms are recorded straight from the decomposition, exactly as
        # run_seed records them; the rest report through their own arm object
        for name in ("unfiltered_grounded_pool", "ungrounded_topic_only_control"):
            records[name] = dec.rates(toy_pool())

        # Dict[str, Any] because the arms are related by a PROTOCOL, not by inheritance: the
        # filters descend from BaseItemFilter, PositiveControlPlanted stands alone, and the two
        # rerankers form a third hierarchy, so a checker joins the values to `object` and reads
        # arm.report below as an attribute access on object. run_arm_sequence calls the same
        # report(items, seed) duck-typed, and inventing a shared base class in methods.py to
        # satisfy this container's join would change the production hierarchy for a test.
        arms: Dict[str, Any] = {
            "per_option_entailment_filter": methods.PerOptionEntailmentFilter(cfg, dec),
            "llm_judge_faithfulness_filter": methods.LLMJudgeFaithfulnessFilter(cfg, dec),
            "contradiction_polarity_item_filter": methods.ContradictionPolarityItemFilter(cfg, dec),
            "not_entail_polarity_item_filter": methods.NotEntailPolarityItemFilter(cfg, dec),
            "null_random_drop": methods.NullRandomDropFilter(cfg, dec),
            "positive_control_planted": methods.PositiveControlPlanted(
                cfg, FakeProxyLLM(), FakeNLI(), dec),
            "overgenerate_rerank_by_contradiction": methods.OvergenerateContradictionReranker(
                cfg, FakeProxyLLM(), FakeNLI(), FakeEmbedder(), dec),
            "overgenerate_rerank_by_key_similarity": methods.OvergenerateKeySimilarityReranker(
                cfg, FakeProxyLLM(), FakeNLI(), FakeEmbedder(), dec),
        }
        for name, arm in arms.items():
            records[name] = arm.report(toy_pool(), 0)   # a FRESH pool, so no arm sees another's cache

        assert set(records) == set(experiment_config.ALL_ARM_NAMES)
        for name, rec in records.items():
            for key in ("n_distractor_slots_measured", "n_distractor_slots_excluded",
                        "n_key_slots_excluded"):
                assert key in rec, f"{evaluator}/{name} publishes no {key}"
                assert isinstance(rec[key], int), f"{evaluator}/{name}.{key} -> {rec[key]!r}"
            n = int(rec["n_items"])
            assert n > 0
            # the two counts PARTITION the item's three distractor slots
            assert (rec["n_distractor_slots_measured"]
                    + rec["n_distractor_slots_excluded"]) == 3 * n, f"{evaluator}/{name}"
            assert rec["n_distractor_slots_excluded"] == 0    # nothing was imputed in this pool
            assert rec["n_key_slots_excluded"] == 0
            assert rec["key_hallucination_rate"] is not None  # every key slot was measured
            for key in analysis.RATE_KEYS:
                v = rec[key]
                assert v is None or (isinstance(v, float) and math.isfinite(v)), \
                    f"{evaluator}/{name}.{key} -> {v!r}"
            # the record the harness would write passes the gate's own contract
            assert analysis.enforce_rates_contract(0, name, rec) is None


def test_both_bootstrap_warnings_survive_one_record():
    """A degraded scheme and a zero-width interval are INDEPENDENT diagnostics, and an arm that
    raises both is at its least trustworthy.

    They were written to the same scalar out["warning"] in sequence, so the degradation text --
    the one fact saying the registered scheme was not the scheme used -- was overwritten by the
    degeneracy text and never reached bootstrap_warnings, the payload flags or the gates.

    Both conditions are forced here: rows_a holds an item id rows_b does not, which breaks the
    subset relation nested requires and degrades it to paired over the three shared ids; and the
    three shared ids carry the SAME per-item rate in both tables, so every paired resample
    subtracts an array from an identical array and hi - lo is EXACTLY 0.0.

    The shared rates are only 0/3 and 3/3 on purpose. Those two are exact in binary floating
    point; k/3 for k in (1, 2) is not, so a fixture built from thirds leaves a last-ulp spread
    between resamples of different composition and `hi - lo == 0.0` -- the exact test the code
    declares -- would be False for a comparison that is degenerate in every meaningful sense."""
    st = analysis.Statistics(_cfg())
    dec = _decomposer()

    def dc_item(item_id: int, n_dc: int) -> data.QuizItem:
        """One item whose first n_dc distractor slots are supported under BOTH stems."""
        p = [0.9] + [0.9 if d < n_dc else 0.1 for d in range(3)]
        return make_item(item_id, 0, p, p)

    # shared ids 0, 1, 2 at per-item dc rates 1.0, 0.0, 1.0 in BOTH tables
    a_items = [dc_item(0, 3), dc_item(1, 0), dc_item(2, 3)]
    b_items = [dc_item(0, 3), dc_item(1, 0), dc_item(2, 3)]
    a_items.append(dc_item(9, 3))          # present only in A, so A is not a subset of B
    rows_a = dec.flag_rows(a_items, "a")
    rows_b = dec.flag_rows(b_items, "b")

    rec = st.paired_bootstrap_diff(rows_a, rows_b, mode="nested")
    assert rec["requested_scheme"] == "nested"
    assert rec["degraded"] is True
    assert rec["scheme"] == "paired"       # three shared ids, so it degrades to paired
    assert rec["n_items"] == 3
    assert rec["zero_width"] is True
    lo, hi = ci_bounds(rec)
    assert hi - lo == pytest.approx(0.0, abs=TOL)
    assert rec["diff"] == pytest.approx(0.0, abs=TOL)

    # BOTH diagnostics are carried out of the call; neither overwrites the other
    assert len(rec["warnings"]) == 2
    joined = rec["warning"]
    assert any("degraded to" in w for w in rec["warnings"])
    assert any("degenerate" in w for w in rec["warnings"])
    # the scalar key every existing reader gates on now carries both texts, not just the last
    assert "degraded to" in joined            # this assertion fails while the breach is present
    assert "degenerate" in joined
    for w in rec["warnings"]:
        assert w in joined

    # and the aggregate's single-string copy therefore reports both facts for the arm
    assert joined == "; ".join(rec["warnings"])


def test_the_shared_scoring_pass_stores_the_judge_validity_mask():
    """ItemScorer.score_llm_judge must store the mask of the pass that produced the numbers.

    It called p_yes_batch, the wrapper that DROPS the mask, so an imputed
    NAN_P_YES_SUBSTITUTE = 0.0 was written into llm_pyes indistinguishably from a measured 0.0
    and then read as a measured non-support. On a key slot that is a counted key hallucination
    inside key_hallucination_rate, which feeds h2_supported and h3_supported; on a distractor
    slot it deflates every rate by a silent amount. Nothing downstream can recover the
    difference, because the only other trace is one aggregate substitution counter."""
    cfg = _cfg()
    # the judge's forward pass was not finite for the second prompt row of each batch
    llm = FakeProxyLLM().impute_pyes_rows([1])
    scorer = methods.ItemScorer(cfg, llm, FakeNLI(), FakeEmbedder())
    items = [make_item(0, 0, [0.9, 0.9, 0.9, 0.9], [0.9, 0.9, 0.9, 0.9])]
    assert items[0].llm_pyes_valid is None                 # nothing scored by the LLM judge yet

    scorer.score_llm_judge(items, "orig")
    scorer.score_llm_judge(items, "rewrite")
    it = items[0]
    # this is the assertion that fails while the breach is present: the mask is never stored
    assert list(it.llm_pyes_valid) == [True, False, True, True]
    assert list(it.llm_pyes_rewrite_valid) == [True, False, True, True]
    assert float(it.llm_pyes[1]) == pytest.approx(0.0, abs=TOL)      # the substitute ...
    assert float(it.llm_pyes[2]) == pytest.approx(0.9, abs=TOL)      # ... beside a measurement

    dec = methods.FlagDecomposer(cfg, "llm")
    r = dec.rates(items)
    assert r["n_distractor_slots_measured"] == 2
    assert r["n_distractor_slots_excluded"] == 1
    assert r["n_key_slots_excluded"] == 0
    # both MEASURED distractor slots are supported under both stems; the imputed one enters
    # neither the numerator nor the denominator, so the rate is 2/2 and not 2/3
    assert r["distractor_caused_csd_rate"] == pytest.approx(1.0, abs=TOL)
    assert r["corpus_supported_rate"] == pytest.approx(1.0, abs=TOL)
    assert r["key_hallucination_rate"] == pytest.approx(0.0, abs=TOL)
    assert analysis.n_rows(dec.flag_rows(items, "x")) == 2
    assert analysis.enforce_rates_contract(0, "unfiltered_grounded_pool", r) is None

    # the NLI evaluator substitutes nothing (score_pairs raises on a non-finite row), so its
    # mask is all-True and its denominators are exactly 3n and n
    nli_dec = methods.FlagDecomposer(cfg, "nli")
    assert nli_dec.valid_mask(items, "orig").all()
    assert nli_dec.rates(items)["n_distractor_slots_measured"] == 3


def test_a_masked_judge_return_must_be_the_pair():
    """probs_of / validity_mask_of exist and are STRICT about the shape they unpack.

    np.asarray on a (p, valid) pair of two length-n arrays does not raise -- it returns a (2, n)
    array -- so a caller that forgot the pair would read the validity mask as a row of
    probabilities and never find out. The strictness is what turns that into a programming error
    that main re-raises rather than a number that reaches the results."""
    p = np.full((3, 4), 0.9, dtype=np.float32)
    ok = np.ones(12, dtype=bool)

    # a bare array is NOT a masked judge return, however array-like it looks
    with pytest.raises(TypeError):
        methods.probs_of(p)
    with pytest.raises(TypeError):
        methods.validity_mask_of(p, 3, 4)

    assert methods.probs_of((p, ok)).shape == (3, 4)
    mask = methods.validity_mask_of((p, ok), 3, 4)
    assert mask.shape == (3, 4)
    assert mask.dtype == np.dtype(bool)
    assert mask.all()

    # a mask that does not describe the rows it arrived with excludes the WRONG slot
    with pytest.raises(ValueError):
        methods.validity_mask_of((p, np.ones(5, dtype=bool)), 3, 4)


def test_the_rerank_report_runs_the_llm_judge_path_with_a_live_proxy():
    """The one call sequence that touches all three of the rerank arms' historical breaches.

    Under a promoted LLM judge, report() reaches _rescore_selected, which unpacks
    p_yes_batch_with_validity through probs_of / validity_mask_of, and then reaches
    proxy.exclusion_record(). When those two helpers existed nowhere the call raised NameError,
    which run_arm_sequence filed as a data-dependent arm failure, so both H4 arms reported
    "failed" on every seed with the payload naming the arm rather than the missing symbol; when
    exclusion_record did not exist it raised AttributeError, which both guards re-raise, ending
    the run with zero seeds recorded after the models, corpus, index, calibration and pilot had
    all been paid for.

    build_arm_objects always passes a LIVE proxy, so that is what this drives -- with a real
    QuizItem cache and injected fakes only, no checkpoint and no network."""
    cfg = _cfg()
    cfg.n_overgen_candidates = 4
    dec = methods.FlagDecomposer(cfg, "llm")               # the promoted-judge path
    llm = FakeProxyLLM()
    proxy = methods.FunctionalityProxy(cfg, llm)
    proxy.set_samples(4)

    for cls, criterion in ((methods.OvergenerateContradictionReranker, "descending_p_contradict"),
                           (methods.OvergenerateKeySimilarityReranker, "descending_cosine_to_key")):
        # a FRESH cache per arm, so neither arm sees the other's selections
        items = [candidate_item(0, [False, False, False, False], [0.9, 0.7, 0.5, 0.3]),
                 candidate_item(1, [False, False, False, False], [0.8, 0.6, 0.4, 0.2])]
        arm = cls(cfg, llm, FakeNLI(), FakeEmbedder(), dec, proxy)
        r = arm.report(items, seed=0)

        assert r["selection_criterion"] == criterion
        assert len(arm.evaluated_items()) == 2             # 3 of 4 candidates selected per item
        # the judge mask was refreshed for the NEW option text, so every slot is measured
        for new in arm.evaluated_items():
            assert list(new.llm_pyes_valid) == [True, True, True, True]
        assert r["n_distractor_slots_measured"] == 6
        assert r["n_distractor_slots_excluded"] == 0
        assert r["n_key_slots_excluded"] == 0

        # the proxy answered for both contexts and the arm published the counts beside its rate
        assert r["functional_contexts_measured"] == 2
        assert r["functional_contexts_excluded"] == 0
        assert r["functional_distractor_rate"] is not None
        assert 0.0 <= r["functional_distractor_rate"] <= 1.0

        # the record the harness would write passes the gate's own contract
        for key in analysis.RATE_KEYS:
            v = r[key]
            assert v is None or (isinstance(v, float) and math.isfinite(v)), f"{key} -> {v!r}"
        assert analysis.enforce_rates_contract(0, arm.name, r) is None
        assert analysis.carries_rate_keys(r) is True       # a real record, not a failure record


def test_every_arm_with_a_live_proxy_publishes_its_functional_context_counts():
    """Every arm that reports a functional rate must be able to say what that rate stands on.

    main.build_arm_objects ALWAYS passes a live FunctionalityProxy, so this is the configuration
    the run really uses -- and the one the rest of the suite does not cover, because the other
    exclusion-count test constructs the filters with proxy=None. When FunctionalityProxy defined
    no exclusion_record, the reranker's `r.update(self.proxy.exclusion_record())` raised
    AttributeError on the first seed; run_arm_sequence and main's seed loop both re-raise that as
    a programming error, so the process exited 1 with no recorded seed and no payload, discarding
    the model load, the corpus build, the index embedding, the calibration and the pilot.

    Ten arms, one live proxy, injected fakes only: no checkpoint, no network, no GPU."""
    cfg = _cfg()
    cfg.functionality_samples_per_context = 4
    dec = _decomposer()
    llm = FakeProxyLLM()
    proxy = methods.FunctionalityProxy(cfg, llm)

    # the interface the rerank arms consume exists, and reads zero before any call
    assert proxy.exclusion_record() == {"functional_contexts_measured": 0,
                                        "functional_contexts_excluded": 0}

    records: Dict[str, Any] = {}
    # the two generator arms are recorded in run_seed straight from the decomposition plus one
    # proxy call; functional_record is the same pairing of rate and counts, read in one step
    records["unfiltered_grounded_pool"] = methods.functional_record(proxy, toy_pool(), 0)
    records["ungrounded_topic_only_control"] = methods.functional_record(proxy, toy_pool(), 0)

    # Dict[str, Any] for the same reason as the other all-arm test: BaseItemFilter's subclasses,
    # PositiveControlPlanted and the two rerankers share no base but `object`, so an unannotated
    # container makes the arm.report call below an attribute access on object. The arms answer
    # one PROTOCOL -- report(items, seed) -> dict -- which is exactly what run_arm_sequence calls.
    arms: Dict[str, Any] = {
        "per_option_entailment_filter": methods.PerOptionEntailmentFilter(cfg, dec, proxy),
        "llm_judge_faithfulness_filter": methods.LLMJudgeFaithfulnessFilter(cfg, dec, proxy),
        "contradiction_polarity_item_filter": methods.ContradictionPolarityItemFilter(cfg, dec, proxy),
        "not_entail_polarity_item_filter": methods.NotEntailPolarityItemFilter(cfg, dec, proxy),
        "null_random_drop": methods.NullRandomDropFilter(cfg, dec, proxy),
        "positive_control_planted": methods.PositiveControlPlanted(
            cfg, llm, FakeNLI(), dec, proxy),
        "overgenerate_rerank_by_contradiction": methods.OvergenerateContradictionReranker(
            cfg, llm, FakeNLI(), FakeEmbedder(), dec, proxy),
        "overgenerate_rerank_by_key_similarity": methods.OvergenerateKeySimilarityReranker(
            cfg, llm, FakeNLI(), FakeEmbedder(), dec, proxy),
    }
    for name, arm in arms.items():
        records[name] = arm.report(toy_pool(), 0)   # a FRESH pool, so no arm sees another's cache

    assert set(records) == set(experiment_config.ALL_ARM_NAMES)
    for name, rec in records.items():
        for key in ("functional_distractor_rate", "functional_contexts_measured",
                    "functional_contexts_excluded"):
            assert key in rec, f"{name} publishes no {key}"
        measured = rec["functional_contexts_measured"]
        excluded = rec["functional_contexts_excluded"]
        assert isinstance(measured, int) and isinstance(excluded, int), name
        # nothing was imputed in this pool, so every context the arm evaluated was measured
        assert excluded == 0, name
        assert measured > 0, name
        rate = rec["functional_distractor_rate"]
        assert rate is not None and 0.0 <= float(rate) <= 1.0, name

    # the counts are per CALL, so the last arm's record describes the last arm's retained set and
    # no arm inherits the pool-level count of the arm before it
    assert proxy.exclusion_record()["functional_contexts_measured"] == (
        records["overgenerate_rerank_by_key_similarity"]["functional_contexts_measured"])
    assert proxy.n_contexts_excluded == 0


def test_the_module_level_rescore_selected_is_not_a_second_implementation():
    """The module-level name must never again be a working copy of the reranker's method.

    A duplicate `def _rescore_selected(self, sel_items)` sat at the end of methods.py taking a
    `self` no caller supplies. It was unreachable, and it was the only definition that unpacked
    p_yes_batch_with_validity correctly -- so the correct code was the code that could not run,
    and an edit landing there would have had no effect on any arm.

    What stands at the name now is a guard, not an implementation: it accepts any arity and
    raises RuntimeError naming the live method, so a caller reaching for the wrong one is told
    which is right instead of getting a silent no-op or an argument-count error."""
    with pytest.raises(RuntimeError):
        methods._rescore_selected([])
    with pytest.raises(RuntimeError):
        methods._rescore_selected(None, [])        # the dead duplicate's (self, sel_items) arity
    with pytest.raises(RuntimeError):
        methods._rescore_selected()

    # and the live one is the METHOD on the reranker, a different object entirely
    live = methods.OvergenerateContradictionReranker._rescore_selected
    assert callable(live)
    assert methods._rescore_selected is not live
    assert methods.OvergenerateKeySimilarityReranker._rescore_selected is live


def test_the_two_substitutions_are_counted_and_flagged_separately():
    """The judge's 0.0 and the answerer's uniform 0.25 are DIFFERENT substitutions.

    They shared one counter and one flag sentence, so a payload reporting 40 substitutions could
    not say whether the generative judge or the functionality-proxy answerer produced the
    non-finite rows -- and the sentence called the judge's 0.0 "a neutral value", which is the
    opposite of what it is for: it sits strictly below every entail threshold in (0, 1] so an
    imputed row can never read as support.

    The flag text lives in analysis for the reason this module exists -- the offline suite may
    not import main or models -- so the gate can drive the payload rule directly."""
    kinds = analysis.substitution_kinds()
    assert kinds == ("judge_p_yes", "answerer_letter_probs")
    assert len(set(kinds)) == 2

    # nothing substituted: no flag at all, not a line saying zero
    assert analysis.substitution_flag_lines({}, 0.0, 0.25) == []
    assert analysis.substitution_flag_lines({"judge_p_yes": 0, "answerer_letter_probs": 0},
                                            0.0, 0.25) == []

    # ONE kind fires: exactly one line, naming that kind's count and ITS substituted value
    judge_only = analysis.substitution_flag_lines({"judge_p_yes": 3, "answerer_letter_probs": 0},
                                                  0.0, 0.25)
    assert len(judge_only) == 1
    assert judge_only[0].startswith("3 ")
    assert "0.0" in judge_only[0]
    assert "judge" in judge_only[0]
    assert "answerer" not in judge_only[0] and "letter" not in judge_only[0]

    answerer_only = analysis.substitution_flag_lines({"answerer_letter_probs": 7}, 0.0, 0.25)
    assert len(answerer_only) == 1
    assert answerer_only[0].startswith("7 ")
    assert "0.25" in answerer_only[0]
    assert "answerer" in answerer_only[0]
    assert "judge" not in answerer_only[0]

    # BOTH fire: two lines in the declared order, neither merged into the other
    both = analysis.substitution_flag_lines({"judge_p_yes": 3, "answerer_letter_probs": 7},
                                            0.0, 0.25)
    assert len(both) == 2
    assert both[0] == judge_only[0]
    assert both[1] == answerer_only[0]

    # no line may call a substituted value neutral, and each must say the rows are UNMEASURED
    for line in both:
        assert "neutral" not in line.lower()
        assert "UNMEASURED" in line
        # a design note about a substitution is not a clock-conditional omission
        assert analysis.flag_safe(line) == line

    # a kind with no declared text is still REPORTED, by name; it must never vanish, and it must
    # not cost a run its measured seeds either, so this returns rather than raising
    unknown = analysis.substitution_flag_lines({"some_future_kind": 2}, 0.0, 0.25)
    assert len(unknown) == 1
    assert "some_future_kind" in unknown[0]
    assert unknown[0].startswith("2 ")


def test_the_whole_offline_chain_runs_on_a_toy_pool_with_fakes_only():
    """ITEM 44 end to end: the gate's whole offline path, on the toy pool, with injected fakes.

    The suite could not run against methods.py at all -- rates published none of the exclusion
    counts, the decomposer had no valid_mask, the proxy had no context counters, and the fake
    answerer's (probs, ok) pair was fed to np.asarray -- so the gate blocked before any smoke run
    and no spot test could say which link was broken. This drives the FULL chain in one test:
    decompose, row-table, item bootstrap, aggregate over three seeds, gates, printed lines and
    the payload contract, on a two-to-four-item toy pool whose values are hand-computed, with a
    fake judge, generator, embedder and proxy passed as constructor arguments and no checkpoint,
    no network and no GPU anywhere in it."""
    cfg = _cfg()
    cfg.seeds = [0, 1, 2]
    cfg.functionality_samples_per_context = 4
    dec = _decomposer()
    proxy = methods.FunctionalityProxy(cfg, FakeProxyLLM())
    st = analysis.Statistics(cfg)

    per_seed: List[Dict[str, Any]] = []
    for seed in cfg.seeds:
        pool = toy_pool(seed=seed)          # the same three items, only the (seed, id) keys move
        rates = dec.rates(pool)
        rates.update(methods.functional_record(proxy, pool, seed))
        # every arm carries the same measured record here, which is what makes the aggregate's
        # arithmetic hand-checkable while still exercising all ten arm names
        conditions: Dict[str, Any] = {}
        rows: Dict[str, Any] = {}
        for name in experiment_config.ALL_ARM_NAMES:
            conditions[name] = dict(rates)
            rows[name] = dec.flag_rows(pool, name)
            # the record the harness would be handed passes the gate's own contract
            assert analysis.enforce_rates_contract(seed, name, conditions[name]) is None
        rows[analysis.NLI_REF_TABLE_KEY] = dec.flag_rows(pool, analysis.NLI_REF_TABLE_KEY)
        per_seed.append({"seed": seed, "status": "ok", "conditions": conditions, "rows": rows,
                         analysis.NLI_REFERENCE_RATES_KEY: dict(rates)})

    # the per-seed record itself: three measured slots per item, none excluded, and the proxy
    # answered for all three contexts
    first = per_seed[0]["conditions"]["unfiltered_grounded_pool"]
    assert first["n_distractor_slots_measured"] == 9
    assert first["n_distractor_slots_excluded"] == 0
    assert first["n_key_slots_excluded"] == 0
    assert first["functional_contexts_measured"] == 3
    assert first["functional_contexts_excluded"] == 0
    assert first["distractor_caused_csd_rate"] == pytest.approx(1.0 / 9.0, abs=TOL)

    agg = analysis.aggregate(cfg, per_seed, st, experiment_config.ALL_ARM_NAMES)
    arm = agg["conditions"]["unfiltered_grounded_pool"]
    assert arm["n_seeds"] == 3 and arm["n_failed"] == 0
    # every seed carries 1/9, so the mean over seed ids is 1/9 and so is the pooled rate
    assert arm["mean"] == pytest.approx(1.0 / 9.0, abs=TOL)
    assert arm["primary_metric"] == pytest.approx(1.0 / 9.0, abs=TOL)
    assert arm["primary_metric_estimator"] == analysis.PRIMARY_ESTIMATOR
    assert arm["pooled"]["n_items"] == 9                 # 3 seeds x 3 items
    assert arm["pooled"]["n_distractors"] == 27          # x 3 measured slots
    assert arm["pooled"][analysis.PRIMARY] == pytest.approx(1.0 / 9.0, abs=TOL)
    assert set(arm["per_seed_by_seed"]) == {0, 1, 2}
    assert "per_seed" not in arm
    # the NLI-flagged reference was found under the ONE shared key
    assert agg["unfiltered_pool_nli_reference"]["mean"] == pytest.approx(1.0 / 9.0, abs=TOL)
    assert agg["unfiltered_pool_nli_reference"]["n_distractors"] == 27
    assert isinstance(agg["bootstrap_warnings"], list)

    gates = analysis.decide_gates(cfg, agg, "nli")
    assert gates["h1_base_rate"] == pytest.approx(1.0 / 9.0, abs=TOL)
    assert gates["h1_grounded_minus_ungrounded"] == pytest.approx(0.0, abs=TOL)
    assert gates["null_arm_effect"] == pytest.approx(0.0, abs=TOL)
    assert isinstance(gates["gate_secondary_seed_counts"], dict)
    assert analysis.assert_no_nonfinite(gates) is None

    # the printed lines cover every arm and can never carry the token nan
    lines = analysis.summary_lines(agg["conditions"], experiment_config.ALL_ARM_NAMES)
    joined = "\n".join(lines)
    for name in experiment_config.ALL_ARM_NAMES:
        assert name in joined
    assert "nan" not in joined.lower()
    assert sum(1 for ln in lines if ln.startswith("SUMMARY:")) == 1

    # and the whole thing serialises through the one write path's two guards
    extra = complete_extra()
    extra["summary"] = agg
    extra["gates"] = gates
    extra["per_seed"] = [{k: v for k, v in s.items() if k != "rows"} for s in per_seed]
    assert analysis.check_payload_contract(extra, len(per_seed)) is None
    assert analysis.assert_no_nonfinite(extra) is None
    text = analysis.dumps(extra)
    assert "NaN" not in text and "Infinity" not in text


def test_the_module_level_paired_bootstrap_diff_is_not_a_second_implementation():
    """The module-level name must never again be a working copy of the Statistics method.

    A duplicate `def paired_bootstrap_diff(self, rows_a, ...)` sat at the end of analysis.py
    taking a `self` no caller supplies. It was unreachable, and it was the ONLY definition that
    collected both diagnostics into out["warnings"] -- so the correct code was the code that
    could not run, and the live method went on overwriting out["warning"].

    What stands at the name now is a guard, not an implementation: it accepts any arity and
    raises RuntimeError naming the live method."""
    with pytest.raises(RuntimeError):
        analysis.paired_bootstrap_diff()
    with pytest.raises(RuntimeError):
        analysis.paired_bootstrap_diff(None, None)   # the dead duplicate's (self, rows_a) arity
    with pytest.raises(RuntimeError):
        analysis.paired_bootstrap_diff(None, None, None, "dc", "nested")

    # and the live one is the METHOD on Statistics, a different object entirely
    live = analysis.Statistics.paired_bootstrap_diff
    assert callable(live)
    assert analysis.paired_bootstrap_diff is not live

    # the method still answers on a real comparison, so the guard replaced nothing that ran
    st = analysis.Statistics(_cfg())
    dec = _decomposer()
    pool = dec.flag_rows(toy_pool(), "pool")
    sub = dec.flag_rows(toy_pool()[:2], "sub")
    rec = st.paired_bootstrap_diff(sub, pool, mode="nested")
    assert rec["scheme"] == "nested" and rec["degraded"] is False
    assert "warning" not in rec and "warnings" not in rec


def test_the_programming_error_classes_are_declared_where_a_test_can_reach_them():
    """The three always-a-defect classes must be bound before a run starts, and testable.

    main.py re-raises AttributeError, NameError and TypeError instead of filing them as arm
    failures, and it once declared the tuple BELOW its `if __name__ == "__main__":` guard. A
    module executes top to bottom, so the guard called main() while that name was still unbound:
    the first exception to reach any handler raised NameError, which replaced the real cause,
    was not caught by the `except Exception` recovery branch on the same try, and ended the run
    with zero seeds recorded and no payload -- the results carried no condition at all, so
    null_random_drop and positive_control_planted were missing along with the other eight.

    The rule now lives in analysis, which main imports at the top, so it is bound during the
    import block. This suite may not import main, which is exactly why the breach survived every
    gate before; analysis it can import, so the declaration is now under test."""
    classes = analysis.programming_errors()
    assert isinstance(classes, tuple)
    assert classes == (AttributeError, NameError, TypeError)
    for cls in classes:
        assert isinstance(cls, type) and issubclass(cls, Exception)

    # the tuple is usable as an except clause and matches exactly those three
    for raised in (AttributeError("renamed method"), NameError("missing symbol"),
                   TypeError("a tuple consumed as a list")):
        caught = False
        try:
            raise raised
        except analysis.programming_errors():
            caught = True
        assert caught, f"{type(raised).__name__} must be re-raised as a programming error"

    # a data-dependent failure is NOT one of them: it has to reach the recording handler
    for other in (data.SeedAbort("grounded pool is empty", stage="generation"),
                  ValueError("candidate cache was empty"),
                  RuntimeError("the judge forward pass is numerically broken")):
        try:
            raise other
        except analysis.programming_errors():
            raise AssertionError(f"{type(other).__name__} must be recorded, never re-raised")
        except Exception:
            pass

    # ImportError is deliberately absent: get_harness catches it to fall back to the stand-in
    assert ImportError not in classes
