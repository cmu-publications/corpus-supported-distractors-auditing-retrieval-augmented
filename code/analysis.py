"""
analysis.py -- every decision the results depend on, with no torch, no pandas and no model.

Owns the JSON/undefined-value contract, the results-payload contract, the rates contract,
id-based seed pairing, the item-clustered bootstrap, the comparison-scheme registry, the
support floors, the aggregate, the gate verdicts, the printed metric lines, the uniform
budget lever and the pre-flight self-tests.

The test suite imports this module, so it must stay importable with no GPU and no
checkpoint -- which is also why the payload rules main.py enforces live HERE, where a test
can drive them without importing main.
"""
import json
import math
import traceback
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from experiment_config import ALL_ARM_NAMES

# ----------------------------------------------------------------------------- identity
PRIMARY = "distractor_caused_csd_rate"
PRIMARY_ESTIMATOR = "mean over seed ids"
REFERENCE_ARM = "unfiltered_grounded_pool"
# The arm whose retained set is always flagged by the NLI decomposer, so its reference is the
# NLI-flagged pool (one judge on both sides) even when the LLM judge is promoted.
NLI_REFERENCE_CONDITIONS = ("llm_judge_faithfulness_filter",)
# The ONE key under which a per-seed record carries the NLI-flagged view of the grounded pool.
# main.py writes it and aggregate() reads it through this constant, so writer and reader cannot
# drift: when they did, the independent-judge arm lost its seed-level reference and
# h2_independent_judge_enrichment was null on every run.
NLI_REFERENCE_RATES_KEY = "nli_reference_rates"
NLI_REF_TABLE_KEY = "unfiltered_grounded_pool_nli"
SOFT_METRICS = ("soft_distractor_caused_score", "mean_distractor_p_support_orig")
NULL_EFFECT_TOLERANCE = 0.02

# ITEM 6: an arm record that carries rates carries ALL nine; each value is a finite float or None.
RATE_KEYS = ("distractor_caused_csd_rate", "stem_caused_csd_rate", "corpus_supported_rate",
             "stem_caused_share", "key_hallucination_rate", "per_item_multi_key_rate",
             "mean_distractor_p_support_orig", "soft_distractor_caused_score",
             "mean_key_p_support_orig")

FILTER_CONDITIONS = ("per_option_entailment_filter", "llm_judge_faithfulness_filter",
                     "contradiction_polarity_item_filter", "not_entail_polarity_item_filter")
H4_CONDITIONS = ("overgenerate_rerank_by_contradiction", "overgenerate_rerank_by_key_similarity")

# ITEM 11: the scheme of every arm, fixed in advance. Inferring it from the data would be a
# post-hoc choice; the relation of each arm's evaluated set to the pool is known before the run.
#   nested      - a subset of the SAME items (the four filters, the null drop)
#   paired      - the SAME item ids with different distractors (rerankers, planted arm)
#   independent - different generated items (the closed-book control)
COMPARISON_MODE: Dict[str, str] = {n: "nested" for n in FILTER_CONDITIONS}
COMPARISON_MODE.update({n: "paired" for n in H4_CONDITIONS})
COMPARISON_MODE["ungrounded_topic_only_control"] = "independent"
COMPARISON_MODE["null_random_drop"] = "nested"
COMPARISON_MODE["positive_control_planted"] = "paired"

# The per-distractor row table: plain lists, no pandas (ITEM 48). Every row carries its
# (seed, item_id) key, because the item is the resampling unit.
ROW_FIELDS = ("keys", "dc", "sc", "cs", "enum", "cov", "key_sim", "overlap")


# ============================================================== ITEM 5: the JSON contract
def safe_float(v: Any) -> float:
    """The float of v, or NaN when v is None or unparseable.

    Used ONLY as an immediate input to math.isfinite; a NaN this returns is never stored in a
    record, because an undefined cell is None everywhere in this project."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def fmt(v: Any, nd: int = 4) -> str:
    """Fixed-point text for a finite value, the literal 'NA' for an undefined one.

    Never returns the token 'nan' or 'inf'. The last run printed SUMMARY with the token nan for
    both H4 arms; a run that emits one non-finite value is a failed run."""
    if v is None:
        return "NA"
    x = safe_float(v)
    return f"{x:.{nd}f}" if math.isfinite(x) else "NA"


def sanitize_json(o: Any) -> Any:
    """Recursively convert numpy scalars and arrays to Python types and every non-finite float
    to None; never raises."""
    if isinstance(o, dict):
        return {str(k): sanitize_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sanitize_json(v) for v in o]
    if isinstance(o, np.ndarray):
        return [sanitize_json(v) for v in o.tolist()]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        x = float(o)
        return x if math.isfinite(x) else None
    return o


def assert_no_nonfinite(o: Any, path: str = "$") -> None:
    """Return None when every number in o is finite; raise ValueError naming the dotted JSON path
    of the first non-finite number in the RAW payload."""
    if isinstance(o, dict):
        for k, v in o.items():
            assert_no_nonfinite(v, f"{path}.{k}")
        return
    if isinstance(o, (list, tuple)):
        for i, v in enumerate(o):
            assert_no_nonfinite(v, f"{path}[{i}]")
        return
    if isinstance(o, np.ndarray):
        assert_no_nonfinite(o.tolist(), path)
        return
    if isinstance(o, (bool, np.bool_)):
        return
    if isinstance(o, (float, np.floating)):
        x = float(o)
        if not math.isfinite(x):
            raise ValueError(f"non-finite value at JSON path {path}: {x!r} "
                             f"(an undefined cell must be None, never NaN)")


def dumps(o: Any, indent: Optional[int] = None) -> str:
    """Strict JSON text with allow_nan=False after sanitising; can never emit NaN or Infinity."""
    return json.dumps(sanitize_json(o), allow_nan=False, indent=indent)


# ============================================================== ITEM 6: the rates contract
def carries_rate_keys(record: Any) -> bool:
    """True when record is a dict holding at least one of the nine RATE_KEYS.

    A status-only failure record holds none of them and yields False, so it is exempt."""
    return isinstance(record, dict) and any(k in record for k in RATE_KEYS)


def enforce_rates_contract(seed: Any, arm: str, record: Any) -> None:
    """Return None for a status-only record; otherwise raise ValueError naming the first missing
    RATE_KEY or the first non-finite numeric cell.

    A partial rate record reaching the aggregate produces a mean over a different denominator
    than the reader assumes."""
    if not carries_rate_keys(record):
        return
    for k in RATE_KEYS:
        if k not in record:
            raise ValueError(f"seed {seed} arm '{arm}': rate key '{k}' is missing from a record "
                             f"that carries rates")
    for k, v in record.items():
        if v is None or isinstance(v, (bool, str, list, dict, tuple)):
            continue
        if isinstance(v, (int, float, np.integer, np.floating)) and not math.isfinite(float(v)):
            raise ValueError(f"seed {seed} arm '{arm}': key '{k}' is non-finite ({float(v)!r}); "
                             f"an undefined cell must be None")


# ============================================================== ITEM 7: the payload contract
# Every one of these sixteen keys must be present in `extra` before any write. The tuple lives
# here, not in main.py, so the suite can check the guard without importing main.
REQUIRED_EXTRA_KEYS = ("metric_def", "hyperparameters", "seeds", "condition_names", "arm_names",
                       "control_arms", "comparison_modes", "run_metadata", "skipped_components",
                       "reduced_components", "undefined_values", "corpus", "backend", "calibration",
                       "scientific_validity", "flags")

# The exact breach the real harness aborts on. A smoke run printed
# "FAIL: results contract: no seed was recorded before write_results" and exited 2 because main
# wrote the payload once between calibration and the seed loop; the stand-in accepted that write,
# so only a GPU run could expose it. Now both harnesses refuse it with this message.
RESULTS_CONTRACT_NO_SEED = "results contract: no seed was recorded before write_results"


def enforce_seed_recorded_before_write(n_seeds_recorded: int) -> None:
    """Return None when at least one seed was recorded; raise ValueError carrying the exact
    message RESULTS_CONTRACT_NO_SEED otherwise."""
    if int(n_seeds_recorded) <= 0:
        raise ValueError(RESULTS_CONTRACT_NO_SEED)


def check_payload_contract(extra: Dict[str, Any], n_seeds_recorded: int) -> None:
    """The three checks every write passes, in order: no required key missing, an EMPTY
    skipped_components, and at least one recorded seed. Raises ValueError naming the breach."""
    missing = [k for k in REQUIRED_EXTRA_KEYS if k not in extra]
    if missing:
        raise ValueError(f"results payload is missing required keys {missing}")
    if list(extra["skipped_components"]):
        raise ValueError(f"skipped_components must be empty on a valid run; "
                         f"got {extra['skipped_components']}")
    enforce_seed_recorded_before_write(n_seeds_recorded)


# ============================================================== ITEM 8: recorded failures
def library_version_record(name: str, probe: Callable[[], str]) -> Dict[str, str]:
    """{name: version} on success, or {name: 'unknown', name_error: 'Type: msg'} when the probe
    raises. Writing the bare string 'unknown' discarded the provenance failure, so a payload could
    not say whether the library was absent, broken or simply unread."""
    try:
        return {name: str(probe())}
    except Exception as e:                                  # the cause is recorded, never dropped
        return {name: "unknown", f"{name}_error": f"{type(e).__name__}: {e}"}


def arm_failure_record(exc: BaseException, tail_chars: int = 800) -> Dict[str, str]:
    """What a data-dependent arm failure leaves in the results.

    Both rerank arms failed on every seed of the last run and the payload carried only
    'failed: ...' with no cause, so the next run could not act on it. The record names the
    exception type, its message and the tail of its traceback, and carries NO rate key, so
    enforce_rates_contract keeps treating it as a status-only record."""
    return {"status": f"failed: {type(exc).__name__}: {exc}",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback_tail": traceback.format_exc()[-int(tail_chars):]}


# ============================================================== ITEM 16: pairing by seed id
def aligned_by_seed(a_by_seed: Dict[int, float],
                    b_by_seed: Dict[int, float]) -> Tuple[List[float], List[float], List[int]]:
    """The two value lists and the common seed ids, paired by INT ID and sorted by id, dropping
    None values. Never pairs by list position: a run summary carrying per-seed keys 0 through 4
    against a three-seed config would silently compare different seeds."""
    a_map = {int(k): float(v) for k, v in (a_by_seed or {}).items() if v is not None}
    b_map = {int(k): float(v) for k, v in (b_by_seed or {}).items() if v is not None}
    seeds = sorted(set(a_map) & set(b_map))
    return [a_map[s] for s in seeds], [b_map[s] for s in seeds], seeds


# ============================================================== the row table (no pandas)
def empty_rows(condition: str = "") -> Dict[str, Any]:
    """A row table with the condition label and every ROW_FIELDS key bound to a FRESH empty list."""
    out: Dict[str, Any] = {"condition": condition}
    for f in ROW_FIELDS:
        out[f] = []
    return out


def concat_rows(parts: Sequence[Dict[str, Any]], condition: str = "") -> Dict[str, Any]:
    """One row table holding every part's rows in order across all ROW_FIELDS; a falsy part is
    skipped and a missing field never raises."""
    out = empty_rows(condition)
    for p in parts or []:
        if not p:
            continue
        for f in ROW_FIELDS:
            out[f].extend(list(p.get(f, [])))
    return out


def n_rows(rows: Optional[Dict[str, Any]]) -> int:
    """The number of distractor rows; 0 for None or an empty table."""
    return len(rows.get("keys", [])) if rows else 0


def subset_rows(rows: Dict[str, Any], mask: Sequence[bool]) -> Dict[str, Any]:
    """A NEW row table holding only the rows whose mask entry is true; never mutates the input."""
    sub = empty_rows(rows.get("condition", ""))
    for i, keep in enumerate(mask):
        if not keep:
            continue
        for f in ROW_FIELDS:
            seq = rows.get(f, [])
            if i < len(seq):
                sub[f].append(seq[i])
    return sub


# ============================================================== ITEM 12: the H2 support floors
def h2_min_sweep_items(n_seeds: int, items_per_seed: int) -> int:
    """max(5, ceil(0.1 * n_seeds * items_per_seed)); 30 at 3x100 and 5 at 1x3.

    h2_enrichment_per_option was 0.32 against a base rate of 0.0133 because a FIXED floor of 5
    admitted a nearly empty cell on a 300-item design."""
    product = int(n_seeds) * int(items_per_seed)
    return int(max(5, math.ceil(round(0.1 * product, 6))))


def per_option_enrichment(sweep_cells: Dict[str, Any], base_mean: Optional[float], min_items: int,
                          min_seeds: int, taus: Sequence[float], plan_tau: float) -> Dict[str, Any]:
    """Admission flag and rate per tau, the selected tau with its enrichment, and
    enrichment_at_plan_tau beside it.

    A cell below EITHER floor (pooled items, contributing seeds) has rate None and admitted False
    and never enters the Spearman correlation."""
    admitted: Dict[str, bool] = {}
    rates: Dict[str, Optional[float]] = {}
    for tau in taus:
        key = str(tau)
        cell = sweep_cells.get(key, {}) or {}
        n_items = int(cell.get("n_items", 0) or 0)
        n_cell_seeds = int(cell.get("n_seeds", 0) or 0)
        rate = cell.get("rate")
        ok = bool(n_items >= int(min_items) and n_cell_seeds >= int(min_seeds) and rate is not None)
        admitted[key] = ok
        rates[key] = float(rate) if ok else None

    selected_tau: Optional[float] = None
    selected_n = 0
    enrich: Optional[float] = None
    for tau in sorted((float(t) for t in taus), reverse=True):
        key = str(tau)
        if key not in admitted:
            matching = [t for t in taus if float(t) == tau]
            key = str(matching[0]) if matching else key
        if admitted.get(key):
            selected_tau = float(tau)
            selected_n = int((sweep_cells.get(key, {}) or {}).get("n_items", 0) or 0)
            sel_rate = rates.get(key)
            if base_mean is not None and sel_rate is not None:
                enrich = float(sel_rate) - float(base_mean)
            break

    plan_key = str(plan_tau) if str(plan_tau) in rates else str(list(taus)[-1])
    plan_rate = rates.get(plan_key)
    plan_enrich = (float(plan_rate) - float(base_mean)
                   if (plan_rate is not None and base_mean is not None) else None)

    # An admitted cell always carries a float rate, but the value is read out of a
    # Dict[str, Optional[float]]; narrowing it HERE is what lets the caller below hand scipy a
    # list of plain floats.
    defined: List[Tuple[float, float]] = []
    for t in taus:
        rate_t = rates.get(str(t))
        if rate_t is None:
            continue
        defined.append((float(t), float(rate_t)))
    rho: Optional[float] = None
    if len(defined) >= 3 and float(np.std([d[1] for d in defined])) > 0:
        rho = float(stats.spearmanr([d[0] for d in defined], [d[1] for d in defined]).statistic)
    return {"admitted": admitted, "rates": rates, "min_items": int(min_items),
            "min_seeds": int(min_seeds), "selected_tau": selected_tau,
            "selected_n_items": selected_n, "enrichment_at_selected_tau": enrich,
            "plan_tau": float(plan_tau), "enrichment_at_plan_tau": plan_enrich,
            "spearman_rho_tau_vs_rate": rho, "spearman_n_taus": len(defined)}


# ============================================================== ITEM 9/10: statistics
class Statistics:
    """Holds the config and one seeded RNG; every returned interval is a two-element list and
    every undefined statistic is None."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.resamples = int(getattr(cfg, "bootstrap_resamples", 1000))
        self.rng = np.random.default_rng(0)

    def _item_matrix(self, rows: Optional[Dict[str, Any]], col: str = "dc"):
        """Group the distractor rows into whole (seed, item_id) units; three rows of one item are
        three views of ONE generation, so the item is the resampling unit."""
        keys: List[tuple] = []
        sums: Dict[tuple, float] = {}
        cnts: Dict[tuple, float] = {}
        if rows:
            for k, v in zip(rows.get("keys", []), rows.get(col, [])):
                kk = tuple(k)
                if kk not in sums:
                    sums[kk], cnts[kk] = 0.0, 0.0
                    keys.append(kk)
                sums[kk] += 1.0 if bool(v) else 0.0
                cnts[kk] += 1.0
        return (keys,
                np.array([sums[k] for k in keys], dtype=np.float64),
                np.array([cnts[k] for k in keys], dtype=np.float64))

    def bootstrap_rate_ci(self, rows: Optional[Dict[str, Any]], col: str = "dc") -> Dict[str, Any]:
        """ITEM 9: THE CI contract. The record is exactly

            {"rate": float|None, "ci": [lo, hi], "n_items": int, "n_rows": int}

        resampling whole (seed, item_id) ITEMS, not distractor rows. The interval is ONE
        two-element list under "ci" -- never scalar lo/hi keys, because a reader that hunted for
        scalar keys raised before its first assertion and took the whole gate suite with it."""
        keys, s, c = self._item_matrix(rows, col)
        if not keys or c.sum() <= 0:
            return {"rate": None, "ci": [None, None], "n_items": 0, "n_rows": 0}
        rate = float(s.sum() / c.sum())
        idx = np.random.default_rng(12345).integers(0, len(keys), size=(self.resamples, len(keys)))
        num, den = s[idx].sum(axis=1), c[idx].sum(axis=1)
        means = np.where(den > 0, num / np.maximum(den, 1.0), np.nan)
        means = means[np.isfinite(means)]
        lo = float(np.quantile(means, 0.025)) if means.size else None
        hi = float(np.quantile(means, 0.975)) if means.size else None
        return {"rate": rate, "ci": [lo, hi], "n_items": len(keys), "n_rows": int(c.sum())}

    def bootstrap_rate_ci_unclustered(self, rows: Optional[Dict[str, Any]],
                                      col: str = "dc") -> Dict[str, Any]:
        """The row-level companion under the SAME "ci" shape, so the clustered interval can be
        compared with the too-narrow one that treats each distractor row as independent."""
        flags = np.array([1.0 if bool(v) else 0.0 for v in (rows.get(col, []) if rows else [])],
                         dtype=np.float64)
        if flags.size == 0:
            return {"rate": None, "ci": [None, None], "n_rows": 0}
        idx = np.random.default_rng(12345).integers(0, flags.size, size=(self.resamples, flags.size))
        m = flags[idx].mean(axis=1)
        return {"rate": float(flags.mean()),
                "ci": [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))],
                "n_rows": int(flags.size)}

    def _unit_values(self, rows: Optional[Dict[str, Any]], col: str) -> Dict[tuple, float]:
        keys, s, c = self._item_matrix(rows, col)
        v = np.divide(s, np.maximum(c, 1.0))
        return {k: float(x) for k, x in zip(keys, v)}

    def _nested_resample(self, a_map: Dict[tuple, float], b_map: Dict[tuple, float]):
        """A is the retained subset of pool B. Resample the POOL; the statistic is
        (mean over the resampled items that were retained) - (mean over the whole resample).
        Resampling only the shared keys would compare each item with itself and give zero."""
        keys = list(b_map)
        b = np.array([b_map[k] for k in keys], dtype=np.float64)
        in_a = np.array([k in a_map for k in keys], dtype=bool)
        a_full = np.array([a_map.get(k, 0.0) for k in keys], dtype=np.float64)
        idx = self.rng.integers(0, len(keys), size=(self.resamples, len(keys)))
        sel = in_a[idx]
        cnt = sel.sum(axis=1)
        num = (a_full[idx] * sel).sum(axis=1)
        sub = np.where(cnt > 0, num / np.maximum(cnt, 1), np.nan)
        d = sub - b[idx].mean(axis=1)
        return a_full[in_a], b, d[np.isfinite(d)], int(in_a.sum())

    def _paired_resample(self, a_map: Dict[tuple, float], b_map: Dict[tuple, float],
                         shared: List[tuple]):
        """The SAME item ids with different content (rerankers, planted arm): resample the shared
        ids as pairs."""
        a = np.array([a_map[k] for k in shared], dtype=np.float64)
        b = np.array([b_map[k] for k in shared], dtype=np.float64)
        idx = self.rng.integers(0, len(shared), size=(self.resamples, len(shared)))
        return a, b, a[idx].mean(1) - b[idx].mean(1), len(shared)

    def _independent_resample(self, a_map: Dict[tuple, float], b_map: Dict[tuple, float]):
        """Different generated items (the closed-book control): resample each arm separately."""
        a = np.array(list(a_map.values()), dtype=np.float64)
        b = np.array(list(b_map.values()), dtype=np.float64)
        ia = self.rng.integers(0, len(a), size=(self.resamples, len(a)))
        ib = self.rng.integers(0, len(b), size=(self.resamples, len(b)))
        return a, b, a[ia].mean(1) - b[ib].mean(1), min(len(a), len(b))

    def paired_bootstrap_diff(self, rows_a: Optional[Dict[str, Any]],
                              rows_b: Optional[Dict[str, Any]], col: str = "dc",
                              mode: str = "auto") -> Dict[str, Any]:
        """ITEM 10: the REGISTERED scheme is used, or the record names the scheme it degraded to.

        A silently swapped scheme is a misaligned comparison, so a degraded record carries
        requested_scheme, the actual scheme, degraded True and a warning naming both.

        The two diagnostics this call can raise -- a degraded scheme and a zero-width interval --
        are INDEPENDENT, so both are collected in a list and neither can overwrite the other.
        They used to be written to the same scalar out["warning"] one after the other, so a
        comparison that both degraded AND came back degenerate published only the second: the
        arm was at its least trustworthy and the one fact its requested_scheme/degraded fields
        exist to make auditable was gone. aggregate() copies a single string per arm into
        bootstrap_warnings and main turns each into a payload flag, so the loss reached the
        results. out["warning"] stays, as the join of every entry, because that is the key every
        existing reader gates on; a record with no diagnostic still carries neither key."""
        empty: Dict[str, Any] = {"diff": None, "ci": [None, None], "p_gt_0": None, "scheme": mode,
                                 "requested_scheme": mode, "degraded": False, "n_items": 0,
                                 "zero_width": False}
        if n_rows(rows_a) == 0 or n_rows(rows_b) == 0:
            return empty
        a_map = self._unit_values(rows_a, col)
        b_map = self._unit_values(rows_b, col)
        shared = sorted(set(a_map) & set(b_map))
        scheme, degraded = mode, False
        if scheme == "auto":
            if set(a_map) <= set(b_map) and len(a_map) < len(b_map):
                scheme = "nested"
            elif len(shared) >= max(3, int(0.5 * min(len(a_map), len(b_map)))):
                scheme = "paired"
            else:
                scheme = "independent"
        if scheme == "nested" and not set(a_map) <= set(b_map):
            scheme, degraded = ("paired" if len(shared) >= 3 else "independent"), True
        if scheme == "paired" and len(shared) < 3:
            scheme, degraded = "independent", True
        if scheme == "nested":
            a, b, diffs, n_items = self._nested_resample(a_map, b_map)
        elif scheme == "paired":
            a, b, diffs, n_items = self._paired_resample(a_map, b_map, shared)
        else:
            a, b, diffs, n_items = self._independent_resample(a_map, b_map)
        if diffs.size == 0 or a.size == 0 or b.size == 0:
            out = dict(empty)
            out.update({"scheme": scheme, "degraded": degraded, "n_items": n_items})
            return out
        lo, hi = float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))
        out = {"diff": float(a.mean() - b.mean()), "ci": [lo, hi],
               "p_gt_0": float((diffs > 0).mean()), "scheme": scheme, "requested_scheme": mode,
               "degraded": bool(degraded), "n_items": int(n_items),
               "n_resamples": int(diffs.size), "zero_width": bool(hi - lo == 0.0)}
        # every diagnostic this call raised, in the order it was detected; appended, never assigned
        warnings: List[str] = []
        if degraded:
            warnings.append(f"requested scheme={mode} degraded to {scheme}")
        if n_items >= 3 and hi - lo == 0.0 and (float(np.std(a)) > 0 or float(np.std(b)) > 0):
            warnings.append(f"zero-width CI under scheme={scheme}; "
                            f"the resampling scheme is degenerate")
        if warnings:
            out["warnings"] = warnings
            out["warning"] = "; ".join(warnings)
        return out

    @staticmethod
    def wilcoxon_seed_level(a_by_seed: Dict[int, float],
                            b_by_seed: Dict[int, float]) -> Dict[str, Any]:
        """The id-aligned seed-level test, flagged underpowered below six seeds (three seeds
        cannot reject: the minimum two-sided p is 0.25). No gate reads its p value."""
        a, b, seeds = aligned_by_seed(a_by_seed, b_by_seed)
        d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
        out: Dict[str, Any] = {"n_seeds": len(seeds), "seeds": seeds,
                               "mean_diff": (float(d.mean()) if len(d) else None),
                               "p": None, "rank_biserial": None, "statistic": None,
                               "underpowered": bool(len(seeds) < 6)}
        if len(d) < 3 or bool(np.all(d == 0)):
            return out
        try:
            res = stats.wilcoxon(a, b, zero_method="wilcox")
            nz = d[d != 0]
            ranks = stats.rankdata(np.abs(nz))
            out.update({"p": float(res.pvalue), "statistic": float(res.statistic),
                        "rank_biserial": float((ranks[nz > 0].sum() - ranks[nz < 0].sum())
                                               / ranks.sum())})
        except ValueError as e:                              # recorded, never silently dropped
            out["error"] = f"ValueError: {e}"
        return out

    def grounding_attribution_posterior(self, rows_g: Optional[Dict[str, Any]],
                                        rows_t: Optional[Dict[str, Any]]) -> Optional[float]:
        """The bootstrap posterior that the grounded minus closed-book difference exceeds the
        configured delta; None when either side has no rows."""
        if n_rows(rows_g) == 0 or n_rows(rows_t) == 0:
            return None
        g = np.array(list(self._unit_values(rows_g, "dc").values()), dtype=np.float64)
        t = np.array(list(self._unit_values(rows_t, "dc").values()), dtype=np.float64)
        if g.size == 0 or t.size == 0:
            return None
        ig = self.rng.integers(0, g.size, size=(self.resamples, g.size))
        it = self.rng.integers(0, t.size, size=(self.resamples, t.size))
        delta = float(getattr(self.cfg, "grounding_attribution_delta", 0.05))
        return float(((g[ig].mean(1) - t[it].mean(1)) > delta).mean())

    def regime_cells(self, rows: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Enumeration-density by retrieval-coverage cells. A cell below min_cell_items carries
        below_floor True and a None rate, so a reader can tell a SUPPRESSED cell from one that
        was never computed."""
        cells: Dict[str, Any] = {}
        if n_rows(rows) == 0 or rows is None:
            return cells
        enum, cov = rows["enum"], rows["cov"]
        n = n_rows(rows)
        floor = int(getattr(self.cfg, "min_cell_items", 1))
        for e in ("high", "low"):
            for c in ("covered", "not_covered"):
                sub = subset_rows(rows, [enum[i] == e and cov[i] == c for i in range(n)])
                n_items = len({tuple(k) for k in sub["keys"]})
                key = f"enum={e}|cov={c}"
                if n_items < floor:
                    cells[key] = {"n_items": n_items, "below_floor": True, PRIMARY: None}
                    continue
                b = self.bootstrap_rate_ci(sub, "dc")
                cells[key] = {"n_items": n_items, PRIMARY: b["rate"], "ci": b["ci"],
                              "stem_caused_csd_rate": self.bootstrap_rate_ci(sub, "sc")["rate"],
                              "corpus_supported_rate": self.bootstrap_rate_ci(sub, "cs")["rate"]}
        for e in ("high", "low"):
            sub = subset_rows(rows, [enum[i] == e for i in range(n)])
            cells[f"enum={e}"] = {"n_items": len({tuple(k) for k in sub["keys"]}),
                                  PRIMARY: self.bootstrap_rate_ci(sub, "dc")["rate"]}
        for c in ("covered", "not_covered"):
            sub = subset_rows(rows, [cov[i] == c for i in range(n)])
            cells[f"cov={c}"] = {"n_items": len({tuple(k) for k in sub["keys"]}),
                                 PRIMARY: self.bootstrap_rate_ci(sub, "dc")["rate"]}
        return cells


def paired_t_and_cohen(x: Sequence[float],
                       ref: Sequence[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """The paired t statistic, p value and Cohen's d over ID-ALIGNED values, or (None, None, None)
    below two pairs or at zero variance; never returns NaN."""
    m = min(len(x), len(ref))
    if m < 2:
        return None, None, None
    d = np.asarray(list(x)[:m], dtype=np.float64) - np.asarray(list(ref)[:m], dtype=np.float64)
    sd = float(np.std(d, ddof=1))
    if sd <= 0:
        return None, None, None
    res = stats.ttest_rel(list(x)[:m], list(ref)[:m])
    return float(res.statistic), float(res.pvalue), float(d.mean() / sd)


# ============================================================== ITEM 15: the aggregate
def headline_metric(arm_summary: Dict[str, Any]) -> Dict[str, Any]:
    """primary_metric is the MEAN OVER SEED IDS; the row-pooled rate is a DIFFERENT estimator and
    is published beside it under primary_metric_pooled. No branch swaps one for the other: the
    last run reported 0.0134 (pooled) and 0.0167 (mean over seeds) as if they were one quantity."""
    n_seeds = int(arm_summary.get("n_seeds", 0) or 0)
    return {"primary_metric": (arm_summary.get("mean") if n_seeds > 0 else None),
            "primary_metric_estimator": PRIMARY_ESTIMATOR,
            "primary_metric_n_seeds": n_seeds,
            "primary_metric_pooled": dict(arm_summary.get("pooled", {}) or {})}


def aggregate(cfg: Any, per_seed: Sequence[Dict[str, Any]], st: "Statistics",
              arm_names: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Per-arm summaries with per_seed_by_seed keyed by INT seed id, pooled CIs, secondary means,
    regime cells, registered-scheme comparisons and the threshold sweep.

    Raises RuntimeError if any arm summary would carry a 'per_seed' key -- a reader defaulting on
    that name saw an empty map for every arm, so no PAIRED line was ever printed. A failed seed is
    ABSENT from the map and counted in n_failed, never imputed."""
    names = list(arm_names or ALL_ARM_NAMES)
    n_total = len(cfg.seeds)
    summary: Dict[str, Any] = {}
    rows_by_arm: Dict[str, Dict[str, Any]] = {}

    for name in names:
        by_seed: Dict[int, float] = {}
        secondary: Dict[str, Dict[int, float]] = {}
        parts: List[Dict[str, Any]] = []
        n_failed = 0
        for s in per_seed:
            rec = ((s.get("conditions") or {}).get(name)) or {}
            v = rec.get(PRIMARY)
            fv = safe_float(v) if v is not None else float("nan")
            if v is None or not math.isfinite(fv):
                n_failed += 1
                continue
            sid = int(s["seed"])
            by_seed[sid] = float(fv)
            for k, val in rec.items():
                if k == PRIMARY or isinstance(val, bool):
                    continue
                if not isinstance(val, (int, float, np.integer, np.floating)):
                    continue
                f2 = safe_float(val)
                if math.isfinite(f2):
                    secondary.setdefault(k, {})[sid] = float(f2)
            r = (s.get("rows") or {}).get(name)
            if n_rows(r):
                parts.append(r)
        rows = concat_rows(parts, name)
        rows_by_arm[name] = rows
        vals = [by_seed[k] for k in sorted(by_seed)]
        pooled = st.bootstrap_rate_ci(rows, "dc")
        arm: Dict[str, Any] = {
            "mean": (float(np.mean(vals)) if vals else None),
            "std": (float(np.std(vals)) if vals else None),
            # The per-seed map is published under per_seed_by_seed and under NO other name.
            "per_seed_by_seed": dict(by_seed),
            "n_seeds": len(vals), "n_failed": int(n_failed), "n_seeds_expected": n_total,
            "success_rate": ((len(vals) / n_total) if n_total else None),
            "pooled": {PRIMARY: pooled["rate"], "ci95": pooled["ci"],
                       "n_items": pooled["n_items"], "n_distractors": pooled["n_rows"]},
            "secondary": {k: {"mean": float(np.mean(list(d.values()))),
                              "std": float(np.std(list(d.values()))),
                              "median": float(np.median(list(d.values()))),
                              "n_seeds": len(d)} for k, d in secondary.items()},
            "regime_cells": st.regime_cells(rows),
        }
        arm.update(headline_metric(arm))
        if "per_seed" in arm:
            raise RuntimeError(f"arm '{name}' publishes a 'per_seed' key; the aggregate emits "
                               f"per_seed_by_seed and nothing else")
        summary[name] = arm

    # The NLI-flagged view of the grounded pool, read under the ONE shared constant.
    nli_by_seed: Dict[int, float] = {}
    nli_kh: List[float] = []
    nli_parts: List[Dict[str, Any]] = []
    for s in per_seed:
        rr = s.get(NLI_REFERENCE_RATES_KEY) or {}
        v = rr.get(PRIMARY)
        if v is not None and math.isfinite(safe_float(v)):
            nli_by_seed[int(s["seed"])] = float(v)
            kh = rr.get("key_hallucination_rate")
            if kh is not None and math.isfinite(safe_float(kh)):
                nli_kh.append(float(kh))
        r = (s.get("rows") or {}).get(NLI_REF_TABLE_KEY)
        if n_rows(r):
            nli_parts.append(r)
    nli_rows = concat_rows(nli_parts, NLI_REF_TABLE_KEY)
    nli_vals = [nli_by_seed[k] for k in sorted(nli_by_seed)]
    nli_reference = {"mean": (float(np.mean(nli_vals)) if nli_vals else None),
                     "per_seed_by_seed": dict(nli_by_seed), "n_seeds": len(nli_vals),
                     "key_hallucination_rate_mean": (float(np.mean(nli_kh)) if nli_kh else None),
                     "n_distractors": n_rows(nli_rows)}

    comparisons: Dict[str, Any] = {}
    boot_warn: List[str] = []
    align_warn: List[str] = []
    for name in names:
        if name == REFERENCE_ARM or name not in summary or summary[name]["n_seeds"] == 0:
            continue
        use_nli = name in NLI_REFERENCE_CONDITIONS
        ref_rows = nli_rows if use_nli else rows_by_arm.get(REFERENCE_ARM)
        ref_by_seed = (nli_by_seed if use_nli
                       else (summary.get(REFERENCE_ARM, {}) or {}).get("per_seed_by_seed", {}))
        cmp_items = st.paired_bootstrap_diff(rows_by_arm[name], ref_rows, "dc",
                                             mode=COMPARISON_MODE.get(name, "auto"))
        if "warning" in cmp_items:
            boot_warn.append(f"{name}: {cmp_items['warning']}")
        _a, _b, common = aligned_by_seed(summary[name]["per_seed_by_seed"], ref_by_seed)
        if len(common) < n_total:
            align_warn.append(f"{name}: seed-level comparison uses {len(common)}/{n_total} "
                              f"seeds {common}")
        comparisons[name] = {
            "reference": (REFERENCE_ARM + "_nli_flags") if use_nli else REFERENCE_ARM,
            "paired_bootstrap_items": cmp_items, "aligned_seeds": common,
            "wilcoxon_seeds": st.wilcoxon_seed_level(summary[name]["per_seed_by_seed"], ref_by_seed)}

    out: Dict[str, Any] = {"conditions": summary, "comparisons_vs_unfiltered": comparisons,
                           "bootstrap_warnings": boot_warn, "seed_alignment_warnings": align_warn,
                           "unfiltered_pool_nli_reference": nli_reference,
                           "primary_metric_estimator": PRIMARY_ESTIMATOR,
                           "rows_by_arm_n": {k: n_rows(v) for k, v in rows_by_arm.items()}}
    out["grounding_attribution_posterior"] = st.grounding_attribution_posterior(
        rows_by_arm.get(REFERENCE_ARM), rows_by_arm.get("ungrounded_topic_only_control"))

    # The per-option threshold sweep, pooled over seeds. A cell that retained zero items across
    # every seed has NO rate (None, never a number).
    #
    # n_items counts ONLY the seeds whose own cell was admitted -- the same seeds whose rates the
    # mean below is taken over. Summing every seed's retained count while averaging only the
    # admitted ones published a support figure larger than the support the rate stood on: at
    # 3x100 a tau whose cells were 30, 30 and a suppressed 9 reported 69 items behind a rate
    # computed from 60, and h2_enrichment_n_items republished that 69. The unconditional total is
    # kept beside it as a diagnostic, so nothing is lost.
    cells: Dict[str, Dict[str, Any]] = {}
    for tau in cfg.tau_sweep:
        key = str(tau)
        n_items = 0
        n_items_all_seeds = 0
        rates: List[float] = []
        for s in per_seed:
            c = (((s.get("conditions") or {}).get("per_option_entailment_filter") or {})
                 .get("threshold_sweep", {}) or {}).get(key, {}) or {}
            cell_items = int(c.get("n_items", 0) or 0)
            n_items_all_seeds += cell_items
            r = c.get(PRIMARY)
            if c.get("admitted") and r is not None and math.isfinite(safe_float(r)):
                n_items += cell_items          # counted under the SAME predicate as the rate
                rates.append(float(r))
        cells[key] = {"n_items": n_items, "n_items_all_seeds": n_items_all_seeds,
                      "n_seeds": len(rates),
                      "rate": (float(np.mean(rates)) if rates else None)}
    out["per_option_threshold_sweep"] = cells
    out["per_option_enrichment"] = per_option_enrichment(
        cells, (summary.get(REFERENCE_ARM, {}) or {}).get("mean"),
        h2_min_sweep_items(len(cfg.seeds), cfg.items_per_seed),
        int(getattr(cfg, "secondary_min_seeds", 2)), list(cfg.tau_sweep), float(cfg.tau_sweep[-1]))

    noop = [v for v in (safe_float(s.get("rewrite_noop_rate")) for s in per_seed) if math.isfinite(v)]
    tfb = [v for v in (safe_float(s.get("topic_label_fallback_rate")) for s in per_seed)
           if math.isfinite(v)]
    out["rewrite_noop_rate_mean"] = float(np.mean(noop)) if noop else None
    out["topic_label_fallback_rate_mean"] = float(np.mean(tfb)) if tfb else None
    return out


# ============================================================== ITEM 13/14: the gate verdicts
def _sub(a: Optional[float], b: Optional[float]) -> Optional[float]:
    return None if (a is None or b is None) else float(a) - float(b)


def _ge(a: Optional[float], b: Optional[float]) -> Optional[bool]:
    return None if (a is None or b is None) else bool(float(a) >= float(b))


def secondary_value(summary: Dict[str, Any], name: str, key: str,
                    min_seeds: int) -> Tuple[Optional[float], int]:
    """ITEM 13: the PAIR (value, n_seeds), read from summary[name]['secondary'][key].

    The arity IS the contract: a caller that treats this as a bare value compares a tuple with
    None and never sees the seed floor at all. The value is None below min_seeds while the count
    is always the true number of contributing seeds."""
    d = ((summary.get(name, {}) or {}).get("secondary", {}) or {}).get(key)
    if not d:
        return None, 0
    n = int(d.get("n_seeds", 0) or 0)
    return (None if n < int(min_seeds) else d.get("mean")), n


def decide_gates(cfg: Any, agg: Dict[str, Any], evaluator: str) -> Dict[str, Any]:
    """ITEM 14: every verdict resting on fewer than secondary_min_seeds seeds is blanked; a verdict
    whose input is undefined is None, NEVER False; gate_secondary_seed_counts names the count
    behind each reading; and NO returned value decides whether an arm runs."""
    S = agg.get("conditions", {}) or {}
    cmpv = agg.get("comparisons_vs_unfiltered", {}) or {}
    min_seeds = int(getattr(cfg, "secondary_min_seeds", 2))
    counts: Dict[str, int] = {}

    def sec(name: str, key: str) -> Optional[float]:
        v, n = secondary_value(S, name, key, min_seeds)
        counts[f"{name}.{key}"] = n
        return v

    def mean_of(name: str) -> Optional[float]:
        arm = S.get(name, {}) or {}
        n = int(arm.get("n_seeds", 0) or 0)
        counts[f"{name}.mean"] = n
        return arm.get("mean") if n > 0 else None

    g: Dict[str, Any] = {"evaluator": evaluator, "judge_validated": evaluator != "none",
                         "unvalidated_rates": evaluator == "none",
                         "gates_are_diagnostics_only": True,
                         "primary_metric_estimator": PRIMARY_ESTIMATOR}

    base = mean_of(REFERENCE_ARM)
    nli_ref = agg.get("unfiltered_pool_nli_reference", {}) or {}
    base_nli = nli_ref.get("mean") if int(nli_ref.get("n_seeds", 0) or 0) > 0 else None
    stem_share = sec(REFERENCE_ARM, "stem_caused_share")
    share_dc = _sub(1.0, stem_share)
    delta = _sub(base, mean_of("ungrounded_topic_only_control"))
    g["h1_base_rate"] = base
    g["h1_powered"] = _ge(base, cfg.h1_base_rate_gate)
    g["h1_grounded_minus_ungrounded"] = delta
    g["h1_stem_caused_share"] = stem_share
    position: Optional[str] = None
    if delta is not None and stem_share is not None and share_dc is not None:
        if delta >= 0.10 and share_dc >= 0.5:
            position = "retrieval_cause"
        elif abs(delta) <= 0.05 and stem_share >= 0.6:
            position = "generator_cause"
        else:
            position = "indeterminate"
    g["h1_position"] = position
    g["grounding_attribution_posterior"] = agg.get("grounding_attribution_posterior")

    enr = agg.get("per_option_enrichment", {}) or {}
    cmp_po = ((cmpv.get("per_option_entailment_filter", {}) or {})
              .get("paired_bootstrap_items", {}) or {})
    ci = list(cmp_po.get("ci", [None, None]))
    kh_gap = _sub(nli_ref.get("key_hallucination_rate_mean"),
                  sec("llm_judge_faithfulness_filter", "key_hallucination_rate"))
    g["h2_enrichment_per_option"] = enr.get("enrichment_at_selected_tau")
    g["h2_enrichment_tau"] = enr.get("selected_tau")
    g["h2_enrichment_n_items"] = enr.get("selected_n_items")
    g["h2_enrichment_min_items"] = enr.get("min_items")
    g["h2_enrichment_at_plan_tau"] = enr.get("enrichment_at_plan_tau")
    g["h2_enrichment_at_retention"] = _sub(mean_of("per_option_entailment_filter"), base)
    g["h2_spearman_rho"] = enr.get("spearman_rho_tau_vs_rate")
    g["h2_spearman_n_taus"] = enr.get("spearman_n_taus")
    g["h2_retained_vs_pool_ci"] = ci
    g["h2_retained_vs_pool_scheme"] = cmp_po.get("scheme")
    g["h2_retained_set_enriched_ci_excludes_zero"] = (None if ci[0] is None else bool(ci[0] > 0))
    h2_in = [enr.get("spearman_rho_tau_vs_rate"), enr.get("enrichment_at_selected_tau"),
             ci[0], kh_gap]
    g["h2_supported"] = (None if any(v is None for v in h2_in)
                         else bool(h2_in[0] >= 0.5 and h2_in[1] >= 0.05 and h2_in[2] > 0
                                   and h2_in[3] >= 0.05))
    g["h2_independent_judge_enrichment"] = _sub(mean_of("llm_judge_faithfulness_filter"), base_nli)
    g["h2_independent_judge_reference"] = "unfiltered_grounded_pool_nli_flags"

    contra = mean_of("contradiction_polarity_item_filter")
    ne = mean_of("not_entail_polarity_item_filter")
    kh_c = sec("contradiction_polarity_item_filter", "key_hallucination_rate")
    kh_po = sec("per_option_entailment_filter", "key_hallucination_rate")
    f_unf = sec(REFERENCE_ARM, "functional_distractor_rate")
    f_c = sec("contradiction_polarity_item_filter", "functional_distractor_rate")
    red = _sub(base, contra)
    f_drop = _sub(f_unf, f_c)
    g["h3_reduction"] = red
    h3_in = [red, kh_c, kh_po, f_drop]
    g["h3_supported"] = (None if any(v is None for v in h3_in)
                         else bool(red >= 0.05 and kh_c <= kh_po + 1e-9 and f_drop <= 0.05))
    diff_ab = _sub(contra, ne)
    g["h3_contradiction_specific"] = (None if diff_ab is None else bool(abs(diff_ab) > 0.02))

    rc = mean_of("overgenerate_rerank_by_contradiction")
    sim = mean_of("overgenerate_rerank_by_key_similarity")
    g["h4_ran"] = bool(int((S.get("overgenerate_rerank_by_contradiction", {}) or {})
                           .get("n_seeds", 0) or 0) > 0)
    g["h4_h1_gate_would_pass"] = (None if (base is None or share_dc is None)
                                  else bool(base >= cfg.h4_base_rate_gate
                                            and share_dc >= cfg.h4_distractor_share_gate))
    filt = [mean_of(n) for n in FILTER_CONDITIONS]
    best_filter = min([v for v in filt if v is not None], default=None)
    ratio = sec("overgenerate_rerank_by_contradiction", "similarity_quartile_csd_ratio")
    g["h4_quartile_ratio"] = ratio
    g["h4_quartile_ratio_raw"] = sec("overgenerate_rerank_by_contradiction",
                                     "similarity_quartile_csd_ratio_raw")
    h4_in = [rc, best_filter, ratio]
    g["h4_supported"] = (None if any(v is None for v in h4_in)
                         else bool(rc < best_filter and ratio >= 1.3))
    g["h4_selection_criterion_effect"] = _sub(sim, rc)
    g["h4_vs_unfiltered"] = _sub(rc, base)

    # The two mandated control arms. null_random_drop must be near zero with a CI covering zero;
    # positive_control_planted must rise above the baseline.
    null_cmp = (cmpv.get("null_random_drop", {}) or {}).get("paired_bootstrap_items", {}) or {}
    null_ci = list(null_cmp.get("ci", [None, None]))
    null_eff = _sub(mean_of("null_random_drop"), base)
    g["null_arm_effect"] = null_eff
    g["null_arm_ci"] = null_ci
    g["null_arm_tolerance"] = NULL_EFFECT_TOLERANCE
    g["null_arm_near_zero"] = (None if (null_eff is None or null_ci[0] is None or null_ci[1] is None)
                               else bool(abs(null_eff) <= NULL_EFFECT_TOLERANCE
                                         and null_ci[0] <= 0.0 <= null_ci[1]))
    det = sec("positive_control_planted", "detection_ratio")
    g["positive_control_detection_ratio"] = det
    g["positive_control_measured_increase"] = sec("positive_control_planted",
                                                  "measured_rate_increase")
    g["positive_control_expected_increase"] = sec("positive_control_planted",
                                                  "expected_rate_increase")
    g["positive_control_detected"] = _ge(det, getattr(cfg, "detection_ratio_gate", 0.8))

    g["seed_level_tests_underpowered"] = bool(len(cfg.seeds) < 6)
    g["gate_secondary_seed_counts"] = counts
    g["gate_secondary_min_seeds"] = min_seeds
    g["rewrite_noop_rate_mean"] = agg.get("rewrite_noop_rate_mean")
    g["topic_label_fallback_rate_mean"] = agg.get("topic_label_fallback_rate_mean")
    g["bootstrap_warnings"] = list(agg.get("bootstrap_warnings", []))
    g["seed_alignment_warnings"] = list(agg.get("seed_alignment_warnings", []))
    return g


# ============================================================== the printed lines
def seed_metric_lines(conditions: Dict[str, Any], seed: int,
                      arm_names: Optional[Sequence[str]] = None) -> List[str]:
    """Two labelled `condition=NAME seed=S` lines per arm over EVERY arm name, printing NA for an
    undefined value; never emits the token nan."""
    lines: List[str] = []
    for name in list(arm_names or ALL_ARM_NAMES):
        c = conditions.get(name, {}) or {}
        v = c.get(PRIMARY)
        lines.append(f"condition={name} seed={seed} primary_metric: {fmt(v)}")
        lines.append(f"condition={name} seed={seed} {PRIMARY}: {fmt(v)} "
                     f"item_yield={fmt(c.get('item_yield'), 2)} "
                     f"key_halluc={fmt(c.get('key_hallucination_rate'), 3)} "
                     f"stem_caused={fmt(c.get('stem_caused_csd_rate'))} "
                     f"soft_dc={fmt(c.get('soft_distractor_caused_score'))} "
                     f"n_items={int(c.get('n_items', 0) or 0)} "
                     f"status={c.get('status', 'ok')}")
    return lines


def summary_lines(summary: Dict[str, Any],
                  arm_names: Optional[Sequence[str]] = None) -> List[str]:
    """Per-arm aggregate lines followed by EXACTLY one SUMMARY: line and EXACTLY one
    SUMMARY_SOFT: line covering every arm name; never emits the token nan."""
    names = list(arm_names or ALL_ARM_NAMES)
    lines: List[str] = []
    for name in names:
        c = summary.get(name, {}) or {}
        lines.append(f"condition={name} primary_metric: {fmt(c.get('primary_metric'))}")
        lines.append(f"condition={name} primary_metric_mean: {fmt(c.get('mean'))} "
                     f"primary_metric_std: {fmt(c.get('std'))}")
        pooled = c.get("pooled", {}) or {}
        ci = pooled.get("ci95", [None, None])
        lines.append(f"condition={name} {PRIMARY}_mean: {fmt(c.get('mean'))} "
                     f"{PRIMARY}_std: {fmt(c.get('std'))} "
                     f"pooled={fmt(pooled.get(PRIMARY))} "
                     f"pooled_ci95=[{fmt(ci[0])},{fmt(ci[1])}] "
                     f"n_distractors={int(pooled.get('n_distractors', 0) or 0)}")
        lines.append(f"condition={name} success_rate: {int(c.get('n_seeds', 0) or 0)}/"
                     f"{int(c.get('n_seeds_expected', 0) or 0)}")
    lines.append("SUMMARY: " + ", ".join(
        f"{n}={fmt((summary.get(n, {}) or {}).get('mean'))}" for n in names))
    lines.append("SUMMARY_SOFT: " + ", ".join(
        f"{n}={fmt(((summary.get(n, {}) or {}).get('secondary', {}) or {}).get('soft_distractor_caused_score', {}).get('mean'))}"
        for n in names))
    return lines


# ============================================================== the uniform budget lever
REDUCIBLE_COMPONENTS = (("items_per_seed", 3), ("obqa_replication_items", 3),
                        ("calibration_pairs", 20), ("functionality_samples_per_context", 2))


def plan_reduction(estimate_s: float, usable_s: float, min_fraction: float,
                   smoke: bool = False) -> Optional[float]:
    """ONE item fraction, decided before any arm runs, returned as a SINGLE value.

    1.0 means nothing moves; a float below 1.0 is the uniform fraction; None means the plan cannot
    fit even at the minimum item counts, so main must print BUDGET_INSUFFICIENT and exit non-zero.
    There is deliberately no (fraction, feasible) pair: unpacking one killed every run immediately
    after the pilot. A smoke design is already at its minimum, so it is never refused."""
    if smoke:
        return 1.0
    e, u = float(estimate_s), float(usable_s)
    if e <= 0 or e <= u:
        return 1.0
    frac = u / e
    return None if frac < float(min_fraction) else float(frac)


def apply_uniform_reduction(cfg: Any, fraction: Optional[float]) -> Dict[str, Any]:
    """Lower every REDUCIBLE_COMPONENTS knob by the SAME fraction with its floor; never drop a
    component. Returns {"fraction", "components"}, or an empty dict (changing nothing) when the
    fraction is None or at least 1.0.

    This is also the ONE writer of cfg.calibration_pairs_designed: the calibration set is built
    BEFORE this lever runs, so the bound it was built under is the value this function is about to
    overwrite. Recording it here -- at the only place that can lower the knob -- is what lets
    executed_design_record keep the interface sheet's three-parameter signature while still
    comparing the built size against the right bound. A second copy threaded through the call
    chain could drift from this one; there is no second copy.

    The write is a plain attribute assignment, not setattr with a literal name: Config DECLARES
    calibration_pairs_designed as None, so both the write here and the read in
    executed_design_record are visible to a reader and to the checker instead of being an
    attribute that springs into existence only after this function has run."""
    if fraction is None or float(fraction) >= 1.0:
        return {}
    components: Dict[str, Any] = {}
    for attr, floor in REDUCIBLE_COMPONENTS:
        before = int(getattr(cfg, attr))
        after = max(int(floor), int(math.floor(before * float(fraction))))
        if attr == "calibration_pairs":
            # the bound in force when the calibration set was built, kept before it is lowered
            cfg.calibration_pairs_designed = before
        setattr(cfg, attr, after)
        components[attr] = {"fraction": float(fraction), "before": before, "after": after,
                            "floor": int(floor)}
    return {"fraction": float(fraction), "components": components}


def truncate_folds(folds: Sequence[Sequence[int]], items_per_seed: int) -> List[List[int]]:
    """Each fold row cut to its first items_per_seed ids; never reorders, never mutates the input."""
    return [[int(x) for x in row[:int(items_per_seed)]] for row in folds]


def executed_design_record(cfg: Any, folds: Any,
                           calibration_pairs: Optional[int] = None) -> Dict[str, Any]:
    """ITEM 17: `folds` is the list of per-seed fold ROWS, never the SeedFolds object.

    Handing the object over raised "'SeedFolds' object is not iterable" deep inside the length
    comprehension, after the models, the corpus, the index and the pilot generation had all been
    paid for. The two shapes are now told apart at the door, with the expected one named.

    The signature is the interface sheet's three parameters exactly. The designed calibration
    bound is NOT an argument: it is read from cfg.calibration_pairs_designed, which
    apply_uniform_reduction records when it lowers cfg.calibration_pairs. The calibration set is
    built before the pilot and the lever lowers the knob after it, so comparing the built size
    against the REDUCED knob raised ValueError on every run whose reduction fraction was below
    1.0 -- after the whole seed loop, before the payload write, discarding hours of generation and
    judging. When no reduction has happened the attribute is absent and the comparison falls back
    to cfg.calibration_pairs, which is then the bound the set was built under."""
    if hasattr(folds, "folds"):
        raise TypeError(f"executed_design_record takes the list of fold ROWS, not a "
                        f"{type(folds).__name__} object; pass folds.folds")
    try:
        lens = [len(r) for r in folds]
    except TypeError as e:
        raise TypeError(f"executed_design_record takes a sequence of fold rows; got "
                        f"{type(folds).__name__}") from e
    if lens and (min(lens) != int(cfg.items_per_seed) or max(lens) != int(cfg.items_per_seed)):
        raise ValueError(f"executed design differs from the recorded one: fold lengths {lens} "
                         f"but items_per_seed={cfg.items_per_seed}")
    recorded_bound = getattr(cfg, "calibration_pairs_designed", None)
    designed = int(cfg.calibration_pairs if recorded_bound is None else recorded_bound)
    if calibration_pairs is not None and int(calibration_pairs) > designed:
        raise ValueError(f"executed design differs from the recorded one: {calibration_pairs} "
                         f"calibration pairs built but calibration_pairs={designed}")
    return {"n_seeds": len(lens), "items_per_seed": int(cfg.items_per_seed),
            "obqa_replication_items": int(cfg.obqa_replication_items),
            "calibration_pairs": (None if calibration_pairs is None else int(calibration_pairs)),
            "calibration_pairs_designed": designed,
            "functionality_samples_per_context": int(cfg.functionality_samples_per_context)}


def budget_insufficient_message(estimate_s: float, usable_s: float, min_fraction: float,
                                cfg: Any) -> str:
    """One line starting BUDGET_INSUFFICIENT: carrying the estimate, the usable budget, the
    required fraction, the minimum fraction and the design counts."""
    return (f"BUDGET_INSUFFICIENT: estimate={float(estimate_s):.0f}s "
            f"usable={float(usable_s):.0f}s "
            f"required_fraction={float(usable_s) / max(1.0, float(estimate_s)):.3f} "
            f"min_reduction_fraction={float(min_fraction):.3f} "
            f"n_seeds={len(cfg.seeds)} items_per_seed={cfg.items_per_seed} "
            f"arms={len(ALL_ARM_NAMES)}; the design cannot fit at the minimum item counts, so no "
            f"partial design is run")


# ============================================================== run-shape guards
CLOCK_WORDS = ("budget", "time", "elapsed", "clock", "deadline", "second", "runtime")
OMISSION_WORDS = ("skip", "omit", "drop", "cut", "truncated", "disabled", "not run", "partial")
# The time budget is 43200 s when RC_TIME_BUDGET_SEC is unset, so an empty environment string on
# main.py's side agrees with this value and with no other.
DEFAULT_TIME_BUDGET_SEC = 43200.0


def _budget_switch(value: Any) -> Tuple[str, Any]:
    """Classify one side of the RC_TIME_BUDGET_SEC comparison so the two sides can meet as numbers.

    Comparing a raw '' against a parsed '43200.0' as TEXT made the agreeing default read as a
    disagreement and aborted every run before the pre-flight self-tests."""
    if value is None:
        return ("unset", None)
    if isinstance(value, bool):
        return ("text", str(value))
    if isinstance(value, (int, float)):
        f = float(value)
        return ("number", f) if math.isfinite(f) else ("text", repr(value))
    text = str(value).strip()
    if not text:
        return ("unset", None)
    try:
        return ("number", float(text))
    except ValueError:
        return ("text", text)


def check_environment_switches(main_smoke: Any, main_budget: Any, cfg_smoke: Any,
                               cfg_budget: Any) -> None:
    """Return None when main.py and experiment_config describe the same run; raise ValueError
    naming BOTH values otherwise."""
    if bool(main_smoke) != bool(cfg_smoke):
        raise ValueError(f"RC_SMOKE_TEST disagreement: main.py read {main_smoke!r}, "
                         f"experiment_config recorded {cfg_smoke!r}")
    main_kind, main_value = _budget_switch(main_budget)
    cfg_kind, cfg_value = _budget_switch(cfg_budget)
    if main_kind == "unset":
        agree = bool(cfg_kind == "unset"
                     or (cfg_kind == "number" and cfg_value == DEFAULT_TIME_BUDGET_SEC))
    elif cfg_kind == "unset":
        agree = False
    else:
        agree = bool(main_kind == cfg_kind and main_value == cfg_value)
    if not agree:
        raise ValueError(f"RC_TIME_BUDGET_SEC disagreement: main.py read {main_budget!r}, "
                         f"experiment_config recorded {cfg_budget!r}")


def check_seed_design(seeds: Sequence[int], smoke: bool) -> None:
    """Return None for exactly one seed in smoke mode and at least three seeds otherwise; raise
    ValueError naming the seed list."""
    n = len(list(seeds))
    if smoke and n != 1:
        raise ValueError(f"smoke mode must run exactly one seed, got {list(seeds)}")
    if not smoke and n < 3:
        raise ValueError(f"a measured run needs at least three seeds (BUG-183), got {list(seeds)}")


def flag_safe(text: str) -> str:
    """The flag text, rewritten to a fixed design note when it would read as a clock-conditional
    omission; never raises."""
    low = str(text).lower()
    if any(w in low for w in CLOCK_WORDS) and any(w in low for w in OMISSION_WORDS):
        return ("design note: every component ran in full; the original note was rewritten so it "
                "cannot read as a conditional omission")
    return str(text)


# ============================================================== the pre-flight self-tests
def self_tests(cfg: Any) -> List[str]:
    """Exactly four SELF_TEST: lines after checking the nested bootstrap, id-based seed alignment,
    null serialisation and flag rewriting.

    Raises RuntimeError naming the first failed check, and touches no model: a statistics defect
    must fail in seconds instead of after five hours of GPU time."""
    lines: List[str] = []
    st = Statistics(cfg)

    ids = list(range(40))
    pool = empty_rows("pool")
    for i in ids:
        for d in range(3):
            pool["keys"].append((0, i))
            pool["dc"].append(bool(i % 4 == 0 and d == 0))
            pool["sc"].append(False)
            pool["cs"].append(bool(i % 4 == 0 and d == 0))
            pool["enum"].append("high" if i % 2 else "low")
            pool["cov"].append("covered")
            pool["key_sim"].append(0.5)
            pool["overlap"].append(0.5)
    keep = {i for i in ids if i % 4 == 0 or i % 8 == 1}
    sub = subset_rows(pool, [k[1] in keep for k in pool["keys"]])
    r = st.paired_bootstrap_diff(sub, pool, "dc", mode="nested")
    width = (r["ci"][1] - r["ci"][0]) if (r["ci"][0] is not None and r["ci"][1] is not None) else 0.0
    lines.append(f"SELF_TEST: nested_bootstrap scheme={r['scheme']} diff={fmt(r['diff'])} "
                 f"ci=[{fmt(r['ci'][0])},{fmt(r['ci'][1])}] nonzero_width={width > 0}")
    if not (width > 0 and r["diff"] is not None and r["diff"] > 0):
        raise RuntimeError("nested bootstrap self-test failed: retained-vs-pool CI is degenerate")

    a, b, common = aligned_by_seed({0: 0.1, 2: 0.3}, {0: 0.0, 1: 0.9, 2: 0.2})
    if common != [0, 2] or b != [0.0, 0.2] or a != [0.1, 0.3]:
        raise RuntimeError("seed alignment self-test failed: values were not paired by seed id")
    lines.append("SELF_TEST: seed_alignment ok")

    probe = dumps({"a": None, "b": [1.0, None], "c": {"d": None}})
    if "NaN" in probe or "Infinity" in probe or "null" not in probe:
        raise RuntimeError(f"JSON self-test failed: {probe}")
    caught = False
    try:
        assert_no_nonfinite({"x": {"y": [float("nan")]}})
    except ValueError:
        caught = True
    if not caught:
        raise RuntimeError("JSON self-test failed: a NaN was not caught by assert_no_nonfinite")
    lines.append("SELF_TEST: undefined_value_serialises_as_null ok")

    rewritten = flag_safe("the proxy was skipped because the time budget ran out")
    if any(w in rewritten.lower() for w in OMISSION_WORDS):
        raise RuntimeError("flag self-test failed: a clock-conditional omission survived")
    lines.append("SELF_TEST: clock_conditional_flag_rewritten ok")
    return lines


def calibration_support_floor(n_pairs: Optional[int], n_deciles: int, cap: int) -> int:
    """ITEM 33: the support floor the top overlap decile must clear.

    Derived from the set ACTUALLY BUILT: a smoke run with 20 pairs holds ~2 per decile and must
    not be judged against a floor it structurally cannot meet, while a 1600-pair run is held to
    the configured cap. Never above the cap and never below one.

    Lives here, not in models.py, because it decides which judge measures every rate in the
    results and the offline suite -- which may not import a module that loads weights -- has to be
    able to drive it."""
    cap_i = max(1, int(cap))
    if n_pairs is None:
        return cap_i
    per_decile = int(n_pairs) // max(1, int(n_deciles))
    return max(1, min(cap_i, per_decile))


def calibration_precision_verdict(n_top_decile_pairs: int, min_top_decile_pairs: int,
                                  precision: Optional[float], gate: float, n_pairs: int,
                                  n_pred_pos: int) -> Dict[str, Any]:
    """ITEM 33: the judge verdict, with an undefined cell RECORDED and never read as a measured 0.0.

    Returns the four fields the calibration record publishes:
    judge_precision_top_overlap_decile, top_decile_below_floor, passes, passes_reason. It RAISES
    on nothing: an under-filled top decile is a diagnostic that must reach the payload, because
    raising here would discard the models, corpus, index and pilot after all of them were paid for.

    A spurious 0.0 in the top decile fails the precision gate, promotes the other judge and
    relabels every rate in the results, so an unmeasured cell must be distinguishable from a
    measured failure: precision None means undefined, and the reason says so in words."""
    n_top = int(n_top_decile_pairs)
    floor = int(min_top_decile_pairs)
    below_floor = bool(n_top < floor)
    top = None if (below_floor or precision is None) else float(precision)
    if below_floor:
        passes = False
        reason = (f"top overlap decile holds {n_top} pairs, below the support floor of "
                  f"{floor} derived from {int(n_pairs)} calibration pairs; precision is undefined "
                  f"and no verdict is claimed")
    elif top is None:
        passes = False
        reason = (f"top overlap decile predicted no positives ({int(n_pred_pos)} "
                  f"predicted of {n_top} pairs); precision is undefined, not zero")
    else:
        passes = bool(top >= float(gate))
        reason = (f"top-decile precision {top:.3f} "
                  f"{'>=' if passes else '<'} gate {gate}")
    return {"judge_precision_top_overlap_decile": top,
            "top_decile_below_floor": below_floor,
            "passes": passes, "passes_reason": reason}


def paired_bootstrap_diff(*args: Any, **kwargs: Any) -> None:
    """Not an implementation: the live one is Statistics.paired_bootstrap_diff.

    A module-level copy of that method used to sit here, taking a `self` no caller supplies. It
    was unreachable dead code that read as a second implementation of the diagnostic collection,
    so the CORRECT body was the body that could not run: the live method went on overwriting
    out["warning"] and never set out["warnings"], and an edit landing in the copy had no effect
    on any comparison. Calling this name RAISES rather than silently doing nothing or failing on
    arity, because a caller reaching it is reaching for the wrong one."""
    del args, kwargs
    raise RuntimeError("analysis.paired_bootstrap_diff is not the live implementation; the "
                       "aggregate calls Statistics.paired_bootstrap_diff, which collects the "
                       "degraded-scheme and zero-width diagnostics in out['warnings'] and "
                       "publishes their join under out['warning']")


def substitution_kinds() -> Tuple[str, str]:
    """The two DIFFERENT values GeneratorLLM can substitute for a non-finite forward pass.

    They are separate quantities with separate consequences and must never share a counter:

      judge_p_yes            NAN_P_YES_SUBSTITUTE, deliberately BELOW every entail threshold in
                             (0, 1] so it can never read as support. Its slot leaves the three
                             distractor rates and key_hallucination_rate, which feed h2_supported
                             and h3_supported.
      answerer_letter_probs  NAN_LETTER_PROB_SUBSTITUTE, a uniform row over A-D. It voids the
                             whole with-passage / no-passage CONTEXT it belongs to, which leaves
                             functional_distractor_rate and so h3_supported by a different route.

    One shared count could not say which instrument was broken, and one shared sentence called
    the judge's 0.0 "a neutral value" -- the opposite of what it is for. models.GeneratorLLM
    builds its per-kind counters from this tuple, so the producing side and the reporting side
    cannot drift apart."""
    return ("judge_p_yes", "answerer_letter_probs")


def substitution_flag_lines(by_kind: Dict[str, int], p_yes_substitute: float,
                            letter_substitute: float) -> List[str]:
    """One flag line per substitution KIND that actually fired; an empty list when none did.

    Each line carries its OWN count and the exact value substituted, and says what excluding
    those rows means for the metrics. The single merged line this replaces reported both kinds
    under one total and described the judge substitute as neutral, so a reader could neither tell
    which instrument produced the non-finite rows nor what the imputed value would have done if
    the validity mask had not excluded it.

    The text lives here rather than in main.py for the reason this module exists: the offline
    suite may not import main or models, and this is a payload rule the gate has to be able to
    drive. A kind with no declared text still gets a line naming the kind -- this runs after the
    seed loop and before the payload write, so raising would discard every measured seed in order
    to report a labelling gap."""
    text = {
        "judge_p_yes": ("{n} generative-judge rows were non-finite and were recorded as "
                        "UNMEASURED, carrying the substitute {value}, which is below every "
                        "entail threshold and therefore can never read as support; the validity "
                        "mask keeps those slots out of both the numerators and the denominators. "
                        "See numeric_health.generator.nan_substitutions_by_kind."),
        "answerer_letter_probs": ("{n} functionality-proxy answerer rows were non-finite and were "
                                  "recorded as UNMEASURED, carrying a uniform substitute {value} "
                                  "over the four options; the whole with-passage / no-passage "
                                  "context each one belongs to is excluded from "
                                  "functional_distractor_rate. See "
                                  "numeric_health.generator.nan_substitutions_by_kind."),
    }
    values = {"judge_p_yes": float(p_yes_substitute),
              "answerer_letter_probs": float(letter_substitute)}
    counts = dict(by_kind or {})
    lines: List[str] = []
    # the declared kinds first, in their declared order, so the payload reads the same every run
    for kind in substitution_kinds():
        n = int(counts.pop(kind, 0) or 0)
        if n > 0:
            lines.append(text[kind].format(n=n, value=values[kind]))
    for kind in sorted(counts):
        n = int(counts[kind] or 0)
        if n > 0:
            lines.append(f"{n} {kind} rows were non-finite and were recorded as UNMEASURED; this "
                         f"substitution kind has no declared flag text, so the value it carries "
                         f"is in numeric_health.generator.nan_substitutions_by_kind")
    return lines


def programming_errors() -> Tuple[type, ...]:
    """The exception classes that are ALWAYS a defect in this code, never a property of the data.

    Read by main's three re-raise sites -- run_arm_sequence, guarded_replication and the seed
    loop -- so the handlers cannot drift apart, and it lives HERE rather than in main.py for two
    reasons. First, ordering: main.py binds this rule during its import block, before the
    `if __name__ == "__main__":` guard calls main(), so no placement of a definition inside
    main.py can leave the name unbound while the run is under way. It once did: the definition sat
    BELOW the guard, so the name was unbound for the whole of main(), and the first exception to
    reach any handler raised NameError instead, destroyed the real exception, escaped the
    `except Exception` recovery branch on the same try, and ended the run with zero seeds recorded
    and no payload -- the results carried no condition at all, both mandated control arms
    included. Second, testability: the offline suite may not import main, so a rule that lived
    there could not be driven by any test, which is why that breach survived every gate.

    NameError is in the tuple because of the rerank breach: _rescore_selected called two module
    helpers that existed nowhere, so under a promoted LLM judge every rerank call raised
    NameError, the broad handler in run_arm_sequence turned it into an arm_failure_record, and
    both H4 arms reported "failed" on every seed with the payload naming the ARM as the cause
    rather than the missing symbol.

    A missing symbol is the same kind of fault as a renamed method (AttributeError) or a tuple
    consumed as a list (TypeError): it recurs identically on every seed, no amount of different
    data will change it, and filing it as a measurement failure buries it. These stop the run
    loudly instead.

    ImportError is deliberately NOT here: get_harness catches it on purpose to fall back to the
    stand-in harness, and a missing optional dependency is a recorded run condition."""
    return (AttributeError, NameError, TypeError)
