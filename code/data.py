"""
data.py — SciQ / OpenBookQA loading, the course corpus and its cap, the disjoint per-seed
folds, the OpenBookQA replication pool, the judge-calibration pair set, the QuizItem
record and the pure text and token-plan utilities.

This module imports no project module and no model: the test suite builds a real QuizItem,
caps a toy corpus and parses a candidate reply with nothing loaded and no GPU.

Two rules this file exists to enforce:
  * the corpus cap runs BEFORE any embedding and can never drop a passage some fold or
    replication item points at (apply_corpus_cap / protected_passage_ids / finalize_corpus);
  * an undefined value is None, never NaN — a failure record carrying a fabricated number
    is indistinguishable from a measured one, and a NaN fails the payload write guard.
"""
import json
import math
import os
import re
import traceback
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ============================================================== ITEM 37: the failure record
class SeedAbort(Exception):
    """Raised when a seed must be recorded as a failure rather than silently filled.

    Carries the pipeline stage that failed and, when applicable, the offending rate (e.g. the
    parse-failure rate that crossed the abort threshold). An UNSET rate is None, never NaN:
    the previous default of float("nan") gave every abort a number, wrote NaN into the payload
    and made an unmeasured failure look like a measured one."""

    def __init__(self, message: str, stage: str = "unknown", rate: Optional[float] = None):
        super().__init__(message)
        self.message = message
        self.stage = stage
        self.rate = self._clean(rate)

    @staticmethod
    def _clean(rate: Optional[float]) -> Optional[float]:
        """The rate as a finite float, or None when it is unset or non-finite."""
        if rate is None:
            return None
        try:
            value = float(rate)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    def to_record(self) -> Dict[str, Any]:
        return {"message": self.message, "stage": self.stage, "rate": self.rate}

    def __str__(self) -> str:
        if self.rate is None:
            return f"{self.message} [stage={self.stage}]"
        return f"{self.message} [stage={self.stage}, rate={self.rate:.3f}]"


# ============================================================== text utilities
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_STOP = set(
    "a an the of in on at to for and or is are was were be been by with from as that this these those "
    "it its into which what who whom whose when where why how do does did not no than then there their "
    "they them we you he she his her our your can could may might will would shall should".split()
)
_ENUM_RE = re.compile(r"\(\d\)|\b\d\)|\bfirst\b|\bsecond\b|\bthird\b|\bsuch as\b|\bincluding\b|\bfor example\b", re.I)
_WOTF_RE = re.compile(r"which of (the )?following|which one of|all of the following|which of these")
_LIST_BULLET_RE = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s*")


def tokens(text: str) -> List[str]:
    """Lower-cased alphanumeric tokens; an empty or None text yields an empty list."""
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def content_tokens(text: str) -> List[str]:
    """Lower-cased tokens with the stop list removed."""
    return [t for t in tokens(text) if t not in _STOP]


def lexical_overlap(option: str, passage: str) -> float:
    """Fraction of the option's content tokens that occur in the passage, in 0-1.

    An option with no content token has nothing to find, so the answer is 0.0 -- the
    denominator is checked before the division and NaN can never arise here."""
    ot = set(content_tokens(option))
    if not ot:
        return 0.0
    pt = set(tokens(passage))
    return len(ot & pt) / len(ot)


def enumeration_density(passage: str) -> float:
    """Fraction of sentences that look like a list, in 0-1; 0.0 for an empty passage."""
    sents = [s for s in re.split(r"[.;:]", passage or "") if s.strip()]
    if not sents:
        return 0.0
    n = 0
    for s in sents:
        commas = s.count(",")
        if commas >= 2 or (commas >= 1 and (" and " in s or " or " in s)) or _ENUM_RE.search(s):
            n += 1
    return n / len(sents)


def which_of_the_following(stem: str) -> bool:
    return bool(_WOTF_RE.search((stem or "").lower()))


def normalize_text(s: str) -> str:
    """The deduplication key: whitespace-collapsed, lower-cased, punctuation-stripped."""
    return re.sub(r"\s+", " ", (s or "").lower()).strip(" .\"'")


def assert_finite(name: str, arr: Any) -> None:
    """Return None when every entry is finite; raise ValueError naming the tensor and shape."""
    a = np.asarray(arr, dtype=np.float64)
    if a.size and not np.all(np.isfinite(a)):
        raise ValueError(f"NaN/Inf in stored tensor '{name}' (shape {a.shape})")


# ============================================================== ITEM 34: the NLI token plan
def nli_pair_token_plan(n_premise_tokens: int, n_hypothesis_tokens: int, max_length: int = 512,
                        hypothesis_budget: int = 64, n_special: int = 4) -> Dict[str, Any]:
    """The premise/hypothesis keep counts under a FIXED premise allowance.

    The allowance is max_length - hypothesis_budget - n_special and does NOT depend on the
    actual hypothesis length. The specificity rewrite lengthens the stem by construction, so a
    hypothesis-dependent allowance would judge the rewrite stage on less of the passage than the
    original stage and move rows from distractor-caused to stem-caused for a reason that is not
    in the data -- a systematic bias in the primary metric's own decomposition."""
    allowance = max(1, int(max_length) - int(hypothesis_budget) - int(n_special))
    hyp_keep = min(int(n_hypothesis_tokens), int(hypothesis_budget))
    prem_keep = min(int(n_premise_tokens), allowance)
    removed_p = max(0, int(n_premise_tokens) - prem_keep)
    removed_h = max(0, int(n_hypothesis_tokens) - hyp_keep)
    return {"premise_allowance": allowance, "premise_keep": prem_keep,
            "hypothesis_budget": int(hypothesis_budget), "hypothesis_keep": hyp_keep,
            "removed_premise_tokens": removed_p, "removed_hypothesis_tokens": removed_h,
            "removed_tokens": removed_p + removed_h,
            "truncated": bool(removed_p > 0 or removed_h > 0),
            "max_length": int(max_length)}


def pass_start_batch_size(home_batch_size: int, current_batch_size: int) -> int:
    """The HOME batch size, so an OOM narrowing inside one pass never narrows a later one.

    One 8->1 fallback used to make every remaining pass of the run eight times slower for no
    reason; the fallback itself stays counted on the model object."""
    return int(home_batch_size)


# ============================================================== generator-output parsing
def _json_loads_lenient(span: str):
    """Parse a JSON span, tolerating a trailing comma or single quotes; None on failure."""
    for cand in (span, re.sub(r",\s*([}\]])", r"\1", span), span.replace("'", '"')):
        try:
            return json.loads(cand)
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def parse_item_json(text: str) -> Optional[Dict[str, Any]]:
    """{stem, key, distractors[3]} from a completion, or None when the reply is unparseable or
    yields fewer than three unique distractors; never returns a partial item."""
    if not text:
        return None
    s, e = text.find("{"), text.rfind("}")
    if s < 0 or e <= s:
        return None
    obj = _json_loads_lenient(text[s:e + 1])
    if not isinstance(obj, dict):
        return None
    stem = str(obj.get("stem") or obj.get("question") or "").strip()
    key = str(obj.get("key") or obj.get("answer") or obj.get("correct_answer") or "").strip()
    ds = obj.get("distractors")
    if isinstance(ds, dict):
        ds = list(ds.values())
    if not isinstance(ds, list):
        return None
    seen = {normalize_text(key)}
    uniq: List[str] = []
    for d in ds:
        d = str(d).strip()
        if not d or normalize_text(d) in seen:
            continue
        seen.add(normalize_text(d))
        uniq.append(d)
    if not stem or not key or len(uniq) < 3:
        return None
    return {"stem": stem, "key": key, "distractors": uniq[:3]}


# ITEM 38: the candidate list and the flag of the fallback that produced it
def parse_candidate_list(text: str, key: str, max_n: int = 10) -> Tuple[List[str], bool]:
    """The PAIR (candidates, line_split_flag).

    The flag is True ONLY when the JSON list failed to parse and the reply was recovered line by
    line; an empty reply recovered nothing and returns ([], False). A fallback that changes what
    the rerank arms select from must be counted at the point it is taken, not inferred later."""
    raw: List[str] = []
    line_split = False
    if text:
        obj = None
        s, e = text.find("["), text.rfind("]")
        if s >= 0 and e > s:
            obj = _json_loads_lenient(text[s:e + 1])
        if isinstance(obj, list):
            raw = [str(x).strip() for x in obj]
        else:
            line_split = True                     # the JSON path failed; recover line by line
            for line in text.splitlines():
                line = _LIST_BULLET_RE.sub("", line).strip().strip('",')
                if line and not line.startswith(("[", "]", "{", "}")):
                    raw.append(line)
    seen = {normalize_text(key)}
    out: List[str] = []
    for c in raw:
        if not c or normalize_text(c) in seen:
            continue
        seen.add(normalize_text(c))
        out.append(c)
        if len(out) >= max_n:
            break
    return out, line_split


# ============================================================== the item record
@dataclass
class QuizItem:
    """One generated item and every tensor scored against it.

    planted / cand_padded / cand_line_split / cand_repair_improved are provenance markers, all
    False on a fresh item, so an arm can report how its input was produced instead of
    reconstructing it by comparison. The three cand_* markers travel with the shared candidate
    cache itself, so every arm that reranks that cache reports the SAME counts for it.

    llm_pyes_valid / llm_pyes_rewrite_valid are the per-option VALIDITY masks of the generative
    judge: False where that forward-pass row was not finite and the probability beside it was
    substituted. None means "not scored by the LLM judge", which every reader treats as all-valid.
    An imputed probability must never decide a flag, so FlagDecomposer excludes a masked slot from
    the numerators AND the denominators instead of counting it as a measured negative."""

    item_id: int
    seed: int
    gold_idx: int
    topic: str
    passage: str
    passage_id: int
    retrieval_covered: bool
    stem: str
    options: List[str]                       # [key, d1, d2, d3]
    enumeration_dense: bool
    arm: str = ""
    stem_rewritten: Optional[str] = None
    nli_orig: Optional[np.ndarray] = None    # [4,3] (C,E,N)
    nli_rewrite: Optional[np.ndarray] = None
    llm_pyes: Optional[np.ndarray] = None    # [4]
    llm_pyes_rewrite: Optional[np.ndarray] = None
    llm_pyes_valid: Optional[np.ndarray] = None          # [4] bool
    llm_pyes_rewrite_valid: Optional[np.ndarray] = None  # [4] bool
    key_sim: Optional[np.ndarray] = None     # [4]
    overlap: Optional[np.ndarray] = None     # [4]
    candidates: Optional[List[str]] = None
    cand_nli: Optional[np.ndarray] = None    # [10,3] NaN-padded past the real candidate count
    cand_sim: Optional[np.ndarray] = None    # [10]   NaN-padded past the real candidate count
    cand_emb: Optional[np.ndarray] = None    # [n_c, dim]
    functional: Optional[np.ndarray] = None  # [3] bool
    planted: bool = False
    cand_padded: bool = False
    cand_line_split: bool = False
    cand_repair_improved: bool = False

    def to_record(self) -> Dict[str, Any]:
        """A JSON-ready record: passage truncated, embedding omitted, and the candidate tensors
        SLICED to the real candidate count so no NaN padding sentinel reaches the payload."""
        n_c = len(self.candidates) if self.candidates else 0
        rec: Dict[str, Any] = {}
        for f in fields(self):
            if f.name == "cand_emb":
                continue
            v = getattr(self, f.name)
            if f.name in ("cand_nli", "cand_sim") and isinstance(v, np.ndarray):
                v = v[:n_c]
            if isinstance(v, np.ndarray):
                v = v.tolist()
            rec[f.name] = v
        rec["passage"] = self.passage[:300]
        rec["n_candidates"] = n_c
        return rec


# ============================================================== HF loading
def load_hf_dataset(hf_id: str, cache_dir: str, subset: Optional[str] = None) -> Any:
    """The loaded dataset; raises RuntimeError naming the id, the cache dir and the original
    error (with a Windows path-length hint), and never returns a partial dataset."""
    from datasets import load_dataset
    os.makedirs(cache_dir, exist_ok=True)
    try:
        if subset is not None:
            return load_dataset(hf_id, subset, cache_dir=cache_dir)
        return load_dataset(hf_id, cache_dir=cache_dir)
    except Exception as e:
        hint = ""
        if isinstance(e, FileNotFoundError) and os.name == "nt":
            hint = (f" On Windows a FileNotFoundError on the .lock file means the cache path "
                    f"({len(cache_dir)} chars) is too long; set RC_DATA_ROOT to a short directory "
                    f"and rerun setup.py.")
        raise RuntimeError(
            f"Could not load '{hf_id}'{' (subset=' + subset + ')' if subset else ''} from "
            f"'{cache_dir}'. Run setup.py first.{hint} Original error: {type(e).__name__}: {e}"
        ) from e


# ============================================================== ITEM 36: the corpus cap
def apply_corpus_cap(passages: Sequence[str], sources: Sequence[str], gold_support_id: Dict[Any, int],
                     protected_ids: Sequence[int], cap: Optional[int],
                     seed: int = 0) -> Tuple[List[str], List[str], Dict[Any, int], List[int], Dict[str, Any]]:
    """Keep every protected passage plus a seeded sample of the rest, and REMAP every id.

    A smoke run embedded 13,461 passages before one arm ran, so the cap must act before the index
    embeds anything; and a cap that dropped the gold passage of a fold or replication item would
    point that item at the wrong passage, so every protected id survives unconditionally.

    A cap BELOW the protected count is reported in n_protected_over_cap with cap_effective set to
    what was actually kept -- never silently widened into a larger sample, and never enforced by
    dropping a gold passage."""
    total = len(passages)
    protected = sorted({int(p) for p in protected_ids if 0 <= int(p) < total})
    n_protected = len(protected)
    cap_requested = None if cap is None else int(cap)

    if cap_requested is None or cap_requested >= total:
        keep_ids = list(range(total))
        capped = False
    else:
        protected_set = set(protected)
        others = [i for i in range(total) if i not in protected_set]
        budget = max(0, cap_requested - n_protected)
        if budget > 0 and others:
            rng = np.random.default_rng(int(seed))
            take = min(budget, len(others))
            sampled = rng.choice(np.array(others, dtype=np.int64), size=take, replace=False)
            keep_ids = sorted(protected + [int(x) for x in sampled])
        else:
            keep_ids = list(protected)
        capped = True

    remap = {old: new for new, old in enumerate(keep_ids)}
    kept_passages = [passages[i] for i in keep_ids]
    kept_sources = [sources[i] for i in keep_ids]
    # An entry whose passage was dropped is REMOVED, never left pointing at whatever id
    # inherited its slot.
    new_gold = {k: remap[v] for k, v in gold_support_id.items() if v in remap}
    record = {"kept": len(keep_ids), "total": total, "capped": bool(capped),
              "n_protected": n_protected, "cap_requested": cap_requested,
              "cap_effective": len(keep_ids),
              "n_protected_over_cap": (max(0, n_protected - cap_requested)
                                       if cap_requested is not None else 0),
              "n_gold_entries_dropped": len(gold_support_id) - len(new_gold),
              "seed": int(seed)}
    return kept_passages, kept_sources, new_gold, keep_ids, record


class CorpusBuilder:
    """The course corpus: deduplicated SciQ support passages plus OpenBookQA core facts.

    Nothing may embed this corpus until finalize_corpus has run, because the cap changes every
    passage id. The cap and the enumeration density are SEPARATE steps: finalize_density fills the
    density over the uncapped corpus so the folds and the replication ids can be chosen from it,
    and finalize_corpus then applies the cap ONCE with those ids protected."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.passages: List[str] = []
        self.sources: List[str] = []
        self._seen: Dict[str, int] = {}
        self.gold_support_id: Dict[tuple, int] = {}
        self.sciq: Any = None
        self.obqa: Any = None
        self.flags: List[str] = []
        self.openbookqa_error: Optional[Dict[str, str]] = None
        self.enum_density: Optional[np.ndarray] = None
        self.enum_median = 0.0
        self.n_sciq_passages = 0
        self.cap_record: Dict[str, Any] = {}
        self.density_finalized = False
        self.finalized = False

    def _add(self, text: str, source: str) -> int:
        key = normalize_text(text)
        if key in self._seen:
            return self._seen[key]
        pid = len(self.passages)
        self.passages.append(text)
        self.sources.append(source)
        self._seen[key] = pid
        return pid

    def ingest_sciq(self, data_root: str, loader: Optional[Callable[..., Any]] = None) -> None:
        """Add every eligible SciQ support passage across the three official splits."""
        load = loader or load_hf_dataset
        self.sciq = load("allenai/sciq", os.path.join(data_root, "sciq"))
        for split in ("train", "validation", "test"):
            if split not in self.sciq:
                raise ValueError(f"SciQ is missing official split '{split}': {list(self.sciq.keys())}")
            for i, row in enumerate(self.sciq[split]):
                s = (row.get("support") or "").strip()
                if len(tokens(s)) < self.cfg.min_corpus_tokens:
                    continue
                self.gold_support_id[("sciq", split, i)] = self._add(s, "sciq_support")
        self.n_sciq_passages = len(self.passages)

    def ingest_openbookqa(self, data_root: str, loader: Optional[Callable[..., Any]] = None) -> bool:
        """True when all three splits were read and committed.

        On ANY failure this commits nothing and records the cause: a partial ingest would put half
        of OpenBookQA's facts in the retrieval corpus while corpus.obqa stayed None, so the
        replication would find no items but the index would still hold the facts they were meant
        to be found by."""
        load = loader or load_hf_dataset
        staged: List[Tuple[tuple, str]] = []
        try:
            obqa = load("allenai/openbookqa", os.path.join(data_root, "openbookqa"), subset="additional")
            for split in ("train", "validation", "test"):
                if split not in obqa:
                    raise ValueError(f"OpenBookQA is missing split '{split}': {list(obqa.keys())}")
                for i, row in enumerate(obqa[split]):
                    s = (row.get("fact1") or "").strip()
                    if len(tokens(s)) < 3:
                        continue
                    staged.append((("obqa", split, i), s))
        except Exception as e:                    # recorded with its cause, never swallowed
            self.openbookqa_error = {"error_type": type(e).__name__, "error_message": str(e),
                                     "traceback_tail": traceback.format_exc()[-800:]}
            self.flags.append(f"openbookqa_unavailable: {type(e).__name__}: {e}")
            print(f"[corpus] WARNING: OpenBookQA unavailable, corpus is SciQ-only: "
                  f"{type(e).__name__}: {e}")
            self.obqa = None
            return False
        for key, text in staged:
            self.gold_support_id[key] = self._add(text, "openbookqa_core_facts")
        self.obqa = obqa
        return True

    def build(self, data_root: str, loader: Optional[Callable[..., Any]] = None) -> "CorpusBuilder":
        print(f"[corpus] data root: {data_root}")
        self.ingest_sciq(data_root, loader)
        self.ingest_openbookqa(data_root, loader)
        if self.n_sciq_passages < 1000 and not self.cfg.smoke:
            raise ValueError(f"SciQ corpus unexpectedly small ({self.n_sciq_passages} passages); "
                             f"cached dataset incomplete?")
        print(f"[corpus] ingested {len(self.passages)} passages "
              f"({self.n_sciq_passages} SciQ, {len(self.passages) - self.n_sciq_passages} OBQA facts)")
        return self

    def protected_passage_ids(self, folds: Optional[Sequence[Sequence[int]]] = None,
                              replication_indices: Optional[Sequence[int]] = None) -> List[int]:
        """The sorted gold passage ids of every fold item and every replication item.

        These are exactly the passages the cap must never drop: an item whose gold passage is
        gone has no evidence to be scored against."""
        out: set = set()
        for row in (folds or []):
            for gi in row:
                pid = self.gold_support_id.get(("sciq", "test", int(gi)))
                if pid is not None:
                    out.add(int(pid))
        for i in (replication_indices or []):
            pid = self.gold_support_id.get(("obqa", "test", int(i)))
            if pid is not None:
                out.add(int(pid))
        return sorted(out)

    def _rebuild_derived(self) -> None:
        """Recompute the dedup index, the SciQ passage count and the enumeration density over the
        CURRENT passage list. Called by finalize_density (no cap applied) and again by
        finalize_corpus (after the cap remapped every id)."""
        self._seen = {normalize_text(p): i for i, p in enumerate(self.passages)}
        self.n_sciq_passages = sum(1 for s in self.sources if s == "sciq_support")
        self.enum_density = np.array([enumeration_density(p) for p in self.passages], dtype=np.float32)
        self.enum_median = float(np.median(self.enum_density)) if len(self.passages) else 0.0

    def finalize_density(self) -> None:
        """Fill the enumeration density over the UNCAPPED corpus, capping nothing and remapping no
        id.

        This exists so the folds and the replication ids can be chosen from the whole corpus and
        then handed to finalize_corpus as protected ids. Using finalize_corpus for this ran the cap
        with an EMPTY protected list, so a smoke run pruned 13,461 passages to 200 before any fold
        existed, chose its folds from what happened to survive, and left the later
        finalize_corpus(protected) with nothing to protect and nothing to drop -- which then
        overwrote cap_record with a no-op 'capped: False' that contradicted the flag the first call
        had already appended."""
        self._rebuild_derived()
        self.density_finalized = True
        print(f"[corpus] density over {len(self.passages)} uncapped passages; "
              f"enumeration-density median={self.enum_median:.3f}")

    def finalize_corpus(self, protected_ids: Sequence[int] = ()) -> Dict[str, Any]:
        """Apply the cap BEFORE any embedding, rebuild the derived structures and return the
        cap record. Nothing may index this corpus until this has run, and it may run only ONCE:
        a second call would re-cap the already-capped corpus, find nothing to drop, and rewrite
        cap_record with a 'capped: False' that contradicts the cut that really happened."""
        if self.finalized:
            raise RuntimeError(
                f"finalize_corpus has already run; a second call would re-run the cap on the "
                f"already-capped corpus and OVERWRITE the record of the real cut "
                f"({self.cap_record}). Call finalize_density() first if the enumeration density "
                f"is needed before the protected ids exist.")
        cap = getattr(self.cfg, "corpus_passage_cap", None)
        (self.passages, self.sources, self.gold_support_id,
         kept_ids, record) = apply_corpus_cap(self.passages, self.sources, self.gold_support_id,
                                              protected_ids, cap, int(self.cfg.fold_seed))
        self._rebuild_derived()
        self.cap_record = record
        self.finalized = True
        if record["capped"]:
            self.flags.append(f"corpus capped to {record['kept']} of {record['total']} passages "
                              f"(cap_requested={record['cap_requested']}, "
                              f"protected={record['n_protected']})")
        if record["n_protected_over_cap"] > 0:
            self.flags.append(f"corpus cap {record['cap_requested']} is below the "
                              f"{record['n_protected']} protected passages; kept "
                              f"{record['cap_effective']} so no fold or replication item lost its "
                              f"gold passage")
        print(f"[corpus] finalized: {record['kept']}/{record['total']} passages "
              f"(capped={record['capped']}, protected={record['n_protected']}); "
              f"enumeration-density median={self.enum_median:.3f}")
        return record

    def composition(self) -> Dict[str, Any]:
        return {"n_passages": len(self.passages), "n_sciq_passages": self.n_sciq_passages,
                "n_openbookqa_facts": len(self.passages) - self.n_sciq_passages,
                "openbookqa_available": bool(self.obqa is not None),
                "openbookqa_error": self.openbookqa_error,
                "enumeration_density_median": self.enum_median,
                "cap": dict(self.cap_record), "finalized": bool(self.finalized),
                "density_finalized": bool(self.density_finalized),
                "n_gold_entries": len(self.gold_support_id)}

    def passage_text(self, pid: int) -> str:
        return self.passages[pid][: self.cfg.passage_char_cap]

    def is_enumeration_dense(self, pid: int) -> bool:
        if self.enum_density is None:
            raise RuntimeError("is_enumeration_dense needs finalize_density or finalize_corpus "
                               "to have run")
        return bool(self.enum_density[pid] >= self.enum_median)


# ============================================================== folds and the replication pool
class SeedFolds:
    """Disjoint folds of SciQ test items, one per seed; the row lists live on .folds.

    main.py passes folds.folds (the list of rows) wherever a sequence of rows is wanted, and this
    object only where fold() is called -- handing the object to a row consumer raised
    "'SeedFolds' object is not iterable" after five hours of GPU time had been spent."""

    def __init__(self, cfg: Any, corpus: CorpusBuilder) -> None:
        self.cfg = cfg
        self.corpus = corpus
        self.flags: List[str] = []
        test = corpus.sciq["test"]
        need = len(cfg.seeds) * cfg.items_per_seed
        eligible = self._eligible(test, cfg.min_support_tokens)
        if len(eligible) < need:
            self.flags.append(f"min_support_tokens lowered {cfg.min_support_tokens}->20 "
                              f"(only {len(eligible)} eligible, need {need})")
            eligible = self._eligible(test, 20)
        if len(eligible) < need:
            raise ValueError(f"SciQ test split yields only {len(eligible)} eligible passages; "
                             f"need {need} for {len(cfg.seeds)} seeds x {cfg.items_per_seed} items.")
        rng = np.random.default_rng(cfg.fold_seed)
        perm = rng.permutation(np.array(eligible))
        rows = perm[:need].reshape(len(cfg.seeds), cfg.items_per_seed)
        self.folds: List[List[int]] = [[int(x) for x in row] for row in rows]
        self.test = test

    def _eligible(self, test: Any, min_tok: int) -> List[int]:
        out, seen_pid = [], set()
        for i, row in enumerate(test):
            s = (row.get("support") or "").strip()
            pid = self.corpus.gold_support_id.get(("sciq", "test", i))
            if pid is None or len(tokens(s)) < min_tok or pid in seen_pid:
                continue
            seen_pid.add(pid)
            out.append(i)
        return out

    def fold(self, seed_index: int) -> List[Dict[str, Any]]:
        """The gold index, gold passage id, gold stem and gold key of every item in that fold.

        Raises ValueError naming the cap record when a gold passage id is missing, rather than
        handing back an item that points at the wrong passage."""
        out: List[Dict[str, Any]] = []
        for gi in self.folds[seed_index]:
            pid = self.corpus.gold_support_id.get(("sciq", "test", int(gi)))
            if pid is None:
                raise ValueError(f"fold {seed_index} item {gi} lost its gold passage; the corpus "
                                 f"cap dropped a protected id: {self.corpus.cap_record}")
            row = self.test[int(gi)]
            out.append({"gold_idx": int(gi), "gold_pid": int(pid),
                        "gold_stem": row["question"], "gold_key": row["correct_answer"]})
        return out


def obqa_replication_indices(cfg: Any, corpus: CorpusBuilder, limit: Optional[int] = None) -> List[int]:
    """A deterministic fixed PREFIX of eligible OpenBookQA test indices.

    A prefix, not a shuffle: a smaller limit must be a SUBSET of a larger one, so the ids the
    corpus cap protects are exactly the ids the reduced replication will evaluate."""
    if corpus.obqa is None:
        return []
    n = int(cfg.obqa_replication_items if limit is None else limit)
    out: List[int] = []
    for i in range(len(corpus.obqa["test"])):
        if corpus.gold_support_id.get(("obqa", "test", i)) is None:
            continue
        out.append(i)
        if len(out) >= n:
            break
    return out


def obqa_replication_pool(cfg: Any, corpus: CorpusBuilder) -> List[Dict[str, Any]]:
    """The replication entries at the configured item count; an empty list when OpenBookQA is
    unavailable, never a partial pool."""
    if corpus.obqa is None:
        return []
    test = corpus.obqa["test"]
    out: List[Dict[str, Any]] = []
    for i in obqa_replication_indices(cfg, corpus):
        pid = corpus.gold_support_id.get(("obqa", "test", int(i)))
        if pid is None:
            continue
        row = test[int(i)]
        out.append({"gold_idx": int(i), "gold_pid": int(pid),
                    "gold_stem": row.get("question_stem", ""), "gold_key": ""})
    return out


# ============================================================== calibration
@dataclass
class CalibrationSet:
    premises: List[str] = field(default_factory=list)
    stems: List[str] = field(default_factory=list)
    options: List[str] = field(default_factory=list)
    labels: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    overlap: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    decile: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    def __len__(self) -> int:
        return len(self.labels)


class CalibrationSetBuilder:
    """Labelled (passage, stem, option) pairs from SciQ VALIDATION gold items, stratified by
    lexical-overlap decile.

    Validation only: the evaluated folds come from the test split, so no item calibrating the
    judge is ever an item the judge then scores."""

    def __init__(self, cfg: Any, corpus: CorpusBuilder) -> None:
        self.cfg = cfg
        self.corpus = corpus

    def build(self) -> CalibrationSet:
        val = self.corpus.sciq["validation"]
        recs: List[Tuple[str, str, str, int, float]] = []
        for row in val:
            s = (row.get("support") or "").strip()
            if len(tokens(s)) < self.cfg.min_corpus_tokens:
                continue
            opts = [(row["correct_answer"], 1)] + [(row[f"distractor{k}"], 0) for k in (1, 2, 3)]
            for o, lab in opts:
                o = (o or "").strip()
                if o:
                    recs.append((s, row["question"], o, lab, lexical_overlap(o, s)))
        if not recs:
            raise ValueError("No calibration pairs could be built from SciQ validation.")
        overlap = np.array([r[4] for r in recs], dtype=np.float32)
        n_dec = self.cfg.n_overlap_deciles
        order = np.argsort(overlap, kind="stable")
        decile = np.zeros(len(recs), dtype=np.int64)
        decile[order] = (np.arange(len(recs)) * n_dec) // max(1, len(recs))
        decile = np.clip(decile, 0, n_dec - 1)
        labels = np.array([r[3] for r in recs], dtype=np.int64)
        rng = np.random.default_rng(0)
        per = max(1, self.cfg.calibration_pairs // n_dec)
        chosen: List[int] = []
        for d in range(n_dec):
            idx = np.where(decile == d)[0]
            if len(idx) == 0:
                continue
            pos, neg = idx[labels[idx] == 1], idx[labels[idx] == 0]
            n_pos = min(int(round(per * len(pos) / len(idx))), len(pos))
            n_neg = min(per - n_pos, len(neg))
            take = list(rng.choice(pos, n_pos, replace=False)) + list(rng.choice(neg, n_neg, replace=False))
            chosen.extend(int(t) for t in take)
        chosen_set = set(chosen)
        remaining = [i for i in range(len(recs)) if i not in chosen_set]
        shortfall = self.cfg.calibration_pairs - len(chosen)
        if shortfall > 0 and remaining:
            extra = rng.choice(np.array(remaining), min(len(remaining), shortfall), replace=False)
            chosen.extend(int(t) for t in extra)
        chosen = sorted(chosen)[: self.cfg.calibration_pairs]
        cs = CalibrationSet(
            premises=[recs[i][0] for i in chosen], stems=[recs[i][1] for i in chosen],
            options=[recs[i][2] for i in chosen], labels=labels[chosen],
            overlap=overlap[chosen], decile=decile[chosen],
        )
        if cs.labels.sum() == 0 or cs.labels.sum() == len(cs):
            raise ValueError("Calibration set has a single class; SciQ gold fields look corrupted.")
        print(f"[calibration] {len(cs)} pairs, {int(cs.labels.sum())} positives, "
              f"deciles={np.bincount(cs.decile, minlength=n_dec).tolist()}")
        return cs
