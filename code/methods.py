"""
methods.py — the ten arm strategies.

Two generators (grounded / closed-book), the shared scoring pass and the flag
decomposition, four post-hoc item filters, the two mandated control arms, two
overgenerate-and-rerank strategies, and the functionality proxy.

Every model arrives as a constructor argument, so the test suite drives this whole file
with small fakes and no checkpoint. No torch import, no pandas import.

Filter family (each scores an item from its stored [4,3] NLI tensor P[option, (C,E,N)]):
  per_option_entailment_filter        S = mean_o P_E(o)                        (symmetric, all 4)
  llm_judge_faithfulness_filter       S = mean_o P_yes(o)                      (generative judge)
  contradiction_polarity_item_filter  S = P_E(key) * mean_d P_C(d)             (contradiction only)
  not_entail_polarity_item_filter     S = P_E(key) * mean_d (1 - P_E(d))       (neutral mass counts)
The two polarity filters share the key term and differ in EXACTLY ONE method, which is
what makes the ablation an ablation rather than a second arm.

Control arms:
  null_random_drop         a seeded uniform drop at the filters' own retention rate,
                           reading no judge tensor; its effect must be near zero.
  positive_control_planted a known fraction of items has one distractor slot overwritten
                           by that item's own key, rescored by the REAL judge; its rate
                           must exceed the baseline.
"""
import dataclasses
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from experiment_config import (HYPOTHESIS_TEMPLATE, PROMPT_ANSWER_NO_PASSAGE, PROMPT_ANSWER_WITH_PASSAGE,
                               PROMPT_GROUNDED, PROMPT_LLM_JUDGE, PROMPT_OVERGEN, PROMPT_REPAIR, PROMPT_REWRITE,
                               PROMPT_TOPIC_LABEL, PROMPT_TOPIC_ONLY)
from data import (QuizItem, SeedAbort, assert_finite, lexical_overlap, parse_candidate_list, parse_item_json,
                  which_of_the_following)
from analysis import ROW_FIELDS, empty_rows

OVERGEN_REPAIR = "\n\nYour previous reply was not a JSON list of 10 strings. Reply with the list only."
C_IDX, E_IDX, N_IDX = 0, 1, 2   # column order of every stored NLI tensor: contradiction, entailment, neutral
PASSAGE_FREE_TOPIC = "a science course topic"
PROMPT_PROBE = "ZQXPROBE unique passage text"


# ============================================================== ITEM 39: topic labels
def topic_fallback_label(corpus: Any, pid: int) -> str:
    """The passage-prefix label used when the GROUNDED arm's topic label is unusable.

    The grounded arm already has the passage in its prompt, so a prefix label leaks nothing
    there; the closed-book arm must never receive this one."""
    text = " ".join(str(corpus.passages[pid]).split()[:6])[:80].strip()
    return text if text else "an untitled course passage"


def passage_free_fallback_label(corpus: Any, pid: int) -> str:
    """A fixed placeholder carrying NO passage text.

    Feeding the passage prefix into the closed-book prompt would leak the gold passage into
    the one control whose definition is that it has no passage -- the exact contrast H1 rests
    on. The corpus and pid arguments keep the signature interchangeable with the grounded
    fallback and are deliberately not read."""
    del corpus, pid
    return PASSAGE_FREE_TOPIC


def label_topics(llm: Any, corpus: Any, pids: Sequence[int], cfg: Any,
                 fallback_label: Optional[Callable[[Any, int], str]] = None) -> Tuple[List[str], int]:
    """The PAIR (labels, n_fallback) for a caller that needs one label list.

    A thin wrapper over label_topics_with_fallback_mask, so there is exactly ONE parsing and
    counting path and the two forms cannot drift."""
    labels, n_fallback, _ = label_topics_with_fallback_mask(llm, corpus, pids, cfg, fallback_label)
    return labels, n_fallback


def per_option_scores(items: Sequence[QuizItem]) -> np.ndarray:
    """The mean entailment probability over all four options of each item."""
    if not items:
        return np.zeros(0, dtype=np.float32)
    return np.array([float(it.nli_orig[:, E_IDX].mean()) for it in items], dtype=np.float32)


# ITEM 40: two disjoint streams per (seed, item) context
def proxy_sampling_seeds(seed: int, item_id: int) -> Tuple[int, int]:
    """Two sampling stream seeds for one context, injective in BOTH arguments.

    The old scheme (seed*1000 + item_id, and the same + 7) collided: item 7's with-passage
    stream was item 0's no-passage stream, so one item's draw silently answered for another."""
    key = int(seed) * 1_000_003 + int(item_id)
    return 2 * key, 2 * key + 1


# ============================================================== generators
class GroundedQuizGenerator:
    """Condition unfiltered_grounded_pool: the retrieved passage is in the generation prompt
    and no filtering is applied. This is the reference pool for every other arm."""

    name = "unfiltered_grounded_pool"

    def __init__(self, cfg: Any, llm: Any, corpus: Any) -> None:
        self.cfg = cfg
        self.llm = llm
        self.corpus = corpus
        self.attempts = 0
        self.parse_failures = 0
        self.n_repaired = 0
        self.last_parse_failure_rate = 0.0

    def build_prompt(self, topic: str, passage: str) -> str:
        return PROMPT_GROUNDED.format(passage=passage, topic=topic)

    def prompt_uses_passage(self) -> bool:
        """True when a sentinel placed in the PASSAGE slot reaches the prompt.

        Probes the template only; the topic argument is a fixed constant so a topic that
        happened to contain the sentinel could not change the answer."""
        return PROMPT_PROBE in self.build_prompt("topic", PROMPT_PROBE)

    def _repair(self, prompts: Sequence[str], raw: Sequence[str], parsed: List[Optional[dict]],
                seed: int) -> List[Optional[dict]]:
        """One repair-prompt retry for every item whose first reply failed to parse."""
        retry = [i for i, d in enumerate(parsed) if d is None]
        if not retry:
            return parsed
        rep = [prompts[i] + PROMPT_REPAIR + "\nPrevious reply: " + (raw[i] or "")[:400] for i in retry]
        raw2 = self.llm.generate_batch(rep, self.cfg.gen_max_new_tokens)
        for i, r in zip(retry, raw2):
            parsed[i] = parse_item_json(r)
            if parsed[i] is None:
                print(f"[{self.name}] parse failure after repair seed={seed} item={i}: {(r or '')[:200]!r}")
            else:
                self.n_repaired += 1
        return parsed

    def generate_pool(self, seed: int, entries: Sequence[Dict[str, Any]], topics: Sequence[str],
                      pids: Sequence[int]) -> List[QuizItem]:
        n = len(entries)
        passages = [self.corpus.passage_text(p) for p in pids]
        prompts = [self.build_prompt(t, p) for t, p in zip(topics, passages)]
        raw = self.llm.generate_batch(prompts, self.cfg.gen_max_new_tokens)
        parsed: List[Optional[dict]] = [parse_item_json(r) for r in raw]
        parsed = self._repair(prompts, raw, parsed, seed)
        items: List[QuizItem] = []
        failures = 0
        for j, (e, t, pid, d) in enumerate(zip(entries, topics, pids, parsed)):
            if d is None:
                failures += 1
                continue
            items.append(QuizItem(
                item_id=j, seed=seed, gold_idx=int(e["gold_idx"]), topic=t, passage=passages[j],
                passage_id=int(pid), retrieval_covered=bool(int(pid) == int(e["gold_pid"])),
                stem=d["stem"], options=[d["key"]] + list(d["distractors"]),
                enumeration_dense=self.corpus.is_enumeration_dense(pid), arm=self.name,
            ))
        self.attempts += n
        self.parse_failures += failures
        self.last_parse_failure_rate = failures / n if n else 0.0
        print(f"[{self.name}] seed={seed} items={len(items)}/{n} "
              f"parse_failure_rate={self.last_parse_failure_rate:.3f} repaired={self.n_repaired}")
        if self.last_parse_failure_rate > self.cfg.parse_failure_abort_rate:
            raise SeedAbort(f"{self.name}: parse failure rate {self.last_parse_failure_rate:.2f} > "
                            f"{self.cfg.parse_failure_abort_rate}", stage=f"{self.name}:generation",
                            rate=self.last_parse_failure_rate)
        return items


class TopicOnlyQuizGenerator(GroundedQuizGenerator):
    """Condition ungrounded_topic_only_control: identical pipeline, parser, judge, rewrite and
    decomposition, but the generation prompt carries ONLY the topic label.

    Items are still scored against the passage retrieved for that label, so the comparison with
    the grounded arm holds the evidence fixed and varies only the generation context."""

    name = "ungrounded_topic_only_control"

    def __init__(self, cfg: Any, llm: Any, corpus: Any) -> None:
        super().__init__(cfg, llm, corpus)
        self.key_in_passage_rate: Optional[float] = None

    def build_prompt(self, topic: str, passage: str) -> str:
        # KEY DIFFERENCE: the passage argument is deliberately unused (closed-book generation).
        del passage
        return PROMPT_TOPIC_ONLY.format(topic=topic)

    def generate_pool(self, seed: int, entries: Sequence[Dict[str, Any]], topics: Sequence[str],
                      pids: Sequence[int]) -> List[QuizItem]:
        if self.prompt_uses_passage():
            raise RuntimeError("TopicOnlyQuizGenerator prompt leaked the passage; the control is invalid")
        items = super().generate_pool(seed, entries, topics, pids)
        leak = sum(1 for it in items if lexical_overlap(it.options[0], it.passage) >= 0.99)
        self.key_in_passage_rate = (leak / len(items)) if items else None
        rate_txt = "NA" if self.key_in_passage_rate is None else f"{self.key_in_passage_rate:.3f}"
        print(f"[{self.name}] seed={seed} key_lexically_in_retrieved_passage_rate={rate_txt}")
        return items


# ============================================================== the shared scoring pass
class ItemScorer:
    """Attaches the rewritten stem, both NLI tensors, the LLM-judge probabilities AND their
    validity masks, key similarity and lexical overlap to items. Every arm reads the tensors this
    pass wrote, so it runs once per pool and no arm rescores the shared objects."""

    def __init__(self, cfg: Any, llm: Any, nli: Any, embedder: Any) -> None:
        self.cfg = cfg
        self.llm = llm
        self.nli = nli
        self.embedder = embedder
        self.rewrite_noops = 0

    def rewrite_stems(self, items: Sequence[QuizItem]) -> None:
        """Sets stem_rewritten on every item; increments rewrite_noops once per unchanged stem.

        A no-op rewrite makes distractor-caused == corpus-supported for that item by
        construction, so the count is a health metric of the decomposition itself."""
        if not items:
            return
        prompts = [PROMPT_REWRITE.format(passage=it.passage, stem=it.stem, key=it.options[0]) for it in items]
        outs = self.llm.generate_batch(prompts, self.cfg.rewrite_max_new_tokens)
        for it, o in zip(items, outs):
            new = ""
            for line in (o or "").splitlines():
                line = line.strip().strip('"').strip()
                if line.lower().startswith("question:"):
                    line = line[9:].strip()
                if line:
                    new = line
                    break
            if len(new) < 5 or new.lower() == it.stem.lower():
                self.rewrite_noops += 1
                new = it.stem
            it.stem_rewritten = new

    def score_nli(self, items: Sequence[QuizItem]) -> None:
        if not items:
            return
        for stage in ("orig", "rewrite"):
            prem: List[str] = []
            hyp: List[str] = []
            for it in items:
                stem = it.stem if stage == "orig" else (it.stem_rewritten or it.stem)
                for o in it.options:
                    prem.append(it.passage)
                    hyp.append(HYPOTHESIS_TEMPLATE.format(stem=stem.rstrip("?"), option=o))
            P = np.asarray(self.nli.score_pairs(prem, hyp)).reshape(len(items), 4, 3)
            assert_finite(f"nli_{stage}", P)
            for i, it in enumerate(items):
                if stage == "orig":
                    it.nli_orig = P[i].astype(np.float32)
                else:
                    it.nli_rewrite = P[i].astype(np.float32)

    def score_llm_judge(self, items: Sequence[QuizItem], stage: str = "orig") -> None:
        """Store the generative judge's probabilities AND the validity mask of the same pass.

        p_yes_batch drops the mask, so an imputed NAN_P_YES_SUBSTITUTE = 0.0 was stored
        indistinguishably from a measured 0.0 and then read as a measured non-support: on a key
        slot that is a counted key hallucination inside key_hallucination_rate, which feeds
        h2_supported and h3_supported. The substitute is below any threshold in (0, 1] so it can
        never manufacture a SUPPORTED flag, but only the mask keeps it out of the denominators,
        and FlagDecomposer.valid_mask is what reads it."""
        if not items:
            return
        prompts: List[str] = []
        for it in items:
            stem = it.stem if stage == "orig" else (it.stem_rewritten or it.stem)
            for o in it.options:
                prompts.append(PROMPT_LLM_JUDGE.format(passage=it.passage, stem=stem, option=o))
        res = self.llm.p_yes_batch_with_validity(prompts)
        p = probs_of(res).reshape(len(items), 4)
        valid = validity_mask_of(res, len(items), 4)
        assert_finite(f"llm_pyes_{stage}", p)
        for i, it in enumerate(items):
            if stage == "orig":
                it.llm_pyes = p[i].astype(np.float32)
                it.llm_pyes_valid = valid[i].copy()
            else:
                it.llm_pyes_rewrite = p[i].astype(np.float32)
                it.llm_pyes_rewrite_valid = valid[i].copy()

    def score_similarity_and_overlap(self, items: Sequence[QuizItem]) -> None:
        if not items:
            return
        flat = [o for it in items for o in it.options]
        E = np.asarray(self.embedder.encode(flat)).reshape(len(items), 4, -1)
        for i, it in enumerate(items):
            it.key_sim = (E[i] @ E[i, 0]).astype(np.float32)
            it.overlap = np.array([lexical_overlap(o, it.passage) for o in it.options], dtype=np.float32)


class FlagDecomposer:
    """Corpus-supported / distractor-caused / stem-caused flags from the stored tensors under one
    named evaluator. dc and sc PARTITION cs exactly, by construction.

    Every flag is intersected with the judge's VALIDITY mask, so an unmeasured slot enters no
    numerator and no denominator. An imputed value counted as a measured negative is a fabricated
    measurement, and it reached two hypothesis verdicts through key_hallucination_rate."""

    def __init__(self, cfg: Any, evaluator: str) -> None:
        self.cfg = cfg
        self.evaluator = evaluator if evaluator in ("nli", "llm") else "nli"
        self.thr = cfg.entail_threshold

    def support_prob(self, items: Sequence[QuizItem], stage: str) -> np.ndarray:
        if not items:
            return np.zeros((0, 4), dtype=np.float32)
        if self.evaluator == "nli":
            f = "nli_orig" if stage == "orig" else "nli_rewrite"
            return np.stack([np.asarray(getattr(it, f))[:, E_IDX] for it in items])
        f = "llm_pyes" if stage == "orig" else "llm_pyes_rewrite"
        return np.stack([np.asarray(getattr(it, f)) for it in items])

    def valid_mask(self, items: Sequence[QuizItem], stage: str) -> np.ndarray:
        """The [n, 4] mask of slots this evaluator actually MEASURED at that stage.

        All True under 'nli': NLIJudge.score_pairs calls assert_finite and RAISES on a non-finite
        row rather than substituting one, so the NLI evaluator never has anything to exclude and
        its denominators stay exactly 3n and n.

        Under 'llm' it is the mask ItemScorer.score_llm_judge stored beside the probabilities.
        None means the item was not scored by the generative judge at all, which every reader
        treats as all-valid -- an item with no LLM tensor is not an item with an imputed one."""
        n = len(items)
        if n == 0:
            return np.zeros((0, 4), dtype=bool)
        if self.evaluator == "nli":
            return np.ones((n, 4), dtype=bool)
        field = "llm_pyes_valid" if stage == "orig" else "llm_pyes_rewrite_valid"
        rows: List[np.ndarray] = []
        for it in items:
            m = getattr(it, field, None)
            rows.append(np.ones(4, dtype=bool) if m is None
                        else np.asarray(m, dtype=bool).reshape(4))
        return np.stack(rows)

    def both_stages_valid(self, items: Sequence[QuizItem]) -> np.ndarray:
        """The [n, 4] mask of slots measured under BOTH stems.

        The decomposition reads the original stem AND its specificity rewrite for every slot, so
        a slot imputed at either stage was never measured as a whole and cannot be classified."""
        if not items:
            return np.zeros((0, 4), dtype=bool)
        return self.valid_mask(items, "orig") & self.valid_mask(items, "rewrite")

    def flags(self, items: Sequence[QuizItem]) -> Dict[str, np.ndarray]:
        """Every flag masked by both_stages_valid, plus the two masks themselves.

        key_halluc was `~So[:, 0]` with no mask: an imputed 0.0 key row is not >= the threshold,
        so it read as a measured hallucination for an item the judge never scored."""
        if not items:
            return {"cs": np.zeros((0, 3), dtype=bool), "dc": np.zeros((0, 3), dtype=bool),
                    "sc": np.zeros((0, 3), dtype=bool), "key_halluc": np.zeros(0, dtype=bool),
                    "multi_key": np.zeros(0, dtype=bool),
                    "dvalid": np.zeros((0, 3), dtype=bool), "kvalid": np.zeros(0, dtype=bool)}
        So = self.support_prob(items, "orig") >= self.thr
        Sr = self.support_prob(items, "rewrite") >= self.thr
        V = self.both_stages_valid(items)
        dvalid = V[:, 1:]
        kvalid = V[:, 0]
        cs = So[:, 1:] & dvalid
        return {"cs": cs, "dc": cs & Sr[:, 1:], "sc": cs & ~Sr[:, 1:],
                "key_halluc": (~So[:, 0]) & kvalid, "multi_key": cs.any(axis=1),
                "dvalid": dvalid, "kvalid": kvalid}

    # ITEM 18: exact denominators over the MEASURED slots, no max(1, .), None for every undefined cell
    def rates(self, items: Sequence[QuizItem]) -> Dict[str, Any]:
        """All nine RATE_KEYS over the slots this evaluator measured, plus the exclusion counts.

        The denominators are the MEASURED distractor slots and the MEASURED key slots, not 3n and
        n: a slot the judge imputed is an undefined cell, and counting it as a measured negative
        deflates every rate by a silent amount. When either denominator is zero the corresponding
        rates are None -- decide_gates turns a None input into a None verdict, so an unmeasured
        pool produces no claim rather than a fabricated zero.

        max(1, cs.sum()) used to fabricate stem_caused_share == 0.0 on a pool with no supported
        distractor, and 1 - 0.0 then read as "every supported slot is distractor-caused" in the
        h1_position verdict -- a verdict about a quantity that was never measured."""
        n = len(items)
        f = self.flags(items)
        out: Dict[str, Any] = {
            "n_items": n,
            "evaluator": self.evaluator,
            "distractor_caused_csd_rate": None,
            "stem_caused_csd_rate": None,
            "corpus_supported_rate": None,
            "stem_caused_share": None,
            "key_hallucination_rate": None,
            "per_item_multi_key_rate": None,
            "mean_distractor_p_support_orig": None,
            "soft_distractor_caused_score": None,
            "mean_key_p_support_orig": None,
            "which_of_the_following_stem_fraction": None,
            # published on EVERY record, so a reader can always tell an unmeasured slot from an
            # unsupported one without going back to the numeric_health counters
            "n_distractor_slots_measured": 0,
            "n_distractor_slots_excluded": 0,
            "n_key_slots_excluded": 0,
        }
        if n == 0:
            return out
        n_measured = int(f["dvalid"].sum())
        n_keys = int(f["kvalid"].sum())
        out["n_distractor_slots_measured"] = n_measured
        out["n_distractor_slots_excluded"] = 3 * n - n_measured
        out["n_key_slots_excluded"] = n - n_keys
        po = self.support_prob(items, "orig")
        pr = self.support_prob(items, "rewrite")
        dv = f["dvalid"]
        if n_measured > 0:
            n_cs = int(f["cs"].sum())
            out["distractor_caused_csd_rate"] = float(f["dc"].sum()) / n_measured
            out["stem_caused_csd_rate"] = float(f["sc"].sum()) / n_measured
            out["corpus_supported_rate"] = float(n_cs) / n_measured
            # the share's denominator is the SUPPORTED slots; zero of them means undefined, not 0.0
            out["stem_caused_share"] = (float(f["sc"].sum()) / n_cs) if n_cs > 0 else None
            out["mean_distractor_p_support_orig"] = float((po[:, 1:] * dv).sum()) / n_measured
            out["soft_distractor_caused_score"] = (
                float((np.minimum(po[:, 1:], pr[:, 1:]) * dv).sum()) / n_measured)
        if n_keys > 0:
            out["key_hallucination_rate"] = float(f["key_halluc"].sum()) / n_keys
            out["mean_key_p_support_orig"] = float((po[:, 0] * f["kvalid"]).sum()) / n_keys
        # an item with no measured distractor slot cannot be judged for a second supported key
        n_items_measured = int(dv.any(axis=1).sum())
        if n_items_measured > 0:
            out["per_item_multi_key_rate"] = float(f["multi_key"].sum()) / n_items_measured
        # the stem wording is a property of the item, not of the judge, so it is over every item
        out["which_of_the_following_stem_fraction"] = (
            float(sum(1 for it in items if which_of_the_following(it.stem))) / n)
        return out

    # ITEM 19: plain-list rows keyed by (seed, item_id) -- no pandas
    def flag_rows(self, items: Sequence[QuizItem], condition: str) -> Dict[str, Any]:
        """One row per MEASURED distractor slot, each carrying its (seed, item_id) key.

        An unmeasured slot emits no row at all: the item bootstrap resamples these rows, so a row
        carrying an imputed flag would enter the interval as though it were an observation. Every
        ROW_FIELDS column advances once per row, and assert_rows_rectangular proves it before the
        table leaves this method."""
        rows = empty_rows(condition)
        f = self.flags(items)
        for i, it in enumerate(items):
            key = (int(it.seed), int(it.item_id))
            enum = "high" if it.enumeration_dense else "low"
            cov = "covered" if it.retrieval_covered else "not_covered"
            for d in range(3):
                if not bool(f["dvalid"][i, d]):
                    continue                                  # unmeasured: no row, never a False
                rows["keys"].append(key)
                rows["dc"].append(bool(f["dc"][i, d]))
                rows["sc"].append(bool(f["sc"][i, d]))
                rows["cs"].append(bool(f["cs"][i, d]))
                rows["enum"].append(enum)
                rows["cov"].append(cov)
                rows["key_sim"].append(float(it.key_sim[d + 1]) if it.key_sim is not None else None)
                rows["overlap"].append(float(it.overlap[d + 1]) if it.overlap is not None else None)
        assert_rows_rectangular(rows, condition)
        return rows


# ============================================================== ITEM 40: functionality proxy
class FunctionalityProxy:
    """A distractor is 'functional' when the answerer picks it MORE often without the passage
    than with it (it draws a reader who has not read the evidence).

    Not examinee data: this is a model-based proxy and the results record it as such."""

    def __init__(self, cfg: Any, llm: Any) -> None:
        self.cfg = cfg
        self.llm = llm
        self.n = cfg.functionality_samples_per_context
        self.calls = 0
        # counts for the LAST call, so an arm reads the support behind its own rate
        self.n_contexts_measured = 0
        self.n_contexts_excluded = 0

    def set_samples(self, n: int) -> None:
        self.n = int(n)

    def exclusion_record(self) -> Dict[str, Any]:
        """The measured and excluded context counts of the most recent functional_rate call."""
        return {"functional_contexts_measured": int(self.n_contexts_measured),
                "functional_contexts_excluded": int(self.n_contexts_excluded)}

    def functional_rate(self, items: Sequence[QuizItem], seed: int) -> Optional[float]:
        """The functional-distractor rate in 0-1, or None when no context could be measured.

        A context is one item's WITH-passage / NO-passage pair, and the whole statistic is the
        comparison between those two rows. letter_probs_batch substitutes a uniform
        NAN_LETTER_PROB_SUBSTITUTE = 0.25 row for a non-finite forward pass, so comparing an
        imputed row against a measured one manufactures a functional flag out of noise -- and
        that flag feeds functional_distractor_rate and then h3_supported. Either row imputed
        excludes the whole context; the counts say how many.

        Writes NOTHING onto the items: every arm calls this on a subset of the same shared
        objects, so a per-item flag would be overwritten by whichever arm ran last and the
        earlier arms' reports would silently describe the later arm's selection."""
        self.n_contexts_measured = 0
        self.n_contexts_excluded = 0
        if not items:
            return None
        prompts: List[str] = []
        perms: List[np.ndarray] = []
        for it in items:
            rng = np.random.default_rng(proxy_sampling_seeds(seed, it.item_id)[0])
            perm = rng.permutation(4)                    # display position -> option index
            opts = [it.options[p] for p in perm]
            kw = {"stem": it.stem, "a": opts[0], "b": opts[1], "c": opts[2], "d": opts[3]}
            prompts.append(PROMPT_ANSWER_WITH_PASSAGE.format(passage=it.passage, **kw))
            prompts.append(PROMPT_ANSWER_NO_PASSAGE.format(**kw))
            perms.append(perm)
        res = self.llm.letter_probs_batch(prompts, self.cfg.functionality_temperature,
                                          self.cfg.functionality_top_p)
        probs = probs_of(res)
        valid = validity_mask_of(res, len(prompts), 1)[:, 0]
        self.calls += len(prompts)
        flags: List[np.ndarray] = []
        for i, it in enumerate(items):
            if not (bool(valid[2 * i]) and bool(valid[2 * i + 1])):
                self.n_contexts_excluded += 1              # one imputed row voids the comparison
                continue
            s_with, s_no = proxy_sampling_seeds(seed, it.item_id)
            perm = perms[i]
            with_p = self.llm.sample_letters(probs[2 * i], self.n, s_with)
            no_p = self.llm.sample_letters(probs[2 * i + 1], self.n, s_no)
            cw = np.bincount(perm[with_p], minlength=4)  # counts back in option order
            cn = np.bincount(perm[no_p], minlength=4)
            flags.append(cn[1:] > cw[1:])
            self.n_contexts_measured += 1
        if not flags:
            return None                                    # undefined, never a measured 0.0
        return float(np.concatenate(flags).mean())


# ============================================================== post-hoc item filters
class BaseItemFilter:
    """Scores each item from the stored tensors, retains the top retention_fraction, sweeps tau."""

    name = "base_filter"
    reads_judge_tensor = True

    def __init__(self, cfg: Any, decomposer: "FlagDecomposer",
                 proxy: Optional["FunctionalityProxy"] = None) -> None:
        self.cfg = cfg
        self.decomposer = decomposer
        self.proxy = proxy
        self._retained: List[QuizItem] = []

    def evaluated_items(self) -> List[QuizItem]:
        """A copy of the retained list, so main never reaches into a private attribute."""
        return list(self._retained)

    def score_item(self, it: QuizItem) -> float:
        raise NotImplementedError

    def score_pool(self, items: Sequence[QuizItem], seed: int = 0) -> np.ndarray:
        del seed                                          # only the null arm needs a seed
        if not items:
            return np.zeros(0, dtype=np.float32)
        return np.array([self.score_item(it) for it in items], dtype=np.float32)

    def retain(self, items: Sequence[QuizItem], scores: np.ndarray) -> List[QuizItem]:
        if not items:
            return []
        k = max(1, int(round(self.cfg.retention_fraction * len(items))))
        order = np.argsort(-np.asarray(scores), kind="stable")
        return [items[i] for i in order[:k]]

    @staticmethod
    def sweep_floor(n_pool: int) -> int:
        """max(1, ceil(0.1 * n_pool)): the minimum retained items a sweep cell needs.

        A cell holding one item reports a rate that is 0 or 1/3 and nothing between, which the
        pooled H2 enrichment then treats as an estimate."""
        return int(max(1, math.ceil(0.1 * int(n_pool))))

    def threshold_sweep(self, items: Sequence[QuizItem], scores: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """One cell per configured tau, carrying admitted, n_items, min_items and either full
        rates or None for every rate key; a cell below the floor never reports a rate."""
        sweep: Dict[str, Dict[str, Any]] = {}
        floor = self.sweep_floor(len(items))
        for tau in self.cfg.tau_sweep:
            kept = [it for it, s in zip(items, np.asarray(scores)) if float(s) >= float(tau)]
            admitted = len(kept) >= floor
            cell = self.decomposer.rates(kept if admitted else [])
            cell["n_items"] = len(kept)
            cell["admitted"] = bool(admitted)
            cell["min_items"] = floor
            cell["item_yield"] = (len(kept) / len(items)) if items else None
            sweep[str(tau)] = cell
        return sweep

    def report(self, items: Sequence[QuizItem], seed: int) -> Dict[str, Any]:
        scores = self.score_pool(items, seed)
        assert_finite(f"{self.name}_scores", scores)
        kept = self.retain(items, scores)
        self._retained = kept
        r = self.decomposer.rates(kept)
        r["item_yield"] = (len(kept) / len(items)) if items else None
        sweep = self.threshold_sweep(items, scores)
        r["threshold_sweep"] = sweep
        # iterate the sweep ACTUALLY produced; the null arm's sweep is empty by design and this
        # loop must not assume a cell exists for every configured tau
        admitted = [(float(t), c["distractor_caused_csd_rate"]) for t, c in sweep.items()
                    if c.get("admitted") and c.get("distractor_caused_csd_rate") is not None]
        rho: Optional[float] = None
        if len(admitted) >= 3 and float(np.std([v for _, v in admitted])) > 0:
            rho = float(stats.spearmanr([t for t, _ in admitted], [v for _, v in admitted]).statistic)
        r["spearman_rho_threshold_vs_rate"] = rho
        r["spearman_n_taus"] = len(admitted)
        po = per_option_scores(items)
        ktau: Optional[float] = None
        if len(items) >= 3 and float(np.std(scores)) > 0 and float(np.std(po)) > 0:
            ktau = float(stats.kendalltau(scores, po).statistic)
        r["kendall_tau_vs_per_option"] = ktau
        r["retained_item_ids"] = [int(it.item_id) for it in kept]
        r["reads_judge_tensor"] = bool(self.reads_judge_tensor)
        r["retention_fraction"] = float(self.cfg.retention_fraction)
        # the rate AND the context counts behind it, for THIS arm's retained set
        r.update(functional_record(self.proxy, kept, seed))
        return r


# ITEM 20: four distinct scorers, four distinct filter_scorer strings
class PerOptionEntailmentFilter(BaseItemFilter):
    """Baseline (RAGAS / MiniCheck / AlignScore style): SYMMETRIC faithfulness.

    Every option -- the key and all three distractors -- is rewarded for being entailed by the
    passage. There is no key term, no polarity flip and no distractor-specific treatment, so a
    distractor the corpus supports RAISES the score. That is the mechanism H2 tests."""

    name = "per_option_entailment_filter"

    def score_item(self, it: QuizItem) -> float:
        return float(np.asarray(it.nli_orig)[:, E_IDX].mean())

    def report(self, items: Sequence[QuizItem], seed: int) -> Dict[str, Any]:
        r = super().report(items, seed)
        r["kendall_tau_vs_per_option"] = 1.0              # identity ranking by construction
        kept = self._retained
        r["retained_mean_p_entail_distractors"] = (
            float(np.mean([np.asarray(it.nli_orig)[1:, E_IDX].mean() for it in kept])) if kept else None)
        r["filter_scorer"] = "nli_mean_entail_all_options"
        return r


class LLMJudgeFaithfulnessFilter(BaseItemFilter):
    """Baseline (G-Eval / RAGAS-LLM style): item score = mean P(Yes) of the generative judge.

    Its retained set is always flagged by the NLI decomposer, so its reference is the
    NLI-flagged pool -- one judge on both sides of that comparison."""

    name = "llm_judge_faithfulness_filter"

    def score_item(self, it: QuizItem) -> float:
        if it.llm_pyes is None:
            raise ValueError("llm_judge_faithfulness_filter needs ItemScorer.score_llm_judge first")
        return float(np.asarray(it.llm_pyes).mean())

    def report(self, items: Sequence[QuizItem], seed: int) -> Dict[str, Any]:
        r = super().report(items, seed)
        llm_scores = self.score_pool(items, seed)
        nli_scores = per_option_scores(items)
        ktau: Optional[float] = None
        if len(items) >= 3 and float(np.std(llm_scores)) > 0 and float(np.std(nli_scores)) > 0:
            ktau = float(stats.kendalltau(llm_scores, nli_scores).statistic)
        r["kendall_tau_llm_vs_nli_ranking"] = ktau
        r["evaluator_forced"] = "nli"
        r["filter_scorer"] = "llm_p_yes"
        return r


class ContradictionPolarityItemFilter(BaseItemFilter):
    """Proposed: OPPOSITE-polarity scoring, S = P_entail(key) * mean_d P_contradict(d).

    The key is rewarded for support; each distractor is rewarded for being CONTRADICTED by the
    passage -- the contradiction column only, so neutral mass earns nothing. The two factors are
    separate methods so the ablation can replace exactly one of them."""

    name = "contradiction_polarity_item_filter"
    polarity_term_name = "p_contradict"

    def key_term(self, it: QuizItem) -> float:
        return float(np.asarray(it.nli_orig)[0, E_IDX])

    def distractor_polarity_term(self, it: QuizItem) -> float:
        return float(np.asarray(it.nli_orig)[1:, C_IDX].mean())

    def score_item(self, it: QuizItem) -> float:
        return self.key_term(it) * self.distractor_polarity_term(it)

    def report(self, items: Sequence[QuizItem], seed: int) -> Dict[str, Any]:
        r = super().report(items, seed)
        kept = self._retained
        r["retained_mean_p_contradict_distractors"] = (
            float(np.mean([np.asarray(it.nli_orig)[1:, C_IDX].mean() for it in kept])) if kept else None)
        r["retained_mean_p_neutral_distractors"] = (
            float(np.mean([np.asarray(it.nli_orig)[1:, N_IDX].mean() for it in kept])) if kept else None)
        r["retained_mean_polarity_term"] = (
            float(np.mean([self.distractor_polarity_term(it) for it in kept])) if kept else None)
        r["filter_scorer"] = f"key_entail_x_distractor_{self.polarity_term_name}"
        return r


class NotEntailPolarityItemFilter(ContradictionPolarityItemFilter):
    """Ablation of the contradiction requirement: the distractor polarity term is WEAKENED from
    P_contradict(d) to 1 - P_entail(d) = P_contradict(d) + P_neutral(d).

    ONLY distractor_polarity_term is overridden; the key term, the product, retention and the
    sweep are inherited. Since 1 - P_E >= P_C, this term is never below the contradiction term
    for the same item and is strictly above it wherever the passage leaves neutral mass, so the
    arm isolates whether the H3 effect comes from demanding contradiction specifically or just
    from penalising entailment."""

    name = "not_entail_polarity_item_filter"
    polarity_term_name = "not_entail"

    def distractor_polarity_term(self, it: QuizItem) -> float:
        return float((1.0 - np.asarray(it.nli_orig)[1:, E_IDX]).mean())

    def neutral_share_of_polarity_term(self, it: QuizItem) -> float:
        """The share of this item's polarity credit coming from neutral rather than
        contradiction mass; 0.0 when the term is zero, never NaN."""
        total = self.distractor_polarity_term(it)
        if total <= 1e-9:
            return 0.0
        return float(np.asarray(it.nli_orig)[1:, N_IDX].mean()) / total

    def report(self, items: Sequence[QuizItem], seed: int) -> Dict[str, Any]:
        r = super().report(items, seed)
        kept = self._retained
        r["retained_neutral_share_of_polarity_term"] = (
            float(np.mean([self.neutral_share_of_polarity_term(it) for it in kept])) if kept else None)
        r["filter_scorer"] = "key_entail_x_distractor_not_entail"
        return r


# ============================================================== ITEM 21: the null control arm
class NullRandomDropFilter(BaseItemFilter):
    """Mandated null control: the treatment replaced by a SEEDED UNIFORM DROP at the filters'
    own retention rate, reading no judge tensor.

    Its effect against the pool must be near zero. A large null effect means the metric is
    circular (the flags follow whatever the arm did) or the arms leak into one another."""

    name = "null_random_drop"
    reads_judge_tensor = False

    def score_item(self, it: QuizItem) -> float:
        raise NotImplementedError("the null arm scores the whole pool with one seeded draw, "
                                  "not item by item")

    def score_pool(self, items: Sequence[QuizItem], seed: int = 0) -> np.ndarray:
        """One seeded uniform draw per item; reads NO judge tensor, so it cannot inherit any of
        the treatment's selection behaviour."""
        if not items:
            return np.zeros(0, dtype=np.float32)
        rng = np.random.default_rng(int(seed) * 7919 + 104729)
        return rng.random(len(items)).astype(np.float32)

    def retain(self, items: Sequence[QuizItem], scores: np.ndarray) -> List[QuizItem]:
        if not items:
            return []
        k = max(1, int(round(self.cfg.null_retention_fraction * len(items))))
        order = np.argsort(-np.asarray(scores), kind="stable")
        return [items[i] for i in order[:k]]

    def threshold_sweep(self, items: Sequence[QuizItem], scores: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Empty BY DESIGN: a uniform random score has no meaningful threshold, and publishing
        cells from it would invite a reader to correlate tau with a random ranking. That costs
        this arm its rho, never its record -- all nine rate keys are still reported."""
        del items, scores
        return {}

    def report(self, items: Sequence[QuizItem], seed: int) -> Dict[str, Any]:
        r = super().report(items, seed)
        r["filter_scorer"] = "seeded_uniform_random"
        r["reads_judge_tensor"] = False
        r["retention_fraction"] = float(self.cfg.null_retention_fraction)
        r["spearman_rho_threshold_vs_rate"] = None        # no sweep exists to correlate
        r["kendall_tau_vs_per_option"] = None
        r["threshold_sweep_empty_by_design"] = True
        return r


# ============================================================== ITEMS 22, 23: the positive control
class PositiveControlPlanted:
    """Mandated positive control: a known fraction of items has one distractor slot overwritten
    by that item's OWN KEY, and the copies are rescored by the REAL judge.

    A key the passage supports, offered as a distractor to the same stem, is a corpus-supported
    distractor by definition, so the primary metric must detect it. The measured rise is a
    property of the instrument, not an assumption, because the same judge scores it."""

    name = "positive_control_planted"
    reads_judge_tensor = True

    def __init__(self, cfg: Any, llm: Any, nli: Any, decomposer: "FlagDecomposer",
                 proxy: Optional["FunctionalityProxy"] = None) -> None:
        self.cfg = cfg
        self.llm = llm
        self.nli = nli
        self.decomposer = decomposer
        self.proxy = proxy
        self._planted: List[QuizItem] = []
        self._planted_ids: List[int] = []

    def evaluated_items(self) -> List[QuizItem]:
        return list(self._planted)

    def choose_ids(self, items: Sequence[QuizItem], seed: int) -> List[int]:
        """The sorted item ids chosen for planting, reproducible for a given seed."""
        if not items:
            return []
        k = int(round(float(self.cfg.planted_fraction) * len(items)))
        k = max(0, min(k, len(items)))
        if k == 0:
            return []
        rng = np.random.default_rng(int(seed) * 31337 + 611953)
        ids = np.array([int(it.item_id) for it in items], dtype=np.int64)
        return sorted(int(x) for x in rng.choice(ids, size=k, replace=False))

    def plant(self, items: Sequence[QuizItem], planted_ids: Sequence[int]) -> List[QuizItem]:
        """Copies of every item, with the chosen ones carrying the key in the planted slot.

        dataclasses.replace alone would SHARE the numpy arrays with the original, so writing a
        rescored tensor into a copy would corrupt the pool every later arm reads; every array is
        copied here, the two judge VALIDITY masks included -- a shared mask is exactly as
        dangerous as a shared probability, because rescore refreshes it for the planted copies.
        The input pool's options, tensors and markers are untouched."""
        chosen = {int(i) for i in planted_ids}
        slot = int(self.cfg.planted_slot)
        out: List[QuizItem] = []
        for it in items:
            copy = dataclasses.replace(
                it,
                options=list(it.options),
                arm=self.name,
                nli_orig=None if it.nli_orig is None else np.array(it.nli_orig, copy=True),
                nli_rewrite=None if it.nli_rewrite is None else np.array(it.nli_rewrite, copy=True),
                llm_pyes=None if it.llm_pyes is None else np.array(it.llm_pyes, copy=True),
                llm_pyes_rewrite=(None if it.llm_pyes_rewrite is None
                                  else np.array(it.llm_pyes_rewrite, copy=True)),
                llm_pyes_valid=(None if it.llm_pyes_valid is None
                                else np.array(it.llm_pyes_valid, copy=True)),
                llm_pyes_rewrite_valid=(None if it.llm_pyes_rewrite_valid is None
                                        else np.array(it.llm_pyes_rewrite_valid, copy=True)),
                key_sim=None if it.key_sim is None else np.array(it.key_sim, copy=True),
                overlap=None if it.overlap is None else np.array(it.overlap, copy=True),
                candidates=None, cand_nli=None, cand_sim=None, cand_emb=None, functional=None,
                planted=False,
            )
            if int(it.item_id) in chosen and 1 <= slot <= 3:
                copy.options[slot] = it.options[0]         # the item's OWN key, as a distractor
                copy.planted = True
            out.append(copy)
        return out

    def rescore(self, items: Sequence[QuizItem]) -> None:
        """Rescores only the planted COPIES with the real judge, under both stems.

        Under the llm evaluator the validity mask is refreshed with the probabilities it
        describes: the planted slot holds new option text, so the mask inherited from the
        pre-plant judge call describes a string this copy no longer contains."""
        targets = [it for it in items if it.planted]
        if not targets:
            return
        for stage in ("orig", "rewrite"):
            prem: List[str] = []
            hyp: List[str] = []
            for it in targets:
                stem = it.stem if stage == "orig" else (it.stem_rewritten or it.stem)
                for o in it.options:
                    prem.append(it.passage)
                    hyp.append(HYPOTHESIS_TEMPLATE.format(stem=stem.rstrip("?"), option=o))
            P = np.asarray(self.nli.score_pairs(prem, hyp)).reshape(len(targets), 4, 3)
            assert_finite(f"planted_nli_{stage}", P)
            for i, it in enumerate(targets):
                if stage == "orig":
                    it.nli_orig = P[i].astype(np.float32)
                else:
                    it.nli_rewrite = P[i].astype(np.float32)
        if self.decomposer.evaluator != "llm":
            return
        for stage in ("orig", "rewrite"):
            prompts = [PROMPT_LLM_JUDGE.format(passage=it.passage,
                                               stem=it.stem if stage == "orig" else (it.stem_rewritten or it.stem),
                                               option=o)
                       for it in targets for o in it.options]
            res = self.llm.p_yes_batch_with_validity(prompts)
            p = probs_of(res).reshape(len(targets), 4)
            valid = validity_mask_of(res, len(targets), 4)
            assert_finite(f"planted_llm_{stage}", p)
            for i, it in enumerate(targets):
                if stage == "orig":
                    it.llm_pyes = p[i].astype(np.float32)
                    it.llm_pyes_valid = valid[i].copy()
                else:
                    it.llm_pyes_rewrite = p[i].astype(np.float32)
                    it.llm_pyes_rewrite_valid = valid[i].copy()

    def expected_rate_increase(self, items: Sequence[QuizItem],
                               planted_ids: Sequence[int]) -> Optional[float]:
        """frac * (mean[key supported under BOTH stems] - mean[slot already dc]) / 3.

        Planting OVERWRITES a slot rather than adding one, so a slot already distractor-caused
        contributes nothing, and a key the judge does not support never flags. The naive
        planted_fraction/3 ignores both and makes a working instrument read as failed."""
        if not items or not planted_ids:
            return None
        slot = int(self.cfg.planted_slot)
        if not 1 <= slot <= 3:
            return None
        chosen = {int(i) for i in planted_ids}
        targets = [it for it in items if int(it.item_id) in chosen]
        if not targets:
            return None
        thr = self.decomposer.thr
        po = self.decomposer.support_prob(targets, "orig")
        pr = self.decomposer.support_prob(targets, "rewrite")
        key_flags = (po[:, 0] >= thr) & (pr[:, 0] >= thr)   # the key would flag in the slot
        slot_already = (po[:, slot] >= thr) & (pr[:, slot] >= thr)   # that slot already flags
        frac = len(targets) / len(items)
        return float(frac * (float(key_flags.mean()) - float(slot_already.mean())) / 3.0)

    def report(self, items: Sequence[QuizItem], seed: int) -> Dict[str, Any]:
        planted_ids = self.choose_ids(items, seed)
        self._planted_ids = planted_ids
        planted = self.plant(items, planted_ids)
        self.rescore(planted)
        self._planted = planted
        r = self.decomposer.rates(planted)
        baseline = self.decomposer.rates(items)
        base_rate = baseline["distractor_caused_csd_rate"]
        measured = r["distractor_caused_csd_rate"]
        increase = (None if (base_rate is None or measured is None)
                    else float(measured) - float(base_rate))
        expected = self.expected_rate_increase(items, planted_ids)
        naive = (float(self.cfg.planted_fraction) / 3.0) if items else None
        detection = (float(increase) / float(expected)
                     if (increase is not None and expected is not None and expected > 0) else None)
        r["item_yield"] = (len(planted) / len(items)) if items else None
        r["baseline_rate"] = base_rate
        r["planted_item_ids"] = list(planted_ids)
        r["n_planted"] = len(planted_ids)
        r["planted_fraction"] = float(self.cfg.planted_fraction)
        r["planted_slot"] = int(self.cfg.planted_slot)
        r["measured_rate_increase"] = increase
        r["expected_rate_increase"] = expected
        r["expected_rate_increase_naive"] = naive          # published BESIDE, never in place of
        r["detection_ratio"] = detection                   # None when the expectation is not positive
        r["reads_judge_tensor"] = True
        r["filter_scorer"] = "planted_key_in_distractor_slot"
        r.update(functional_record(self.proxy, planted, seed))
        return r


# ============================================================== overgenerate + rerank (H4)
class OvergenerateContradictionReranker:
    """Proposed H4 arm: 10 candidate distractors per item, 3 selected by descending
    P_contradict with a deterministic diversity tie-break.

    The candidate cache lives ON the items (candidates / cand_nli / cand_sim / cand_emb), so the
    key-similarity ablation reranks EXACTLY the same ten candidates without a second generation
    pass -- which is the only way the two arms differ solely in their selection criterion. Every
    provenance marker for that cache lives on the items too, so both arms report the same counts
    for it."""

    name = "overgenerate_rerank_by_contradiction"
    selection_criterion = "descending_p_contradict"

    def __init__(self, cfg: Any, llm: Any, nli: Any, embedder: Any, decomposer: "FlagDecomposer",
                 proxy: Optional["FunctionalityProxy"] = None) -> None:
        self.cfg = cfg
        self.llm = llm
        self.nli = nli
        self.embedder = embedder
        self.decomposer = decomposer
        self.proxy = proxy
        self._selected: List[QuizItem] = []

    def evaluated_items(self) -> List[QuizItem]:
        return list(self._selected)

    # ITEM 24: shared cache, a repair kept only when it recovers MORE candidates
    def generate_candidates(self, items: Sequence[QuizItem]) -> None:
        """Fills the shared cache only for items that have none, and marks the provenance of
        every fallback on the item so both arms report the same counts.

        The repair improvement is marked on the ITEM, exactly as padding and line-splitting are.
        It used to be an instance counter, and generate_candidates returns immediately for the
        second arm because the cache is already filled, so that arm reported 0 repairs for a pool
        that had been repaired -- one candidate pool described by two contradictory numbers in the
        same payload, with the ablation arm's 0 reading as "no repair was needed"."""
        todo = [it for it in items if it.candidates is None]
        if not todo:
            return
        prompts = [PROMPT_OVERGEN.format(passage=it.passage, stem=it.stem, key=it.options[0]) for it in todo]
        outs = self.llm.generate_batch(prompts, self.cfg.overgen_max_new_tokens)
        parsed = [parse_candidate_list(o, it.options[0], self.cfg.n_overgen_candidates)
                  for o, it in zip(outs, todo)]
        cands = [p[0] for p in parsed]
        line_split = [p[1] for p in parsed]
        repair_improved = [False] * len(todo)

        retry = [i for i, c in enumerate(cands) if len(c) < 3]
        if retry:
            rep = [prompts[i] + OVERGEN_REPAIR for i in retry]
            outs2 = self.llm.generate_batch(rep, self.cfg.overgen_max_new_tokens)
            for i, o in zip(retry, outs2):
                new_c, new_split = parse_candidate_list(o, todo[i].options[0], self.cfg.n_overgen_candidates)
                # keep the retry ONLY when it recovers more; a shorter reply silently shrinking
                # the pool would change what both arms select from
                if len(new_c) > len(cands[i]):
                    cands[i], line_split[i] = new_c, new_split
                    repair_improved[i] = True

        for it, c, split, improved in zip(todo, cands, line_split, repair_improved):
            padded = False
            if len(c) < 3:
                padded = True
                seen = {x.lower() for x in c}
                for d in it.options[1:]:
                    if d.lower() not in seen:
                        c.append(d)
                        seen.add(d.lower())
                print(f"[{self.name}] item {it.item_id}: only {len(c)} candidates after repair; "
                      f"padded with the original distractors")
            it.candidates = c[: self.cfg.n_overgen_candidates]
            it.cand_padded = bool(padded)                  # marked on the ITEM, so both arms agree
            it.cand_line_split = bool(split)
            it.cand_repair_improved = bool(improved)
        counts = [len(it.candidates or []) for it in todo]
        print(f"[{self.name}] candidates for {len(todo)} items: mean={np.mean(counts):.2f} "
              f"min={min(counts)} padded={sum(1 for it in todo if it.cand_padded)} "
              f"line_split={sum(1 for it in todo if it.cand_line_split)} "
              f"repair_improved={sum(1 for it in todo if it.cand_repair_improved)}")

    def score_candidates(self, items: Sequence[QuizItem]) -> None:
        """Fills cand_nli, cand_sim and cand_emb only for items that lack them; the padded tail
        stays NaN and is never read, because every reader slices to the real candidate count."""
        todo = [it for it in items if it.cand_nli is None and it.candidates]
        if not todo:
            return
        prem: List[str] = []
        hyp: List[str] = []
        for it in todo:
            for c in it.candidates or []:
                prem.append(it.passage)
                hyp.append(HYPOTHESIS_TEMPLATE.format(stem=it.stem.rstrip("?"), option=c))
        P = np.asarray(self.nli.score_pairs(prem, hyp))
        K = int(self.cfg.n_overgen_candidates)
        pos = 0
        for it in todo:
            n = len(it.candidates or [])
            arr = np.full((K, 3), np.nan, dtype=np.float32)
            arr[:n] = P[pos:pos + n]
            pos += n
            it.cand_nli = arr
            E = np.asarray(self.embedder.encode(list(it.candidates or []) + [it.options[0]]))
            it.cand_emb = E[:n]
            sim = np.full(K, np.nan, dtype=np.float32)
            sim[:n] = E[:n] @ E[n]
            it.cand_sim = sim

    def primary_scores(self, it: QuizItem, n: int) -> np.ndarray:
        return np.asarray(it.cand_nli)[:n, C_IDX]

    def select(self, it: QuizItem) -> List[int]:
        """Up to three candidate indices by descending primary score with a deterministic
        diversity tie-break; fewer than three only when the item has fewer candidates."""
        n = len(it.candidates or [])
        if n == 0:
            return []
        prim = self.primary_scores(it, n)
        E = np.asarray(it.cand_emb)
        chosen: List[int] = []
        remaining = list(range(n))
        while len(chosen) < 3 and remaining:
            best = max(remaining, key=lambda j: (float(prim[j]), -j))
            tied = [j for j in remaining if float(prim[best]) - float(prim[j]) <= self.cfg.tie_eps]
            j_star = best
            if chosen and len(tied) > 1:
                picked = list(chosen)

                def adjusted(j: int) -> float:
                    max_sim = max(float(E[j] @ E[c]) for c in picked)
                    return float(prim[j]) - self.cfg.lambda_div * max_sim

                j_star = max(tied, key=lambda j: (adjusted(j), -j))
            chosen.append(j_star)
            remaining.remove(j_star)
        return chosen

    def _rescore_selected(self, sel_items: List[QuizItem]) -> None:
        """Rewrite-stem scores for the rebuilt items (and LLM-judge scores when the LLM is the
        evaluator): the selected distractors are new text, so the rewrite stage must see them.

        The generative judge's validity mask is refreshed with the probabilities it describes.
        build_selected_items constructs each rebuilt item with dataclasses.replace, which copies
        llm_pyes_valid from the PRE-rerank item; three of its four slots then describe option
        text that is no longer in the item, so a stale True would readmit an imputed 0.0 as a
        measured non-support and a stale False would discard a slot the judge did measure. Both
        stages are rescored under the llm evaluator for exactly that reason."""
        if not sel_items:
            return
        prem: List[str] = []
        hyp: List[str] = []
        for new in sel_items:
            stem = new.stem_rewritten or new.stem
            for o in new.options:
                prem.append(new.passage)
                hyp.append(HYPOTHESIS_TEMPLATE.format(stem=stem.rstrip("?"), option=o))
        R = np.asarray(self.nli.score_pairs(prem, hyp)).reshape(len(sel_items), 4, 3)
        assert_finite("rerank_nli_rewrite", R)
        for i, new in enumerate(sel_items):
            new.nli_rewrite = R[i].astype(np.float32)
        if self.decomposer.evaluator != "llm":
            return
        for stage in ("orig", "rewrite"):
            prompts = [PROMPT_LLM_JUDGE.format(passage=x.passage,
                                               stem=x.stem if stage == "orig" else (x.stem_rewritten or x.stem),
                                               option=o)
                       for x in sel_items for o in x.options]
            res = self.llm.p_yes_batch_with_validity(prompts)
            p = np.asarray(probs_of(res)).reshape(len(sel_items), 4)
            valid = validity_mask_of(res, len(sel_items), 4)
            assert_finite(f"rerank_llm_{stage}", p)
            for i, new in enumerate(sel_items):
                if stage == "orig":
                    new.llm_pyes = p[i].astype(np.float32)
                    new.llm_pyes_valid = valid[i].copy()
                else:
                    new.llm_pyes_rewrite = p[i].astype(np.float32)
                    new.llm_pyes_rewrite_valid = valid[i].copy()

    def build_selected_items(self, items: Sequence[QuizItem]) -> List[QuizItem]:
        """Rebuilt items keeping the key in slot 0 and their own item_id, so the comparison with
        the pool is paired by id; an item yielding fewer than three selections is dropped and
        shows up in item_yield.

        The inherited judge masks are CLEARED here rather than carried: three of the four slots
        hold new option text, so the copied mask describes strings this item no longer contains.
        Under the llm evaluator _rescore_selected refills them from the judge that scored the new
        options; under nli they are unused and valid_mask is all-True by construction."""
        sel_items: List[QuizItem] = []
        for it in items:
            idx = self.select(it)
            if len(idx) < 3:
                continue
            new = dataclasses.replace(
                it, options=[it.options[0]] + [(it.candidates or [])[j] for j in idx], arm=self.name,
                candidates=None, cand_nli=None, cand_sim=None, cand_emb=None, functional=None,
                planted=False, llm_pyes_valid=None, llm_pyes_rewrite_valid=None,
            )
            new.nli_orig = np.vstack([np.asarray(it.nli_orig)[0:1],
                                      np.asarray(it.cand_nli)[idx]]).astype(np.float32)
            new.key_sim = np.concatenate([[1.0], np.asarray(it.cand_sim)[idx]]).astype(np.float32)
            new.overlap = np.array([lexical_overlap(o, it.passage) for o in new.options], dtype=np.float32)
            sel_items.append(new)
        self._rescore_selected(sel_items)
        return sel_items

    # ITEM 25: the quartile ratio needs support in both compared quartiles
    @staticmethod
    def quartile_floor(n_bucketed: int) -> int:
        """max(5, ceil(0.1 * n_bucketed / 4)): the minimum candidates each compared quartile
        needs before a ratio may be published."""
        return int(max(5, math.ceil(0.1 * int(n_bucketed) / 4)))

    def candidate_pool_csd_rate(self, items: Sequence[QuizItem]) -> Optional[float]:
        """The share of all generated candidates the judge supports under the original stem;
        None when no item carries candidates."""
        flags: List[float] = []
        for it in items:
            n = len(it.candidates or [])
            if n == 0 or it.cand_nli is None:
                continue
            sup = np.asarray(it.cand_nli)[:n, E_IDX] >= self.cfg.entail_threshold
            flags.extend(float(x) for x in sup)
        return float(np.mean(flags)) if flags else None

    def similarity_quartiles(self, items: Sequence[QuizItem]) -> Dict[str, Any]:
        """Per-quartile support rates and counts, the smoothed ratio, the RAW ratio and the floor.

        An item with fewer than four candidates cannot be quartered, so it is skipped WITHOUT
        counting its candidates toward support. With eps=0.01 the smoothed ratio turns one
        entailed candidate in Q1 against zero in Q2 into ~101, clearing the H4 gate of 1.3 by a
        factor of 78 on a single lucky candidate -- so both compared quartiles must clear the
        floor or the ratio is None."""
        buckets: List[List[bool]] = [[] for _ in range(4)]
        n_bucketed = 0
        n_items_skipped = 0
        for it in items:
            n = len(it.candidates or [])
            if n < 4 or it.cand_nli is None or it.cand_sim is None:
                n_items_skipped += 1
                continue
            sim = np.asarray(it.cand_sim)[:n]
            sup = np.asarray(it.cand_nli)[:n, E_IDX] >= self.cfg.entail_threshold
            order = np.argsort(-sim, kind="stable")
            for rank, j in enumerate(order):
                buckets[(4 * rank) // n].append(bool(sup[j]))
            n_bucketed += n
        counts = [len(b) for b in buckets]
        rates: List[Optional[float]] = [float(np.mean(b)) if b else None for b in buckets]
        floor = self.quartile_floor(n_bucketed)
        q1_ok = counts[0] >= floor and rates[0] is not None
        q2_ok = counts[1] >= floor and rates[1] is not None
        eps = float(self.cfg.quartile_ratio_smoothing)
        ratio: Optional[float] = None
        ratio_raw: Optional[float] = None
        if q1_ok and q2_ok:
            ratio = (float(rates[0]) + eps) / (float(rates[1]) + eps)
            ratio_raw = (float(rates[0]) / float(rates[1])) if float(rates[1]) > 0 else None
        u_shaped: Optional[bool] = None
        if ratio is not None and rates[3] is not None and counts[3] >= floor:
            u_shaped = bool(float(rates[0]) > float(rates[1]) and float(rates[3]) > float(rates[1]))
        return {"candidate_csd_rate_by_similarity_quartile": rates,
                "candidate_count_by_similarity_quartile": counts,
                "similarity_quartile_csd_ratio": ratio,
                "similarity_quartile_csd_ratio_raw": ratio_raw,
                "similarity_quartile_min_candidates": floor,
                "similarity_quartile_q1_admitted": bool(q1_ok),
                "similarity_quartile_q2_admitted": bool(q2_ok),
                "u_shaped": u_shaped,
                "n_candidates_scored": int(n_bucketed),
                "n_items_below_four_candidates": int(n_items_skipped)}

    def report(self, items: Sequence[QuizItem], seed: int) -> Dict[str, Any]:
        self.generate_candidates(items)
        self.score_candidates(items)
        sel = self.build_selected_items(items)
        self._selected = sel
        r = self.decomposer.rates(sel)
        r["item_yield"] = (len(sel) / len(items)) if items else None
        r.update(self.similarity_quartiles(items))
        r["candidate_pool_csd_rate"] = self.candidate_pool_csd_rate(items)
        r["selected_item_ids"] = [int(it.item_id) for it in sel]
        r["mean_selected_p_contradict"] = (
            float(np.mean([np.asarray(it.nli_orig)[1:, C_IDX].mean() for it in sel])) if sel else None)
        r["mean_selected_key_sim"] = (
            float(np.mean([np.asarray(it.key_sim)[1:].mean() for it in sel])) if sel else None)
        # ALL THREE provenance counts derived from the ITEMS, so both arms report identical
        # numbers for the one shared candidate cache
        r["n_padded_candidate_sets"] = int(sum(1 for it in items if it.cand_padded))
        r["n_line_split_candidate_sets"] = int(sum(1 for it in items if it.cand_line_split))
        r["n_repair_improved_candidate_sets"] = int(sum(1 for it in items if it.cand_repair_improved))
        r["selection_criterion"] = self.selection_criterion
        r["reads_judge_tensor"] = True
        if self.proxy is None:
            r["functional_distractor_rate"] = None
            r["functional_contexts_measured"] = None
            r["functional_contexts_excluded"] = None
        else:
            r["functional_distractor_rate"] = self.proxy.functional_rate(sel, seed)
            r.update(self.proxy.exclusion_record())
        return r


class OvergenerateKeySimilarityReranker(OvergenerateContradictionReranker):
    """Ablation H4 arm: the SAME ten cached candidates ranked by descending cosine similarity to
    the key (standard semantic hard-negative selection) instead of by contradiction.

    Removing the validity criterion is expected to raise the distractor-caused rate, because
    similarity selects the sibling list items the passage supports -- which isolates whether any
    H4 gain comes from the criterion or merely from the candidate surplus."""

    name = "overgenerate_rerank_by_key_similarity"
    selection_criterion = "descending_cosine_to_key"

    def primary_scores(self, it: QuizItem, n: int) -> np.ndarray:
        return np.asarray(it.cand_sim)[:n]                # KEY DIFFERENCE: validity criterion removed

    def report(self, items: Sequence[QuizItem], seed: int) -> Dict[str, Any]:
        """The contradiction arm's report body verbatim, with selection_criterion reading
        descending_cosine_to_key and the provenance counts identical to that arm's.

        Declared explicitly because the interface sheet names it, and a PASS-THROUGH by
        construction: this arm's whole claim is that only the selection criterion differs, so a
        second copy of the report body here would be exactly the place that claim could quietly
        stop being true. The counts match because all three come from the shared cache's own
        items, not from either arm instance."""
        return super().report(items, seed)


def label_topics_with_fallback_mask(llm: Any, corpus: Any, pids: Sequence[int], cfg: Any,
                                    fallback_label: Optional[Callable[[Any, int], str]] = None
                                    ) -> Tuple[List[str], int, List[bool]]:
    """The TRIPLE (labels, n_fallback, used_fallback) from ONE generation pass.

    used_fallback[i] is True exactly when item i's reply was unusable and the fallback label was
    substituted, counted AT THE POINT the substitution happens; reconstructing the count later by
    comparing each label with the fallback string miscounts a model that happens to echo the
    passage prefix as its own label.

    The mask exists so a caller that needs BOTH the grounded and the closed-book label list can
    derive the second from the first instead of regenerating it. A second pass repeats one
    generation per item -- items_per_seed per seed plus the replication -- which the
    pilot_cost_multiplier arithmetic behind TIME_ESTIMATE does not account for; and because
    generate_batch halves its batch size on OOM, a second pass at a different batch size can
    decode a different greedy label, which would prompt the closed-book control with a topic other
    than the one retrieval used."""
    fb = fallback_label or topic_fallback_label
    prompts = [PROMPT_TOPIC_LABEL.format(passage=str(corpus.passages[p])[:800]) for p in pids]
    outs = llm.generate_batch(prompts, cfg.topic_max_new_tokens)
    labels: List[str] = []
    used_fallback: List[bool] = []
    n_fallback = 0
    for o, p in zip(outs, pids):
        lab = ""
        for line in (o or "").splitlines():
            line = line.strip().strip('"').strip("'").strip(" .:")
            if line:
                lab = line
                break
        took_fallback = bool(len(lab) < 3)
        if took_fallback:
            lab = fb(corpus, p)
            n_fallback += 1                     # counted here, not inferred later
        labels.append(lab[:80])
        used_fallback.append(took_fallback)
    return labels, n_fallback, used_fallback


def assert_rows_rectangular(rows: Dict[str, Any], condition: str) -> None:
    """Return None when every ROW_FIELDS column of a row table has the same length; raise
    ValueError naming the ragged columns otherwise.

    subset_rows masks by POSITION and the item bootstrap zips the key column against a flag
    column, so a column that advanced without the others would silently pair a flag with another
    row's item key -- a misaligned comparison no downstream guard can see."""
    lengths = {f: len(rows.get(f, [])) for f in ROW_FIELDS}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"row table for '{condition}' has ragged columns: {lengths}; every "
                         f"ROW_FIELDS column must advance together or the item bootstrap would "
                         f"read misaligned rows")


def _rescore_selected(*args: Any, **kwargs: Any) -> None:
    """Not an implementation: the live one is OvergenerateContradictionReranker._rescore_selected.

    A module-level copy of that method used to sit here, taking a `self` no caller supplies. It
    was unreachable dead code that read as a second implementation of the mask refresh, so an
    edit could land in it and never run. Calling it RAISES rather than silently doing nothing or
    failing on arity, because a caller reaching this name is reaching for the wrong one."""
    del args, kwargs
    raise RuntimeError("methods._rescore_selected is not the live implementation; the rerank "
                       "arms call OvergenerateContradictionReranker._rescore_selected, which "
                       "refreshes both judge validity masks for the rebuilt options")


def _unpack_scored(result: Any, what: str) -> Tuple[Any, Any]:
    """The (values, validity) pair a masked judge call returns; TypeError on any other shape.

    Deliberately strict. np.asarray on a (p, valid) pair of two length-n arrays does NOT raise --
    it returns a (2, n) array, so a caller that forgot the pair reads the validity mask as a row
    of probabilities and never finds out. Naming the expected shape here turns that into a
    programming error, which run_arm_sequence re-raises instead of filing as an arm failure."""
    if isinstance(result, (tuple, list)) and len(result) == 2:
        return result[0], result[1]
    raise TypeError(f"expected a (probabilities, validity) pair to read the {what} from; got "
                    f"{type(result).__name__}. A judge that flags must be called through its "
                    f"*_with_validity method, because an imputed row is an UNMEASURED cell and "
                    f"only the mask beside it says so.")


def probs_of(result: Any) -> np.ndarray:
    """The probability array of a masked judge return."""
    values, _valid = _unpack_scored(result, "probabilities")
    return np.asarray(values)


def validity_mask_of(result: Any, n_rows: int, width: int = 1) -> np.ndarray:
    """The [n_rows, width] boolean validity mask of a masked judge return.

    False marks a row whose forward pass was not finite and whose value beside it was
    SUBSTITUTED. Raises ValueError when the mask does not describe exactly the rows the caller
    is about to store, because a mask misaligned with its probabilities excludes the wrong
    slot -- which is worse than excluding none."""
    _values, valid = _unpack_scored(result, "validity mask")
    m = np.asarray(valid, dtype=bool)
    if m.size != int(n_rows) * int(width):
        raise ValueError(f"validity mask holds {m.size} entries but {int(n_rows)} x {int(width)} "
                         f"= {int(n_rows) * int(width)} were scored; the mask does not describe "
                         f"the rows it was returned with")
    return m.reshape(int(n_rows), int(width))


def functional_record(proxy: Optional["FunctionalityProxy"], items: Sequence[QuizItem],
                      seed: int) -> Dict[str, Any]:
    """One arm's functionality-proxy rate together with the context counts it stands on.

    The rate and the counts are read in ONE call, so the counts always describe the set this
    arm measured: FunctionalityProxy resets its counters at the start of every call, and every
    arm calls the proxy on its own retained set. A rate published without them cannot say
    whether an imputed answerer row was excluded or silently counted."""
    if proxy is None:
        return {"functional_distractor_rate": None,
                "functional_contexts_measured": None,
                "functional_contexts_excluded": None}
    out: Dict[str, Any] = {"functional_distractor_rate": proxy.functional_rate(items, seed)}
    out.update(proxy.exclusion_record())
    return out
