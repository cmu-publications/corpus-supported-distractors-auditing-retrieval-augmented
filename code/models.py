"""
models.py — the frozen pretrained components and the judge calibration.

Qwen2.5-3B-Instruct (generator, stem rewriter, LLM judge, functionality-proxy answerer),
cross-encoder/nli-deberta-v3-base (the NLI judge), BAAI/bge-small-en-v1.5 (retrieval and
key similarity), the dense retrieval index over the KEPT corpus, and the calibration that
decides which judge measures the study's rates.

This is the only module that loads weights. Nothing is trained: every module goes through
_freeze (train(False) plus requires_grad_(False)) and every softmax is computed in float32.

Three rules this file exists to enforce:
  * a calibration cell with an empty denominator is None, never a measured 0.0 -- that
    number decides the evaluator and so relabels every rate in the results;
  * retrieval happens at ONE fixed depth on every seed, so the evidence never changes
    between seeds and the arms stay comparable;
  * the NLI premise allowance is fixed and independent of the hypothesis length, so the
    stem rewrite cannot move rows between the decomposition's two classes.
"""
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from experiment_config import (BGE_QUERY_INSTRUCTION, HYPOTHESIS_TEMPLATE, MODEL_REVISIONS, PROMPT_LLM_JUDGE,
                               snapshot_cached, snapshot_revision)
from data import CalibrationSet, assert_finite, nli_pair_token_plan, pass_start_batch_size


# ITEM 35: the absent VRAM value is None, so no log line can format inf
def free_vram_bytes() -> Optional[float]:
    """Free device memory in bytes, or None when no CUDA device is visible.

    The old float("inf") reached three log lines that formatted it, so a CPU run printed a
    non-finite value to stdout and the gate's non-finite scan failed the run."""
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        return float(free)
    return None


def _vram_text() -> str:
    """The free-VRAM log fragment; the literal 'n/a' when no device is visible."""
    free = free_vram_bytes()
    return "n/a" if free is None else f"{free / 1e9:.2f} GB"


def _local_kw(role: str, repo_id: str) -> Dict[str, Any]:
    """local_files_only when the snapshot is cached, plus the pinned revision when one is set.

    A cached snapshot must never wait on a rate-limited Hub request mid-run. The ROLE is passed in
    so the pin table is read under its declared key at this call site, not resolved from the repo
    id after the fact."""
    kw: Dict[str, Any] = {"local_files_only": True} if snapshot_cached(repo_id) else {}
    rev = model_pin(role)
    if rev:
        kw["revision"] = str(rev)
    return kw


def _revision_record(role: str, repo_id: str) -> Dict[str, Any]:
    """The repo id, the pinned revision and the resolved snapshot sha of the loaded weights, with
    the PATH that resolved it.

    Read through model_pin(role), the same accessor _local_kw uses, so the revision REQUESTED at
    load time and the revision RECORDED in the payload are one value. Both used to look the table
    up by repo id and both silently got None.

    resolution says whether the sha came from the pin, from the refs file a local_files_only load
    follows, or from the mtime guess of last resort; complete_snapshots lists every cached
    candidate. Without those a run with two cached snapshots could name the commit the load did
    not follow and nothing in the payload would say the value was a guess. The import is
    function-local so this module's import list is unchanged, the same way JudgeCalibrator pulls in
    the calibration floor at its one call site."""
    from experiment_config import snapshot_revision_record

    rec = snapshot_revision_record(repo_id)
    return {"role": role, "repo_id": repo_id, "pinned_revision": model_pin(role),
            "resolved_revision": rec["resolved_revision"],
            "revision_resolution": rec["resolution"],
            "revision_ref_read": rec["ref_read"],
            "n_complete_snapshots": rec["n_complete_snapshots"],
            "complete_snapshots": list(rec["complete_snapshots"])}


def _log(msg: str) -> None:
    print(msg, flush=True)


def _freeze(module: torch.nn.Module) -> torch.nn.Module:
    """Inference mode plus gradients disabled on every parameter; nothing here is trained."""
    module.train(False)
    for p in module.parameters():
        p.requires_grad_(False)
    return module


def _ratio(num: int, den: int) -> Optional[float]:
    """num/den as a float, or None when the denominator is zero.

    ITEM 33: an unmeasured cell is None. Reporting 0.0 for an empty denominator is
    indistinguishable from a judge that predicted positives and got every one wrong."""
    return (float(num) / float(den)) if den > 0 else None


def _r2_on_overlap(overlap: np.ndarray, scores: np.ndarray) -> Optional[float]:
    """R^2 of the linear fit score ~ lexical overlap, or None when the fit is undefined."""
    x = np.asarray(overlap, dtype=np.float64)
    y = np.asarray(scores, dtype=np.float64)
    if len(x) <= 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    rvalue = float(stats.linregress(x, y).rvalue)
    return float(rvalue ** 2) if np.isfinite(rvalue) else None


def _auroc(y: np.ndarray, scores: np.ndarray) -> Optional[float]:
    """AUROC, or None when one class is absent or the metric is undefined; never 0.0 for
    an unmeasured set."""
    if not 0 < int(y.sum()) < len(y):
        return None
    try:
        return float(roc_auc_score(y, scores))
    except ValueError as e:                        # recorded by the caller, never silently 0.0
        _log(f"[calibration] AUROC undefined ({type(e).__name__}: {e}); recorded as null")
        return None


# ============================================================== generator
class GeneratorLLM:
    """Qwen2.5-3B-Instruct: greedy generation, next-token P(Yes), answer-letter distributions.

    Loading policy on an 8 GB GPU: the weights stream straight to the device with
    device_map={"": 0} so the 6 GB checkpoint is never staged through host RAM (that path
    stalled a run); nf4 is attempted only when bitsandbytes is importable and the fallback
    records its cause."""

    # A non-finite judge row is UNDEFINED, not neutral. The substitute is strictly below any
    # entail_threshold in (0, 1], so an imputed row can never manufacture a corpus-supported flag
    # or a key hallucination; the validity mask returned beside it is what actually keeps the row
    # out of the numerators and denominators. The old value was 0.5 -- exactly the threshold, and
    # the flag rule is >=, so every fabricated row resolved to "supported".
    NAN_P_YES_SUBSTITUTE = 0.0
    # The proxy's letter distribution is a sampling distribution, not a flag input, so a uniform
    # row decides nothing on its own; but the WITH/WITHOUT comparison it feeds turns two rows into
    # one flag, so an imputed row still has to be excluded. It is returned beside the mask for
    # exactly that reason, and it is still counted and reported.
    NAN_LETTER_PROB_SUBSTITUTE = 0.25
    # The two substitutions are DIFFERENT quantities with different consequences, so they are
    # counted under different keys and flagged in different sentences. One shared counter could
    # not say which instrument produced the non-finite rows, and the one shared sentence called
    # the judge's 0.0 "a neutral value" -- precisely what it is designed not to be.
    JUDGE_SUBSTITUTION = "judge_p_yes"
    ANSWERER_SUBSTITUTION = "answerer_letter_probs"
    SUBSTITUTION_LABELS = {JUDGE_SUBSTITUTION: "P(yes)",
                           ANSWERER_SUBSTITUTION: "letter-probability"}

    def __init__(self, cfg: Any) -> None:
        from analysis import substitution_kinds

        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.revisions = _revision_record("generator", cfg.generator_id)
        lk = _local_kw("generator", cfg.generator_id)
        _assert_pin_honoured("generator", cfg.generator_id, lk)
        _log(f"[llm] loading tokenizer {cfg.generator_id} (local_only={bool(lk.get('local_files_only'))})")
        self.tok = AutoTokenizer.from_pretrained(cfg.generator_id, **lk)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        # typed Any, not Optional[torch.nn.Module]: torch declares Module.__getattr__ as returning
        # Tensor | Module, so self.model.generate(...) reads to the checker as calling a Tensor
        self.model: Any = None
        self.precision = "unloaded"
        self.quantization_error: Optional[str] = None
        t0 = time.time()
        if cfg.load_4bit_nf4 and self.device.type == "cuda":
            try:
                import bitsandbytes  # noqa: F401
                from transformers import BitsAndBytesConfig
                q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                       bnb_4bit_compute_dtype=torch.float16,
                                       bnb_4bit_use_double_quant=True)
                _log("[llm] loading generator in nf4 (bitsandbytes) ...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    cfg.generator_id, quantization_config=q, device_map={"": 0}, **lk)
                self.precision = "nf4"
            except (ImportError, RuntimeError, ValueError, OSError) as e:
                # the cause is RECORDED, never swallowed: a payload that says float16 must
                # also say why nf4 was not used
                self.quantization_error = f"{type(e).__name__}: {e}"
                _log(f"[llm] 4-bit nf4 unavailable ({self.quantization_error}); "
                     f"loading float16 weights directly onto the GPU")
                self.model = None
                torch.cuda.empty_cache()
        if self.model is None:
            self.model = self._load_dense(cfg.generator_id, lk)
        _freeze(self.model)
        self.home_gen_batch_size = int(cfg.gen_batch_size)
        self.home_score_batch_size = int(cfg.judge_batch_size)
        self.gen_batch_size = self.home_gen_batch_size
        self.score_batch_size = self.home_score_batch_size
        self.yes_ids = self._first_ids(["Yes", " Yes", "yes", " yes"])
        self.no_ids = self._first_ids(["No", " No", "no", " no"])
        self.letter_ids = [self._first_ids([letter, " " + letter]) for letter in "ABCD"]
        self.n_generate_calls = 0
        self.generate_seconds = 0.0
        # observability counters: nothing substituted or downsized goes unrecorded. The kinds come
        # from analysis.substitution_kinds so the side that COUNTS and the side that FLAGS cannot
        # name them differently; nan_substitutions is a derived total, never a second tally.
        self.nan_substitutions_by_kind: Dict[str, int] = {k: 0 for k in substitution_kinds()}
        for declared in (self.JUDGE_SUBSTITUTION, self.ANSWERER_SUBSTITUTION):
            if declared not in self.nan_substitutions_by_kind:
                raise RuntimeError(f"substitution kind {declared!r} is not among the declared "
                                   f"kinds {sorted(self.nan_substitutions_by_kind)}; its rows "
                                   f"would be counted under a key no flag line reports")
        self.nan_rows_last_pass = 0
        self.nucleus_uniform_fallbacks = 0
        self.oom_fallbacks = 0
        _log(f"[llm] {cfg.generator_id} loaded precision={self.precision} device={self.device} "
             f"in {time.time() - t0:.0f}s; free VRAM={_vram_text()}")

    @property
    def nan_substitutions(self) -> int:
        """Every substituted row, of either kind.

        DERIVED from the per-kind counters rather than maintained beside them, so the total and
        its parts cannot disagree. Readers that only need "did anything get substituted" keep
        working; readers that need to know WHICH instrument read nan_substitutions_by_kind."""
        return int(sum(self.nan_substitutions_by_kind.values()))

    def _load_dense(self, repo_id: str, lk: Dict[str, Any]) -> Any:
        on_cuda = self.device.type == "cuda"
        dtype = torch.float16 if on_cuda else torch.float32
        kw = dict(lk)
        if on_cuda:
            kw["device_map"] = {"": 0}
        _log(f"[llm] loading generator weights dtype={dtype} device_map={kw.get('device_map')} "
             f"... (this can take a few minutes)")
        try:
            model = AutoModelForCausalLM.from_pretrained(repo_id, dtype=dtype, **kw)
        except TypeError:                          # older transformers spells it torch_dtype
            model = AutoModelForCausalLM.from_pretrained(repo_id, torch_dtype=dtype, **kw)
        if not on_cuda:
            model.to(self.device)
        self.precision = str(dtype).replace("torch.", "")
        return model

    def _first_ids(self, variants: Sequence[str]) -> List[int]:
        ids = set()
        for v in variants:
            enc = self.tok.encode(v, add_special_tokens=False)
            if enc:
                ids.add(int(enc[0]))
        return sorted(ids)

    def _chat_text(self, prompt: str) -> str:
        return self.tok.apply_chat_template([{"role": "user", "content": prompt}],
                                            tokenize=False, add_generation_prompt=True)

    def _encode(self, prompts: Sequence[str]):
        return self.tok([self._chat_text(p) for p in prompts], return_tensors="pt",
                        padding=True).to(self.device)

    @torch.no_grad()
    def generate_batch(self, prompts: Sequence[str], max_new_tokens: int) -> List[str]:
        """One decoded completion per prompt in order.

        Each pass STARTS at the home batch size: an OOM halving inside one pass used to
        persist for the whole run, so a single 4->1 fallback made every later pass four times
        slower for no reason. The fallback itself stays counted."""
        outs: List[str] = [""] * len(prompts)
        bs = pass_start_batch_size(self.home_gen_batch_size, self.gen_batch_size)
        i = 0
        t0 = time.time()
        while i < len(prompts):
            chunk = prompts[i:i + bs]
            try:
                enc = self._encode(chunk)
                gen = self.model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens,
                                          pad_token_id=self.tok.pad_token_id)
                new = gen[:, enc["input_ids"].shape[1]:]
                for j, d in enumerate(self.tok.batch_decode(new, skip_special_tokens=True)):
                    outs[i + j] = d
                i += len(chunk)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
                self.gen_batch_size = bs
                self.oom_fallbacks += 1
                _log(f"[llm] OOM during generation; batch size -> {bs} "
                     f"(fallback #{self.oom_fallbacks})")
        self.n_generate_calls += len(prompts)
        self.generate_seconds += time.time() - t0
        return outs

    @torch.no_grad()
    def _last_logits(self, prompts: Sequence[str]) -> torch.Tensor:
        enc = self._encode(prompts)
        try:
            out = self.model(**enc, logits_to_keep=1)
        except TypeError:
            out = self.model(**enc)
        return out.logits[:, -1, :].float()        # left padding -> the last position is valid

    def _score_chunks(self, prompts: Sequence[str],
                      fn: Callable[[torch.Tensor], Tuple[np.ndarray, np.ndarray]],
                      out_dim: int) -> Tuple[np.ndarray, np.ndarray]:
        """The scored rows and their per-row VALIDITY mask.

        Both are written at the true chunk offset AFTER fn returns, so an OOM retry at a smaller
        batch size cannot leave the mask misaligned with the rows it describes."""
        out = np.zeros((len(prompts), out_dim), dtype=np.float32)
        valid = np.ones(len(prompts), dtype=bool)
        bs = pass_start_batch_size(self.home_score_batch_size, self.score_batch_size)
        i = 0
        while i < len(prompts):
            chunk = prompts[i:i + bs]
            try:
                values, ok = fn(self._last_logits(chunk))
                out[i:i + len(chunk)] = values
                valid[i:i + len(chunk)] = ok
                i += len(chunk)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
                self.score_batch_size = bs
                self.oom_fallbacks += 1
                _log(f"[llm] OOM during scoring; batch size -> {bs} "
                     f"(fallback #{self.oom_fallbacks})")
        return out, valid

    def _account_nan(self, p: torch.Tensor, kind: str) -> np.ndarray:
        """The per-row VALIDITY mask (True where every entry is finite), counted UNDER ITS KIND
        before any substitution; never silent.

        The mask is the point: a substituted row is an UNMEASURED cell, and this project excludes
        an unmeasured cell rather than counting it as a measured negative. The kind is the second
        point: the judge's substitute and the answerer's substitute are different values with
        different downstream effects, and a single counter could not tell a broken judge from a
        broken answerer. An undeclared kind RAISES -- it can only come from a literal typo at a
        call site, never from the data, so main's programming_errors() re-raises it."""
        if kind not in self.nan_substitutions_by_kind:
            raise KeyError(f"_account_nan got substitution kind {kind!r}, which is not declared "
                           f"in {sorted(self.nan_substitutions_by_kind)}; its rows would be "
                           f"counted under a key no flag line reports")
        finite = torch.isfinite(p).all(dim=-1)
        bad = int((~finite).sum().item())
        if bad:
            self.nan_substitutions_by_kind[kind] += bad
            self.nan_rows_last_pass += bad
            label = self.SUBSTITUTION_LABELS.get(kind, kind)
            _log(f"[llm] WARNING: {bad} non-finite {label} rows substituted and marked INVALID "
                 f"({kind} total {self.nan_substitutions_by_kind[kind]}, "
                 f"all kinds {self.nan_substitutions})")
        return finite.cpu().numpy().astype(bool)

    def _check_nan_fraction(self, n_rows: int, what: str) -> None:
        frac = self.nan_rows_last_pass / max(1, n_rows)
        if frac > self.cfg.judge_nan_abort_fraction:
            raise RuntimeError(f"{what}: {self.nan_rows_last_pass}/{n_rows} rows ({frac:.1%}) "
                               f"produced NaN logits; the generator forward pass is "
                               f"numerically broken")

    def p_yes_batch_with_validity(self, prompts: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        """The PAIR (p_yes[n], valid[n]).

        valid[i] is False exactly when row i's logits were not finite and the value beside it was
        substituted. The substitute is NAN_P_YES_SUBSTITUTE, strictly below any entail_threshold
        in (0, 1], so even a caller that ignores the mask cannot turn a broken forward pass into a
        corpus-supported distractor or a key hallucination."""
        self.nan_rows_last_pass = 0

        def fn(logits: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
            yes = torch.logsumexp(logits[:, self.yes_ids], dim=-1)
            no = torch.logsumexp(logits[:, self.no_ids], dim=-1)
            p = torch.softmax(torch.stack([yes, no], dim=-1), dim=-1)[:, :1]
            ok = self._account_nan(p, self.JUDGE_SUBSTITUTION)
            sub = float(self.NAN_P_YES_SUBSTITUTE)
            return torch.nan_to_num(p, nan=sub, posinf=sub, neginf=sub).cpu().numpy(), ok

        out, valid = self._score_chunks(prompts, fn, 1)
        self._check_nan_fraction(len(prompts), "llm_judge")
        return out[:, 0], valid

    def p_yes_batch(self, prompts: Sequence[str]) -> np.ndarray:
        """P(Yes) / (P(Yes) + P(No)) from the next-token logits, shape [n].

        A thin wrapper over p_yes_batch_with_validity for callers that only score (the judge
        calibration); anything that turns these numbers into a FLAG must take the mask too."""
        p, _valid = self.p_yes_batch_with_validity(prompts)
        return p

    def letter_probs_batch(self, prompts: Sequence[str], temperature: float,
                           top_p: float) -> Tuple[np.ndarray, np.ndarray]:
        """The PAIR (probs[n, 4], valid[n]): a nucleus-truncated categorical over A-D per row,
        and the mask of the rows that were actually MEASURED.

        The mask used to be computed here and dropped on the way out, so the functionality proxy
        could not tell an imputed uniform NAN_LETTER_PROB_SUBSTITUTE row from a genuinely flat
        distribution. The proxy turns a WITH-passage row and a NO-passage row into ONE functional
        flag, so an imputed row on either side fabricates that flag out of noise -- and it feeds
        functional_distractor_rate and then h3_supported. Returning the pair is what lets
        FunctionalityProxy exclude the whole context and say how many it excluded."""
        self.nan_rows_last_pass = 0

        def fn(logits: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
            stacked = torch.stack([torch.logsumexp(logits[:, ids], dim=-1)
                                   for ids in self.letter_ids], dim=-1)
            probs_t = torch.softmax(stacked / max(temperature, 1e-4), dim=-1)
            ok = self._account_nan(probs_t, self.ANSWERER_SUBSTITUTION)
            sub = float(self.NAN_LETTER_PROB_SUBSTITUTE)
            probs = torch.nan_to_num(probs_t, nan=sub, posinf=sub, neginf=sub).cpu().numpy()
            rows = []
            for row in probs:
                p, fell_back = self._nucleus(row, top_p)
                if fell_back:
                    self.nucleus_uniform_fallbacks += 1
                rows.append(p)
            return np.stack(rows), ok

        out, valid = self._score_chunks(prompts, fn, 4)
        self._check_nan_fraction(len(prompts), "functionality_proxy")
        return out, valid

    @staticmethod
    def _nucleus(probs: np.ndarray, top_p: float) -> Tuple[np.ndarray, bool]:
        """The nucleus-truncated row and whether the uniform fallback was taken."""
        order = np.argsort(-probs)
        cum = np.cumsum(probs[order])
        keep = (cum - probs[order]) < top_p
        mask = np.zeros_like(probs, dtype=bool)
        mask[order[keep]] = True
        p = probs * mask
        s = float(p.sum())
        if s > 0:
            return p / s, False
        return np.full_like(probs, 0.25), True

    @staticmethod
    def sample_letters(probs: np.ndarray, n: int, seed: int) -> np.ndarray:
        """n letter indices from a generator seeded with seed; reproducible for the same
        seed and probabilities, so a rerun reproduces the proxy's record."""
        rng = np.random.default_rng(seed)
        p = np.asarray(probs, dtype=np.float64)
        return rng.choice(4, size=n, p=p / p.sum())

    def health(self) -> Dict[str, Any]:
        """The generator's numeric health, with each substitution kind counted and VALUED
        separately: one total under one label could not say which instrument was broken, nor what
        the substituted rows would have meant had the mask not excluded them."""
        return {"nan_substitutions": int(self.nan_substitutions),
                "nan_substitutions_by_kind": dict(self.nan_substitutions_by_kind),
                "n_judge_p_yes_substitutions":
                    int(self.nan_substitutions_by_kind[self.JUDGE_SUBSTITUTION]),
                "n_answerer_letter_substitutions":
                    int(self.nan_substitutions_by_kind[self.ANSWERER_SUBSTITUTION]),
                "nan_p_yes_substitute": float(self.NAN_P_YES_SUBSTITUTE),
                "nan_letter_prob_substitute": float(self.NAN_LETTER_PROB_SUBSTITUTE),
                "nucleus_uniform_fallbacks": int(self.nucleus_uniform_fallbacks),
                "oom_fallbacks": int(self.oom_fallbacks),
                "precision": self.precision,
                "quantization_error": self.quantization_error,
                "home_gen_batch_size": int(self.home_gen_batch_size),
                "final_gen_batch_size": int(self.gen_batch_size),
                "home_score_batch_size": int(self.home_score_batch_size),
                "final_score_batch_size": int(self.score_batch_size),
                "revisions": dict(self.revisions)}


# ============================================================== NLI judge
class NLIJudge:
    """cross-encoder/nli-deberta-v3-base; a 3-way softmax in [contradiction, entailment,
    neutral] order regardless of the checkpoint's own label indices."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.revisions = _revision_record("nli", cfg.nli_id)
        lk = _local_kw("nli", cfg.nli_id)
        _assert_pin_honoured("nli", cfg.nli_id, lk)
        _log(f"[nli] loading {cfg.nli_id} (local_only={bool(lk.get('local_files_only'))})")
        self.tok = AutoTokenizer.from_pretrained(cfg.nli_id, **lk)
        # typed Any and frozen in a separate statement, not bound to _freeze's return value:
        # _freeze is declared to return torch.nn.Module, and torch declares Module.__getattr__ as
        # returning Tensor | Module, so self.model.config.id2label.items() reads to the checker as
        # calling a Tensor. _freeze returns the very module it was handed, so the object bound
        # here and the moment it is frozen are both unchanged.
        self.model: Any = AutoModelForSequenceClassification.from_pretrained(cfg.nli_id, **lk)
        _freeze(self.model)
        id2label = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        l2i = {v: k for k, v in id2label.items()}
        missing = [name for name in ("contradiction", "entailment", "neutral") if name not in l2i]
        if missing:
            raise ValueError(f"NLI model labels {id2label} lack {missing}")
        self.perm = [l2i["contradiction"], l2i["entailment"], l2i["neutral"]]
        self.home_batch_size = int(cfg.nli_batch_size)
        self.batch_size = self.home_batch_size
        self.device = torch.device("cpu")
        self.home_device = torch.device("cpu")
        self.n_pairs_scored = 0
        self.seconds = 0.0
        self.oom_fallbacks = 0
        self.moved_to_cpu_on_oom = False
        # ITEM 34 counters: every truncated pair and every removed token is recorded
        self.n_truncated_pairs = 0
        self.n_premise_tokens_removed = 0
        self.n_hypothesis_tokens_removed = 0

    def to_device(self, device: str) -> None:
        """Move the judge and adopt the new device as HOME, so a later pass returns there."""
        dev = torch.device(device)
        self.model.to(dev)
        self.device = dev
        self.home_device = dev

    def maybe_offload(self) -> None:
        """Move to CPU when free VRAM is below the floor; a no-op with no CUDA device."""
        free = free_vram_bytes()
        if self.device.type == "cuda" and free is not None and free < self.cfg.low_vram_bytes:
            _log(f"[nli] free VRAM {_vram_text()} below the floor; moving the NLI judge to CPU")
            self.to_device("cpu")
            torch.cuda.empty_cache()

    # ITEM 34: a FIXED premise allowance, independent of the hypothesis length
    def _truncate_pair(self, premise: str, hypothesis: str) -> Tuple[str, str]:
        """Cut both sides to the plan from data.nli_pair_token_plan and count what was removed.

        The specificity rewrite lengthens the stem by construction, so a hypothesis-dependent
        premise allowance would judge the rewrite stage on less of the passage than the original
        stage -- moving rows from distractor-caused to stem-caused for a reason that is not in
        the data."""
        p_ids = self.tok.encode(premise, add_special_tokens=False)
        h_ids = self.tok.encode(hypothesis, add_special_tokens=False)
        plan = nli_pair_token_plan(len(p_ids), len(h_ids), max_length=self.cfg.nli_max_length,
                                   hypothesis_budget=self.cfg.nli_hypothesis_token_budget)
        if not plan["truncated"]:
            return premise, hypothesis
        self.n_truncated_pairs += 1
        self.n_premise_tokens_removed += int(plan["removed_premise_tokens"])
        self.n_hypothesis_tokens_removed += int(plan["removed_hypothesis_tokens"])
        p_out = (self.tok.decode(p_ids[:plan["premise_keep"]], skip_special_tokens=True)
                 if plan["removed_premise_tokens"] else premise)
        h_out = (self.tok.decode(h_ids[:plan["hypothesis_keep"]], skip_special_tokens=True)
                 if plan["removed_hypothesis_tokens"] else hypothesis)
        return p_out, h_out

    @torch.no_grad()
    def score_pairs(self, premises: Sequence[str], hypotheses: Sequence[str]) -> np.ndarray:
        """An [n,3] probability array in contradiction, entailment, neutral order.

        assert_finite RAISES on a non-finite row rather than substituting one, which is why the
        NLI evaluator measures every option slot and FlagDecomposer.valid_mask reports all-valid
        for it."""
        n = len(premises)
        if n != len(hypotheses):
            raise ValueError(f"score_pairs got {n} premises and {len(hypotheses)} hypotheses")
        out = np.zeros((n, 3), dtype=np.float32)
        if n == 0:
            return out
        pairs = [self._truncate_pair(p, h) for p, h in zip(premises, hypotheses)]
        prem = [p for p, _ in pairs]
        hyp = [h for _, h in pairs]
        bs = pass_start_batch_size(self.home_batch_size, self.batch_size)
        i = 0
        t0 = time.time()
        while i < n:
            try:
                enc = self.tok(prem[i:i + bs], hyp[i:i + bs], truncation=False, padding=True,
                               return_tensors="pt").to(self.device)
                logits = self.model(**enc).logits.float()
                probs = torch.softmax(logits, dim=-1)[:, self.perm].cpu().numpy()
                out[i:i + len(probs)] = probs
                i += len(probs)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                self.oom_fallbacks += 1
                if bs > self.cfg.nli_min_batch_size:
                    bs = max(self.cfg.nli_min_batch_size, bs // 2)
                    self.batch_size = bs
                    _log(f"[nli] OOM; batch size -> {bs} (fallback #{self.oom_fallbacks})")
                elif self.device.type == "cuda":
                    _log("[nli] OOM at the minimum batch size; moving to CPU for this pass")
                    self.moved_to_cpu_on_oom = True
                    self.model.to(torch.device("cpu"))
                    self.device = torch.device("cpu")
                else:
                    raise
        self.n_pairs_scored += n
        self.seconds += time.time() - t0
        assert_finite("nli_probs", out)
        return out

    def score_options(self, passage: str, stem: str, options: Sequence[str]) -> np.ndarray:
        """An [n_options,3] array for one passage and stem; inherits score_pairs' guarantees."""
        hyps = [HYPOTHESIS_TEMPLATE.format(stem=stem.rstrip("?"), option=o) for o in options]
        return self.score_pairs([passage] * len(options), hyps)

    def health(self) -> Dict[str, Any]:
        return {"oom_fallbacks": int(self.oom_fallbacks),
                "moved_to_cpu_on_oom": bool(self.moved_to_cpu_on_oom),
                "home_batch_size": int(self.home_batch_size),
                "final_batch_size": int(self.batch_size),
                "home_device": str(self.home_device), "device": str(self.device),
                "n_pairs_scored": int(self.n_pairs_scored),
                "max_length": int(self.cfg.nli_max_length),
                "hypothesis_token_budget": int(self.cfg.nli_hypothesis_token_budget),
                "n_truncated_pairs": int(self.n_truncated_pairs),
                "n_premise_tokens_removed": int(self.n_premise_tokens_removed),
                "n_hypothesis_tokens_removed": int(self.n_hypothesis_tokens_removed),
                "revisions": dict(self.revisions)}


# ============================================================== embedder and index
class Embedder:
    """bge-small-en-v1.5 with CLS pooling and L2 normalisation, so cosine is a dot product."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.revisions = _revision_record("embedder", cfg.embedder_id)
        lk = _local_kw("embedder", cfg.embedder_id)
        _assert_pin_honoured("embedder", cfg.embedder_id, lk)
        _log(f"[embedder] loading {cfg.embedder_id} "
             f"(local_only={bool(lk.get('local_files_only'))})")
        self.tok = AutoTokenizer.from_pretrained(cfg.embedder_id, **lk)
        # typed Any for the same reason as the NLI judge: bound to _freeze's declared
        # torch.nn.Module return, self.model.config.hidden_size resolves through
        # Module.__getattr__ to Tensor | Module. _freeze and .to both act on and return the same
        # object, so the module is frozen and moved to the device exactly as before.
        self.model: Any = AutoModel.from_pretrained(cfg.embedder_id, **lk)
        _freeze(self.model)
        self.model.to(self.device)
        self.dim = int(self.model.config.hidden_size)
        self.home_batch_size = int(cfg.embed_batch_size)
        self.batch_size = self.home_batch_size
        self.oom_fallbacks = 0

    @torch.no_grad()
    def encode(self, texts: Sequence[str], batch_size: Optional[int] = None) -> np.ndarray:
        """An [n,dim] float32 array of unit vectors in input order; an empty input yields a
        correctly shaped empty array and each call starts at the home batch size."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        bs = int(batch_size) if batch_size else pass_start_batch_size(self.home_batch_size,
                                                                     self.batch_size)
        out: List[np.ndarray] = []
        i = 0
        while i < len(texts):
            try:
                enc = self.tok(list(texts[i:i + bs]), padding=True, truncation=True,
                               max_length=512, return_tensors="pt").to(self.device)
                h = self.model(**enc).last_hidden_state[:, 0].float()
                out.append(torch.nn.functional.normalize(h, dim=-1).cpu().numpy())
                i += bs
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
                self.batch_size = bs
                self.oom_fallbacks += 1
                _log(f"[embedder] OOM; batch size -> {bs} (fallback #{self.oom_fallbacks})")
        return np.concatenate(out).astype(np.float32)

    def health(self) -> Dict[str, Any]:
        return {"oom_fallbacks": int(self.oom_fallbacks),
                "home_batch_size": int(self.home_batch_size),
                "final_batch_size": int(self.batch_size),
                "dim": int(self.dim), "revisions": dict(self.revisions)}


# ITEM 35: one fixed retrieval depth, over the KEPT passages only
class RetrievalIndex:
    """Exact dense retrieval over the kept corpus passages at ONE fixed depth.

    There is no widening branch. Re-retrieving at a deeper k on the seeds whose coverage
    looked low would score those seeds against different evidence than the others -- and the
    trigger was the data itself, which makes the evidence a post-hoc choice."""

    def __init__(self, cfg: Any, corpus: Any, embedder: "Embedder") -> None:
        self.cfg = cfg
        self.corpus = corpus
        self.embedder = embedder
        if not getattr(corpus, "finalized", False):
            raise RuntimeError("RetrievalIndex requires a finalized corpus; call "
                               "CorpusBuilder.finalize_corpus() before indexing, because the "
                               "cap remaps every passage id")
        t0 = time.time()
        self.emb = embedder.encode(corpus.passages)
        if self.emb.shape[0] != len(corpus.passages):
            raise RuntimeError(f"index has {self.emb.shape[0]} rows for "
                               f"{len(corpus.passages)} kept passages; every retrieval would "
                               f"be misaligned by an unknown offset")
        _log(f"[index] embedded {self.emb.shape[0]} kept passages in {time.time() - t0:.0f}s "
             f"(retrieval depth k={cfg.retrieval_top_k}, fixed for every seed)")

    def retrieve(self, query: str, k: int) -> List[int]:
        """The k best passage ids in descending score order; raises on an empty index."""
        if self.emb.shape[0] == 0:
            raise IndexError("retrieval index is empty; the corpus kept no passages")
        q = self.embedder.encode([BGE_QUERY_INSTRUCTION + query])[0]
        scores = self.emb @ q
        k = max(1, min(int(k), len(scores)))
        top = np.argpartition(-scores, k - 1)[:k]
        return [int(x) for x in top[np.argsort(-scores[top])]]

    def retrieve_for_topic(self, topic: str) -> int:
        """The single best passage id at cfg.retrieval_top_k; the depth never varies."""
        return self.retrieve(topic, self.cfg.retrieval_top_k)[0]


# ============================================================== ITEM 33: calibration
class JudgeCalibrator:
    """Validates the NLI judge and, if it fails, the LLM judge on labelled gold pairs
    stratified by lexical-overlap decile.

    The calibration pairs come from SciQ VALIDATION while every evaluated item comes from the
    TEST split, so no item that calibrates the judge is ever an item the judge then scores.

    The support floor and the precision verdict live in analysis.py and are called from here, so
    there is ONE implementation of the rule that decides which judge measures the study's rates
    and the offline suite -- which may not import this module -- can drive it directly."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg

    def min_top_decile_pairs(self, n_pairs: Optional[int] = None) -> int:
        """The support floor the top overlap decile must clear, from analysis."""
        from analysis import calibration_support_floor

        return calibration_support_floor(n_pairs, self.cfg.n_overlap_deciles,
                                         int(getattr(self.cfg, "judge_min_top_decile_pairs", 5)))

    def _metrics(self, scores: np.ndarray, calib: CalibrationSet, name: str) -> Dict[str, Any]:
        """Every metric is a finite float or None; an empty denominator NEVER becomes 0.0.

        A spurious 0.0 in the top decile fails the precision gate, promotes the other judge and
        relabels every rate in the results -- so an unmeasured cell must be distinguishable
        from a measured failure. An under-filled top decile is recorded and the run continues:
        raising here would discard the models, corpus, index and pilot after all were paid for.

        The verdict itself is analysis.calibration_precision_verdict; the import is function-local
        so this module's import list is unchanged, the same way data.load_hf_dataset pulls in a
        name it needs at one call site."""
        from analysis import calibration_precision_verdict

        thr = self.cfg.entail_threshold
        y = calib.labels.astype(int)
        pred = (np.asarray(scores) >= thr).astype(int)
        n_dec = int(self.cfg.n_overlap_deciles)
        prec: List[Optional[float]] = []
        rec: List[Optional[float]] = []
        n_pos: List[int] = []
        n_all: List[int] = []
        n_pred_pos: List[int] = []
        for d in range(n_dec):
            m = calib.decile == d
            tp = int(((pred == 1) & (y == 1) & m).sum())
            fp = int(((pred == 1) & (y == 0) & m).sum())
            fn = int(((pred == 0) & (y == 1) & m).sum())
            prec.append(_ratio(tp, tp + fp))       # None, not 0.0, when nothing was predicted
            rec.append(_ratio(tp, tp + fn))
            n_pos.append(int((y[m] == 1).sum()))
            n_all.append(int(m.sum()))
            n_pred_pos.append(tp + fp)

        n_top = n_all[n_dec - 1]
        floor = self.min_top_decile_pairs(len(y))
        verdict = calibration_precision_verdict(
            n_top_decile_pairs=n_top, min_top_decile_pairs=floor, precision=prec[n_dec - 1],
            gate=self.cfg.judge_precision_gate, n_pairs=len(y),
            n_pred_pos=n_pred_pos[n_dec - 1])
        below_floor = bool(verdict["top_decile_below_floor"])
        top = verdict["judge_precision_top_overlap_decile"]
        passes = bool(verdict["passes"])
        reason = str(verdict["passes_reason"])

        tp_all = int(((pred == 1) & (y == 1)).sum())
        fp_all = int(((pred == 1) & (y == 0)).sum())
        fn_all = int(((pred == 0) & (y == 1)).sum())

        # robustness view: drop the top-5%-overlap negatives, which may be gold distractors the
        # corpus actually supports (the very phenomenon this study measures)
        neg = np.where(y == 0)[0]
        cut = float(np.quantile(calib.overlap[neg], 0.95)) if len(neg) else 1.0
        keep = ~((y == 0) & (calib.overlap >= cut))
        m9 = (calib.decile == n_dec - 1) & keep
        tp9 = int(((pred == 1) & (y == 1) & m9).sum())
        fp9 = int(((pred == 1) & (y == 0) & m9).sum())
        trimmed = None if below_floor else _ratio(tp9, tp9 + fp9)

        idx9 = np.where(calib.decile == n_dec - 1)[0][:30]
        spot = [{"passage": calib.premises[i][:160], "stem": calib.stems[i],
                 "option": calib.options[i], "label": int(y[i]), "score": float(scores[i])}
                for i in idx9]
        return {
            "judge": name,
            "judge_precision_top_overlap_decile": top,
            "precision_top_decile_trimmed": trimmed,
            "precision_by_decile": prec,
            "recall_by_decile": rec,
            "n_by_decile": n_all, "n_pos_by_decile": n_pos, "n_pred_pos_by_decile": n_pred_pos,
            "n_top_decile_pairs": int(n_top),
            "min_top_decile_pairs": int(floor),
            "top_decile_below_floor": below_floor,
            "precision_overall": _ratio(tp_all, tp_all + fp_all),
            "recall_overall": _ratio(tp_all, tp_all + fn_all),
            "auroc": _auroc(y, np.asarray(scores)),
            "judge_r2_entail_on_overlap": _r2_on_overlap(calib.overlap, np.asarray(scores)),
            "n_pairs": int(len(y)),
            "entail_threshold": float(thr),
            "precision_gate": float(self.cfg.judge_precision_gate),
            "passes": passes, "passes_reason": reason,
            "spot_check_top_decile": spot,
        }

    def run_nli(self, nli: NLIJudge, calib: CalibrationSet) -> Dict[str, Any]:
        hyps = [HYPOTHESIS_TEMPLATE.format(stem=s.rstrip("?"), option=o)
                for s, o in zip(calib.stems, calib.options)]
        probs = nli.score_pairs(list(calib.premises), hyps)
        return self._metrics(probs[:, 1], calib, "nli")

    def run_llm(self, llm: GeneratorLLM, calib: CalibrationSet) -> Dict[str, Any]:
        prompts = [PROMPT_LLM_JUDGE.format(passage=p[: self.cfg.passage_char_cap], stem=s, option=o)
                   for p, s, o in zip(calib.premises, calib.stems, calib.options)]
        return self._metrics(llm.p_yes_batch(prompts), calib, "llm")

    def select_evaluator(self, nli: NLIJudge, llm: GeneratorLLM,
                         calib: CalibrationSet) -> Tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]:
        """The evaluator name with both calibration records; never raises on an undefined cell.

        An under-filled top decile is recorded with precision None and passes False, and the run
        continues to the second judge -- the diagnostic reaches the payload instead of ending a
        run that already paid for every model."""

        def txt(rec: Dict[str, Any], key: str) -> str:
            v = rec.get(key)
            return "NA" if v is None else f"{float(v):.3f}"

        r_nli = self.run_nli(nli, calib)
        _log(f"JUDGE nli precision_top_decile={txt(r_nli, 'judge_precision_top_overlap_decile')} "
             f"auroc={txt(r_nli, 'auroc')} "
             f"r2_overlap={txt(r_nli, 'judge_r2_entail_on_overlap')} "
             f"n_top_decile={r_nli['n_top_decile_pairs']}/{r_nli['min_top_decile_pairs']} "
             f"passes={r_nli['passes']} ({r_nli['passes_reason']})")
        if r_nli["passes"]:
            return "nli", r_nli, None
        r_llm = self.run_llm(llm, calib)
        _log(f"JUDGE llm precision_top_decile={txt(r_llm, 'judge_precision_top_overlap_decile')} "
             f"passes={r_llm['passes']} ({r_llm['passes_reason']})")
        if r_llm["passes"]:
            return "llm", r_nli, r_llm
        return "none", r_nli, r_llm


def load_all_models(cfg: Any) -> Tuple[GeneratorLLM, NLIJudge, Embedder]:
    """The three frozen components, with the NLI judge on GPU only when free VRAM allows.

    Raises on a failed checkpoint load rather than returning an untrained stand-in: a fabricated
    model would produce numbers that look like measurements."""
    llm = GeneratorLLM(cfg)
    nli = NLIJudge(cfg)
    free = free_vram_bytes()
    if cfg.device == "cuda" and free is not None and free >= cfg.nli_gpu_min_free_bytes:
        nli.to_device("cuda")
    else:
        reason = "no CUDA device visible" if free is None else f"free VRAM {_vram_text()}"
        _log(f"[nli] {reason}; the NLI judge stays on CPU (float32)")
    embedder = Embedder(cfg)
    nli.maybe_offload()
    _log(f"[models] loaded; nli device={nli.device}; free VRAM={_vram_text()}")
    return llm, nli, embedder


def model_pin(role: str) -> Optional[str]:
    """The pinned Hub revision declared for ROLE, read under the key the table declares.

    MODEL_REVISIONS is keyed by role ("generator", "nli", "embedder"). Reading it with a repo id
    missed on every lookup and returned None, so `revision=` was never passed: a pinned run loaded
    the repo's default branch while the payload recorded pinned_revision: null, a silent
    substitution of the very weights the results are attributed to. An undeclared role RAISES,
    because a typo must never be indistinguishable from "no pin set"."""
    if role not in MODEL_REVISIONS:
        raise KeyError(f"model_pin: '{role}' is not a declared model role "
                       f"{sorted(MODEL_REVISIONS)}; a pin can only be read under a declared key")
    return MODEL_REVISIONS[role]


def _assert_pin_honoured(role: str, repo_id: str, lk: Dict[str, Any]) -> None:
    """Return None when the load keywords carry exactly the pin declared for role; raise otherwise.

    The failure this guards is silent by nature: from_pretrained without `revision=` succeeds, and
    the run then reports numbers produced by weights nobody chose. Checked at the door of every
    load, so a future edit that filters the keyword dict cannot quietly drop the pin."""
    rev = model_pin(role)
    if rev and lk.get("revision") != rev:
        raise RuntimeError(f"{role}: MODEL_REVISIONS pins {repo_id} at revision {rev!r} but the "
                           f"load keywords carry revision={lk.get('revision')!r}; the run would "
                           f"load weights the payload does not describe")
    if rev is None and "revision" in lk:
        raise RuntimeError(f"{role}: the load keywords carry revision={lk.get('revision')!r} but "
                           f"MODEL_REVISIONS declares no pin for {repo_id}; the recorded and the "
                           f"requested revision would disagree")
