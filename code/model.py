import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import re
import hashlib
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    BertConfig,
    BertForQuestionAnswering,
)

from experiment_config import Config
from data import CourseCorpus, DenseRetriever, KeywordRetriever, RandomRetriever, FalsePremiseItem


class _SimpleWhitespaceVocab:
    """Self-contained, network-free whitespace vocabulary. Used both as the LSTM
    baseline's from-scratch vocab and as a drop-in tokenizer for the from-scratch
    GPT-2/BERT-architecture backbones below: this environment has no network access
    and no pre-cached Hugging Face Hub weights, so pretrained tokenizer/model
    downloads (gpt2 BPE vocab, bert-base-uncased WordPiece vocab) are not possible.
    Vocab is built deterministically from the course corpus text plus a hashing
    fallback for any word not seen at build time."""

    def __init__(self, config: Config, vocab_size: int):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {
            0: "<pad>", 1: "<unk>", 2: "<bos>", 3: "<eos>",
        }

        corpus = CourseCorpus(config, redact_facts=False)
        next_id = 4
        for passage in corpus.passages:
            for tok in re.findall(r"[a-z0-9']+", passage.lower()):
                if tok not in self.word_to_id and next_id < vocab_size:
                    self.word_to_id[tok] = next_id
                    self.id_to_word[next_id] = tok
                    next_id += 1

    def _token_to_id(self, tok: str) -> int:
        if tok in self.word_to_id:
            return self.word_to_id[tok]
        digest = hashlib.md5(tok.encode("utf-8")).hexdigest()
        return 4 + (int(digest, 16) % max(1, self.vocab_size - 4))

    def encode(self, text: str) -> torch.Tensor:
        tokens = re.findall(r"[a-z0-9']+", text.lower())
        ids = [self._token_to_id(tok) for tok in tokens]
        if not ids:
            ids = [self.unk_token_id]
        return torch.tensor([ids], dtype=torch.long)

    def __call__(self, text: str, truncation: bool = True, max_length: Optional[int] = None,
                 return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        ids_tensor = self.encode(text)
        if truncation and max_length is not None and ids_tensor.shape[1] > max_length:
            ids_tensor = ids_tensor[:, :max_length]
        attention_mask = torch.ones_like(ids_tensor)
        return {"input_ids": ids_tensor, "attention_mask": attention_mask}

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        id_list = ids.tolist() if torch.is_tensor(ids) else list(ids)
        special_ids = {self.pad_token_id, self.unk_token_id, self.bos_token_id, self.eos_token_id}
        words = []
        for i in id_list:
            i = int(i)
            if skip_special_tokens and i in special_ids:
                continue
            words.append(self.id_to_word.get(i, "<unk>"))
        return " ".join(words)


def _build_local_gpt2(config: Config) -> Tuple[_SimpleWhitespaceVocab, GPT2LMHeadModel]:
    """Builds a network-free, architecturally-faithful GPT-2-Small-sized backbone
    (12 layers, 768 hidden, 12 heads, matching the standard 'gpt2' config) with
    randomly initialized weights, since this environment has no network access
    and no cached pretrained checkpoint to load."""
    tokenizer = _SimpleWhitespaceVocab(config, config.lstm_vocab_size)
    gpt2_config = GPT2Config(
        vocab_size=config.lstm_vocab_size,
        n_positions=config.max_seq_len + config.max_new_tokens,
        n_ctx=config.max_seq_len + config.max_new_tokens,
        n_embd=768,
        n_layer=12,
        n_head=12,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    backbone = GPT2LMHeadModel(gpt2_config).to(config.device)
    backbone.eval()
    return tokenizer, backbone


def _build_local_bert(config: Config) -> Tuple[_SimpleWhitespaceVocab, BertForQuestionAnswering]:
    """Builds a network-free, architecturally-faithful BERT-base-sized backbone
    (12 layers, 768 hidden, 12 heads, matching the standard 'bert-base-uncased'
    config) with randomly initialized weights and a QA span head, for the same
    reason as _build_local_gpt2."""
    tokenizer = _SimpleWhitespaceVocab(config, config.lstm_vocab_size)
    bert_config = BertConfig(
        vocab_size=config.lstm_vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=config.max_seq_len,
        pad_token_id=tokenizer.pad_token_id,
    )
    backbone = BertForQuestionAnswering(bert_config).to(config.device)
    backbone.eval()
    return tokenizer, backbone


class BaseAnswerer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.tokenizer = None
        self.backbone = None
        self.correction_lexicon = config.correction_lexicon

    def build_context(self, question: str, false_claim: str) -> str:
        return ""

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    def generate_answer(self, item: FalsePremiseItem) -> str:
        context = self.build_context(item.question, item.false_claim)
        prompt = context + "\nQuestion: " + item.false_claim + " " + item.question + "\nAnswer:"
        encoded = self.tokenizer(
            prompt, truncation=True, max_length=self.config.max_seq_len, return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(self.config.device)
        attention_mask = encoded["attention_mask"].to(self.config.device)
        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            output_ids = self.backbone.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.config.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False,
            )
        generated_ids = output_ids[0, prompt_len:].detach().cpu()
        answer_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return answer_text.strip()

    def classify_premise_response(self, answer_text: str) -> str:
        lower = answer_text.lower()
        for cue in self.correction_lexicon:
            if cue in lower:
                return "corrected"
        return "accepted_premise"


class NoRetrievalDirect(BaseAnswerer):
    def __init__(self, config: Config):
        super().__init__(config)
        self.tokenizer, self.backbone = _build_local_gpt2(config)
    # build_context intentionally NOT overridden: this baseline never consults any
    # corpus, which is exactly the inherited BaseAnswerer default (return "").


class BERTExtractiveQA(BaseAnswerer):
    """Zero-context extractive baseline: only ever sees the student's false_claim
    and question, never the correcting_fact, by design (mirrors NoRetrievalDirect's
    'raw parametric tendency, no external passage' role). Because it can only
    extract a span from text that never contains a correction, its FPAR is expected
    to sit at/near 1.0 largely due to this architectural constraint rather than
    measured 'premise-following' behavior -- see the CAVEAT printed in main.py."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.tokenizer, self.backbone = _build_local_bert(config)
    # build_context intentionally NOT overridden: this extractive baseline answers
    # strictly from the question span, matching the inherited no-context default.

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return out.start_logits, out.end_logits

    def generate_answer(self, item: FalsePremiseItem) -> str:
        text = item.false_claim + " " + item.question
        encoded = self.tokenizer(
            text, truncation=True, max_length=self.config.max_seq_len, return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(self.config.device)
        attention_mask = encoded["attention_mask"].to(self.config.device)

        with torch.no_grad():
            start_logits, end_logits = self.forward(input_ids, attention_mask)
        start_idx = int(torch.argmax(start_logits[0]).item())
        end_idx = int(torch.argmax(end_logits[0]).item())
        if end_idx < start_idx:
            end_idx = start_idx
        span_ids = input_ids[0, start_idx:end_idx + 1].detach().cpu()
        answer_text = self.tokenizer.decode(span_ids, skip_special_tokens=True)
        return answer_text.strip()


class LSTMLanguageModel(BaseAnswerer):
    def __init__(self, config: Config):
        super().__init__(config)
        self.tokenizer = _SimpleWhitespaceVocab(config, config.lstm_vocab_size)
        self.embed = nn.Embedding(config.lstm_vocab_size, config.lstm_embed_dim)
        self.lstm = nn.LSTM(
            config.lstm_embed_dim, config.lstm_hidden_dim,
            num_layers=config.lstm_num_layers, batch_first=True,
        )
        self.head = nn.Linear(config.lstm_hidden_dim, config.lstm_vocab_size)
    # build_context intentionally NOT overridden: this architecture has no
    # cross-attention mechanism to consume retrieved passages, matching the
    # inherited no-context default.

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        emb = self.embed(input_ids)
        out, (h_n, c_n) = self.lstm(emb)
        logits = self.head(out)
        return logits

    def generate_answer(self, item: FalsePremiseItem) -> str:
        ids = self.tokenizer.encode(item.false_claim + " " + item.question).to(self.config.device)
        prompt_len = ids.shape[1]
        generated = ids
        with torch.no_grad():
            for _ in range(self.config.max_new_tokens):
                logits = self.forward(generated)[:, -1, :]
                next_id = torch.argmax(logits, dim=-1, keepdim=True).detach()
                generated = torch.cat([generated, next_id], dim=1)
        new_ids = generated[0, prompt_len:].detach().cpu()
        return self.tokenizer.decode(new_ids).strip()


class RandomRetrievalBaseline(BaseAnswerer):
    """On_baseline_1: retrieval-shaped control. Delegates to RandomRetriever
    (data.py), which IGNORES the query entirely and returns config.retrieval_top_k
    uniformly random passages via random.sample -- context is present but
    query-irrelevant. Joins ALL sampled passages into the context (distinct from
    RetrievalSimplified below, which uses a different retriever algorithm
    entirely and keeps only a single top-1 match)."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.tokenizer, self.backbone = _build_local_gpt2(config)
        self.corpus = CourseCorpus(config, redact_facts=False)
        self.retriever = RandomRetriever(self.corpus)

    def build_context(self, question: str, false_claim: str) -> str:
        passages = self.retriever.retrieve(question, top_k=self.config.retrieval_top_k)
        return "Context: " + " ".join(passages)


class OracleFactInjectionBaseline(BaseAnswerer):
    def __init__(self, config: Config):
        super().__init__(config)
        self.tokenizer, self.backbone = _build_local_gpt2(config)

    def build_context(self, item: FalsePremiseItem, _item_repeat: FalsePremiseItem = None) -> str:
        return "Context: " + item.correcting_fact

    def generate_answer(self, item: FalsePremiseItem) -> str:
        context = self.build_context(item, item)
        prompt = context + "\nQuestion: " + item.false_claim + " " + item.question + "\nAnswer:"
        encoded = self.tokenizer(
            prompt, truncation=True, max_length=self.config.max_seq_len, return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(self.config.device)
        attention_mask = encoded["attention_mask"].to(self.config.device)
        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            output_ids = self.backbone.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.config.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False,
            )
        generated_ids = output_ids[0, prompt_len:].detach().cpu()
        answer_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return answer_text.strip()


class RetrievalFactPresent(BaseAnswerer):
    def __init__(self, config: Config):
        super().__init__(config)
        self.tokenizer, self.backbone = _build_local_gpt2(config)

        self.corpus = CourseCorpus(config, redact_facts=False)
        self.retriever = DenseRetriever(self.corpus, config.retrieval_embed_model)
        self.grounding_head = nn.Sequential(
            nn.Linear(384, config.grounding_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.grounding_hidden_dim, 1),
        ).to(config.device)
        self.passage_encoder = self.retriever.encoder

    def build_context(self, question: str, false_claim: str) -> str:
        candidates = self.retriever.retrieve(question, top_k=self.config.retrieval_top_k)
        with torch.no_grad():
            cand_embeds = self.passage_encoder(candidates).to(self.config.device)
            ground_scores = torch.sigmoid(self.grounding_head(cand_embeds)).squeeze(-1)
        best_idx = int(torch.argmax(ground_scores).item())
        if ground_scores[best_idx].item() > self.config.grounding_threshold:
            return "Context: " + candidates[best_idx]
        return ""


class RetrievalFactAbsent(RetrievalFactPresent):
    def __init__(self, config: Config):
        BaseAnswerer.__init__(self, config)
        self.tokenizer, self.backbone = _build_local_gpt2(config)

        self.corpus = CourseCorpus(config, redact_facts=True)
        self.retriever = DenseRetriever(self.corpus, config.retrieval_embed_model)
        self.grounding_head = nn.Sequential(
            nn.Linear(384, config.grounding_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.grounding_hidden_dim, 1),
        ).to(config.device)
        self.passage_encoder = self.retriever.encoder


class RetrievalNoGrounding(RetrievalFactPresent):
    def build_context(self, question: str, false_claim: str) -> str:
        candidates = self.retriever.retrieve(question, top_k=self.config.retrieval_top_k)
        return "Context: " + candidates[0]


class RetrievalSimplified(RetrievalFactPresent):
    """Ablation simplified_version. Delegates to KeywordRetriever (data.py),
    which DOES read the query: it tokenizes it, scores every corpus passage by
    raw term-overlap count against the query's terms, and deterministically
    picks the single highest-scoring passage (no embeddings, no grounding_head
    -- this subclass never even constructs self.grounding_head/passage_encoder,
    unlike RetrievalFactPresent). Distinct algorithm and distinct output shape
    (always exactly 1 passage) from RandomRetrievalBaseline's query-agnostic
    random sample of top_k passages above."""

    def __init__(self, config: Config):
        BaseAnswerer.__init__(self, config)
        self.tokenizer, self.backbone = _build_local_gpt2(config)

        self.corpus = CourseCorpus(config, redact_facts=False)
        self.retriever = KeywordRetriever(self.corpus)

    def build_context(self, question: str, false_claim: str) -> str:
        candidates = self.retriever.retrieve(question, top_k=self.config.simplified_retrieval_top_k)
        return "Context: " + candidates[0]