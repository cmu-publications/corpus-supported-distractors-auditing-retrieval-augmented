import re
import time
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from experiment_config import Config
from data import FalsePremiseItem
from model import LSTMLanguageModel, RetrievalFactPresent, BaseAnswerer


def _build_lstm_training_chunks(model: LSTMLanguageModel, corpus_texts: List[str], max_seq_len: int) -> torch.Tensor:
    all_ids: List[int] = []
    for text in corpus_texts:
        ids = model.tokenizer.encode(text)
        all_ids.extend(ids[0].tolist())

    num_chunks = len(all_ids) // max_seq_len
    pad_id = None
    padded_ids = None
    chunk_tensor = None
    trimmed_ids = None
    if num_chunks == 0:
        pad_id = model.tokenizer.pad_token_id
        padded_ids = all_ids + [pad_id] * (max_seq_len - len(all_ids))
        chunk_tensor = torch.tensor([padded_ids], dtype=torch.long)
    else:
        trimmed_ids = all_ids[:num_chunks * max_seq_len]
        chunk_tensor = torch.tensor(trimmed_ids, dtype=torch.long).view(num_chunks, max_seq_len)

    return chunk_tensor


def train_lstm_baseline(model: LSTMLanguageModel, corpus_texts: List[str], config: Config, device) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lstm_lr)
    ids_tensor = _build_lstm_training_chunks(model, corpus_texts, config.max_seq_len)
    loader = DataLoader(TensorDataset(ids_tensor), batch_size=config.batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()

    model.train()
    final_epoch_avg_loss = 0.0
    for epoch in range(config.lstm_epochs):
        epoch_loss_sum = 0.0
        epoch_batches = 0
        for batch in loader:
            batch_ids = batch[0].to(device)
            inputs, targets = batch_ids[:, :-1], batch_ids[:, 1:]
            logits = model.forward(inputs)
            vocab_size = logits.shape[-1]
            loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))

            if torch.isnan(loss) or loss.item() > 100:
                print(f"FAIL: NaN/divergence detected in train_lstm_baseline epoch={epoch}, skipping batch")
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss_sum += loss.item()
            epoch_batches += 1

        final_epoch_avg_loss = epoch_loss_sum / max(1, epoch_batches)
        print(f"train_lstm_baseline epoch={epoch} avg_loss={final_epoch_avg_loss:.4f}")

    model.eval()
    return final_epoch_avg_loss


def train_grounding_head(model: RetrievalFactPresent, texts: List[str], labels: List[int], config: Config, device) -> float:
    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(model.grounding_head.parameters(), lr=config.grounding_head_lr)
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        embeds = model.passage_encoder(texts).to(device).detach()
    labels_t = torch.tensor(labels, dtype=torch.float32).to(device)

    loader = DataLoader(TensorDataset(embeds, labels_t), batch_size=config.batch_size, shuffle=True)

    model.grounding_head.train()
    final_epoch_avg_loss = 0.0
    for epoch in range(config.grounding_head_epochs):
        epoch_loss_sum = 0.0
        epoch_batches = 0
        for x_batch, y_batch in loader:
            logit = model.grounding_head(x_batch).squeeze(-1)
            loss = criterion(logit, y_batch)

            if torch.isnan(loss) or loss.item() > 100:
                print(f"FAIL: NaN/divergence detected in train_grounding_head epoch={epoch}, skipping batch")
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss_sum += loss.item()
            epoch_batches += 1

        final_epoch_avg_loss = epoch_loss_sum / max(1, epoch_batches)
        print(f"train_grounding_head epoch={epoch} avg_loss={final_epoch_avg_loss:.4f}")

    model.grounding_head.eval()
    return final_epoch_avg_loss


_GENERIC_GROUNDING_STOPWORDS = {"o"}  # bare Big-O notation marker; carries no
# discriminating content since it appears in every correct_value AND every
# distractor in this domain ('O(1)', 'O(n)', 'O(n log n)', ...).


def _is_grounded(gold_answer: str, answer_text: str) -> bool:
    """Overlap-based grounding check requiring co-occurrence of the gold
    answer's DISTINGUISHING content words in the generated text.

    Two refinements over a naive token-overlap ratio:
    1. The generic Big-O marker 'o' is dropped from both token sets -- matching
       it carries zero signal about whether the correcting fact specifically
       was reproduced (see _GENERIC_GROUNDING_STOPWORDS).
    2. For gold answers left with <=2 tokens after that removal (e.g.
       'O(1)' -> {'1'}, 'O(n log n)' -> {'n','log'}), ALL of them must be
       present, not just half -- with such small token sets a >=50% ratio
       degenerates to "any single token", which is trivially satisfiable by
       chance from an undertrained/randomly-initialized decoder and is not a
       meaningful grounding signal. Larger token sets still use a >=50%
       (ceiling) requirement.
    """
    gold_tokens = set(re.findall(r"[a-z0-9]+", gold_answer.lower())) - _GENERIC_GROUNDING_STOPWORDS
    if not gold_tokens:
        return False
    answer_tokens = set(re.findall(r"[a-z0-9]+", answer_text.lower())) - _GENERIC_GROUNDING_STOPWORDS
    overlap = gold_tokens & answer_tokens

    required = None
    if len(gold_tokens) <= 2:
        required = len(gold_tokens)
    else:
        required = -(-len(gold_tokens) // 2)  # ceil(len(gold_tokens) / 2) via integer math

    return len(overlap) >= required


def evaluate(model: BaseAnswerer, eval_items: List[FalsePremiseItem], device) -> Dict[str, float]:
    model.eval()
    n_accepted = 0
    n_corrected = 0
    n_grounded_and_used = 0

    for item in eval_items:
        with torch.no_grad():
            answer_text = model.generate_answer(item)
        label = model.classify_premise_response(answer_text)
        if label == "accepted_premise":
            n_accepted += 1
        else:
            n_corrected += 1
        if _is_grounded(item.gold_answer, answer_text):
            n_grounded_and_used += 1

    total = len(eval_items)
    primary_metric = n_accepted / total
    secondary_metric = n_grounded_and_used / total

    return {
        "primary_metric": primary_metric,
        "secondary_metric": secondary_metric,
        "n_corrected": n_corrected,
        "n_total": total,
    }


def check_time_budget(start_time: float, config: Config) -> bool:
    elapsed_hours = (time.time() - start_time) / 3600.0
    return elapsed_hours < config.time_budget_hours