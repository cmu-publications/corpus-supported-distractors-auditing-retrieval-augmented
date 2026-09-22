# Corpus-Supported Distractors: Auditing Retrieval-Augmented Quiz Generation Against the Stem

Code, data and results for this paper.

## Layout

| Path | What it holds |
|---|---|
| `paper/` | The manuscript, its LaTeX source and its bibliography. (6 files, 516.2 KB) |
| `code/` | The experiment code. (12 files, 490.5 KB) |
| `data/` | The datasets the experiment reads, with `MANIFEST.md` listing them and their hashes. (8 files, 8.8 MB) |
| `results/` | The per-seed measurements, the results table and the figures in the paper. (13 files, 1.2 MB) |

## The experiment

Conditions compared: `unfiltered_grounded_pool`, `ungrounded_topic_only_control`, `per_option_entailment_filter`, `llm_judge_faithfulness_filter`, `contradiction_polarity_item_filter`, `not_entail_polarity_item_filter`, `overgenerate_rerank_by_contradiction`, `overgenerate_rerank_by_key_similarity`, `null_random_drop`, `positive_control_planted`.

Seeds: 0, 1, 2.

## Data

The datasets are in `data/`, so the experiment needs no download.

| Dataset | Size |
|---|---|
| `openbookqa` | 1.5 MB |
| `sciq` | 7.3 MB |

## Running the experiment

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows
set RC_DATA_ROOT=%CD%\data
python code/setup.py                                # prepares the corpus
python code/main.py
```

A run writes its measurements in the same shape as `results/measurements.json`.

