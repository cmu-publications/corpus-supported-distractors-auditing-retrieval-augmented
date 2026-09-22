"""
config.py — compatibility shim, and nothing else.

Every setting lives in experiment_config.py, which is named that way because `config` is a
real pip package name: a stale `import config` on a machine where that package is installed
would otherwise bind to something else entirely. This module re-exports the settings so any
surviving `import config` reaches THIS project's arm registry, prompts and resolvers.

# ITEM 50: the shim re-exports and defines nothing.
The arm registry is the live risk. main.py runs TEN arms -- the plan's eight CONDITION_NAMES
plus CONTROL_ARM_NAMES, joined as ALL_ARM_NAMES -- and a stale import that saw only the eight
would iterate an eight-arm world while the harness aborts on an arm set differing from the
expected one. All three names are re-exported here, and no name is defined locally, so the
two modules cannot diverge.

A from-import binds the SAME OBJECT, which is what makes
    config.ALL_ARM_NAMES is experiment_config.ALL_ARM_NAMES
true for every list, dict, function and class below.

model_revision is re-exported beside MODEL_REVISIONS because the pin table is declared by
role while models.py holds only the repo id at the point of the lookup; the resolver is the
one place those two names meet, and a caller that reached this shim but not the resolver
would be back to reading the table under a name it does not carry.

DEVICE_FALLBACK_REASON is deliberately NOT re-exported: detect_device rebinds that module
global at call time (None on success, "Type: message" on failure), and a from-import would
freeze this module's copy at the value it held during import. A caller reading
config.DEVICE_FALLBACK_REASON would then see a stale None while experiment_config named a
real CPU-fallback cause -- exactly the diverging copy this file exists to prevent. Read it
from experiment_config, where it is always current.
"""
from experiment_config import (ALL_ARM_NAMES, ARMS_ALWAYS_RUN, BGE_QUERY_INSTRUCTION,  # noqa: F401
                               CONDITION_NAMES, CONTROL_ARM_NAMES, ENV_SMOKE_AT_IMPORT,
                               ENV_TIME_BUDGET_AT_IMPORT, HYPOTHESIS_TEMPLATE, MODEL_IDS, MODEL_REVISIONS,
                               PROJECT_DATA_DIR, PROJECT_DIR, PROMPT_ANSWER_NO_PASSAGE,
                               PROMPT_ANSWER_WITH_PASSAGE, PROMPT_GROUNDED, PROMPT_LLM_JUDGE, PROMPT_OVERGEN,
                               PROMPT_REPAIR, PROMPT_REWRITE, PROMPT_TOPIC_LABEL, PROMPT_TOPIC_ONLY, SMOKE,
                               TIME_BUDGET_SEC, Config, detect_device, hf_hub_cache_root, model_revision,
                               resolve_data_root, snapshot_cached, snapshot_revision)