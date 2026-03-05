# mnema

**MNEMA Architecture — Cognitive AI for frozen language models.**

A Python library that wraps any HuggingFace CausalLM with a complete cognitive architecture: relational memory, goal-driven retrieval, behavioral state evolution, and self-modeling — all without touching the base model weights.

```python
pip install mnema
```

---

## Quick Start

```python
from mnema import MNEMA, MNEMAConfig
from mnema.model import load_model

# Load any HuggingFace CausalLM
model, tokenizer = load_model("LiquidAI/LFM2.5-1.2B-Instruct")

# Attach the cognitive architecture
brain = MNEMA(model=model, tokenizer=tokenizer, profile="ken")

# Chat
response = brain.chat("Hi, my name is Ken")
print(response)

# Stream
for token in brain.stream("What do you remember about me?"):
    print(token, end="", flush=True)

# Get monologue + response
response, monologue = brain.chat("Tell me something", think=True)

# Clean shutdown
brain.close()
```

Memory persists automatically at `~/.mnema/user/` across sessions.

---

## What MNEMA Gives Your Model

| Without MNEMA | With MNEMA |
|---|---|
| Stateless — forgets everything | Remembers across sessions |
| Flat retrieval | Relational memory graph with contradiction detection |
| No goals | 5 explicit goals shape what gets remembered |
| Fixed personality | 8-axis behavioral state evolves over time |
| No self-awareness | Tracks own correction rate and confidence |
| Hardware-agnostic | Auto-adjusts for available VRAM |

---

## The Four Systems

**Relational Memory Graph** — memories are nodes with typed edges (temporal, refines, contradicts, causal). Contradictions are auto-detected and resolved. Multi-hop traversal expands context beyond direct matches.

**Goal & Utility Layer** — five explicit goals (minimize corrections, match tone, be concise, remember context, build trust) re-rank retrieved memories by purpose, not just similarity.

**Adaptive State Core (ASC)** — an 8-axis behavioral state vector (curiosity, warmth, formality, verbosity, confidence, playfulness, depth, caution) that evolves every turn via signal equations. No backpropagation. Personality accumulates over months.

**Meta-Cognition** — tracks correction rate, confidence per memory type, and reliability over time. Injects self-awareness into the thinking process: *"I've been corrected about this before — I should hedge."*

---

## Multiple Profiles

```python
# Fully isolated memory per user or project
ken   = MNEMA(model=model, tokenizer=tokenizer, profile="ken")
alice = MNEMA(model=model, tokenizer=tokenizer, profile="alice")
# ~/.mnema/ken/   and   ~/.mnema/alice/   are independent
```

---

## Configuration

```python
from mnema import MNEMA, MNEMAConfig

config = MNEMAConfig(
    name="MNEMA",
    show_thinking=False,
    top_k_memories=5,
    decay_interval_hours=6.0,
    consolidation_trigger_count=15,
    lora_r=8,
    lora_alpha=16,
    load_in_4bit=True,
)

brain = MNEMA(model=model, tokenizer=tokenizer, config=config)
```

---

## Context Manager

```python
with MNEMA(model=model, tokenizer=tokenizer) as brain:
    response = brain.chat("Hello")
# scheduler stopped automatically on exit
```

---

## Inspection

```python
brain.memory    # RelationalMemoryGraph
brain.asc       # AdaptiveStateCore
brain.goals     # GoalUtilityLayer
brain.metacog   # MetaCognition
brain.hw        # HardwareMonitor

print(brain.status())   # full system status
brain.clear()           # wipe memory for this profile
```

---

## Supported Models

Any `AutoModelForCausalLM` HuggingFace model. Tested on:

| Model | VRAM (4-bit) |
|---|---|
| `LiquidAI/LFM2.5-1.2B-Instruct` | ~0.9GB |
| `LiquidAI/LFM2.5-3B-Instruct` | ~2.0GB |
| `microsoft/Phi-3-mini-4k-instruct` | ~2.2GB |
| `google/gemma-2-2b-it` | ~1.5GB |
| `Qwen/Qwen2.5-3B-Instruct` | ~2.0GB |
| `meta-llama/Llama-3.2-3B-Instruct` | ~2.0GB |

When switching models, update `lora_target_modules` in `MNEMAConfig` to match the new architecture's attention layer names.

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8 or 12.1 (CPU-only works, slower)
- 4GB+ VRAM recommended

---

## Research

Built on the MNEMA Architecture. Read the full research paper:
[MNEMA-v2-research.pdf](https://github.com/Ainix-dev/Project-MNEMA/blob/main/MNEMA-v2-research.pdf)

Source repository: [github.com/Ainix-dev/Project-MNEMA](https://github.com/Ainix-dev/Project-MNEMA)

---

## Citation

```bibtex
@software{mnema2026,
  author  = {Ainix-dev and {Claude Sonnet 4.6}},
  title   = {MNEMA Architecture},
  year    = {2026},
  url     = {https://github.com/Ainix-dev/mnema}
}
```

---

## License

Apache License 2.0
