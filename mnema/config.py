# mnema/config.py
"""
MNEMAConfig — all tunable parameters in one place.
Pass to MNEMA() to override any default.
"""

from dataclasses import dataclass, field


@dataclass
class MNEMAConfig:
    # ── Identity ──────────────────────────────────────────────────────────────
    name: str = "MNEMA"

    # ── Storage ───────────────────────────────────────────────────────────────
    storage_dir: str = "~/.mnema"        # base dir — profile subdirs created here

    # ── Generation ────────────────────────────────────────────────────────────
    show_thinking:        bool  = False   # monologue off by default
    thinking_tokens:      int   = 250
    response_tokens:      int   = 400
    history_budget:       int   = 1200   # token budget for conversation history
    history_max_turns:    int   = 40     # hard cap on conversation turns kept

    # ── Memory ────────────────────────────────────────────────────────────────
    top_k_memories:       int   = 5
    min_importance:       float = 0.4    # below this → no memory stored
    embedding_model:      str   = "all-MiniLM-L6-v2"
    contradiction_threshold: float = 0.72

    # ── Decay ─────────────────────────────────────────────────────────────────
    decay_interval_hours: float = 6.0
    min_strength:         float = 0.05   # archive below this

    # ── Consolidation ─────────────────────────────────────────────────────────
    consolidation_trigger_count: int   = 15
    consolidation_epochs:        int   = 2
    consolidation_interval_min:  float = 30.0
    ewc_lambda:                  float = 5000.0

    # ── LoRA ──────────────────────────────────────────────────────────────────
    lora_r:               int   = 8
    lora_alpha:           int   = 16
    lora_dropout:         float = 0.05
    lora_target_modules:  list  = field(
        default_factory=lambda: ["q_proj", "v_proj", "out_proj"]
    )
    lora_layers_to_transform: list = field(
        default_factory=lambda: [2, 5, 8, 10, 12, 14]
    )
    load_in_4bit:         bool  = True

    # ── Hardware ──────────────────────────────────────────────────────────────
    hardware_check_every: int   = 3      # turns between hardware checks
    vram_full_gb:         float = 1.2
    vram_reduced_gb:      float = 0.7
    vram_minimal_gb:      float = 0.4

    # ── Goals ─────────────────────────────────────────────────────────────────
    goal_weights: dict = field(default_factory=lambda: {
        "minimize_corrections": 0.30,
        "match_tone":           0.25,
        "be_concise":           0.15,
        "remember_context":     0.20,
        "build_trust":          0.10,
    })

    # ── Importance weights per memory type ───────────────────────────────────
    importance_weights: dict = field(default_factory=lambda: {
        "correction": 1.0,
        "preference": 0.8,
        "fact":       0.5,
        "casual":     0.1,
    })
