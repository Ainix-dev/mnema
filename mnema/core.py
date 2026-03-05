# mnema/core.py
"""
MNEMA — primary entry point.

Usage:
    from mnema import MNEMA

    brain = MNEMA(model=model, tokenizer=tokenizer)
    response = brain.chat("Hi my name is Ken")

    for token in brain.stream("What do you remember?"):
        print(token, end="", flush=True)

    response, monologue = brain.chat("Tell me something", think=True)
    brain.close()
"""

import os
import gc
import torch
from pathlib import Path
from typing import Optional, Generator

from .config import MNEMAConfig
from .memory.graph import RelationalMemoryGraph
from .memory.extractor import MemoryExtractor
from .memory.composer import ContextComposer
from .memory.fade import MultiSpeedDecay
from .cognition.goals import GoalUtilityLayer
from .cognition.metacog import MetaCognition
from .cognition.asc import AdaptiveStateCore
from .system.hardware import HardwareMonitor
from .system.scheduler import MNEMAScheduler
from .model.inference import generate, build_thinking_prompt, build_response_prompt


class MNEMA:
    """
    Cognitive architecture wrapper.
    Attach to any frozen HuggingFace CausalLM model.

    Parameters
    ----------
    model       : loaded HuggingFace model (frozen, with LoRA adapter)
    tokenizer   : corresponding tokenizer
    profile     : name for isolated storage (~/.mnema/<profile>/)
    config      : MNEMAConfig instance — override any default
    """

    def __init__(
        self,
        model,
        tokenizer,
        profile:  str = "default",
        config:   Optional[MNEMAConfig] = None,
    ):
        self.model     = model
        self.tokenizer = tokenizer
        self.profile   = profile
        self.cfg       = config or MNEMAConfig()

        # ── Storage ───────────────────────────────────────────────────────────
        base = Path(self.cfg.storage_dir).expanduser()
        self.storage = base / profile
        self.storage.mkdir(parents=True, exist_ok=True)

        db_path      = str(self.storage / "memory_graph.db")
        chroma_path  = str(self.storage / "chroma")
        adapter_path = str(self.storage / "lora_adapter")

        # ── Core components ───────────────────────────────────────────────────
        self.memory   = RelationalMemoryGraph(db_path, chroma_path, self.cfg)
        self.goals    = GoalUtilityLayer(db_path, self.cfg)
        self.metacog  = MetaCognition(db_path, self.cfg)
        self.asc      = AdaptiveStateCore(db_path, self.cfg)
        self.hw       = HardwareMonitor(db_path, self.cfg)
        self._decay   = MultiSpeedDecay(self.memory, self.cfg)
        self._composer = ContextComposer(self.cfg)
        self._extractor = MemoryExtractor(self.cfg)

        # ── Background scheduler ──────────────────────────────────────────────
        self._scheduler = MNEMAScheduler(self._decay, self.memory, self.cfg)
        self._scheduler.start()

        # ── Session state ─────────────────────────────────────────────────────
        self._history:      list  = []
        self._turn:         int   = 0
        self._adapter_path: str   = adapter_path

        print(f"[MNEMA] Profile: {profile} | Storage: {self.storage}")
        print(f"[MNEMA] {self.hw.display_status()}")

    # ── Primary API ───────────────────────────────────────────────────────────

    def chat(
        self,
        message:  str,
        think:    Optional[bool] = None,
    ) -> str | tuple[str, str]:
        """
        Send a message and receive a response.

        Parameters
        ----------
        message : user message string
        think   : if True, returns (response, monologue) tuple
                  if None, uses config.show_thinking

        Returns
        -------
        str                    if think is False/None
        tuple[str, str]        if think is True — (response, monologue)
        """
        show_thinking = think if think is not None else self.cfg.show_thinking
        self._turn += 1

        # ── Hardware check ────────────────────────────────────────────────────
        if self._turn % self.cfg.hardware_check_every == 0:
            tier_changed = self.hw.update()
            if tier_changed:
                print(f"\n  [Hardware] {tier_changed}\n")

        hw_cfg = self.hw.get_config()

        # ── Signal detection ──────────────────────────────────────────────────
        signals = self.goals.detect_signals(message)

        # ── Memory retrieval ──────────────────────────────────────────────────
        memories = self.memory.retrieve(message, top_k=hw_cfg["top_k_memories"])
        memories = self.goals.tag_with_utility(memories, signals)

        # ── Context composition ───────────────────────────────────────────────
        memory_block = self._composer.compose(memories, hw_cfg["history_budget"])

        # ── ASC update ────────────────────────────────────────────────────────
        self.asc.update(signals, memories)
        asc_guidance  = self.asc.behavioral_summary()
        metacog_note  = self.metacog.self_note()

        # ── Pass 1: Thinking (if allowed by hardware) ─────────────────────────
        monologue = ""
        thinking_tokens = hw_cfg.get("thinking_tokens", 0)

        if show_thinking and thinking_tokens > 0:
            think_prompt = build_thinking_prompt(
                message, memory_block, asc_guidance,
                metacog_note, self._get_history(hw_cfg["history_budget"])
            )
            monologue = generate(
                self.model, self.tokenizer,
                think_prompt, thinking_tokens
            )
            self._clear_vram()

        # ── Pass 2: Response ──────────────────────────────────────────────────
        response_prompt = build_response_prompt(
            message, memory_block, monologue,
            self._get_history(hw_cfg["history_budget"]),
            self.cfg.name
        )
        response = generate(
            self.model, self.tokenizer,
            response_prompt, hw_cfg["response_tokens"]
        )
        self._clear_vram()

        # ── Post-turn updates ─────────────────────────────────────────────────
        self._history.append({"role": "user",      "content": message})
        self._history.append({"role": "assistant", "content": response})
        self._trim_history()

        # Store memories
        new_mems = self._extractor.extract(message, self._turn)
        for mem in new_mems:
            self.memory.add(mem, self._turn)
            print(f"  [memory: {mem['type']} · importance={mem['importance']}]")

        # Update goals + metacog
        self.goals.score_turn(signals)
        if signals.get("correction"):
            self.metacog.record_correction(message, new_mems)
        else:
            self.metacog.record_access(new_mems)

        # Periodic VRAM cleanup
        if self._turn % 10 == 0:
            self._clear_vram(full=True)

        # Scheduler pause/resume
        if self.hw.should_pause_scheduler():
            self._scheduler.pause()
        else:
            self._scheduler.resume()

        if think:
            return response, monologue
        return response

    def stream(
        self,
        message: str,
    ) -> Generator[str, None, None]:
        """
        Stream response tokens as a generator.

        Usage:
            for token in brain.stream("Hello"):
                print(token, end="", flush=True)
        """
        self._turn += 1

        # Hardware + memory pipeline (same as chat)
        if self._turn % self.cfg.hardware_check_every == 0:
            self.hw.update()

        hw_cfg       = self.hw.get_config()
        signals      = self.goals.detect_signals(message)
        memories     = self.memory.retrieve(message, top_k=hw_cfg["top_k_memories"])
        memories     = self.goals.tag_with_utility(memories, signals)
        memory_block = self._composer.compose(memories, hw_cfg["history_budget"])

        self.asc.update(signals, memories)

        response_prompt = build_response_prompt(
            message, memory_block, "",
            self._get_history(hw_cfg["history_budget"]),
            self.cfg.name
        )

        # Stream tokens
        yield from self._stream_generate(
            response_prompt, hw_cfg["response_tokens"]
        )

        # Post-turn (history + memory) handled after stream exhausted
        # Caller must exhaust the generator for side effects to apply

    # ── Inspection ────────────────────────────────────────────────────────────

    def status(self) -> str:
        lines = [
            f"\n── MNEMA Status ── profile: {self.profile}",
            f"  Turn:     {self._turn}",
            f"  History:  {len(self._history)//2} turns",
            self.hw.display_status(),
            self.asc.display(),
            self.goals.display(),
            self.metacog.display(),
        ]
        return "\n".join(lines)

    def clear(self):
        """Wipe all memory for this profile and reinitialize."""
        self.memory._init_db()
        self.goals._init_db()
        self.metacog._init_db()
        self.asc._init_db()
        self.hw._init_db()
        self._history = []
        self._turn    = 0
        print("  [Memory cleared]")

    def close(self):
        """Shutdown background scheduler cleanly."""
        self._scheduler.stop()
        self._clear_vram(full=True)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_history(self, budget: int) -> list:
        """Return history trimmed to token budget."""
        from .model.inference import trim_history_to_budget
        return trim_history_to_budget(
            self._history, self.tokenizer, budget
        )

    def _trim_history(self):
        """Hard cap on history length."""
        max_turns = self.cfg.history_max_turns * 2   # pairs
        if len(self._history) > max_turns:
            self._history = self._history[-max_turns:]

    def _clear_vram(self, full: bool = False):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if full:
            gc.collect()

    def _stream_generate(
        self, messages: list, max_tokens: int
    ) -> Generator[str, None, None]:
        """Token-by-token streaming generation."""
        from .model.inference import stream_generate
        yield from stream_generate(
            self.model, self.tokenizer, messages, max_tokens
        )

    # ── Context manager support ───────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self):
        return (f"MNEMA(profile={self.profile!r}, "
                f"turn={self._turn}, "
                f"memories={self.memory.count()})")
