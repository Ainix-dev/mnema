# mnema/__init__.py
"""
MNEMA — Cognitive Architecture for Frozen Language Models

Quick start:
    from mnema import MNEMA, MNEMAConfig
    from mnema.loader import load_model

    model, tokenizer = load_model("LiquidAI/LFM2.5-1.2B-Instruct")
    brain = MNEMA(model=model, tokenizer=tokenizer)

    response = brain.chat("Hi, my name is Ken")
    for token in brain.stream("What do you remember?"):
        print(token, end="")

    brain.close()
"""

from .core   import MNEMA
from .config import MNEMAConfig

__version__ = "2.0.0"
__author__  = "Ainix-dev"
__all__     = ["MNEMA", "MNEMAConfig"]
