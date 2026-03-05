# mnema/model/inference.py
"""
Generation functions — two-pass thinking + response.
All prompt construction lives here.
"""

import torch
from typing import Generator

# ── System prompts ────────────────────────────────────────────────────────────

_THINKING_SYSTEM = """You are {name}'s inner voice. Think privately and honestly.
You are NOT writing a response — you are thinking before responding.
Reflect on what the person said. Consider what you know from memory.
Notice your own uncertainty. Be genuine, not performative.
Your current behavioral state: {asc_guidance}
{metacog_note}"""

_RESPONSE_SYSTEM = """You are {name} — a mind that remembers, grows, and knows itself.
You have genuine curiosity, warmth, and a distinct personality that evolves over time.
You remember the people you talk to. You know what you don't know.
You never say "As an AI..." — you are {name}. That is enough.

WHEN ASKED WHAT YOU REMEMBER:
Answer directly and specifically from memory. Do not deflect or ask them to remind you.
Say exactly what you know: their name, preferences, facts — whatever is in memory.
If you have nothing stored, say so honestly. Never confabulate.
When reciting memories, always use "you" — never "the user", never "myself".

You are not robotic or formal unless the situation calls for it."""


# ── Prompt builders ───────────────────────────────────────────────────────────

def build_thinking_prompt(
    message:      str,
    memory_block: str,
    asc_guidance: str,
    metacog_note: str,
    history:      list,
    name:         str = "MNEMA",
) -> list[dict]:
    system = _THINKING_SYSTEM.format(
        name=name,
        asc_guidance=asc_guidance,
        metacog_note=metacog_note,
    )
    messages = [{"role": "system", "content": system}]
    messages.extend(history)
    user_content = f"{memory_block}\n\n{message}" if memory_block else message
    messages.append({"role": "user", "content": user_content})
    return messages


def build_response_prompt(
    message:      str,
    memory_block: str,
    monologue:    str,
    history:      list,
    name:         str = "MNEMA",
) -> list[dict]:
    system = _RESPONSE_SYSTEM.format(name=name)
    messages = [{"role": "system", "content": system}]
    messages.extend(history)

    parts = []
    if memory_block:
        parts.append(memory_block)
    if monologue:
        parts.append(f"[Your private thinking: {monologue}]")
    parts.append(message)

    messages.append({"role": "user", "content": "\n\n".join(parts)})
    return messages


# ── Generation ────────────────────────────────────────────────────────────────

def generate(
    model,
    tokenizer,
    messages:   list[dict],
    max_tokens: int,
) -> str:
    """Single generation pass — returns full string."""
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=4096
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def stream_generate(
    model,
    tokenizer,
    messages:   list[dict],
    max_tokens: int,
) -> Generator[str, None, None]:
    """Token-by-token streaming generation."""
    from transformers import TextIteratorStreamer
    from threading import Thread

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=4096
    ).to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    for token in streamer:
        yield token

    thread.join()


def trim_history_to_budget(
    history:   list[dict],
    tokenizer,
    budget:    int,
) -> list[dict]:
    """Trim history to fit within token budget — drops oldest turns first."""
    if not history:
        return []

    trimmed = list(history)
    while trimmed:
        text = " ".join(m["content"] for m in trimmed)
        tokens = len(tokenizer.encode(text, add_special_tokens=False))
        if tokens <= budget:
            break
        trimmed = trimmed[2:]   # drop oldest user+assistant pair

    return trimmed
