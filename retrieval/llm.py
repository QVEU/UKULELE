"""
A single call_llm(prompt) -> str, chosen by env var. This is the ONLY place a
provider is named; everything upstream is provider-agnostic.

  KU_LLM=openai   + OPENAI_API_KEY   (+ optional OPENAI_BASE_URL for local/compatible)
  KU_LLM=anthropic+ ANTHROPIC_API_KEY
  KU_LLM=echo     -> returns the prompt (no network; for inspecting grounding)
"""
import os

def _openai(prompt):
    from openai import OpenAI
    client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL"))  # None = default
    r = client.chat.completions.create(
        model=os.getenv("KU_LLM_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,  # grounded synthesis wants determinism, not creativity
    )
    return r.choices[0].message.content

def _anthropic(prompt):
    import anthropic
    client = anthropic.Anthropic()
    r = client.messages.create(
        model=os.getenv("KU_LLM_MODEL", "claude-haiku-4-5-20251001"),
        max_tokens=1024, temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in r.content if hasattr(b, "text"))

def call_llm(prompt: str) -> str:
    backend = os.getenv("KU_LLM", "echo").lower()
    if backend == "openai":    return _openai(prompt)
    if backend == "anthropic": return _anthropic(prompt)
    if backend == "echo":      return "[echo — no LLM called]\n\n" + prompt
    raise ValueError(f"Unknown KU_LLM backend: {backend}")
