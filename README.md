# model-rank

Benchmarks LLMs via the OpenRouter API for use as a voice agent backend.
Measures time-to-first-token (TTFT), total latency, token usage, and cost per request
across a configurable set of models and personal-assistant style prompts.

## What it tests

Five prompts designed to reflect real voice assistant usage:

- **Meeting invite** — draft a product strategy sync invite with agenda
- **Follow-up email** — polite but firm client chaser
- **Trip planning** — 3-day Tokyo itinerary on a budget
- **Book recommendations** — 5 picks based on a known favourite
- **Task breakdown** — launch a company blog in 6 weeks

All models are prompted with a voice-agent system instruction: conversational tone,
no markdown or formatting, brief and direct.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export OPENROUTER_API_KEY=sk-or-v1-...
python test_suite.py
```

## Configuration

Edit the top of `test_suite.py` to change models, prompts, or constraints.

### Adding or changing models

Models are defined as search terms resolved against the live OpenRouter models list at runtime:

```python
MODEL_SEARCHES: dict[str, tuple[str, dict]] = {
    "Claude Haiku":  ("claude-haiku",  {"sort": "latency"}),
    "GPT-4o":        ("gpt-4o",        {"sort": "latency"}),
    "Kimi K2":       ("kimi-k2",       {"sort": "latency"}),
    "MiniMax":       ("minimax-m",     {"sort": "latency"}),
    "Qwen3":         ("qwen3",         {"sort": "latency"}),
}
```

The search term is matched against model IDs from `/api/v1/models`. The shortest
non-`:free` match is selected as the canonical version. The second element is an
OpenRouter [provider preferences](https://openrouter.ai/docs/guides/routing/provider-selection)
object — use `{"sort": "latency"}` to auto-route to the fastest available provider,
or `{"order": ["Anthropic"], "allow_fallbacks": false}` to hard-pin one.

### Voice agent settings

```python
MAX_TOKENS = 350       # hard cap to keep responses speakable
SYSTEM_PROMPT = "..."  # instructs models to reply without formatting
```

## Output

Results are printed to the terminal as two rich tables:

- **Model Comparison Summary** — per-prompt breakdown of TTFT, total time, token counts, and cost
- **Per-Model Averages** — aggregated metrics across all prompts per model

Full response text and all metrics are also written to `results.json` for offline review.

## Key findings (as of April 2026)

| Model | Avg TTFT | Avg Total | Avg Out tok | Total Cost (5 prompts) |
|---|---|---|---|---|
| GPT-4o | 0.64s | 2.40s | 168 | $0.0034 |
| Claude Haiku | 0.92s | 3.38s | 205 | $0.0029 |
| Kimi K2 | 1.39s | 8.65s | 216 | $0.0016 |
| MiniMax | 3.58s* | 4.86s | 335 | $0.0007 |
| Qwen3-8b | 5.67s | 6.94s | 627 | $0.0006 |

*MiniMax does not stream reliably — responses arrive in a single chunk, making true TTFT unmeasurable and unsuitable for voice.
