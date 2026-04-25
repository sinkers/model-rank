#!/usr/bin/env python3
"""
OpenRouter Model Test Suite
Tests personal assistant style requests across multiple models,
measuring response quality, latency, and cost.
"""

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional
import httpx
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
if not API_KEY:
    sys.exit("Error: OPENROUTER_API_KEY environment variable is not set.")

BASE_URL = "https://openrouter.ai/api/v1"

# Voice agent constraints
MAX_TOKENS = 350  # keep responses short enough to speak aloud
SYSTEM_PROMPT = (
    "You are a voice assistant. Reply conversationally as if speaking aloud. "
    "Never use markdown, bullet points, headers, emojis, or any formatting. "
    "Be brief and direct — a few sentences at most. "
    "Write exactly as you would say it."
)

# Search terms used to resolve model IDs from the OpenRouter models API.
# Each value is (search_term, provider_preferences).
# The search_term is matched against model IDs; the best (shortest non-free) match wins.
#
# Provider prefs reference: https://openrouter.ai/docs/guides/routing/provider-selection
#   {"sort": "latency"}                         — auto-pick fastest provider
#   {"order": ["Anthropic"], "allow_fallbacks": False}  — hard-pin a provider
MODEL_SEARCHES: dict[str, tuple[str, dict]] = {
    "Claude Haiku":  ("claude-haiku",  {"sort": "latency"}),
    "GPT-4o":        ("gpt-4o",        {"sort": "latency"}),
    "Kimi K2":       ("kimi-k2",       {"sort": "latency"}),
    "MiniMax":       ("minimax-m",     {"sort": "latency"}),
    "Qwen3":         ("qwen3",         {"sort": "latency"}),
}

# Personal assistant style prompts
PROMPTS: list[dict] = [
    {
        "id": "schedule",
        "label": "Meeting invite",
        "message": (
            "Draft a professional meeting invite for a 1-hour product strategy sync "
            "with 5 team members, scheduled for next Monday at 10 am. Include an agenda, "
            "a video-call link placeholder, and prep instructions."
        ),
    },
    {
        "id": "email",
        "label": "Follow-up email",
        "message": (
            "Write a polite but firm follow-up email to a client who hasn't responded "
            "to our $15,000 software development proposal in two weeks."
        ),
    },
    {
        "id": "plan",
        "label": "Trip planning",
        "message": (
            "Plan a 3-day weekend itinerary for two people visiting Tokyo for the first time "
            "with a $2,000 budget. Include accommodation, food, and key sights."
        ),
    },
    {
        "id": "recommend",
        "label": "Book recs",
        "message": (
            "Recommend 5 books for someone who loved The Hitchhiker's Guide to the Galaxy. "
            "Briefly explain why each one would appeal to them."
        ),
    },
    {
        "id": "tasks",
        "label": "Task breakdown",
        "message": (
            "I need to launch a company blog from scratch in 6 weeks. "
            "Break this into a prioritised task list with rough time estimates."
        ),
    },
]

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Result:
    model_name: str
    model_id: str
    prompt_id: str
    prompt_label: str
    response_text: str = ""
    ttft_s: Optional[float] = None       # time to first token
    total_s: Optional[float] = None      # total wall-clock time
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    generation_id: Optional[str] = None
    error: Optional[str] = None

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/sinkers/model-rank",
    "X-Title": "ClawtalkModelTest",
}


async def resolve_models(
    client: httpx.AsyncClient,
    searches: dict[str, tuple[str, dict]],
) -> dict[str, tuple[str, dict]]:
    """
    Fetch the OpenRouter models list and resolve each search term to a concrete model ID.
    Picks the shortest matching ID that isn't a :free variant (those often have lower limits).
    """
    resp = await client.get(f"{BASE_URL}/models", headers=HEADERS, timeout=15)
    resp.raise_for_status()
    all_models: list[dict] = resp.json().get("data", [])
    all_ids = [m["id"] for m in all_models]

    resolved: dict[str, tuple[str, dict]] = {}
    for display_name, (term, prefs) in searches.items():
        matches = [mid for mid in all_ids if term.lower() in mid.lower() and ":free" not in mid]
        if not matches:
            # Fall back to including :free if nothing else matches
            matches = [mid for mid in all_ids if term.lower() in mid.lower()]
        if not matches:
            resolved[display_name] = (f"UNRESOLVED:{term}", prefs)
        else:
            # Prefer the shortest ID — usually the canonical/flagship version
            best = min(matches, key=len)
            resolved[display_name] = (best, prefs)

    return resolved


async def fetch_generation_cost(
    client: httpx.AsyncClient,
    gen_id: str,
    retries: int = 8,
    delay: float = 1.0,
) -> Optional[float]:
    """
    Poll the OpenRouter generation endpoint until cost data is available.
    OpenRouter processes cost asynchronously so it may not be ready immediately.
    """
    # Generation cost is processed asynchronously — give it a moment before polling
    await asyncio.sleep(2.0)
    for attempt in range(retries):
        try:
            resp = await client.get(
                f"{BASE_URL}/generation",
                params={"id": gen_id},
                headers=HEADERS,
                timeout=10,
            )
            if resp.status_code == 200:
                cost = resp.json().get("data", {}).get("total_cost")
                if cost is not None:
                    return cost
        except Exception:
            pass
        if attempt < retries - 1:
            await asyncio.sleep(delay)
    return None


async def run_prompt(
    client: httpx.AsyncClient,
    model_name: str,
    model_id: str,
    prompt: dict,
    provider_prefs: dict | None = None,
) -> Result:
    """Stream a single chat completion and collect metrics."""
    result = Result(
        model_name=model_name,
        model_id=model_id,
        prompt_id=prompt["id"],
        prompt_label=prompt["label"],
    )

    payload: dict = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt["message"]},
        ],
        "stream": True,
        "max_tokens": MAX_TOKENS,
        "usage": {"include": True},
    }
    if provider_prefs:
        payload["provider"] = provider_prefs

    chunks: list[str] = []
    start = time.perf_counter()
    first_token_at: Optional[float] = None

    try:
        async with client.stream(
            "POST",
            f"{BASE_URL}/chat/completions",
            headers=HEADERS,
            json=payload,
            timeout=120,
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                result.error = f"HTTP {resp.status_code}: {body.decode()[:200]}"
                return result

            async for raw_line in resp.aiter_lines():
                line = raw_line.strip()
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # The generation ID is on every chunk — capture it once
                if result.generation_id is None and chunk.get("id"):
                    result.generation_id = chunk["id"]

                # Usage arrives on the final chunk
                usage = chunk.get("usage")
                if usage:
                    result.input_tokens = usage.get("prompt_tokens")
                    result.output_tokens = usage.get("completion_tokens")
                    if "total_cost" in usage:
                        result.cost_usd = usage["total_cost"]

                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        chunks.append(content)

    except httpx.TimeoutException:
        result.error = "Request timed out after 120s"
        return result
    except Exception as exc:
        result.error = str(exc)
        return result

    end = time.perf_counter()
    result.total_s = end - start
    result.ttft_s = (first_token_at - start) if first_token_at else None
    result.response_text = "".join(chunks)

    # Poll for cost if it wasn't included inline in the usage chunk
    if result.cost_usd is None and result.generation_id:
        result.cost_usd = await fetch_generation_cost(client, result.generation_id)

    return result

# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

console = Console()


def print_summary_table(results: list[Result]) -> None:
    table = Table(
        title="Model Comparison Summary",
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold cyan",
    )
    table.add_column("Model", style="bold", min_width=14)
    table.add_column("Prompt", min_width=16)
    table.add_column("TTFT", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("In tok", justify="right")
    table.add_column("Out tok", justify="right")
    table.add_column("Cost (USD)", justify="right")
    table.add_column("Status")

    for r in results:
        if r.error:
            table.add_row(
                r.model_name, r.prompt_label,
                "—", "—", "—", "—", "—",
                f"[red]{r.error[:40]}[/red]",
            )
        else:
            ttft  = f"{r.ttft_s:.2f}s"    if r.ttft_s  is not None else "—"
            total = f"{r.total_s:.2f}s"   if r.total_s is not None else "—"
            tin   = str(r.input_tokens)   if r.input_tokens  is not None else "—"
            tout  = str(r.output_tokens)  if r.output_tokens is not None else "—"
            cost  = f"${r.cost_usd:.6f}"  if r.cost_usd is not None else "—"
            table.add_row(
                r.model_name, r.prompt_label,
                ttft, total, tin, tout, cost,
                "[green]OK[/green]",
            )

    console.print(table)


def print_per_model_averages(results: list[Result]) -> None:
    from collections import defaultdict

    groups: dict[str, list[Result]] = defaultdict(list)
    for r in results:
        if not r.error:
            groups[r.model_name].append(r)

    table = Table(
        title="Per-Model Averages",
        box=box.ROUNDED,
        header_style="bold magenta",
    )
    table.add_column("Model", style="bold")
    table.add_column("Model ID", style="dim")
    table.add_column("Avg TTFT", justify="right")
    table.add_column("Avg Total", justify="right")
    table.add_column("Avg Out tok", justify="right")
    table.add_column("Total Cost", justify="right")
    table.add_column("OK", justify="right")

    for model_name, res in sorted(groups.items()):
        ttfts    = [r.ttft_s        for r in res if r.ttft_s        is not None]
        totals   = [r.total_s       for r in res if r.total_s       is not None]
        out_toks = [r.output_tokens for r in res if r.output_tokens is not None]
        costs    = [r.cost_usd      for r in res if r.cost_usd      is not None]

        table.add_row(
            model_name,
            res[0].model_id,
            f"{sum(ttfts)/len(ttfts):.2f}s"          if ttfts    else "—",
            f"{sum(totals)/len(totals):.2f}s"         if totals   else "—",
            f"{sum(out_toks)/len(out_toks):.0f}"      if out_toks else "—",
            f"${sum(costs):.6f}"                      if costs    else "—",
            str(len(res)),
        )

    console.print(table)

# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

async def main() -> None:
    async with httpx.AsyncClient() as client:
        # Resolve model search terms → concrete IDs from the live models list
        console.print("[dim]Resolving model IDs from OpenRouter API…[/dim]")
        models = await resolve_models(client, MODEL_SEARCHES)

        console.print(Panel.fit(
            "[bold cyan]OpenRouter Personal Assistant Model Test Suite[/bold cyan]\n"
            f"Models: {len(models)}  |  Prompts: {len(PROMPTS)}  |  "
            f"Total requests: {len(models) * len(PROMPTS)}",
            border_style="cyan",
        ))

        # Show resolved IDs
        for display, (mid, _) in models.items():
            flag = "[red]UNRESOLVED[/red]" if mid.startswith("UNRESOLVED") else "[green]✓[/green]"
            console.print(f"  {flag} {display}: [dim]{mid}[/dim]")
        console.print()

        all_results: list[Result] = []

        for model_name, (model_id, provider_prefs) in models.items():
            if model_id.startswith("UNRESOLVED"):
                console.print(f"[red]Skipping {model_name} — no matching model ID found[/red]")
                continue

            console.print(f"[bold yellow]━━ {model_name}[/bold yellow] [dim]({model_id})[/dim]")

            for prompt in PROMPTS:
                console.print(f"  [cyan]►[/cyan] {prompt['label']} …", end="")

                result = await run_prompt(client, model_name, model_id, prompt, provider_prefs)
                all_results.append(result)

                if result.error:
                    console.print(f" [red]FAILED[/red]")
                    console.print(f"    [red]{result.error[:120]}[/red]")
                else:
                    ttft  = f"{result.ttft_s:.2f}s"       if result.ttft_s  else "—"
                    total = f"{result.total_s:.2f}s"      if result.total_s else "—"
                    cost  = f"${result.cost_usd:.6f}"     if result.cost_usd is not None else "n/a"
                    console.print(
                        f" [green]done[/green] "
                        f"[dim]ttft={ttft} total={total} cost={cost}[/dim]"
                    )

    console.print()
    print_summary_table(all_results)
    console.print()
    print_per_model_averages(all_results)

    output = [
        {
            "model_name":    r.model_name,
            "model_id":      r.model_id,
            "prompt_id":     r.prompt_id,
            "prompt_label":  r.prompt_label,
            "response_text": r.response_text,
            "ttft_s":        r.ttft_s,
            "total_s":       r.total_s,
            "input_tokens":  r.input_tokens,
            "output_tokens": r.output_tokens,
            "cost_usd":      r.cost_usd,
            "error":         r.error,
        }
        for r in all_results
    ]
    with open("results.json", "w") as f:
        json.dump(output, f, indent=2)
    console.print("[dim]Full responses saved to results.json[/dim]")


if __name__ == "__main__":
    asyncio.run(main())
