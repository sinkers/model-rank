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
# HTML report
# ---------------------------------------------------------------------------

def _ttft_color(ttft_s: Optional[float]) -> str:
    if ttft_s is None:
        return "#888"
    if ttft_s < 1.0:
        return "#22c55e"   # green
    if ttft_s < 3.0:
        return "#eab308"   # yellow
    if ttft_s < 8.0:
        return "#f97316"   # orange
    return "#ef4444"       # red


def generate_html_report(results: list[Result], output_path: str = "results.html") -> None:
    from collections import defaultdict
    from datetime import datetime, timezone
    from html import escape

    run_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # --- per-model aggregates ---
    groups: dict[str, list[Result]] = defaultdict(list)
    for r in results:
        if not r.error:
            groups[r.model_name].append(r)

    def avg(vals: list[float]) -> Optional[float]:
        return sum(vals) / len(vals) if vals else None

    # -----------------------------------------------------------------------
    # Summary cards
    # -----------------------------------------------------------------------
    cards_html = ""
    # rank by avg TTFT (models with no TTFT go last)
    def sort_key(item: tuple) -> float:
        _, res = item
        ttfts = [r.ttft_s for r in res if r.ttft_s is not None]
        return avg(ttfts) if ttfts else 9999

    for model_name, res in sorted(groups.items(), key=sort_key):
        mid        = res[0].model_id
        ttfts      = [r.ttft_s        for r in res if r.ttft_s        is not None]
        totals     = [r.total_s       for r in res if r.total_s       is not None]
        out_toks   = [r.output_tokens for r in res if r.output_tokens is not None]
        costs      = [r.cost_usd      for r in res if r.cost_usd      is not None]

        avg_ttft   = avg(ttfts)
        avg_total  = avg(totals)
        avg_out    = avg(out_toks)
        total_cost = sum(costs) if costs else None

        ttft_str  = f"{avg_ttft:.2f}s"        if avg_ttft   is not None else "—"
        total_str = f"{avg_total:.2f}s"        if avg_total  is not None else "—"
        out_str   = f"{avg_out:.0f}"           if avg_out    is not None else "—"
        cost_str  = f"${total_cost:.6f}"       if total_cost is not None else "—"
        color     = _ttft_color(avg_ttft)

        cards_html += f"""
        <div class="card">
          <div class="card-name">{escape(model_name)}</div>
          <div class="card-id">{escape(mid)}</div>
          <div class="card-metrics">
            <div class="metric">
              <span class="metric-value" style="color:{color}">{ttft_str}</span>
              <span class="metric-label">avg TTFT</span>
            </div>
            <div class="metric">
              <span class="metric-value">{total_str}</span>
              <span class="metric-label">avg total</span>
            </div>
            <div class="metric">
              <span class="metric-value">{out_str}</span>
              <span class="metric-label">avg tokens</span>
            </div>
            <div class="metric">
              <span class="metric-value">{cost_str}</span>
              <span class="metric-label">total cost</span>
            </div>
          </div>
          <div class="card-ok">{len(res)}/{len(res) + sum(1 for r in results if r.model_name == model_name and r.error)} prompts OK</div>
        </div>"""

    # -----------------------------------------------------------------------
    # Detail rows
    # -----------------------------------------------------------------------
    rows_html = ""
    prev_model = None
    for r in results:
        model_class = "model-first" if r.model_name != prev_model else ""
        prev_model = r.model_name

        if r.error:
            rows_html += f"""
        <tr class="error-row {model_class}">
          <td>{escape(r.model_name)}</td>
          <td>{escape(r.prompt_label)}</td>
          <td colspan="5" class="error-msg">{escape(r.error[:120])}</td>
          <td><span class="badge badge-error">ERROR</span></td>
        </tr>"""
        else:
            color    = _ttft_color(r.ttft_s)
            ttft_str = f"{r.ttft_s:.2f}s"   if r.ttft_s        is not None else "—"
            tot_str  = f"{r.total_s:.2f}s"  if r.total_s       is not None else "—"
            tin_str  = str(r.input_tokens)  if r.input_tokens  is not None else "—"
            tout_str = str(r.output_tokens) if r.output_tokens is not None else "—"
            cost_str = f"${r.cost_usd:.6f}" if r.cost_usd      is not None else "—"
            preview  = escape(r.response_text[:120]).replace("\n", " ")
            if len(r.response_text) > 120:
                preview += "…"
            full_text = escape(r.response_text)

            rows_html += f"""
        <tr class="{model_class}">
          <td><strong>{escape(r.model_name)}</strong></td>
          <td>{escape(r.prompt_label)}</td>
          <td style="color:{color};font-weight:600">{ttft_str}</td>
          <td>{tot_str}</td>
          <td>{tin_str}</td>
          <td>{tout_str}</td>
          <td>{cost_str}</td>
          <td>
            <details>
              <summary class="response-preview">{preview}</summary>
              <div class="response-full">{full_text}</div>
            </details>
          </td>
        </tr>"""

    # -----------------------------------------------------------------------
    # Full HTML
    # -----------------------------------------------------------------------
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Model Rank — Results</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0f1117;
      color: #e2e8f0;
      padding: 2rem;
      line-height: 1.5;
    }}

    h1 {{ font-size: 1.6rem; font-weight: 700; color: #f8fafc; }}
    h2 {{ font-size: 1.1rem; font-weight: 600; color: #94a3b8; text-transform: uppercase;
          letter-spacing: .08em; margin: 2rem 0 1rem; }}

    .meta {{ color: #64748b; font-size: .85rem; margin-top: .3rem; }}

    /* --- cards --- */
    .cards {{ display: flex; flex-wrap: wrap; gap: 1rem; margin-top: 1rem; }}
    .card {{
      background: #1e2330;
      border: 1px solid #2d3448;
      border-radius: .75rem;
      padding: 1.2rem 1.4rem;
      min-width: 200px;
      flex: 1 1 200px;
    }}
    .card-name {{ font-size: 1rem; font-weight: 700; color: #f1f5f9; }}
    .card-id   {{ font-size: .72rem; color: #64748b; margin-top: .15rem; margin-bottom: .8rem;
                  font-family: monospace; }}
    .card-metrics {{ display: flex; gap: 1.2rem; flex-wrap: wrap; }}
    .metric {{ display: flex; flex-direction: column; align-items: center; }}
    .metric-value {{ font-size: 1.2rem; font-weight: 700; }}
    .metric-label {{ font-size: .7rem; color: #64748b; text-transform: uppercase;
                     letter-spacing: .06em; margin-top: .1rem; }}
    .card-ok {{ font-size: .75rem; color: #475569; margin-top: .8rem; }}

    /* --- table --- */
    .table-wrap {{ overflow-x: auto; margin-top: 1rem; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: .85rem;
    }}
    thead th {{
      background: #1e2330;
      color: #94a3b8;
      text-transform: uppercase;
      font-size: .72rem;
      letter-spacing: .07em;
      padding: .65rem 1rem;
      text-align: left;
      border-bottom: 1px solid #2d3448;
      white-space: nowrap;
    }}
    tbody tr {{ border-bottom: 1px solid #1e2330; }}
    tbody tr:hover {{ background: #1a1f2e; }}
    tbody tr.model-first {{ border-top: 2px solid #2d3448; }}
    td {{ padding: .55rem 1rem; vertical-align: top; }}
    .error-row td {{ color: #f87171; }}
    .error-msg {{ font-family: monospace; font-size: .8rem; }}

    /* --- badges --- */
    .badge {{ display: inline-block; padding: .2rem .5rem; border-radius: .3rem;
               font-size: .72rem; font-weight: 600; }}
    .badge-error {{ background: #450a0a; color: #fca5a5; }}

    /* --- response expand --- */
    details summary {{
      cursor: pointer;
      list-style: none;
      color: #64748b;
      font-size: .8rem;
    }}
    details summary::-webkit-details-marker {{ display: none; }}
    details[open] summary {{ color: #94a3b8; margin-bottom: .5rem; }}
    .response-preview {{ white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
                         max-width: 320px; display: block; }}
    .response-full {{
      white-space: pre-wrap;
      font-size: .82rem;
      color: #cbd5e1;
      background: #0f1117;
      border: 1px solid #2d3448;
      border-radius: .4rem;
      padding: .75rem 1rem;
      max-width: 600px;
    }}

    /* --- legend --- */
    .legend {{ display: flex; gap: 1.5rem; font-size: .78rem; color: #64748b;
               margin-top: .5rem; flex-wrap: wrap; }}
    .legend-dot {{ width: 10px; height: 10px; border-radius: 50%;
                   display: inline-block; margin-right: .3rem; }}
  </style>
</head>
<body>
  <h1>Model Rank</h1>
  <p class="meta">Personal assistant benchmark via OpenRouter &nbsp;·&nbsp; {run_time}</p>

  <h2>Summary</h2>
  <div class="cards">{cards_html}</div>

  <div class="legend">
    <span>TTFT colour scale:</span>
    <span><span class="legend-dot" style="background:#22c55e"></span>&lt; 1 s</span>
    <span><span class="legend-dot" style="background:#eab308"></span>1 – 3 s</span>
    <span><span class="legend-dot" style="background:#f97316"></span>3 – 8 s</span>
    <span><span class="legend-dot" style="background:#ef4444"></span>&gt; 8 s</span>
  </div>

  <h2>Detail</h2>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th>Prompt</th>
          <th>TTFT</th>
          <th>Total</th>
          <th>In tok</th>
          <th>Out tok</th>
          <th>Cost (USD)</th>
          <th>Response</th>
        </tr>
      </thead>
      <tbody>{rows_html}
      </tbody>
    </table>
  </div>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)


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

    generate_html_report(all_results, "results.html")
    console.print("[dim]HTML report saved to results.html[/dim]")


if __name__ == "__main__":
    asyncio.run(main())
