#!/usr/bin/env python3
"""
OpenRouter Model Test Suite
Tests LLMs via OpenRouter across configurable prompts, measuring TTFT,
latency, token usage, cost, and tool-calling behaviour.

Usage:
  python test_suite.py                        # uses config/default.toml
  python test_suite.py --config config/tool_capable.toml
"""

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Optional

import httpx
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Load .env if present
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        if _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        os.environ.setdefault(_k.strip(), _v.strip())

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
if not API_KEY:
    sys.exit("Error: OPENROUTER_API_KEY environment variable is not set.")

BASE_URL     = "https://openrouter.ai/api/v1"
FIXTURES_DIR = Path(__file__).parent / "fixtures"
RESULTS_DIR  = Path(__file__).parent / "results"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type":  "application/json",
    "HTTP-Referer":  "https://github.com/sinkers/model-rank",
    "X-Title":       "ModelRank",
}

console = Console()

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def models_from_config(cfg: dict) -> dict[str, tuple[str, dict]]:
    """Return {display_name: (model_id_or_search, provider_prefs)}."""
    out = {}
    for m in cfg.get("models", []):
        out[m["name"]] = (m["id"], dict(m.get("provider", {})))
    return out

# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------

def load_system_prompts() -> dict[str, str]:
    sp_dir = FIXTURES_DIR / "system_prompts"
    return {p.stem: p.read_text().strip() for p in sorted(sp_dir.glob("*.txt"))}


def load_tools() -> dict[str, dict]:
    tools_dir = FIXTURES_DIR / "tools"
    return {p.stem: json.loads(p.read_text()) for p in sorted(tools_dir.glob("*.json"))}


def load_prompts(
    system_prompts: dict[str, str],
    tools: dict[str, dict],
    prompt_ids: list[str] | None = None,
) -> list[dict]:
    prompts = []
    for path in sorted((FIXTURES_DIR / "prompts").glob("*.json")):
        data = json.loads(path.read_text())

        if prompt_ids and data["id"] not in prompt_ids:
            continue

        sp_name = data.get("system_prompt")
        if sp_name:
            if sp_name not in system_prompts:
                sys.exit(f"Error: prompt '{path.name}' references unknown system_prompt '{sp_name}'.")
            data["system_prompt_text"] = system_prompts[sp_name]
        else:
            data["system_prompt_text"] = None

        resolved_tools = []
        for name in data.get("tools", []):
            if name not in tools:
                sys.exit(f"Error: prompt '{path.name}' references unknown tool '{name}'.")
            resolved_tools.append(tools[name])
        data["tool_definitions"] = resolved_tools

        prompts.append(data)
    return prompts

# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------

async def resolve_models(
    client: httpx.AsyncClient,
    searches: dict[str, tuple[str, dict]],
) -> dict[str, tuple[str, dict]]:
    resp = await client.get(f"{BASE_URL}/models", headers=HEADERS, timeout=15)
    resp.raise_for_status()
    all_ids = [m["id"] for m in resp.json().get("data", [])]

    resolved: dict[str, tuple[str, dict]] = {}
    for display_name, (term, prefs) in searches.items():
        if "/" in term:
            resolved[display_name] = (term if term in all_ids else f"UNRESOLVED:{term}", prefs)
            continue
        matches = [mid for mid in all_ids if term.lower() in mid.lower() and ":free" not in mid]
        if not matches:
            matches = [mid for mid in all_ids if term.lower() in mid.lower()]
        resolved[display_name] = (
            min(matches, key=len) if matches else f"UNRESOLVED:{term}",
            prefs,
        )
    return resolved

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Result:
    model_name:    str
    model_id:      str
    prompt_id:     str
    prompt_label:  str
    response_text: str = ""
    ttft_s:        Optional[float] = None
    total_s:       Optional[float] = None
    input_tokens:  Optional[int]   = None
    output_tokens: Optional[int]   = None
    cost_usd:      Optional[float] = None
    generation_id: Optional[str]   = None
    tool_calls:    Optional[list[dict]] = None
    error:         Optional[str]   = None

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

async def fetch_generation_cost(
    client: httpx.AsyncClient,
    gen_id: str,
    retries: int = 8,
    delay: float = 1.0,
) -> Optional[float]:
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
    client:        httpx.AsyncClient,
    model_name:    str,
    model_id:      str,
    prompt:        dict,
    provider_prefs: dict | None = None,
    timeout_s:     int = 120,
    max_tokens:    int = 350,
) -> Result:
    result = Result(
        model_name=model_name, model_id=model_id,
        prompt_id=prompt["id"], prompt_label=prompt["label"],
    )

    messages: list[dict] = []
    if prompt.get("system_prompt_text"):
        messages.append({"role": "system", "content": prompt["system_prompt_text"]})
    messages.append({"role": "user", "content": prompt["message"]})

    payload: dict = {
        "model":      model_id,
        "messages":   messages,
        "stream":     True,
        "max_tokens": max_tokens,
        "usage":      {"include": True},
    }
    if provider_prefs:
        payload["provider"] = provider_prefs
    if prompt.get("tool_definitions"):
        payload["tools"]       = prompt["tool_definitions"]
        payload["tool_choice"] = "auto"

    chunks:   list[str]       = []
    tc_accum: dict[int, dict] = {}
    start = time.perf_counter()
    first_token_at: Optional[float] = None

    try:
        async with client.stream(
            "POST", f"{BASE_URL}/chat/completions",
            headers=HEADERS, json=payload, timeout=timeout_s,
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

                if result.generation_id is None and chunk.get("id"):
                    result.generation_id = chunk["id"]

                usage = chunk.get("usage")
                if usage:
                    result.input_tokens  = usage.get("prompt_tokens")
                    result.output_tokens = usage.get("completion_tokens")
                    if "total_cost" in usage:
                        result.cost_usd = usage["total_cost"]

                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})

                for tc in delta.get("tool_calls", []):
                    idx = tc.get("index", 0)
                    if idx not in tc_accum:
                        tc_accum[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc.get("id"):
                        tc_accum[idx]["id"] = tc["id"]
                    fn = tc.get("function", {})
                    if fn.get("name"):
                        tc_accum[idx]["name"] = fn["name"]
                    if fn.get("arguments"):
                        tc_accum[idx]["arguments"] += fn["arguments"]
                    if first_token_at is None:
                        first_token_at = time.perf_counter()

                content = delta.get("content", "")
                if content:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    chunks.append(content)

    except httpx.TimeoutException:
        result.error = f"Request timed out after {timeout_s}s"
        return result
    except Exception as exc:
        result.error = str(exc)
        return result

    result.total_s        = time.perf_counter() - start
    result.ttft_s         = (first_token_at - start) if first_token_at else None
    result.response_text  = "".join(chunks)

    if tc_accum:
        result.tool_calls = []
        for tc in tc_accum.values():
            try:
                args = json.loads(tc["arguments"]) if tc["arguments"] else {}
            except json.JSONDecodeError:
                args = {"_raw": tc["arguments"]}
            result.tool_calls.append({"name": tc["name"], "arguments": args})

    if result.cost_usd is None and result.generation_id:
        result.cost_usd = await fetch_generation_cost(client, result.generation_id)

    return result

# ---------------------------------------------------------------------------
# Terminal display
# ---------------------------------------------------------------------------

def print_summary_table(results: list[Result]) -> None:
    table = Table(title="Model Comparison", box=box.ROUNDED, show_lines=True, header_style="bold cyan")
    table.add_column("Model",      style="bold", min_width=14)
    table.add_column("Prompt",     min_width=16)
    table.add_column("TTFT",       justify="right")
    table.add_column("Total",      justify="right")
    table.add_column("In tok",     justify="right")
    table.add_column("Out tok",    justify="right")
    table.add_column("Cost (USD)", justify="right")
    table.add_column("Tool calls")
    table.add_column("Status")

    for r in results:
        tc = ", ".join(t["name"] for t in r.tool_calls) if r.tool_calls else ""
        if r.error:
            table.add_row(r.model_name, r.prompt_label, "—","—","—","—","—","—",
                          f"[red]{r.error[:40]}[/red]")
        else:
            table.add_row(
                r.model_name, r.prompt_label,
                f"{r.ttft_s:.2f}s"   if r.ttft_s        is not None else "—",
                f"{r.total_s:.2f}s"  if r.total_s       is not None else "—",
                str(r.input_tokens)  if r.input_tokens  is not None else "—",
                str(r.output_tokens) if r.output_tokens is not None else "—",
                f"${r.cost_usd:.6f}" if r.cost_usd      is not None else "—",
                f"[magenta]{tc}[/magenta]" if tc else "—",
                "[green]OK[/green]",
            )
    console.print(table)


def print_leaderboard(results: list[Result]) -> None:
    from collections import defaultdict
    groups: dict[str, list[Result]] = defaultdict(list)
    for r in results:
        if not r.error:
            groups[r.model_name].append(r)

    def avg(vals): return sum(vals) / len(vals) if vals else None

    rows = []
    for name, res in groups.items():
        ttfts  = [r.ttft_s    for r in res if r.ttft_s    is not None]
        totals = [r.total_s   for r in res if r.total_s   is not None]
        costs  = [r.cost_usd  for r in res if r.cost_usd  is not None]
        tool_ok = sum(1 for r in res if r.tool_calls)
        rows.append({
            "name":       name,
            "model_id":   res[0].model_id,
            "avg_ttft":   avg(ttfts),
            "avg_total":  avg(totals),
            "total_cost": sum(costs) if costs else None,
            "ok":         len(res),
            "tool_ok":    tool_ok,
        })

    rows.sort(key=lambda r: (r["avg_ttft"] or 9999))

    table = Table(title="Leaderboard", box=box.ROUNDED, header_style="bold magenta")
    table.add_column("Rank", justify="right")
    table.add_column("Model",      style="bold")
    table.add_column("Avg TTFT",   justify="right")
    table.add_column("Avg Total",  justify="right")
    table.add_column("Total Cost", justify="right")
    table.add_column("Tool calls", justify="right")
    table.add_column("Prompts OK", justify="right")

    medals = {1: "1st", 2: "2nd", 3: "3rd"}
    for i, r in enumerate(rows, 1):
        table.add_row(
            medals.get(i, str(i)),
            r["name"],
            f"{r['avg_ttft']:.2f}s"    if r["avg_ttft"]   is not None else "—",
            f"{r['avg_total']:.2f}s"   if r["avg_total"]  is not None else "—",
            f"${r['total_cost']:.6f}"  if r["total_cost"] is not None else "—",
            str(r["tool_ok"]) if r["tool_ok"] else "—",
            str(r["ok"]),
        )
    console.print(table)

# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def _ttft_color(v: Optional[float]) -> str:
    if v is None:   return "#888"
    if v < 1.0:     return "#22c55e"
    if v < 3.0:     return "#eab308"
    if v < 8.0:     return "#f97316"
    return "#ef4444"


def generate_html_report(
    results: list[Result],
    run_cfg: dict,
    output_path: Path,
) -> None:
    from collections import defaultdict

    run_name  = run_cfg.get("run", {}).get("name", "run")
    run_desc  = run_cfg.get("run", {}).get("description", "")
    run_time  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # --- aggregates ---
    groups: dict[str, list[Result]] = defaultdict(list)
    for r in results:
        if not r.error:
            groups[r.model_name].append(r)

    def avg(vals): return sum(vals) / len(vals) if vals else None

    board_rows = []
    for name, res in groups.items():
        ttfts  = [r.ttft_s   for r in res if r.ttft_s   is not None]
        totals = [r.total_s  for r in res if r.total_s  is not None]
        costs  = [r.cost_usd for r in res if r.cost_usd is not None]
        out_t  = [r.output_tokens for r in res if r.output_tokens is not None]
        tool_ok = sum(1 for r in res if r.tool_calls)
        board_rows.append({
            "name":       name,
            "model_id":   res[0].model_id,
            "avg_ttft":   avg(ttfts),
            "avg_total":  avg(totals),
            "avg_out":    avg(out_t),
            "total_cost": sum(costs) if costs else None,
            "ok":         len(res),
            "tool_ok":    tool_ok,
            "errors":     sum(1 for r in results if r.model_name == name and r.error),
        })
    board_rows.sort(key=lambda r: (r["avg_ttft"] or 9999))

    # --- leaderboard HTML ---
    medals = ["1st", "2nd", "3rd"]
    leaderboard_html = ""
    for i, r in enumerate(board_rows):
        color   = _ttft_color(r["avg_ttft"])
        badge   = medals[i] if i < 3 else str(i + 1)
        ttft_s  = f"{r['avg_ttft']:.2f}s"   if r["avg_ttft"]   is not None else "—"
        tot_s   = f"{r['avg_total']:.2f}s"  if r["avg_total"]  is not None else "—"
        cost_s  = f"${r['total_cost']:.6f}" if r["total_cost"] is not None else "—"
        tools_s = f"{r['tool_ok']} / {r['ok']}" if r["tool_ok"] else "—"
        tool_cls = "tool-yes" if r["tool_ok"] else "tool-no"

        leaderboard_html += f"""
      <div class="lb-card rank-{min(i+1,4)}">
        <div class="lb-rank">{badge}</div>
        <div class="lb-body">
          <div class="lb-name">{escape(r['name'])}</div>
          <div class="lb-id">{escape(r['model_id'])}</div>
          <div class="lb-stats">
            <span class="stat"><span class="stat-val" style="color:{color}">{ttft_s}</span><span class="stat-lbl">TTFT</span></span>
            <span class="stat"><span class="stat-val">{tot_s}</span><span class="stat-lbl">total</span></span>
            <span class="stat"><span class="stat-val">{cost_s}</span><span class="stat-lbl">cost</span></span>
            <span class="stat {tool_cls}"><span class="stat-val">{tools_s}</span><span class="stat-lbl">tool calls</span></span>
          </div>
        </div>
      </div>"""

    # --- detail rows ---
    detail_rows = ""
    for r in results:
        tc = ""
        if r.tool_calls:
            for tc_item in r.tool_calls:
                args_str = escape(json.dumps(tc_item["arguments"], indent=2))
                tc += f'<span class="tool-name">{escape(tc_item["name"])}</span><pre class="tool-args">{args_str}</pre>'
        color   = _ttft_color(r.ttft_s)
        ttft_s  = f"{r.ttft_s:.2f}"   if r.ttft_s        is not None else "—"
        tot_s   = f"{r.total_s:.2f}"  if r.total_s       is not None else "—"
        tin_s   = str(r.input_tokens) if r.input_tokens  is not None else "—"
        tout_s  = str(r.output_tokens)if r.output_tokens is not None else "—"
        cost_s  = f"{r.cost_usd:.6f}" if r.cost_usd      is not None else "—"
        status  = f'<span class="badge-err">ERROR</span>' if r.error else '<span class="badge-ok">OK</span>'
        err_td  = f'<td colspan="6" class="err-msg">{escape(r.error or "")}</td>' if r.error else \
                  f'<td data-val="{ttft_s}"  style="color:{color};font-weight:600">{ttft_s}s</td>' \
                  f'<td data-val="{tot_s}">{tot_s}s</td>' \
                  f'<td data-val="{tin_s}">{tin_s}</td>' \
                  f'<td data-val="{tout_s}">{tout_s}</td>' \
                  f'<td data-val="{cost_s}">{cost_s}</td>' \
                  f'<td>{tc or "—"}</td>'

        preview  = escape(r.response_text[:120]).replace("\n", " ")
        if len(r.response_text) > 120:
            preview += "…"
        full_text = escape(r.response_text)

        detail_rows += f"""
      <tr>
        <td>{escape(r.model_name)}</td>
        <td>{escape(r.prompt_label)}</td>
        {err_td}
        <td>
          <details><summary class="resp-preview">{preview or "<em>no text</em>"}</summary>
          <div class="resp-full">{full_text}</div></details>
        </td>
        <td>{status}</td>
      </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Model Rank — {escape(run_name)}</title>
  <style>
    *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
          background:#0f1117;color:#e2e8f0;padding:2rem;line-height:1.5}}
    h1{{font-size:1.5rem;font-weight:700;color:#f8fafc}}
    h2{{font-size:.9rem;font-weight:600;color:#64748b;text-transform:uppercase;
        letter-spacing:.08em;margin:2rem 0 .8rem}}
    .meta{{color:#64748b;font-size:.82rem;margin-top:.25rem}}

    /* leaderboard */
    .leaderboard{{display:flex;flex-wrap:wrap;gap:.75rem;margin-top:.5rem}}
    .lb-card{{display:flex;gap:1rem;background:#1e2330;border:1px solid #2d3448;
              border-radius:.75rem;padding:1rem 1.2rem;flex:1 1 260px;align-items:flex-start}}
    .rank-1{{border-color:#f59e0b}}
    .rank-2{{border-color:#94a3b8}}
    .rank-3{{border-color:#b45309}}
    .lb-rank{{font-size:1.3rem;font-weight:800;color:#64748b;min-width:2.5rem;padding-top:.1rem}}
    .rank-1 .lb-rank{{color:#f59e0b}}
    .rank-2 .lb-rank{{color:#94a3b8}}
    .rank-3 .lb-rank{{color:#b45309}}
    .lb-name{{font-size:.95rem;font-weight:700;color:#f1f5f9}}
    .lb-id{{font-size:.7rem;color:#475569;font-family:monospace;margin:.15rem 0 .6rem}}
    .lb-stats{{display:flex;flex-wrap:wrap;gap:.8rem}}
    .stat{{display:flex;flex-direction:column;align-items:center}}
    .stat-val{{font-size:1rem;font-weight:700}}
    .stat-lbl{{font-size:.65rem;color:#64748b;text-transform:uppercase;letter-spacing:.05em}}
    .tool-yes .stat-val{{color:#22c55e}}
    .tool-no  .stat-val{{color:#475569}}

    /* table */
    .tbl-wrap{{overflow-x:auto;margin-top:.5rem}}
    table{{width:100%;border-collapse:collapse;font-size:.83rem}}
    thead th{{background:#1e2330;color:#94a3b8;text-transform:uppercase;font-size:.7rem;
              letter-spacing:.07em;padding:.6rem .9rem;text-align:left;
              border-bottom:1px solid #2d3448;white-space:nowrap;user-select:none}}
    thead th.sortable{{cursor:pointer}}
    thead th.sortable:hover{{color:#e2e8f0}}
    thead th.sort-asc::after{{content:" ▲";font-size:.65rem}}
    thead th.sort-desc::after{{content:" ▼";font-size:.65rem}}
    tbody tr{{border-bottom:1px solid #161b27}}
    tbody tr:hover{{background:#1a1f2e}}
    td{{padding:.5rem .9rem;vertical-align:top}}
    .err-msg{{font-family:monospace;font-size:.78rem;color:#f87171}}
    .badge-ok{{background:#14532d;color:#86efac;font-size:.7rem;
               font-weight:600;padding:.15rem .45rem;border-radius:.25rem}}
    .badge-err{{background:#450a0a;color:#fca5a5;font-size:.7rem;
                font-weight:600;padding:.15rem .45rem;border-radius:.25rem}}
    .tool-name{{background:#2d1b69;color:#c4b5fd;font-size:.7rem;font-weight:600;
                padding:.1rem .4rem;border-radius:.25rem;font-family:monospace;display:inline-block}}
    .tool-args{{font-size:.72rem;color:#94a3b8;background:#0f1117;border:1px solid #2d3448;
                border-radius:.3rem;padding:.3rem .5rem;margin-top:.25rem;
                white-space:pre-wrap;max-width:280px}}
    details summary{{cursor:pointer;list-style:none;color:#64748b;font-size:.78rem}}
    details summary::-webkit-details-marker{{display:none}}
    details[open] summary{{color:#94a3b8;margin-bottom:.4rem}}
    .resp-preview{{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
                   max-width:300px;display:block}}
    .resp-full{{white-space:pre-wrap;font-size:.8rem;color:#cbd5e1;background:#0f1117;
                border:1px solid #2d3448;border-radius:.4rem;padding:.65rem .9rem;max-width:560px}}

    /* legend */
    .legend{{display:flex;gap:1.2rem;font-size:.75rem;color:#64748b;margin-top:.5rem;flex-wrap:wrap}}
    .dot{{width:9px;height:9px;border-radius:50%;display:inline-block;margin-right:.25rem}}
  </style>
</head>
<body>
  <h1>Model Rank — {escape(run_name)}</h1>
  <p class="meta">{escape(run_desc)} &nbsp;·&nbsp; {run_time}</p>

  <h2>Leaderboard</h2>
  <div class="leaderboard">{leaderboard_html}</div>

  <div class="legend">
    <span>TTFT:</span>
    <span><span class="dot" style="background:#22c55e"></span>&lt; 1s</span>
    <span><span class="dot" style="background:#eab308"></span>1–3s</span>
    <span><span class="dot" style="background:#f97316"></span>3–8s</span>
    <span><span class="dot" style="background:#ef4444"></span>&gt; 8s</span>
  </div>

  <h2>Detail <span style="font-weight:400;color:#475569;font-size:.8rem">(click column headers to sort)</span></h2>
  <div class="tbl-wrap">
    <table id="detail">
      <thead>
        <tr>
          <th class="sortable" data-col="0">Model</th>
          <th class="sortable" data-col="1">Prompt</th>
          <th class="sortable" data-col="2">TTFT (s)</th>
          <th class="sortable" data-col="3">Total (s)</th>
          <th class="sortable" data-col="4">In tok</th>
          <th class="sortable" data-col="5">Out tok</th>
          <th class="sortable" data-col="6">Cost (USD)</th>
          <th>Tool calls</th>
          <th>Response</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>{detail_rows}</tbody>
    </table>
  </div>

  <script>
  (function(){{
    const table = document.getElementById('detail');
    const tbody = table.querySelector('tbody');
    let sortCol = -1, sortDir = 1;

    table.querySelectorAll('th.sortable').forEach(th => {{
      th.addEventListener('click', () => {{
        const col = +th.dataset.col;
        if (sortCol === col) {{ sortDir *= -1; }}
        else {{ sortCol = col; sortDir = 1; }}

        table.querySelectorAll('th').forEach(h => h.classList.remove('sort-asc','sort-desc'));
        th.classList.add(sortDir === 1 ? 'sort-asc' : 'sort-desc');

        const rows = Array.from(tbody.querySelectorAll('tr'));
        rows.sort((a, b) => {{
          const getVal = row => {{
            const cells = row.querySelectorAll('td');
            const cell  = cells[col];
            if (!cell) return '';
            return cell.dataset.val ?? cell.textContent.trim();
          }};
          const va = getVal(a), vb = getVal(b);
          const na = parseFloat(va), nb = parseFloat(vb);
          if (!isNaN(na) && !isNaN(nb)) return (na - nb) * sortDir;
          return va.localeCompare(vb) * sortDir;
        }});
        rows.forEach(r => tbody.appendChild(r));
      }});
    }});
  }})();
  </script>
</body>
</html>"""

    output_path.write_text(html)

# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------

def make_results_dir(run_name: str) -> Path:
    ts  = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    slug = run_name.lower().replace(" ", "-")
    d   = RESULTS_DIR / f"{ts}_{slug}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_results(results: list[Result], results_dir: Path, config_path: Path, run_cfg: dict) -> None:
    # Copy the config used for this run
    shutil.copy(config_path, results_dir / "config.toml")

    # JSON results
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
            "tool_calls":    r.tool_calls,
            "error":         r.error,
        }
        for r in results
    ]
    (results_dir / "results.json").write_text(json.dumps(output, indent=2))

    # HTML report
    generate_html_report(results, run_cfg, results_dir / "report.html")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(config_path: Path) -> None:
    cfg        = load_config(config_path)
    run_name   = cfg.get("run", {}).get("name", "run")
    settings   = cfg.get("settings", {})
    max_tokens = settings.get("max_tokens", 350)
    timeout_s  = settings.get("timeout_s", 120)
    prompt_ids = settings.get("prompt_ids")

    system_prompts = load_system_prompts()
    tools          = load_tools()
    prompts        = load_prompts(system_prompts, tools, prompt_ids)
    model_searches = models_from_config(cfg)

    console.print(
        f"[dim]Config: {config_path.name} · "
        f"{len(prompts)} prompts · "
        f"{len(system_prompts)} system prompt(s) · "
        f"{len(tools)} tool(s)[/dim]"
    )

    async with httpx.AsyncClient() as client:
        console.print("[dim]Resolving model IDs…[/dim]")
        models = await resolve_models(client, model_searches)

        console.print(Panel.fit(
            f"[bold cyan]Model Rank — {run_name}[/bold cyan]\n"
            f"Models: {len(models)}  |  Prompts: {len(prompts)}  |  "
            f"Total requests: {len(models) * len(prompts)}",
            border_style="cyan",
        ))

        for display, (mid, _) in models.items():
            flag = "[red]UNRESOLVED[/red]" if mid.startswith("UNRESOLVED") else "[green]✓[/green]"
            console.print(f"  {flag} {display}: [dim]{mid}[/dim]")
        console.print()

        all_results: list[Result] = []

        for model_name, (model_id, provider_prefs) in models.items():
            if model_id.startswith("UNRESOLVED"):
                console.print(f"[red]Skipping {model_name} — no matching model found[/red]")
                continue
            console.print(f"[bold yellow]━━ {model_name}[/bold yellow] [dim]({model_id})[/dim]")

            for prompt in prompts:
                console.print(f"  [cyan]►[/cyan] {prompt['label']} …", end="")
                result = await run_prompt(
                    client, model_name, model_id, prompt,
                    provider_prefs, timeout_s, max_tokens,
                )
                all_results.append(result)

                if result.error:
                    console.print(f" [red]FAILED[/red]\n    [red]{result.error[:120]}[/red]")
                else:
                    ttft  = f"{result.ttft_s:.2f}s"   if result.ttft_s  else "—"
                    total = f"{result.total_s:.2f}s"   if result.total_s else "—"
                    cost  = f"${result.cost_usd:.6f}"  if result.cost_usd is not None else "n/a"
                    tc_str = ""
                    if result.tool_calls:
                        names  = ", ".join(t["name"] for t in result.tool_calls)
                        tc_str = f" [magenta]tool_calls=[{names}][/magenta]"
                    elif prompt.get("tool_definitions"):
                        tc_str = " [yellow]no tool call[/yellow]"
                    console.print(f" [green]done[/green] [dim]ttft={ttft} total={total} cost={cost}[/dim]{tc_str}")

    console.print()
    print_leaderboard(all_results)
    console.print()
    print_summary_table(all_results)

    results_dir = make_results_dir(run_name)
    save_results(all_results, results_dir, config_path, cfg)
    console.print(f"\n[dim]Results saved to {results_dir}/[/dim]")
    console.print(f"[dim]  results.json · report.html · config.toml[/dim]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OpenRouter model benchmarks.")
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).parent / "config" / "default.toml",
        help="Path to TOML config file (default: config/default.toml)",
    )
    args = parser.parse_args()

    if not args.config.exists():
        sys.exit(f"Error: config file not found: {args.config}")

    asyncio.run(main(args.config))
