#!/usr/bin/env python3
"""
Discover OpenRouter models filtered by capability and sorted by latency.

Usage:
  python discover.py                        # all models sorted by latency
  python discover.py --tools                # only models with tool/function-calling support
  python discover.py --tools --limit 20     # top 20 tool-capable by p50 latency
  python discover.py --tools --toml         # emit a config/tool_capable.toml snippet
"""

import argparse
import asyncio
import json
import os
import sys
import httpx

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
if not API_KEY:
    sys.exit("Error: OPENROUTER_API_KEY environment variable is not set.")

BASE_URL  = "https://openrouter.ai/api/v1"
HEADERS   = {"Authorization": f"Bearer {API_KEY}"}
BATCH     = 25   # concurrent endpoint fetches


async def fetch_endpoints(client: httpx.AsyncClient, model_id: str) -> list[dict]:
    try:
        r = await client.get(
            f"{BASE_URL}/models/{model_id}/endpoints",
            headers=HEADERS, timeout=10,
        )
        if r.status_code == 200:
            return r.json().get("data", {}).get("endpoints", [])
    except Exception:
        pass
    return []


async def discover(
    require_tools: bool = False,
    sort_by: str = "latency",
    limit: int = 20,
) -> list[dict]:
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{BASE_URL}/models", headers=HEADERS, timeout=15)
        r.raise_for_status()
        models = [m for m in r.json().get("data", []) if ":free" not in m["id"]]

        print(f"Fetching endpoints for {len(models)} models…", file=sys.stderr)

        results = []
        for i in range(0, len(models), BATCH):
            batch = models[i : i + BATCH]
            endpoints_list = await asyncio.gather(
                *[fetch_endpoints(client, m["id"]) for m in batch]
            )
            for model, endpoints in zip(batch, endpoints_list):
                for ep in endpoints:
                    lat = ep.get("latency_last_30m") or {}
                    thr = ep.get("throughput_last_30m") or {}
                    p50 = lat.get("p50")
                    if p50 is None:
                        continue

                    supports_tools = any(
                        p in ep.get("supported_parameters", [])
                        for p in ("tools", "tool_choice")
                    )
                    if require_tools and not supports_tools:
                        continue

                    results.append({
                        "model_id":      model["id"],
                        "model_name":    model.get("name", model["id"]),
                        "provider":      ep.get("provider_name", "?"),
                        "provider_tag":  ep.get("tag", ""),
                        "supports_tools": supports_tools,
                        "p50_ms":        p50,
                        "p90_ms":        lat.get("p90"),
                        "thr_p50":       thr.get("p50"),
                        "prompt_price":  ep.get("pricing", {}).get("prompt"),
                        "completion_price": ep.get("pricing", {}).get("completion"),
                        "uptime":        ep.get("uptime_last_30m"),
                        "quantization":  ep.get("quantization", ""),
                    })

    if sort_by == "latency":
        results.sort(key=lambda e: e["p50_ms"])
    elif sort_by == "throughput":
        results.sort(key=lambda e: -(e["thr_p50"] or 0))
    elif sort_by == "price":
        results.sort(key=lambda e: float(e["prompt_price"] or 9999))

    return results[:limit]


def print_table(rows: list[dict]) -> None:
    print(
        f"\n{'#':<3} {'Model':<45} {'Provider':<22}"
        f" {'p50':>7} {'p90':>7} {'tok/s':>6}"
        f" {'$/1M in':>9} {'$/1M out':>9} {'tools':>5} {'uptime':>7}"
    )
    print("-" * 120)
    for i, e in enumerate(rows, 1):
        p_in  = f"${float(e['prompt_price'])*1e6:.2f}"     if e["prompt_price"]      else "—"
        p_out = f"${float(e['completion_price'])*1e6:.2f}" if e["completion_price"]   else "—"
        thr   = f"{e['thr_p50']:.0f}"                      if e["thr_p50"]            else "—"
        up    = f"{e['uptime']:.1f}%"                       if e["uptime"]             else "—"
        p90   = f"{e['p90_ms']:.0f}ms"                     if e["p90_ms"]             else "—"
        tools = "yes" if e["supports_tools"] else "no"
        print(
            f"{i:<3} {e['model_id']:<45} {e['provider']:<22}"
            f" {e['p50_ms']:>5.0f}ms {p90:>7} {thr:>6}"
            f" {p_in:>9} {p_out:>9} {tools:>5} {up:>7}"
        )


def emit_toml(rows: list[dict], run_name: str = "discovered") -> str:
    lines = [
        f'[run]',
        f'name = "{run_name}"',
        f'description = "Auto-discovered via discover.py"',
        f'',
        f'[settings]',
        f'max_tokens = 350',
        f'timeout_s  = 120',
        f'',
    ]
    seen: set[str] = set()
    for e in rows:
        key = (e["model_id"], e["provider"])
        if key in seen:
            continue
        seen.add(key)
        name = f"{e['model_name']} ({e['provider']})"
        lines += [
            f'[[models]]',
            f'name     = "{name}"',
            f'id       = "{e["model_id"]}"',
            f'[models.provider]',
            f'order            = ["{e["provider"]}"]',
            f'allow_fallbacks  = false',
            f'',
        ]
    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Discover OpenRouter models by capability.")
    parser.add_argument("--tools",  action="store_true", help="Only show models with tool/function-calling support")
    parser.add_argument("--sort",   default="latency", choices=["latency", "throughput", "price"])
    parser.add_argument("--limit",  type=int, default=20)
    parser.add_argument("--toml",   action="store_true", help="Emit a TOML config snippet instead of a table")
    parser.add_argument("--output", help="Write TOML config to this file path")
    args = parser.parse_args()

    rows = await discover(require_tools=args.tools, sort_by=args.sort, limit=args.limit)

    if args.toml or args.output:
        run_name = "tool-capable" if args.tools else "discovered"
        toml_str = emit_toml(rows, run_name)
        if args.output:
            with open(args.output, "w") as f:
                f.write(toml_str)
            print(f"Written to {args.output}", file=sys.stderr)
        else:
            print(toml_str)
    else:
        print_table(rows)


if __name__ == "__main__":
    asyncio.run(main())
