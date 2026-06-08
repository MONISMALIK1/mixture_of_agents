"""CLI for Mixture-of-Agents.

Usage:
    # 3 proposers, 2 layers (default), default model for all
    python -m mixture_of_agents "What are the trade-offs of microservices?"

    # use specific proposer models and a different aggregator
    python -m mixture_of_agents "..." \\
        --models "openai/gpt-oss-120b:free,meta-llama/llama-3.1-8b-instruct" \\
        --aggregator "openai/gpt-oss-120b:free" --layers 3 --show-trace
"""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .core import baseline, run
from .llm import DEFAULT_MODEL


def _print_trace(res) -> None:
    print("--- mixture-of-agents trace ---")
    for li, layer in enumerate(res.layers, 1):
        print(f"Layer {li} ({len(layer)} proposers):")
        for ai, resp in enumerate(layer, 1):
            head = resp.replace("\n", " ")
            print(f"  [{ai}] {head[:100]}{'...' if len(head) > 100 else ''}")
    print("-------------------------------")


def main() -> int:
    p = argparse.ArgumentParser(
        prog="mixture_of_agents",
        description="Mixture-of-Agents (Wang et al., 2024): layered propose-and-aggregate.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("query", nargs="?", help="The instruction to answer.")
    p.add_argument("--proposers", type=int, default=3,
                   help="Number of proposer agents per layer (default: 3).")
    p.add_argument("--models", default=None,
                   help="Comma-separated proposer model slugs (overrides --proposers).")
    p.add_argument("--layers", type=int, default=2, help="Number of proposer layers (default: 2).")
    p.add_argument("--aggregator", default=None, help="Model for the final synthesis.")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="Proposer sampling temperature for diversity (default: 0.7).")
    p.add_argument("--model", default=None, help=f"Default model (default: {DEFAULT_MODEL}).")
    p.add_argument("--show-trace", action="store_true", help="Print every layer's proposals.")
    p.add_argument("--baseline", action="store_true",
                   help="Also print a single-model answer to compare against MoA.")
    args = p.parse_args()

    if not args.query:
        p.error("provide an instruction to answer")

    proposers = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else args.proposers

    print(f"\nInstruction: {args.query}", file=sys.stderr)
    print(f"Proposers: {proposers} | layers: {args.layers} | "
          f"aggregator: {args.aggregator or args.model or DEFAULT_MODEL}\n",
          file=sys.stderr, flush=True)

    res = run(args.query, proposers=proposers, layers=args.layers,
              aggregator=args.aggregator, temperature=args.temperature, model=args.model)
    if args.show_trace:
        _print_trace(res)
    if args.baseline:
        print("=" * 60)
        print("Single-model baseline:")
        print(baseline(args.query, model=args.model))
    print("=" * 60)
    print("Mixture-of-Agents:" if args.baseline else "")
    print(res.final)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
