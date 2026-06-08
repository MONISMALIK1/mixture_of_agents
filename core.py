"""The Mixture-of-Agents control flow.

Reference: Wang, Wang, Athiwaratkun, Zhang et al. 2024, "Mixture-of-Agents Enhances
Large Language Model Capabilities", https://arxiv.org/abs/2406.04692

    user instruction
        |
    Layer 1: N proposer agents answer it independently           -> N responses
        |
    Layer i (>1): each agent re-answers, given ALL of layer i-1's
                  responses as auxiliary context (Aggregate-and-Synthesize) -> N responses
        |   ... repeat for `layers` layers ...
        |
    Final aggregator: synthesize the last layer's responses        -> the answer

Diversity among proposers comes from temperature (and/or different ``proposers``
models); the aggregator runs greedy for a stable synthesis. Only the model calls
touch the network, via the injectable ``chat_fn``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .llm import chat
from .prompts import build_aggregate_prompt


@dataclass
class MoAResult:
    query: str
    final: str
    layers: list[list[str]] = field(default_factory=list)  # responses per layer
    aggregator_model: str | None = None


def _proposer_models(proposers) -> list:
    """Normalize ``proposers`` (an int count, or a list of model slugs) to a list."""
    if isinstance(proposers, int):
        if proposers < 1:
            raise ValueError("proposers must be >= 1")
        return [None] * proposers
    models = list(proposers)
    if not models:
        raise ValueError("proposers list must be non-empty")
    return models


def _propose(prompt, models, chat_fn, default_model, temperature, max_workers) -> list[str]:
    """Run every proposer on ``prompt``. Proposers are independent, so they run
    concurrently when ``max_workers > 1``; ``map`` preserves input order, keeping
    results deterministic regardless of which thread finishes first."""
    def call(m):
        return chat_fn(prompt, model=(m or default_model), temperature=temperature).strip()

    if max_workers > 1 and len(models) > 1:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(models))) as ex:
            return list(ex.map(call, models))
    return [call(m) for m in models]


def run(
    query: str,
    proposers=3,
    layers: int = 2,
    aggregator: str | None = None,
    temperature: float = 0.7,
    model: str | None = None,
    max_workers: int = 4,
    chat_fn=chat,
) -> MoAResult:
    """Run MoA for ``query``.

    ``proposers`` is the number of agents per layer (int) or a list of model slugs.
    ``layers`` is the number of proposer layers; ``aggregator`` is the model that
    produces the final synthesis (defaults to ``model``). ``max_workers`` runs the
    proposers in a layer concurrently (set to 1 to force sequential).
    """
    if layers < 1:
        raise ValueError("layers must be >= 1")
    models = _proposer_models(proposers)

    # Layer 1: answer the raw instruction.
    current = _propose(query, models, chat_fn, model, temperature, max_workers)
    trace: list[list[str]] = [current]

    # Layers 2..L: each agent refines, given the whole previous layer as context.
    for _ in range(layers - 1):
        ctx = build_aggregate_prompt(query, current)
        current = _propose(ctx, models, chat_fn, model, temperature, max_workers)
        trace.append(current)

    # Final aggregation (greedy).
    final = chat_fn(build_aggregate_prompt(query, current),
                    model=(aggregator or model), temperature=0.0).strip()

    return MoAResult(query=query, final=final, layers=trace, aggregator_model=aggregator)


def baseline(query: str, model: str | None = None, temperature: float = 0.0,
             chat_fn=chat) -> str:
    """A single-model answer to the same instruction — the baseline MoA improves on."""
    return chat_fn(query, model=model, temperature=temperature).strip()


__all__ = ["MoAResult", "run", "baseline"]
