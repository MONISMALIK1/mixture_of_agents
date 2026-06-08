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


def _propose(prompt, models, chat_fn, default_model, temperature) -> list[str]:
    return [
        chat_fn(prompt, model=(m or default_model), temperature=temperature).strip()
        for m in models
    ]


def run(
    query: str,
    proposers=3,
    layers: int = 2,
    aggregator: str | None = None,
    temperature: float = 0.7,
    model: str | None = None,
    chat_fn=chat,
) -> MoAResult:
    """Run MoA for ``query``.

    ``proposers`` is the number of agents per layer (int) or a list of model slugs.
    ``layers`` is the number of proposer layers; ``aggregator`` is the model that
    produces the final synthesis (defaults to ``model``).
    """
    if layers < 1:
        raise ValueError("layers must be >= 1")
    models = _proposer_models(proposers)

    # Layer 1: answer the raw instruction.
    current = _propose(query, models, chat_fn, model, temperature)
    trace: list[list[str]] = [current]

    # Layers 2..L: each agent refines, given the whole previous layer as context.
    for _ in range(layers - 1):
        ctx = build_aggregate_prompt(query, current)
        current = _propose(ctx, models, chat_fn, model, temperature)
        trace.append(current)

    # Final aggregation (greedy).
    final = chat_fn(build_aggregate_prompt(query, current),
                    model=(aggregator or model), temperature=0.0).strip()

    return MoAResult(query=query, final=final, layers=trace, aggregator_model=aggregator)


__all__ = ["MoAResult", "run"]
