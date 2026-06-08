"""mixture_of_agents — MoA from scratch, dependency-free.

Combine several LLM "proposer" responses with an "aggregator" that synthesizes them,
stacked in layers — each layer's agents see the previous layer's outputs. Often beats
any single model on the same backend.

Reference: Wang et al., 2024, "Mixture-of-Agents Enhances Large Language Model
Capabilities", https://arxiv.org/abs/2406.04692

    from mixture_of_agents import run

    result = run("Explain why the sky is blue.", proposers=3, layers=2)
    print(result.final)
"""

from __future__ import annotations

__version__ = "0.1.0"

from .core import MoAResult, baseline, run
from .prompts import build_aggregate_prompt

__all__ = ["__version__", "MoAResult", "run", "baseline", "build_aggregate_prompt"]
