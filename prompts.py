"""The Aggregate-and-Synthesize prompt — the heart of MoA (Wang et al., 2024).

Every non-first layer (and the final aggregator) sees the previous layer's responses
through this prompt: synthesize them into something better than any one of them,
while critically evaluating each (some may be wrong). The marker line "Responses from
models:" lets the offline tests tell an *aggregation* call apart from a plain
first-layer *proposal* call (which just gets the raw instruction).
"""

AGGREGATE_PROMPT = """You have been provided with a set of responses from various models to the \
user instruction below. Your task is to synthesize them into a single, high-quality response.

Critically evaluate the responses — some may be biased or incorrect. Do not merely copy them; \
produce a refined, accurate, and comprehensive answer that is well-structured and coherent.

Responses from models:
{responses}

User instruction:
{query}

Synthesized response:"""


def build_aggregate_prompt(query: str, responses: list[str]) -> str:
    listed = "\n".join(f"{i}. {r}" for i, r in enumerate(responses, 1))
    return AGGREGATE_PROMPT.format(responses=listed, query=query)


__all__ = ["AGGREGATE_PROMPT", "build_aggregate_prompt"]
