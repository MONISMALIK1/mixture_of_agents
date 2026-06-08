# mixture_of_agents

[![tests](https://github.com/MONISMALIK1/mixture_of_agents/actions/workflows/test.yml/badge.svg)](https://github.com/MONISMALIK1/mixture_of_agents/actions/workflows/test.yml)

A from-scratch, dependency-free implementation of **Mixture-of-Agents (MoA)** —
combine several LLM responses by having an **aggregator** synthesize them, and stack
that in **layers** so each layer refines the last. On the same backend it routinely
beats any single response; the paper's open-model MoA even tops GPT-4 Omni on
AlpacaEval 2.0.

> Wang, Wang, Athiwaratkun, Zhang et al. (2024), *Mixture-of-Agents Enhances Large
> Language Model Capabilities.* [arXiv:2406.04692](https://arxiv.org/abs/2406.04692)

## The idea

```
                instruction
                    │
   Layer 1   ┌──────┼──────┐         N proposers answer independently
            A1     A2     A3
             └──────┼──────┘
                    │  (all 3 responses become context)
   Layer 2   ┌──────┼──────┐         each agent RE-answers, having seen the others
            A1     A2     A3         — "Aggregate-and-Synthesize"
             └──────┼──────┘
                    │
   Aggregator ──────┴──────►  one synthesized, high-quality answer
```

Each non-first layer (and the final aggregator) receives **all** of the previous
layer's responses and is asked to *synthesize* them — critically, not just copy —
into something better than any single one. Diversity among proposers comes from
temperature and/or using different models; the final aggregation runs greedy.

## How it differs from the other LLM repos

| | combine multiple responses by… |
| --- | --- |
| [self_consistency](https://github.com/MONISMALIK1/self_consistency) | **majority vote** over sampled chains |
| **mixture_of_agents** (this repo) | an LLM that **synthesizes** them, in **layers** |
| [chain_of_verification](https://github.com/MONISMALIK1/chain_of_verification) | the model **fact-checks** its own answer |

## Install

No third-party dependencies. Python 3.11+.

```bash
git clone https://github.com/MONISMALIK1/mixture_of_agents.git
cd mixture_of_agents && pip install -e .      # optional; or run from the parent dir
```

Point it at any OpenAI-compatible backend:

```bash
export OPENROUTER_API_KEY=sk-or-...                                   # OpenRouter (default)
# …or a local model:
export MOA_BASE_URL=http://localhost:11434/v1/chat/completions        # Ollama
export MOA_MODEL=qwen2.5:7b
```

## Use

```bash
# 3 proposers, 2 layers, default model for all
python -m mixture_of_agents "What are the trade-offs of microservices?"

# specific proposer models + a chosen aggregator, with the per-layer trace
python -m mixture_of_agents "Design a rate limiter." --layers 3 --show-trace \
  --models "openai/gpt-oss-120b:free,meta-llama/llama-3.1-8b-instruct" \
  --aggregator "openai/gpt-oss-120b:free"

# print a single-model answer next to the MoA answer, to see what aggregation adds
python -m mixture_of_agents "Summarize the CAP theorem." --baseline
```

Proposers within a layer are independent, so they run **concurrently** (a layer of
N proposers costs ~one round-trip, not N) — `run(..., max_workers=...)`, or `1` to
force sequential.

As a library:

```python
from mixture_of_agents import run

res = run("Explain why the sky is blue.", proposers=3, layers=2)
print(res.final)          # the synthesized answer
res.layers                # [[layer-1 responses], [layer-2 responses], ...]
```

## Design

The control flow is plain stdlib and unit-tested offline; only the model calls touch
the network (via the injectable `chat_fn`).

| Module | Responsibility |
| --- | --- |
| `prompts.py` | the Aggregate-and-Synthesize prompt + builder |
| `core.py` | the layered propose → aggregate loop → `MoAResult` |
| `llm.py` | backend-agnostic OpenAI-compatible client (OpenRouter or local) |

## Test

```bash
make test        # or: python -m unittest discover -s mixture_of_agents/tests -t . -v
```

13 offline tests with a scripted model: the layer/proposer structure, that non-first
layers are aggregation calls carrying the previous responses, that the aggregator
runs greedy, proposer-model routing, and input validation.

## Cost & limitations

- **It's not free.** Cost ≈ `proposers × layers + 1` model calls per query —
  `--proposers`/`--layers` trade quality for spend.
- **The aggregator is the ceiling.** A weak aggregator can't synthesize good answers
  out of good proposals; the paper finds aggregation skill matters most.
- **Diversity matters.** With one model and temperature 0, the proposers collapse to
  near-identical answers and MoA gains little — use multiple models or a higher
  temperature.

## License

MIT
