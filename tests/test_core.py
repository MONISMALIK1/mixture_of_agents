"""The MoA control flow with a scripted model (offline).

Asserts the layered structure: N proposers per layer, the right number of layers,
that each non-first layer's call is an *aggregation* prompt carrying the previous
layer's responses, and that the final answer comes from the aggregator.
"""

import unittest

from mixture_of_agents.core import run


def make_fake():
    """Records (prompt, model, temperature); proposals vs aggregations by marker."""
    calls = []

    def fake(prompt, model=None, temperature=0.7, **kw):
        calls.append({"prompt": prompt, "model": model, "temperature": temperature})
        if "Responses from models:" in prompt:        # AGGREGATE_PROMPT
            return "SYNTHESIZED"
        return "PROPOSAL"

    return fake, calls


class StructureTests(unittest.TestCase):
    def test_layers_and_proposer_counts(self):
        fake, _ = make_fake()
        res = run("why is the sky blue?", proposers=3, layers=2, chat_fn=fake)
        self.assertEqual(len(res.layers), 2)
        self.assertEqual([len(layer) for layer in res.layers], [3, 3])
        self.assertEqual(res.layers[0], ["PROPOSAL", "PROPOSAL", "PROPOSAL"])
        self.assertEqual(res.final, "SYNTHESIZED")

    def test_first_layer_sees_raw_query_then_aggregation(self):
        fake, calls = make_fake()
        run("Q?", proposers=2, layers=2, chat_fn=fake)
        # 2 proposers (raw) + 2 second-layer (aggregate) + 1 final (aggregate) = 5
        self.assertEqual(len(calls), 5)
        self.assertEqual(sum("Responses from models:" not in c["prompt"] for c in calls), 2)
        # the final aggregation prompt carries the previous layer's responses
        self.assertIn("SYNTHESIZED", calls[-1]["prompt"])  # layer-2 outputs fed to final

    def test_single_layer_is_propose_then_aggregate(self):
        fake, calls = make_fake()
        res = run("Q?", proposers=2, layers=1, chat_fn=fake)
        self.assertEqual(len(res.layers), 1)
        self.assertEqual(len(calls), 3)  # 2 proposals + 1 final aggregation

    def test_final_aggregation_is_greedy(self):
        fake, calls = make_fake()
        run("Q?", proposers=2, layers=1, chat_fn=fake)
        self.assertEqual(calls[-1]["temperature"], 0.0)  # aggregator runs greedy


class ProposerModelTests(unittest.TestCase):
    def test_model_list_used_for_proposers(self):
        fake, calls = make_fake()
        run("Q?", proposers=["m-a", "m-b"], layers=1, chat_fn=fake)
        layer1_models = [c["model"] for c in calls[:2]]
        self.assertEqual(layer1_models, ["m-a", "m-b"])

    def test_aggregator_model_used_for_final(self):
        fake, calls = make_fake()
        run("Q?", proposers=2, layers=1, aggregator="agg-model", chat_fn=fake)
        self.assertEqual(calls[-1]["model"], "agg-model")

    def test_validation(self):
        with self.assertRaises(ValueError):
            run("Q?", proposers=0, chat_fn=make_fake()[0])
        with self.assertRaises(ValueError):
            run("Q?", layers=0, chat_fn=make_fake()[0])
        with self.assertRaises(ValueError):
            run("Q?", proposers=[], chat_fn=make_fake()[0])


if __name__ == "__main__":
    unittest.main()
