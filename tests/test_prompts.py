"""The aggregate-and-synthesize prompt builder."""

import unittest

from mixture_of_agents.prompts import AGGREGATE_PROMPT, build_aggregate_prompt


class PromptTests(unittest.TestCase):
    def test_placeholders(self):
        self.assertIn("{responses}", AGGREGATE_PROMPT)
        self.assertIn("{query}", AGGREGATE_PROMPT)

    def test_numbers_the_responses(self):
        out = build_aggregate_prompt("What is 2+2?", ["four", "4", "the answer is four"])
        self.assertIn("1. four", out)
        self.assertIn("2. 4", out)
        self.assertIn("3. the answer is four", out)
        self.assertIn("What is 2+2?", out)

    def test_has_aggregation_marker(self):
        # the marker the tests/router rely on to spot an aggregation call
        self.assertIn("Responses from models:", build_aggregate_prompt("q", ["a"]))


if __name__ == "__main__":
    unittest.main()
