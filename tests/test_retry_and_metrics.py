import unittest

from crawler.models import StructuredResult
from crawler.nodes.metrics_evaluator import evaluate_metrics
from crawler.routing import route_after_evaluation
from crawler.state import State


class RetryRoutingTests(unittest.TestCase):
    def test_route_ends_when_no_gaps(self):
        state = State(user_query="incubators in india", missing_data_targets=[])
        self.assertEqual(route_after_evaluation(state), "__end__")

    def test_route_retries_when_budget_available(self):
        state = State(
            user_query="incubators in india",
            missing_data_targets=["Y Combinator :: Funding Amount"],
            retry_count=1,
            max_retries=2,
        )
        self.assertEqual(route_after_evaluation(state), "investigator")

    def test_route_ends_when_budget_exhausted(self):
        state = State(
            user_query="incubators in india",
            missing_data_targets=["Y Combinator :: Funding Amount"],
            retry_count=3,
            max_retries=2,
        )
        self.assertEqual(route_after_evaluation(state), "__end__")


class MetricsEvaluatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_targets_use_delimiter_and_increment_retry(self):
        state = State(
            user_query="incubators in india",
            target_metrics=["Funding Amount"],
            structured_results=[
                StructuredResult(name="Y Combinator", properties={}, relationships=[])
            ],
        )

        out = await evaluate_metrics(state)

        self.assertIn("missing_data_targets", out)
        self.assertEqual(out.get("retry_count"), 1)
        self.assertTrue(any("::" in item for item in out["missing_data_targets"]))

    async def test_max_retries_override_from_configurable(self):
        state = State(
            user_query="incubators in india",
            target_metrics=["Location"],
            structured_results=[
                StructuredResult(name="CIIE", properties={}, relationships=[])
            ],
        )

        out = await evaluate_metrics(state, config={"configurable": {"max_retries": 5}})
        self.assertEqual(out.get("max_retries"), 5)


if __name__ == "__main__":
    unittest.main()
