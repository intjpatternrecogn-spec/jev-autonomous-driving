import unittest

from jev_driving import DrivingDecisionEngine, DrivingState


class DrivingDecisionTests(unittest.TestCase):
    def setUp(self):
        self.question = "What is the safest maneuver now?"

    def test_unavailable_lane_is_masked(self):
        state = DrivingState(15, 22, 40, 0, False, True, None, "green")
        result = DrivingDecisionEngine(0.0).decide(
            state, self.question, ["keep lane", "change left", "change right", "brake"]
        )
        self.assertFalse(result.feasible["change left"])
        self.assertEqual(result.probabilities["change left"], 0.0)

    def test_red_light_allows_braking_only(self):
        state = DrivingState(8, 14, 18, -1, True, True, None, "red")
        result = DrivingDecisionEngine(0.0).decide(
            state, self.question, ["keep lane", "change left", "brake"]
        )
        self.assertEqual(result.choice, "brake")
        self.assertEqual(result.probabilities["brake"], 1.0)

    def test_probabilities_sum_to_one(self):
        state = DrivingState(12, 20, 50, 1, True, True, None, "green")
        result = DrivingDecisionEngine(0.0).decide(
            state, self.question, ["keep lane", "change left", "change right", "accelerate"]
        )
        self.assertAlmostEqual(sum(result.probabilities.values()), 1.0)

    def test_pedestrian_masks_non_braking_actions(self):
        state = DrivingState(6, 12, 30, 0, True, True, 5.0, "green")
        result = DrivingDecisionEngine(0.0).decide(
            state, self.question, ["keep lane", "change left", "brake"]
        )
        self.assertEqual(result.choice, "brake")
        self.assertFalse(result.feasible["keep lane"])

    def test_all_infeasible_requests_review(self):
        state = DrivingState(8, 12, 10, -8, False, False, 3.0, "red")
        result = DrivingDecisionEngine().decide(
            state, self.question, ["keep lane", "change left", "change right"]
        )
        self.assertEqual(result.choice, "request_human_review")
        self.assertFalse(result.safe)

    def test_invalid_state_is_rejected(self):
        with self.assertRaises(ValueError):
            DrivingState(-1, 12, 10, 0, True, True, None, "green")


if __name__ == "__main__":
    unittest.main()
