import unittest
import torch
from src.dicefer.losses import dv_bound, shuffle_batch


class DonskerVaradhanTests(unittest.TestCase):
    def test_zero_scores_have_zero_bound(self):
        scores = torch.zeros(8, 1)
        self.assertAlmostEqual(float(dv_bound(scores, scores)), 0.0, places=6)

    def test_known_global_bound(self):
        joint = torch.ones(8, 1)
        marginal = torch.zeros(8, 1)
        self.assertAlmostEqual(float(dv_bound(joint, marginal)), 1.0, places=6)

    def test_local_bound_sums_spatial_sites(self):
        joint = torch.ones(8, 1, 2, 2)
        marginal = torch.zeros(8, 1, 2, 2)
        self.assertAlmostEqual(
            float(dv_bound(joint, marginal, sum_spatial=True)),
            4.0,
            places=6,
        )

    def test_shuffle_is_a_derangement(self):
        values = torch.arange(32)
        shuffled = shuffle_batch(values)
        self.assertTrue(torch.all(values != shuffled))
        self.assertEqual(sorted(values.tolist()), sorted(shuffled.tolist()))


if __name__ == "__main__":
    unittest.main()
