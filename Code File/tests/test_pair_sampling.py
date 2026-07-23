import unittest
from pathlib import Path
from src.fer_pair_dataloader import FERSample, FERPairDataset
class FERPairDatasetTests(unittest.TestCase):
    def test_pairs_have_same_expression_and_different_identity(self):
        dataset = FERPairDataset(
            [
                FERSample(Path("a1.jpg"), "happy", "person_a"),
                FERSample(Path("a2.jpg"), "happy", "person_a"),
                FERSample(Path("b1.jpg"), "happy", "person_b"),
                FERSample(Path("c1.jpg"), "sad", "person_c"),
                FERSample(Path("d1.jpg"), "sad", "person_d"),
            ],
            seed=7,
        )

        for anchor_pos in range(len(dataset)):
            anchor_index = dataset.valid_anchor_indices[anchor_pos]
            partner_index = dataset._sample_partner_index(anchor_index)
            anchor = dataset.samples[anchor_index]
            partner = dataset.samples[partner_index]

            self.assertEqual(anchor.expression, partner.expression)
            self.assertNotEqual(anchor.identity, partner.identity)

    def test_single_identity_expression_is_not_a_valid_anchor(self):
        dataset = FERPairDataset(
            [
                FERSample(Path("a1.jpg"), "happy", "person_a"),
                FERSample(Path("a2.jpg"), "happy", "person_a"),
                FERSample(Path("b1.jpg"), "sad", "person_b"),
                FERSample(Path("c1.jpg"), "sad", "person_c"),
            ]
        )

        valid_expressions = {
            dataset.samples[index].expression for index in dataset.valid_anchor_indices
        }
        self.assertEqual(valid_expressions, {"sad"})

    def test_raises_when_no_expression_has_two_identities(self):
        with self.assertRaisesRegex(ValueError, "No valid FER pairs"):
            FERPairDataset(
                [
                    FERSample(Path("a1.jpg"), "happy", "person_a"),
                    FERSample(Path("a2.jpg"), "happy", "person_a"),
                ]
            )


if __name__ == "__main__":
    unittest.main()
