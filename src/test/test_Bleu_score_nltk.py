import unittest

import torch

from metrics.bleuscore import BLEUScoreNLTK


class TestBLEUScoreNLTK(unittest.TestCase):
    def setUp(self):
        # Create sample tensors for testing
        self.reference_tensor = torch.tensor([[1, 2, 3, 4, 8], [4, 5, 7, 10, 5]])
        self.candidate_tensor = torch.tensor([[1, 2, 3, 4, 8], [5, 5, 7, 10, 5]])

        # Create an instance of the BLEUScoreNLTK class for testing
        self.bleu_score_nltk = BLEUScoreNLTK(self.reference_tensor, self.candidate_tensor)

    def test_initialization(self):
        # Check if attributes are correctly initialized
        self.assertEqual(
            self.bleu_score_nltk.reference_tensor.tolist(),
            self.reference_tensor.tolist(),
        )
        self.assertEqual(
            self.bleu_score_nltk.candidate_tensor.tolist(),
            self.candidate_tensor.tolist(),
        )
        self.assertEqual(self.bleu_score_nltk.bleu_scores, [])

    def test_bleu_score_calculation(self):
        # Calculate BLUE score using class method
        result = self.bleu_score_nltk.calculate_bleu_score()

        # Add assertions based on expected results
        # For example, check if the BLUE score is within an acceptable range
        self.assertIsInstance(result, torch.Tensor)
        scores = torch.tensor([1.0000, 0.6687])
        for i, r in enumerate(result):
            self.assertGreaterEqual(r, 0.0)
            self.assertLessEqual(r, 1.0)
            self.assertAlmostEqual(r.item(), scores[i].item(), places=4)


# Create a test runner and run the tests
runner = unittest.TextTestRunner()
result = runner.run(unittest.makeSuite(TestBLEUScoreNLTK))