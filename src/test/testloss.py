import unittest

import torch
from torch import nn

from metrics.losses import Loss


class TestLoss(unittest.TestCase):
    def setUp(self):
        # Create sample tensors for testing
        self.x = torch.randn(2, 5, 10)  # Example of tensor x
        self.y = torch.randint(0, 10, (2, 5))  # Example of tensor y

        # Create an instance of the Loss class for testing
        self.loss_instance = Loss(loss_fn=nn.CrossEntropyLoss())

    def test_initialization(self):
        # Check if the attribute is correctly initialized
        assert isinstance(self.loss_instance.loss_fn, nn.CrossEntropyLoss)

    def test_forward_pass(self):
        # Execute forward pass with class method
        result = self.loss_instance(self.x, self.y)

        # Add assertions based on expected results
        self.assertIsInstance(result, torch.Tensor)
        self.assertTrue(not (result.requires_grad))
        self.assertFalse(torch.isnan(result).any())


# Create a test runner and run the tests
runner = unittest.TextTestRunner()
result = runner.run(unittest.makeSuite(TestLoss))


class TestNLLLoss(unittest.TestCase):
    def setUp(self):
        # Create sample tensors for testing
        self.x = torch.rand(2, 5, 10)  # Example of tensor x
        self.y = torch.randint(0, 10, (2, 5))  # # Example of tensor y

        # Create an instance of the Loss class for testing
        self.loss_instance = Loss(loss_fn=nn.NLLLoss())

    def test_initialization(self):
        # Check if the attribute is correctly initialized
        assert isinstance(self.loss_instance.loss_fn, nn.NLLLoss)

    def test_forward_pass(self):
        # Execute forward pass with class method
        result = self.loss_instance(self.x, self.y)

        # Add assertions based on expected results
        self.assertIsInstance(result, torch.Tensor)
        self.assertTrue(not (result.requires_grad))
        self.assertFalse(torch.isnan(result).any())


# Create a test runner and run the tests
runner = unittest.TextTestRunner()
result = runner.run(unittest.makeSuite(TestNLLLoss))