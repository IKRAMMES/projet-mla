import unittest
import torch
from decoder import Decoder  # Replace 'your_module' with the actual name of your module

class TestDecoder(unittest.TestCase):
    def setUp(self):
        self.hidden_size = 10
        self.output_size = 5
        self.decoder = Decoder(self.hidden_size, self.output_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_forward(self):
        input_seq = torch.tensor([2], dtype=torch.long, device=self.device)  # Replace [2] with an example sequence
        prev_hidden = self.decoder.initHidden()

        with torch.no_grad():
            output, hidden = self.decoder(input_seq, prev_hidden)

        # Ensure the output shape is correct
        self.assertEqual(output.size(), torch.Size([1, self.output_size]))

    def test_init_hidden(self):
        initial_hidden = self.decoder.initHidden()

        # Ensure the initial hidden state has the correct shape
        self.assertEqual(initial_hidden.size(), torch.Size([1, 1, self.hidden_size]))

if __name__ == '__main__':
    unittest.main()