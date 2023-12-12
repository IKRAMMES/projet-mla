import unittest
import torch
from decoder import Decoder

class TestAttentionDecoder(unittest.TestCase):
    def setUp(self):
        self.hidden_size = 16
        self.output_size = 10
        self.max_length = 20
        self.dropoutP = 0.1

        self.decoder = Attention_Decoder(self.hidden_size, self.output_size, self.max_length, self.dropoutP)

        self.input_seq = torch.tensor([1], dtype=torch.long)
        self.prev_hidden = torch.randn(1, 1, self.hidden_size * 2)
        self.encoder_outputs = torch.randn(1, self.max_length, self.hidden_size * 2)
        self.encoder_outputs = self.encoder_outputs.unsqueeze(0)
        self.encoder_outputs = self.encoder_outputs.permute(0, 3, 1, 2)
        self.encoder_outputs = self.encoder_outputs.squeeze(0)  


    def test_forward(self):
        output, hidden, attn_weights = self.decoder(self.input_seq, self.prev_hidden, self.encoder_outputs)
        self.assertIsInstance(output, torch.Tensor)
        self.assertIsInstance(hidden, torch.Tensor)
        self.assertIsInstance(attn_weights, torch.Tensor)

    def test_init_hidden(self):
        initial_hidden = self.decoder.initHidden()
        self.assertIsInstance(initial_hidden, torch.Tensor)

if __name__ == '__main__':
    unittest.main()