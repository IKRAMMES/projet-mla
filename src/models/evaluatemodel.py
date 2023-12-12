from trainmodel import Seq2SeqTrainer
import torch 
from data_processing.data_proc import Preprocessor
class MySeq2SeqModelEvaluation(nn.Module):
    def __init__(self, encoder, decoder, max_length, input_lang, output_lang, start_token, end_token, device):
        super(MySeq2SeqModelEvaluation, self).__init__()
        # Initialize model components
        self.encoder = encoder
        self.decoder = decoder
        self.max_length = max_length
        self.device = device
        self.input_lang = input_lang
        self.start_token = start_token
        self.end_token = end_token
        self.output_lang = output_lang

    def forward(self, sentence):
        with torch.no_grad():
            # Convert the input sentence into a tensor usable by the model
            input_tensor = tensorFromSentence(self.input_lang, sentence)
            input_length = input_tensor.size()[0]

            # Initialize the hidden state of the encoder
            encoder_hidden = self.encoder.initHidden()

            # Initialize the tensor for encoder outputs
            encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size * 2, device=self.device)

            # Encode the input sequence
            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder(input_tensor[ei])
                encoder_outputs[ei] = encoder_output[0, 0]

            # Initialize the decoder input with the start token
            decoder_input = torch.tensor([[self.start_token]], device=self.device)
            decoder_hidden = encoder_hidden  # Use the final hidden state of the encoder

            decoded_words = []  # List to store decoded words
            decoder_attentions = torch.zeros(self.max_length, self.max_length)

            # Iterative decoding until reaching the maximum length or the end token
            for di in range(self.max_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data

                topv, topi = decoder_output.data.topk(1)
                if topi.item() == self.end_token:
                    # Stop if the end token is reached
                    decoded_words.append('<EOS>')
                    break
                else:
                    # Add the decoded word to the list
                    decoded_words.append(self.output_lang.index2word[topi.item()])

                # Update the decoder input for the next iteration
                decoder_input = topi.squeeze().detach()

        # Return the decoded words and decoder attentions
        return decoded_words, decoder_attentions[:di + 1]