import torch
import numpy as np
import time
import torch.optim as optim
import torch.nn as nn
import random
from data_proc import *

from decoderattn import Attention_Decoder
from decoder import Decoder


class Seq2SeqTrainer:
    def __init__(self, encoder_model, decoder_model, max_sequence_length, start_token, end_token, decoder_type='simple', device='cpu'):
        self.encoder_model = encoder_model
        self.decoder_type = decoder_type
        self.max_sequence_length = max_sequence_length
        self.start_token = start_token
        self.end_token = end_token
        self.device = device

        if decoder_type == 'simple':
            self.decoder_model = Decoder(encoder_model.hidden_size * 2, decoder_model.output_size)
        elif decoder_type == 'attention':
            self.decoder_model = Attention_Decoder(encoder_model.hidden_size, decoder_model.output_size, max_sequence_length)
        else:
            raise ValueError("Invalid decoder type. Supported types are 'simple' and 'attention'.")

    def train(self, input_sequence, target_sequence, encoder_optimizer, decoder_optimizer, loss_criterion, probability=0.5):
        # Initializations
        input_len = input_sequence.size(0)
        target_len = target_sequence.size(0)
        total_loss = 0
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        encoder_hidden_states = torch.zeros(self.max_sequence_length, self.encoder_model.hidden_size * 2, device=self.device)

        # Initialize the decoder
        current_input = torch.tensor([[self.start_token]], device=self.device)

        # Encoding Phase
        hidden_state = self.encoder_model.init_hidden()

        for i in range(input_len):
            encoder_output, hidden_state = self.encoder_model(input_sequence[i])
            encoder_hidden_states[i] = encoder_output[0, 0]

            # Decoder takes the last hidden state of the encoder
            hidden_state_decoder = hidden_state

            # Decoding Phase
            for j in range(target_len):
                decoder_output, hidden_state_decoder, _ = self.decoder_model(current_input, hidden_state_decoder, encoder_hidden_states)
                total_loss += loss_criterion(decoder_output, target_sequence[j])

                explore_decision = (np.random.rand() < probability)
                if explore_decision:
                    current_input = target_sequence[j]
                else:
                    topv, topi = decoder_output.topk(1)
                    current_input = topi.squeeze().detach()

                if current_input.item() == self.end_token:
                    break

        # Backpropagation and weight update
        total_loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        return total_loss.item() / target_len

    def custom_training_function(self, num_epochs, print_interval=500, pairs=0, plot_interval=50, learning_rate=0.01):
        # Initialize loss tracking variables
        all_losses = []
        current_loss_sum = 0
        plot_loss_sum = 0

        # Initialize Adam optimizers for the two networks with the specified learning rate
        optimizer_encoder = optim.Adam(self.encoder_model.parameters(), lr=learning_rate)
        optimizer_decoder = optim.Adam(self.decoder_model.parameters(), lr=learning_rate)

        # Generate training data using random pairs and define the loss criterion
        training_data = [[tensorsFromPair(random.choice(pairs))] for _ in range(num_epochs)]

        for epoch in range(1, num_epochs + 1):
            data_point = training_data[epoch - 1]
            input_data = data_point[0]
            target_data = data_point[1]

            loss = self.train(input_data, target_data, optimizer_encoder, optimizer_decoder, nn.NLLLoss())
            current_loss_sum += loss
            plot_loss_sum += loss

            # Print the average loss at specified intervals
            if epoch % print_interval == 0:
                avg_print_loss = current_loss_sum / print_interval
                current_loss_sum = 0
                print('%s (%d %d%%) %.4f' % (self.calculate_time_elapsed(time.time(), epoch / num_epochs),
                                             epoch, epoch / num_epochs * 100, avg_print_loss))

            # Track losses for plotting at specified intervals
            if epoch % plot_interval == 0:
                avg_plot_loss = plot_loss_sum / plot_interval
                all_losses.append(avg_plot_loss)
                plot_loss_sum = 0

        # Print the list of losses for visualization
        print("Loss Visualization: ", all_losses)

    # Function to calculate elapsed and remaining time
    def calculate_time_elapsed(self, start, progress):
        elapsed_seconds = time.time() - start
        remaining_seconds = (elapsed_seconds / progress) * (1 - progress)
        return f'Time Elapsed: {int(elapsed_seconds)}s, Remaining: {int(remaining_seconds)}s'
  




