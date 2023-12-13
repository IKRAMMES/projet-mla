import torch
import torch.optim as optim
import torch.nn as nn
import random
import time
from data_proc import *

class Seq2SeqTrainer:
    def __init__(self, encoder_model, decoder_model, max_sequence_length, start_token, end_token, device='cpu'):
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
        self.max_sequence_length = max_sequence_length
        self.start_token = start_token
        self.end_token = end_token
        self.device = device

    def train_batch(self, input_sequences, target_sequences, encoder_optimizer, decoder_optimizer, loss_criterion, probability=0.5):
        input_len = input_sequences.size(1)
        target_len = target_sequences.size(1)
        total_loss = 0
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        encoder_hidden_states = torch.zeros(input_sequences.size(0), self.max_sequence_length, self.encoder_model.hidden_size * 2, device=self.device)

        hidden_state = self.encoder_model.init_hidden()

        for i in range(input_len):
            encoder_output, hidden_state = self.encoder_model(input_sequences[:, i])
            encoder_hidden_states[:, i] = encoder_output[:, 0]

        current_input = torch.tensor([[self.start_token]] * input_sequences.size(0), device=self.device)

        hidden_state_decoder = hidden_state

        for j in range(target_len):
            decoder_output, hidden_state_decoder, _ = self.decoder_model(current_input, hidden_state_decoder, encoder_hidden_states)
            total_loss += loss_criterion(decoder_output, target_sequences[:, j])

            explore_decision = (torch.rand(input_sequences.size(0)).to(self.device) < probability)
            current_input = torch.where(explore_decision.unsqueeze(1), target_sequences[:, j], decoder_output.argmax(dim=1))

            if current_input.item() == self.end_token:
                break

        total_loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        return total_loss.item() / target_len

    def custom_training_function(self, num_epochs, batch_size, print_interval=500, pairs=0, plot_interval=50, learning_rate=0.01):
        all_losses = []
        current_loss_sum = 0
        plot_loss_sum = 0

        optimizer_encoder = optim.Adam(self.encoder_model.parameters(), lr=learning_rate)
        optimizer_decoder = optim.Adam(self.decoder_model.parameters(), lr=learning_rate)

        for epoch in range(1, num_epochs + 1):
            random.shuffle(pairs)
            mini_batches = [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size)]

            for mini_batch in mini_batches:
                input_data = [tensorsFromPair(pair) for pair in mini_batch]
                target_data = [tensorsFromPair(pair) for pair in mini_batch]

                input_tensor = torch.stack([data[0] for data in input_data], dim=0)
                target_tensor = torch.stack([data[1] for data in target_data], dim=0)

                loss = self.train_batch(input_tensor, target_tensor, optimizer_encoder, optimizer_decoder, nn.NLLLoss())
                current_loss_sum += loss
                plot_loss_sum += loss

                if epoch % print_interval == 0:
                    avg_print_loss = current_loss_sum / print_interval
                    current_loss_sum = 0
                    print('%s (%d %d%%) %.4f' % (self.calculate_time_elapsed(time.time(), epoch / num_epochs),
                                                 epoch, epoch / num_epochs * 100, avg_print_loss))

                if epoch % plot_interval == 0:
                    avg_plot_loss = plot_loss_sum / plot_interval
                    all_losses.append(avg_plot_loss)
                    plot_loss_sum = 0

        print("Loss Visualization: ", all_losses)

    def calculate_time_elapsed(self, start, progress):
        elapsed_seconds = time.time() - start
        remaining_seconds = (elapsed_seconds / progress) * (1 - progress)
        return f'Time Elapsed: {int(elapsed_seconds)}s, Remaining: {int(remaining_seconds)}s'



