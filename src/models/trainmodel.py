import torch
import torch
import numpy as np
import time
import torch.optim as optim
import torch.nn as nn
import random
from data_proc import *
from sklearn.model_selection import train_test_split


class Seq2SeqTrainer:
    """
    Utility for training sequence-to-sequence models.

    Args:
    - encoder_model (nn.Module): The encoder module.
    - decoder_model (nn.Module): The decoder module.
    - max_sequence_length (int): The maximum length of the output sequence.
    - start_token (int): The token representing the start of a sequence.
    - end_token (int): The token representing the end of a sequence.
    - input_lang (Lang): The language object for input sentences.
    - output_lang (Lang): The language object for target sentences.

    Methods:
    - train(input_sequence, target_sequence, encoder_optimizer, decoder_optimizer, loss_criterion, probability=0.5):
        Trains the models for a given input-target sequence pair.
    - custom_training_function(num_epochs, pairs, print_interval=500, plot_interval=50, learning_rate=0.01):
        Custom training function for the models over multiple epochs.
    - calculate_time_elapsed(start, progress):
        Calculates the elapsed and remaining time during training.
    """

    def __init__(self, encoder_model, decoder_model, max_sequence_length, start_token, end_token, input_lang,
                 output_lang, device):
        self.encoder_model = encoder_model
        # self.decoder_type = decoder_type
        self.max_sequence_length = max_sequence_length
        self.start_token = start_token
        self.end_token = end_token
        self.device = device
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.decoder_model = decoder_model

        '''
        if self.decoder_type == 'simple':
            self.decoder_model = Decoder(encoder_model.hidden_size * 2, decoder_model.output_size)
        elif self.decoder_type == 'attention':
            self.decoder_model = Attention_Decoder(encoder_model.hidden_size, decoder_model.output_size, max_sequence_length)
        else:
            raise ValueError("Invalid decoder type. Supported types are 'simple' and 'attention'.")'''

    def train(self, input_sequence, target_sequence, encoder_optimizer, decoder_optimizer, loss_criterion, istraining=1,
              probability=0.5):
        """
        Trains the encoder-decoder models for a given input-target sequence pair.

        Args:
        - input_sequence (torch.Tensor): Tensor representing the input sequence.
        - target_sequence (torch.Tensor): Tensor representing the target sequence.
        - encoder_optimizer (torch.optim.Optimizer): Optimizer for the encoder model.
        - decoder_optimizer (torch.optim.Optimizer): Optimizer for the decoder model.
        - loss_criterion (torch.nn.Module): Loss criterion for training.
        - probability (float): Probability for exploration during training.

        Returns:
        - float: Normalized loss for the given sequence pair.
        """
        # Initializations
        input_len = input_sequence.size(0)
        target_len = target_sequence.size(0)
        total_loss = 0
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        encoder_outputs = torch.zeros(self.max_sequence_length, self.encoder_model.hidden_size * 2, device=self.device)

        # Initialize the decoder
        current_input = torch.tensor([[self.start_token]], device=self.device)

        for i in range(input_len):
            encoder_output, encoder_hidden_state = self.encoder_model(input_sequence[i])
            encoder_outputs[i] = encoder_output[0, 0]

        # Decoder takes the last hidden state of the encoder
        hidden_state_decoder = encoder_hidden_state

        # explore_decision = True if np.random.rand() < probability else False

        if istraining:
            # Decoding Phase
            for j in range(target_len):
                decoder_output, hidden_state_decoder, _ = self.decoder_model(current_input, hidden_state_decoder,
                                                                             encoder_outputs)
                total_loss += loss_criterion(decoder_output, target_sequence[j])
                current_input = target_sequence[j]
        else:
            for j in range(target_len):
                decoder_output, hidden_state_decoder, _ = self.decoder_model(current_input, hidden_state_decoder,
                                                                             encoder_outputs)
                topv, topi = decoder_output.topk(1)
                current_input = topi.squeeze().detach()
                total_loss += loss_criterion(decoder_output, target_sequence[j])

                if current_input.item() == self.end_token:
                    break

        # Backpropagation and weight update
        total_loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        return total_loss.item() / target_len

    def custom_training_function(self, num_epochs, pairs, print_interval=500, plot_interval=50, learning_rate=0.01):
        """
        Custom training function for the encoder-decoder models.

        Args:
        - num_epochs (int): Number of training epochs.
        - pairs (list): List of input-target sequence pairs.
        - print_interval (int): Interval for printing training progress.
        - plot_interval (int): Interval for plotting training losses.
        - learning_rate (float): Learning rate for optimization.

        Returns:
        - float: Average loss over the training process.
        """
        # Initialize loss tracking variables
        all_train_losses=[]
        all_val_losses = []
        current_loss_sum = 0
        current_val_loss_sum = 0
        plot_loss_sum = 0
        plot_val_loss_sum= 0
        start = time.time()

        # Initialize SGD optimizers for the two networks with the specified learning rate
        optimizer_encoder = optim.SGD(self.encoder_model.parameters(), lr=learning_rate)
        optimizer_decoder = optim.SGD(self.decoder_model.parameters(), lr=learning_rate)

        # Split data into 80% training and 20% validation
        pairs_train, pairs_val = train_test_split(pairs, test_size=0.2, random_state=42)
        
        # Generate training data using random pairs and define the loss criterion
        training_data = [tensorsFromPair(random.choice(pairs_train), self.input_lang, self.output_lang, self.device) for _ in
                         range(num_epochs)]
        
        # Generate validation data using random pairs from the validation set
        validation_data = [tensorsFromPair(random.choice(pairs_val), self.input_lang, self.output_lang, self.device) for _ in range(num_epochs)]

        
        for epoch in range(1, num_epochs + 1):
            data_point = training_data[epoch - 1]
            input_data = data_point[0]
            target_data = data_point[1]

            train_loss = self.train(input_data, target_data, optimizer_encoder, optimizer_decoder, nn.NLLLoss(), 1)
            current_loss_sum += train_loss
            plot_loss_sum += train_loss
            
            # Use validation data for validation
            val_data_point = validation_data[epoch - 1]
            val_input_data = val_data_point[0]
            val_target_data = val_data_point[1]

            val_loss = self.train(val_input_data, val_target_data, optimizer_encoder, optimizer_decoder, nn.NLLLoss(), 0)
            
            current_val_loss_sum += val_loss
            plot_val_loss_sum += val_loss
            
            if epoch % print_interval == 0:
                avg_print_train_loss = current_loss_sum / print_interval
                avg_print_val_loss= current_val_loss_sum / print_interval
                current_loss_sum = 0
                current_val_loss_sum=0
                print('%s (Epochs %d%%) Loss Train: %.4f | Loss Validation: %.4f' % (self.calculate_time_elapsed(start, epoch / num_epochs),
                                                epoch / num_epochs * 100, avg_print_train_loss, avg_print_val_loss))

            # Track losses for plotting at specified intervals
            if epoch % plot_interval == 0:
                avg_plot_train_loss = plot_loss_sum / plot_interval
                all_train_losses.append(avg_plot_train_loss)

                avg_plot_val_loss= plot_val_loss_sum / plot_interval
                all_val_losses.append(avg_plot_val_loss)

        return all_train_losses, all_val_losses

    # Function to calculate elapsed and remaining time
    def calculate_time_elapsed(self, start, progress):
        """
        Calculates the elapsed and remaining time during training.

        Args:
        - start (float): Start time of training.
        - progress (float): Training progress as a fraction.

        Returns:
        - str: Formatted string indicating elapsed and remaining time.
        """
        elapsed_seconds = time.time() - start
        remaining_seconds = (elapsed_seconds / progress) * (1 - progress)
        return f'Time Elapsed: {int(elapsed_seconds)}s, Remaining: {int(remaining_seconds)}s'


  




