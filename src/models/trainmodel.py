import torch
import numpy as np

def train(input_sequence, target_sequence, encoder_model, decoder_model, encoder_optimizer, decoder_optimizer, loss_criterion, max_sequence_length, start_token, end_token, device):
    # Initializations
    probability = 0.5  # Exploration probability
    input_len = input_sequence.size(0)  # Length of the input sequence
    target_len = target_sequence.size(0)  # Length of the target sequence
    total_loss = 0  # Total loss for the entire sequence
    encoder_optimizer.zero_grad()  # Zero the gradients for the encoder optimizer
    decoder_optimizer.zero_grad()  # Zero the gradients for the decoder optimizer
    encoder_hidden_states = torch.zeros(max_sequence_length, encoder_model.hidden_size * 2, device=device)  # Tensor to store encoder hidden states
    
    # Initialize the decoder
    current_input = torch.tensor([[start_token]], device=device)  # Start with the start token as the first input

    # Encoding Phase
    hidden_state = encoder_model.init_hidden()  # Initialize the hidden state of the encoder

    for i in range(input_len):
        encoder_output, hidden_state = encoder_model(input_sequence[i])  # Forward pass through the encoder
        encoder_hidden_states[i] = encoder_output[0, 0]  # Store the encoder hidden state

        # Decoder takes the last hidden state of the encoder
        hidden_state_decoder = hidden_state

        # Decoding Phase
        for j in range(target_len):
            decoder_output, hidden_state_decoder, _ = decoder_model(current_input, hidden_state_decoder, encoder_hidden_states)
            total_loss += loss_criterion(decoder_output, target_sequence[j])  # Compute loss at each step
            
            explore_decision = (np.random.rand() < probability)  # Exploration-exploitation decision
            if explore_decision:
                current_input = target_sequence[j]  # Exploration: use the target as the next input
            else:
                topv, topi = decoder_output.topk(1)
                current_input = topi.squeeze().detach()  # Exploitation: use the top predicted token as the next input, detach from history

            if current_input.item() == end_token:  # If the end token is predicted, break out of the loop
                break

    # Backpropagation and weight update
    total_loss.backward()  # Backward pass to compute gradients
    encoder_optimizer.step()  # Update encoder weights
    decoder_optimizer.step()  # Update decoder weights

    return total_loss.item() / target_len  # Return the average loss per target sequence length

