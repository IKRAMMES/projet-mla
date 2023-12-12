import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention_Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, dropoutP=0.1):
        """
        Args:
            hidden_size (int): The size of the hidden state in the decoder.
            output_size (int): The size of the vocabulary for the output sequence.
            dropout_p (float): Probability of dropout to prevent overfitting.
            max_length (int): Maximum length of input sequence for attention.
        """
        super(Attention_Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length
        self.dropoutP = dropoutP
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

        # Embedding layer to convert output sequence indices to dense vectors
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        
        # Attention mechanism components
        self.Attention_weights_layer = nn.Linear(self.hidden_size * 3, self.max_length)
        self.Attention_combine = nn.Linear(self.hidden_size * 3, self.hidden_size)
        
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(self.dropoutP)
        
        # GRU layer to process input and previous hidden state
        self.gru = nn.GRU(self.hidden_size * 2, self.hidden_size * 2)
        
        # Output layer to produce final predictions
        self.out = nn.Linear(self.hidden_size * 2, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        """
        Forward pass of the decoder.

        Args:
            input (Tensor): The input sequence at a specific time step.
            hidden (Tensor): The hidden state from the previous time step.
            encoder_outputs (Tensor): The outputs from the encoder to attend to.

        Returns:
            output (Tensor): The output sequence prediction.
            hidden (Tensor): The updated hidden state.
            attn_weights (Tensor): The attention weights applied to the encoder outputs.
        """
        # Convert output sequence index to dense vector
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        # Calculate attention weights using a linear layer
        Attention_weights = F.softmax(self.Attention_weights_layer(torch.cat((embedded[0], hidden[0]), 1)))

        # Apply attention to encoder outputs
        Attention_applied = torch.bmm(Attention_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        # Concatenate embedded input and attention-applied context vector

        # and Apply hyperbolic tangent activation function
        output = F.tanh(Attention_applied)
        
        # Process the output through the GRU
        output, hidden = self.gru(output, hidden)

        # Apply log softmax to get final predictions
        output = F.log_softmax(self.out(output[0]), dim=1)
        
        return output, hidden, Attention_weights

    def initHidden(self):
        """
        Initialize the hidden state with zeros.

        Returns:
            Tensor: The initial hidden state.
        """
        return torch.zeros(1, 1, self.hidden_size * 2, device= self.device)


