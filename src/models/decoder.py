import torch
import torch.nn as nn
import torch.nn.functional as F

'''
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        """
        Args:
            hidden_size (int): Size of the hidden state in the decoder.
            output_size (int): Size of the vocabulary for the output sequence.
        """
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        # Embedding layer to convert output sequence indices to dense vectors
        self.embedding = nn.Embedding(output_size, 3*hidden_size)
        
        # GRU layer to process input and previous hidden state
        # Input size: embedding_size (hidden_size), 2*  hidden_size (previous hidden state) = for forward & backward 
        self.gru = nn.GRU(hidden_size * 3, hidden_size)

        # Output layer to produce final predictions
        # Input size: hidden_size, output_size (vocabulary size)
        self.out = nn.Linear(hidden_size, output_size)

        # Softmax activation for probability distribution over the vocabulary
        self.softmax = nn.LogSoftmax(dim=1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, input, hidden):
        """
        Forward pass of the decoder.

        Args:
            input (Tensor): The input sequence at a specific time step.
            hidden (Tensor): The hidden state from the previous time step.

        Returns:
            output (Tensor): The output sequence prediction.
            hidden (Tensor): The updated hidden state.
        """

        # Convert output sequence index to dense vector using embedding layer
        embedded = self.embedding(input).view(1, 1, -1)

        # Apply ReLU activation
        embedded = F.relu(embedded) 

        # Process the input through the GRU layer
        output, hidden = self.gru(embedded, hidden)

        # Apply linear layer and softmax to get final predictions
        output = self.softmax(self.out(output[0]))

        return output, hidden

    def initHidden(self):
        """
        Initialize the hidden state with zeros.

        Returns:
            Tensor: The initial hidden state.
        """
        return torch.zeros(1, 1, self.hidden_size, device=self.device)
'''

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropoutP=0.1):
        """
        Args:
            hidden_size (int): The size of the hidden state in the decoder.
            output_size (int): The size of the vocabulary for the output sequence.
            dropout_p (float): Probability of dropout to prevent overfitting.
        """
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropoutP = dropoutP
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Embedding layer to convert output sequence indices to dense vectors
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)

        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(self.dropoutP)

        # GRU layer to process input and previous hidden state
        self.gru = nn.GRU(self.hidden_size * 2, self.hidden_size * 2)

        # Output layer to produce final predictions
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden):
        """
        Forward pass of the decoder.

        Args:
            input (Tensor): The input sequence at a specific time step.
            hidden (Tensor): The hidden state from the previous time step.

        Returns:
            output (Tensor): The output sequence prediction.
            hidden (Tensor): The updated hidden state.
        """
        # Convert output sequence index to dense vector
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        # Process the output through the GRU
        output, hidden = self.gru(2*embedded, hidden)

        # Apply log softmax to get final predictions
        output = F.log_softmax(self.out(output[0]), dim=1)

        return output, hidden

    def initHidden(self):
        """
        Initialize the hidden state with zeros.

        Returns:
            Tensor: The initial hidden state.
        """
        return torch.zeros(1, 1, self.hidden_size*2, device=self.device)
