# classe encoder 
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        """
        Initialiseur de la classe Encoder
        Args:
            input_size (int): Taille du vocabulaire français.
            hidden_size (int): Taille des couches cachées du GRU.
        """
        super(Encoder, self).__init__()
        self.device=device
        self.hidden_size = hidden_size
        # Embedding layer to convert word indices into dense vectors.
        self.embedding = nn.Embedding(input_size, hidden_size)
        # Bidirectional GRU to capture contextual information (both past and future).
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=True)

    
    def initHidden(self):
            """
            Initialize the hidden state to 0.
            Returns : torch.Tensor: hidden state initialised to 0.
            """
            hi0= torch.zeros(1, 1, self.hidden_size, dtype=torch.float32, device=self.device)
            return hi0

    def forward(self, input):
        """
        Forward of the encoder rnn.
        Args: input (torch.Tensor): input in sequence format.

        Returns: output (torch.Tensor): Sortie du GRU.
                 hidden (torch.Tensor): Dernier état caché du GRU.
        """
        # Embedding of the input
        embedded = self.embedding(input).view(1, 1, -1)

        # Apply the GRU
        out, hidden = self.gru(embedded)

        # Reshaping the hidden state to match the expected dimensions.
        hidden = hidden.reshape((1, 1, 2 * self.hidden_size))
        return out, hidden

    




