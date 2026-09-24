# Define the LSTM-based model for review analysis

import config
import torch
from torch import nn

class ReviewAnalystModel(nn.Module):
    def __init__(self, vocab_size, padding_index):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.EMBEDDING_DIM, padding_idx=padding_index) # Use padding index 0 for <PAD> token which is a predefined location in the vocabulary.
        self.lstm = nn.LSTM(config.EMBEDDING_DIM, config.HIDDEN_DIM, batch_first=True) # hidden dim is both hidden and cell states
        self.linear = nn.Linear(config.HIDDEN_DIM, 1)

    def forward(self, x):
        # Forward pass through the model
        # x.shape: [batch_size, seq_len]
        embed = self.embedding(x)
        # embed.shape: [batch_size, seq_len, embedding_dim]
        output, (hn, _) = self.lstm(embed) # (h0, c0) initial hidden and cell states are optional and default to zeros if not provided.
        # output.shape: [batch_size, seq_len, hidden_dim]

        # Get the last hidden state excluding padding for each sequence in the batch
        batch_index = torch.arange(0, output.shape[0])
        length = (x != self.embedding.padding_idx).sum(dim=1) # Get the actual lengths of each sequence excluding padding
        last_hidden = output[batch_index, length - 1, :] # Select the last hidden state for each sequence excluding padding
        # last_hidden.shape: [batch_size, hidden_dim]
        out = self.linear(last_hidden).squeeze(-1) # Remove the last dimension to get shape [batch_size]
        # out.shape: [batch_size]
        return out