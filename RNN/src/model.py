# To define the RNN model: embedding -> RNN -> linear

from torchinfo import summary
import torch
import torch.nn as nn
import config


class InputMethodModel(torch.nn.Module):
    # to inherit init from torch.nn.Module
    def __init__(self, vocab_size):
        """
        Initialize the InputMethodModel.
        Args:
            vocab_size (int): Size of the vocabulary.
        """
        super().__init__() # Initialize with parent class torch.nn.Module
        self.embedding = nn.Embedding(vocab_size, config.EMBEDDING_DIM)
        self.rnn = nn.RNN(input_size=config.EMBEDDING_DIM, hidden_size=config.HIDDEN_DIM, batch_first=True)
        self.linear = nn.Linear(config.HIDDEN_DIM, vocab_size)

    # forward method to define the forward pass
    def forward(self,x):
        """
        Forward propagation of the input through the model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, vocab_size)
        """
        embed = self.embedding(x)
        rnn_out,hn = self.rnn(embed)
        last_hidden_state = rnn_out[:, -1, :]
        out = self.linear(last_hidden_state)
        return out

if __name__ == '__main__':
    model = InputMethodModel(vocab_size=20000).to('cpu')

    # create dummy input for model summary
    dummy_input = torch.randint(
        low=0,
        high=20000,
        size=(config.BATCH_SIZE, config.SEQ_LEN),
        dtype=torch.long,
        device='cpu'
    )

    # print model summary
    summary(model, input_data=dummy_input)
