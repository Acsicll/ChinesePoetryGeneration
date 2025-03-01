import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 3, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, hidden, encoder_outputs):
        # hidden shape [num_layers, batch_size, hidden_dim]
        # encoder_outputs shape [seq_len, batch_size, hidden_dim * 2]
        hidden = hidden[-1].unsqueeze(1).repeat(1, encoder_outputs.shape[0], 1) # [batch_size, seq_len, hidden_dim]
        encoder_outputs = encoder_outputs.permute(1, 0, 2) # [batch_size, seq_len, hidden_dim * 2]
        energy = torch.tanh(self.attn(torch.cat((hidden,encoder_outputs),dim=2)))
        attention = torch.sum(self.v * energy, dim=2)
        return F.softmax(attention, dim=1)