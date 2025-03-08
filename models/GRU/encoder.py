import torch
import torch.nn as nn

class Encoder_GRU(nn.Module):
    def __init__(self,input_dim,emb_dim,hidden_dim,num_layers,dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim,emb_dim)
        self.rnn = nn.GRU(emb_dim,hidden_dim,num_layers,dropout=dropout,bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)

        self.self_attention = nn.MultiheadAttention(embed_dim=hidden_dim * 2,num_heads=4,dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self,src):
        embedded = self.dropout(self.embedding(src)) # [batch_size, seq_len, emb_dim]
        embedded = embedded.permute(1,0,2) # [seq_len, batch_size, emb_dim]

        output,hidden = self.rnn(embedded)

        hidden = hidden.view(self.num_layers,2,-1,self.hidden_dim) # [num_layers, 2,batch_size, hidden_dim]
        hidden = hidden[:, -1, :, :]

        attn_output, _ = self.self_attention(output, output, output)
        output = output + attn_output

        return output, hidden