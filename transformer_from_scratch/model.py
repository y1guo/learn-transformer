import torch
from torch import nn
from utils import DEVICE


MAX_SEQ_LEN = 128  # need fix


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos = torch.arange(MAX_SEQ_LEN)[None, :, None]
        idx = torch.arange(0, d_model, 2)[None, None, :]
        wavlen = 10000 ** (idx / d_model)
        pos_encoding = torch.zeros(1, MAX_SEQ_LEN, d_model)
        pos_encoding[:, :, 0::2] = torch.sin(pos / wavlen)
        pos_encoding[:, :, 1::2] = torch.cos(pos / wavlen)
        self.pos_encoding = pos_encoding.to(DEVICE)

    def forward(self, x: torch.Tensor):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pos_encoding[:, : x.size(1), :]
        return self.dropout(x)
        # Question: There's no weights to learn in this layer, so why dropout?


class InverseEmbedding(nn.Module):
    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.embedding = embedding

    def forward(self, x: torch.Tensor):
        # x: (batch_size, seq_len, d_model)
        return x @ self.embedding.weight.data.T


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        # hyperparameters
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        # embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        # self.inverse_embedding = InverseEmbedding(self.embedding)
        self.inverse_embedding = nn.Linear(d_model, vocab_size)
        # positional encoding layer
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        # transformer layers
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.tgt_mask = torch.triu(torch.full((MAX_SEQ_LEN, MAX_SEQ_LEN), True), diagonal=1).to(DEVICE)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ):
        # src, tgt: (batch_size, seq_len)
        sqrt_d_model = torch.tensor(self.d_model**0.5).to(DEVICE)
        src = self.positional_encoding(self.embedding(src) * sqrt_d_model)
        tgt = self.positional_encoding(self.embedding(tgt) * sqrt_d_model)
        # src, tgt: (batch_size, seq_len, d_model), src_mask, tgt_mask: (batch_size, seq_len)
        out = self.transformer(
            src,
            tgt,
            # src_key_padding_mask=src_mask,
            # tgt_key_padding_mask=tgt_mask,
            # memory_key_padding_mask=tgt_mask,
            tgt_mask=self.tgt_mask,
        )
        # out: (batch_size, seq_len, d_model)
        out = self.inverse_embedding(out)
        # out: (batch_size, seq_len, vocab_size)
        return out
