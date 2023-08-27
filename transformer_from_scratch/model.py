import torch
from torch import nn
from utils import DEVICE


MAX_SEQ_LEN = 128  # need fix


def pos_encoding(seq_len: int, d_model: int):
    pos = torch.arange(0, seq_len)[None, :, None]
    idx = torch.arange(0, d_model)[None, None, :]
    wavlen = 10000 ** (idx / d_model)
    return torch.sin(pos / wavlen) * (1 - idx % 2) + torch.cos(pos / wavlen) * (idx % 2)


class InverseEmbedding(nn.Module):
    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.embedding = embedding
        self.softmax = nn.Softmax(dim=-1)

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
    ):
        super().__init__()
        # hyperparameters
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        # embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.inverse_embedding = InverseEmbedding(self.embedding)
        # positional encoding
        self.pos_encoding = pos_encoding(MAX_SEQ_LEN, d_model).to(DEVICE)
        # transformer layers
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=0,
            batch_first=True,
        )
        self.tgt_mask = torch.triu(
            torch.full((MAX_SEQ_LEN, MAX_SEQ_LEN), True), diagonal=1
        ).to(DEVICE)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        y: torch.Tensor,
        y_mask: torch.Tensor,
    ):
        # x, y: (batch_size, seq_len)
        x = (
            self.embedding(x) * torch.tensor(self.d_model).to(DEVICE) ** 0.5
            + self.pos_encoding
        )
        y = (
            self.embedding(y) * torch.tensor(self.d_model).to(DEVICE) ** 0.5
            + self.pos_encoding
        )
        # x, y: (batch_size, seq_len, d_model), x_mask, y_mask: (batch_size, seq_len)
        output = self.transformer(
            x,
            y,
            src_key_padding_mask=x_mask,
            tgt_key_padding_mask=y_mask,
            memory_key_padding_mask=x_mask,
            tgt_mask=self.tgt_mask,
        )
        # output: (batch_size, seq_len, d_model)
        output = self.inverse_embedding(output)
        # output: (batch_size, seq_len, vocab_size)
        return output
