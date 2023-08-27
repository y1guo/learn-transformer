import torch
from torch import nn
from utils import DEVICE


MAX_SEQ_LEN = 128  # need fix


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        pos = torch.arange(MAX_SEQ_LEN)[None, :, None]
        idx = torch.arange(0, d_model, 2)[None, None, :]
        wavlen = 10000 ** (idx / d_model)
        pos_encoding = torch.zeros(1, MAX_SEQ_LEN, d_model)
        pos_encoding[:, :, 0::2] = torch.sin(pos / wavlen)
        pos_encoding[:, :, 1::2] = torch.cos(pos / wavlen)
        self.pos_encoding = pos_encoding.to(DEVICE)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x
                (batch_size, seq_len, d_model)

        Returns
        -------
        torch.Tensor
                (batch_size, seq_len, d_model)
        """
        return x + self.pos_encoding[:, : x.shape[1], :]


class InverseEmbedding(nn.Module):
    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.embedding = embedding

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x
                (batch_size, seq_len, d_model)

        Returns
        -------
        torch.Tensor
                (batch_size, seq_len, vocab_size)
        """
        return x @ self.embedding.weight.data.T


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.linear = nn.Linear(d_model, d_model)
        self.casal_mask = torch.triu(torch.full((MAX_SEQ_LEN, MAX_SEQ_LEN), float("-inf")), diagonal=1).to(DEVICE)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        q_mask: torch.Tensor,
        k_mask: torch.Tensor,
        causal_mask: bool = False,
    ):
        """Note: Treating V = K, and V mask is not needed.
        Parameters
        ----------
        q, k
                (batch_size, seq_len, d_model)
        q_mask, k_mask
                (batch_size, seq_len) 0: mask, 1: not mask
        causal_mask: bool
                whether to use causal attention mask. Default: False

        Returns
        -------
        torch.Tensor
                (batch_size, seq_len, d_model)
        """
        v = k  # v = k but allow different head representations
        q = self.q_linear(q).reshape(*q.shape[:2], self.nhead, self.d_k).transpose(1, 2)
        k = self.k_linear(k).reshape(*k.shape[:2], self.nhead, self.d_k).transpose(1, 2)
        v = self.v_linear(v).reshape(*v.shape[:2], self.nhead, self.d_k).transpose(1, 2)
        # q, k, v: (batch_size, nhead, seq_len, d_k)
        sqrt_d_k = torch.tensor(self.d_k**0.5).to(DEVICE)
        attention = q @ k.mT / sqrt_d_k  # (batch_size, nhead, q_seq_len, k_seq_len)
        # the order of masking is crucial, do not change
        if causal_mask:
            attention = attention + self.casal_mask[: attention.shape[-2], : attention.shape[-1]]
        attention = attention.masked_fill(k_mask[:, None, None, :] == 0, float("-inf"))
        attention = torch.softmax(attention, dim=-1)  # (batch_size, nhead, q_seq_len, k_seq_len)
        attention = attention.masked_fill(q_mask[:, None, :, None] == 0, 0)
        out = attention @ v  # (batch_size, nhead, q_seq_len, d_k)
        out = out.transpose(1, 2)  # (batch_size, q_seq_len, nhead, d_k)
        out = out.reshape(*out.shape[:2], self.d_model)  # (batch_size, q_seq_len, d_model)
        out = self.linear(out)  # (batch_size, q_seq_len, d_model)
        return out


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x
                (batch_size, seq_len, d_model)

        Returns
        -------
        torch.Tensor
                (batch_size, seq_len, d_model)
        """
        return self.linear2(torch.relu(self.linear1(x)))


class AddNorm(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer_out: torch.Tensor):
        """
        Parameters
        ----------
        x
                (batch_size, seq_len, d_model)
        sublayer_out
                (batch_size, seq_len, d_model)

        Returns
        -------
        torch.Tensor
                (batch_size, seq_len, d_model)
        """
        return self.norm(x + self.dropout(sublayer_out))


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, nhead)
        self.add_norm1 = AddNorm(d_model, dropout)
        self.feed_forward = FeedForward(d_model, dim_feedforward)
        self.add_norm2 = AddNorm(d_model, dropout)

    def forward(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor):
        """
        Parameters
        ----------
        src
                (batch_size, seq_len, d_model)
        src_key_padding_mask
                (batch_size, seq_len)

        Returns
        -------
        torch.Tensor
                (batch_size, seq_len, d_model)
        """
        out = self.add_norm1(src, self.multi_head_attention(src, src, src_key_padding_mask, src_key_padding_mask))
        out = self.add_norm2(out, self.feed_forward(out))
        return out


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.multi_head_attention1 = MultiHeadAttention(d_model, nhead)
        self.add_norm1 = AddNorm(d_model, dropout)
        self.multi_head_attention2 = MultiHeadAttention(d_model, nhead)
        self.add_norm2 = AddNorm(d_model, dropout)
        self.feed_forward = FeedForward(d_model, dim_feedforward)
        self.add_norm3 = AddNorm(d_model, dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ):
        """
        Parameters
        ----------
        tgt
                (batch_size, seq_len, d_model)
        memory
                (batch_size, seq_len, d_model)
        tgt_key_padding_mask
                (batch_size, seq_len)
        memory_key_padding_mask
                (batch_size, seq_len)

        Returns
        -------
        torch.Tensor
                (batch_size, seq_len, d_model)
        """
        out = self.add_norm1(
            tgt, self.multi_head_attention1(tgt, tgt, tgt_key_padding_mask, tgt_key_padding_mask, causal_mask=True)
        )
        out = self.add_norm2(
            out, self.multi_head_attention2(out, memory, tgt_key_padding_mask, memory_key_padding_mask)
        )
        out = self.add_norm3(out, self.feed_forward(out))
        return out


class Encoder(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)]
        )

    def forward(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor):
        """
        Parameters
        ----------
        src
                (batch_size, seq_len, d_model)
        src_key_padding_mask
                (batch_size, seq_len)

        Returns
        -------
        torch.Tensor
                (batch_size, seq_len, d_model)
        """
        for layer in self.layers:
            src = layer(src, src_key_padding_mask)
        return src


class Decoder(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)]
        )

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ):
        """
        Parameters
        ----------
        tgt
                (batch_size, seq_len, d_model)
        memory
                (batch_size, seq_len, d_model)
        tgt_key_padding_mask
                (batch_size, seq_len)
        memory_key_padding_mask
                (batch_size, seq_len)

        Returns
        -------
        torch.Tensor
                (batch_size, seq_len, d_model)
        """
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_key_padding_mask, memory_key_padding_mask)
        return tgt


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
        # embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.inverse_embedding = InverseEmbedding(self.embedding)
        # self.inverse_embedding = nn.Linear(d_model, vocab_size)
        # positional encoding layer
        self.pos_enc = PositionalEncoding(d_model)
        # dropout layer
        self.dropout = nn.Dropout(dropout)
        # transformer layers
        self.encoder = Encoder(d_model, nhead, num_encoder_layers, dim_feedforward, dropout)
        self.decoder = Decoder(d_model, nhead, num_decoder_layers, dim_feedforward, dropout)

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
    ):
        """
        Parameters
        ----------
        src
                (batch_size, seq_len)
        src_key_padding_mask
                (batch_size, seq_len)
        tgt
                (batch_size, seq_len)
        tgt_key_padding_mask
                (batch_size, seq_len)

        Returns
        -------
        torch.Tensor
                (batch_size, seq_len, vocab_size)
        """
        sqrt_d_model = torch.tensor(self.d_model**0.5).to(DEVICE)
        src = self.dropout(self.pos_enc(self.embedding(src) * sqrt_d_model))  # (batch_size, src_seq_len, d_model)
        tgt = self.dropout(self.pos_enc(self.embedding(tgt) * sqrt_d_model))  # (batch_size, tgt_seq_len, d_model)
        mem = self.encoder(src, src_key_padding_mask)  # (batch_size, src_seq_len, d_model)
        out = self.decoder(tgt, mem, tgt_key_padding_mask, src_key_padding_mask)  # (batch_size, tgt_seq_len, d_model)
        out = self.inverse_embedding(out)  # (batch_size, tgt_seq_len, vocab_size)
        return out
