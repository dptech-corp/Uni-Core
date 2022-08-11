# Copyright (c) DP Technology.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import TransformerDecoderLayer, LayerNorm
from .transformer_encoder import relative_position_bucket


def fill_with_neg_inf(t):
    return t.fill_(float("-inf"))


def bulid_future_mask(seq_len):
    return torch.triu(
        fill_with_neg_inf(torch.zeros([seq_len, seq_len])), 1
    )


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layers: int = 6,
        embed_dim: int = 768,
        ffn_embed_dim: int = 3072,
        attention_heads: int = 8,
        emb_dropout: float = 0.1,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        max_seq_len: int = 256,
        activation_fn: str = "gelu",
        rel_pos: bool = True,
        rel_pos_bins: int = 32,
        max_rel_pos: int = 128,
        post_ln: bool = False,
        auto_regressive: bool = True,
    ) -> None:

        super().__init__()
        self.emb_dropout = emb_dropout
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.emb_layer_norm = LayerNorm(self.embed_dim)
        self.auto_regressive = auto_regressive
        if self.auto_regressive:
            self._future_mask = bulid_future_mask(self.max_seq_len)
        else:
            self._future_mask = None
        if not post_ln:
            self.final_layer_norm = LayerNorm(self.embed_dim)
        else:
            self.final_layer_norm = None

        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    embed_dim=self.embed_dim,
                    ffn_embed_dim=ffn_embed_dim,
                    attention_heads=attention_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    post_ln=post_ln,

                )
                for _ in range(decoder_layers)
            ]
        )

        self.rel_pos = rel_pos
        if self.rel_pos:
            assert rel_pos_bins % 2 == 0
            self.rel_pos_bins = rel_pos_bins
            self.max_rel_pos = max_rel_pos
            self.relative_attention_bias = nn.Embedding(
                self.rel_pos_bins, self.attention_heads)
            seq_len = self.max_seq_len
            context_position = torch.arange(seq_len, dtype=torch.long)[:, None]
            memory_position = torch.arange(seq_len, dtype=torch.long)[None, :]
            relative_position = memory_position - context_position
            self.rp_bucket = relative_position_bucket(
                relative_position,
                num_buckets=self.rel_pos_bins,
                max_distance=self.max_rel_pos
            )
            self.rp_bucket -= self.rp_bucket.min()

    def get_rel_pos_bias(self, x):
        # Assume the input is ordered. If your input token is permuted, you may need to update this accordingly
        if self.rp_bucket.device != x.device:
            self.rp_bucket = self.rp_bucket.to(x.device)
        seq_len = x.size(1)
        rp_bucket = self.rp_bucket[:seq_len, :seq_len]
        values = F.embedding(rp_bucket, self.relative_attention_bias.weight)
        values = values.permute([2, 0, 1])
        return values.contiguous()

    def get_future_mask(self, x, attn_mask):
        if not self.auto_regressive:
            return attn_mask
        if self._future_mask.device != x.device:
            self._future_mask = self._future_mask.to(x.device)
        if self._future_mask.dtype != x.dtype:
            self._future_mask = self._future_mask.type_as(x)
        if attn_mask is None:
            ret = self._future_mask[:x.size(1), :x.size(1)]
            ret = ret.contiguous().unsqueeze(0).repeat(
                x.size(0)*self.attention_heads, 1, 1)
            return ret
        else:
            assert list(attn_mask.size()) == [x.size(
                0) * self.attention_heads, x.size(1), x.size(1)]
            return attn_mask + self._future_mask[:x.size(1), :x.size(1)]

    def forward(
        self,
        emb,
        encoder_out: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        encoder_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        seq_len = emb.size(1)
        x = self.emb_layer_norm(emb)
        x = F.dropout(x, p=self.emb_dropout, training=self.training)

        # account for padding while computing the representation
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        rel_pos_bias = self.get_rel_pos_bias(x).repeat(
            x.size(0), 1, 1) if self.rel_pos else None

        if attn_mask is None:
            attn_mask = rel_pos_bias
        elif rel_pos_bias is not None:
            attn_mask += rel_pos_bias

        if self.auto_regressive:
            attn_mask = self.get_future_mask(x, attn_mask)

        if attn_mask is not None and padding_mask is not None:
            # merge key_padding_mask and attn_mask
            attn_mask = attn_mask.view(x.size(0), -1, seq_len, seq_len)
            attn_mask.masked_fill_(
                padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf")
            )
            attn_mask = attn_mask.view(-1, seq_len, seq_len)
            padding_mask = None

        for layer in self.layers:
            x = layer(x, encoder_out=encoder_out, padding_mask=padding_mask, attn_bias=attn_mask,
                      encoder_padding_mask=encoder_padding_mask, encoder_attn_bias=encoder_attn_mask)

        if self.final_layer_norm is not None:
            x = self.final_layer_norm(x)

        return x
