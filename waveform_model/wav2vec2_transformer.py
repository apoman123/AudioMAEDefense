import torch
import torch.nn as nn
from typing import Optional, Tuple, OrderedDict
from transformers.activations import ACT2FN

class FeedForward(nn.Module):
    def __init__(self, activation_dropout, hidden_size, intermediate_size, hidden_act, hidden_dropout):
        super().__init__()
        self.intermediate_dropout = nn.Dropout(activation_dropout)

        self.intermediate_dense = nn.Linear(hidden_size, intermediate_size)
        if isinstance(hidden_act, str):
            self.intermediate_act_fn = ACT2FN[hidden_act]
        else:
            self.intermediate_act_fn = hidden_act

        self.output_dense = nn.Linear(intermediate_size, hidden_size)
        self.output_dropout = nn.Dropout(hidden_dropout)

    def forward(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)

        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states

class Attention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        is_causal=False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.is_causal = is_causal
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self._shape(self.q_proj(hidden_states), tgt_len, bsz)
        # get key, value proj, self_attention
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        is_causal = True if self.is_causal and attention_mask is None and tgt_len > 1 else False

        attn_output = nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal
        )

        if attn_output.size() != (bsz, self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)
        return attn_output

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, bias: bool = True):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.attention = Attention(embed_dim, num_heads, dropout, bias)
        self.feed_forward = FeedForward(dropout, embed_dim, 4 * embed_dim, "gelu", dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, padding_mask: Optional[torch.Tensor] = None):
        attn_residual = hidden_states
        hidden_states = self.attention(hidden_states, padding_mask=padding_mask)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + attn_residual

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.layer_norm(hidden_states)

        return hidden_states