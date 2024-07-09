import torch
import torch.nn as nn
from transformers.activations import ACT2FN

class GroupNormConvLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias, hidden_act="gelu"):
        super(GroupNormConvLayer, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size, stride, bias=bias)
        self.activation = ACT2FN[hidden_act]
        self.layer_norm = nn.GroupNorm(out_channel, out_channel, affine=True)

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class ConvLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias, hidden_act="gelu") -> None:
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size, stride, bias=bias)
        self.activation = ACT2FN[hidden_act]

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class FeatureProjection(nn.Module):
    def __init__(self, last_conv_dim, embed_dim, dropout, eps=1e-5) -> None:
        super(FeatureProjection, self).__init__()
        self.layer_norm = nn.LayerNorm(last_conv_dim, eps=eps)
        self.projection = nn.Linear(last_conv_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states):
        norm_hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class SamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings=128):
        super(SamePadLayer, self).__init__()
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, hidden_states):
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, : -self.num_pad_remove]
        return hidden_states


class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, kernel_size, padding=64, groups=16, hidden_act="gelu") -> None:
        super(PositionalEmbedding, self).__init__()
        self.conv = nn.Conv1d(embed_dim, embed_dim, kernel_size, padding=padding, groups=groups)

        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm

        self.conv = weight_norm(self.conv, name="weight", dim=2)
        self.padding = SamePadLayer()
        self.activation = ACT2FN[hidden_act]

    def forward(self, hidden_states):
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.conv(hidden_states)
        hidden_states = self.padding(hidden_states)
        hidden_states = self.activation(hidden_states)

        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states

class InverseFeatureProjection(nn.Module):
    def __init__(self, embed_dim, last_conv_dim, dropout, eps=1e-5) -> None:
        super(InverseFeatureProjection, self).__init__()
        self.layer_norm = nn.LayerNorm(embed_dim, eps=eps)
        self.projection = nn.Linear(embed_dim, last_conv_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states):
        norm_hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class TransposeConvLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias, out_pad, hidden_act="gelu") -> None:
        super(TransposeConvLayer, self).__init__()
        self.conv = nn.ConvTranspose1d(in_channel, out_channel, kernel_size, stride, bias=bias, output_padding=out_pad)
        self.activation = ACT2FN[hidden_act]

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states

class TransposeBatchNormConvLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias, out_pad, hidden_act="gelu") -> None:
        super(TransposeBatchNormConvLayer, self).__init__()
        self.conv = nn.ConvTranspose1d(in_channel, out_channel, kernel_size, stride, bias=bias, output_padding=out_pad)
        self.activation = ACT2FN[hidden_act]
        self.batch_norm = nn.BatchNorm1d(out_channel)

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states

class GroupNormTransposeConvLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias, out_pad, hidden_act="gelu"):
        super(GroupNormTransposeConvLayer, self).__init__()
        self.conv = nn.ConvTranspose1d(in_channel, out_channel, kernel_size, stride, bias=bias, output_padding=out_pad)
        self.activation = ACT2FN[hidden_act]
        self.layer_norm = nn.GroupNorm(out_channel, out_channel, affine=True)

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states

