from typing import Union
import torch
import torch.nn as nn
from model.convolution_parts import *
from model.transformer_parts import *
import random

KERNELS = [10,3,3,3,3,2,2]
STRIDES = [5,2,2,2,2,2,2]
TRANSPOSE_OUTPADS = [0, 0, 0, 0, 0, 1, 1]
IN_CHANNEL=1
OUT_CHANNEL=512
DROPOUT = 0.1
TRANSFORMER_BLOCK_DEPTH = 12
SAMPLING_RATE = 16000
MAX_SPEECH_POSITIONS = 16000 * 30
PAD_TOKEN_ID=1
BOS_TOKEN_ID=0
EOS_TOKEN_ID=2

class FeatureEncoder(nn.Module):
    def __init__(self, in_cahnnel, out_channel, kernels, strides, bias, embed_dim) -> None:
        super(FeatureEncoder, self).__init__()
        self.kernels = kernels[:]
        self.strides = strides[:]
        self.group_norm_conv = GroupNormConvLayer(in_cahnnel, out_channel, kernels[0], strides[0], bias)
        copied_kernels = kernels[:]
        copied_strides = strides[:]
        del copied_kernels[0]
        del copied_strides[0]
        self.feature_extractor = nn.ModuleList([
            ConvLayer(out_channel, out_channel, kernel, stride, bias) for (kernel, stride) in zip(copied_kernels, copied_strides)
        ])

    # Copied from transformers.models.unispeech.modeling_unispeech.UniSpeechPreTrainedModel._get_feature_vector_padding_mask
    def _get_feature_vector_padding_mask(self, feature_vector_length: int, padding_mask: torch.LongTensor):
        # Effectively padding_mask.sum(-1), but not inplace to be able to run
        # on inference mode.
        non_padded_lengths = padding_mask.cumsum(dim=-1)[:, -1]
        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths).to(torch.long)
        batch_size = padding_mask.shape[0]

        padding_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=padding_mask.dtype, device=padding_mask.device
        )
        # these two operations makes sure that all values before the output lengths idxs are attended to
        padding_mask[(torch.arange(padding_mask.shape[0], device=padding_mask.device), output_lengths - 1)] = 1
        padding_mask = padding_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return padding_mask

    # Copied from transformers.models.unispeech.modeling_unispeech.UniSpeechPreTrainedModel._get_feat_extract_output_lengths
    def _get_feat_extract_output_lengths(self, input_lengths: Union[torch.LongTensor, int]):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        for kernel_size, stride in zip(self.kernels, self.strides):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths

    def forward(self, hidden_states, padding_mask=None):
        hidden_states = self.group_norm_conv(hidden_states)
        for block in self.feature_extractor:
            hidden_states = block(hidden_states)

        if padding_mask != None:
            padding_mask = self._get_feature_vector_padding_mask(hidden_states.shape[2], padding_mask)

        return hidden_states, padding_mask

class Encoder(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, bias, depth) -> None:
        super(Encoder, self).__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout, bias) for _ in range(depth)
        ])

    def forward(self, hidden_states, padding_mask):
        for block in self.blocks:
            hidden_states = block(hidden_states, padding_mask)

        return hidden_states

class Decoder(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, bias, depth) -> None:
        super(Decoder, self).__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout, bias) for _ in range(depth)
        ])

    def forward(self, hidden_states, padding_mask):
        for block in self.blocks:
            hidden_states = block(hidden_states, padding_mask)
        return hidden_states


class WaveRecontructor(nn.Module):
    def __init__(self, in_channel, out_channel, kernels, strides, bias) -> None:
        super(WaveRecontructor, self).__init__()
        output_kernel = kernels[0]
        output_stride = strides[0]
        output_pad = TRANSPOSE_OUTPADS[0]

        copied_kernels = kernels[:]
        copied_strides = strides[:]
        copied_out_pads = TRANSPOSE_OUTPADS[:]

        del copied_kernels[0]
        del copied_strides[0]
        del copied_out_pads[0]

        self.wave_reconstructor = nn.ModuleList([
            TransposeBatchNormConvLayer(in_channel, in_channel, kernel, stride, bias, out_pad) for (kernel, stride, out_pad) in zip(reversed(copied_kernels), reversed(copied_strides), reversed(copied_out_pads))
        ])
        self.output_conv = TransposeBatchNormConvLayer(in_channel, out_channel, output_kernel, output_stride, bias, output_pad)

    def forward(self, hidden_states):
        for block in self.wave_reconstructor:
            hidden_states = block(hidden_states)
        wave = self.output_conv(hidden_states)
        return wave


class WaveMAE(nn.Module):
    def __init__(self, in_channel=1, middle_channel=512, embed_dim=768,
                 num_heads=16, kernels=KERNELS, strides=STRIDES,
                 bias=True, dropout=DROPOUT, depth=12,
                 masking_mode="random", masked_ratio=0.8) -> None:
        super(WaveMAE, self).__init__()
        self.masked_ratio = masked_ratio
        self.prenet = FeatureEncoder(in_channel, middle_channel, kernels, strides, bias, embed_dim)
        self.in_feature_projection = nn.Conv1d(middle_channel, embed_dim, 1, 1)
        self.pos_embeding = SinusoidalPositionalEncoding(embed_dim)
        self.encoder = Encoder(embed_dim, num_heads, dropout, bias, depth)
        self.decoder = Decoder(embed_dim, num_heads, dropout, bias, depth)
        self.out_feature_projection = nn.Conv1d(embed_dim, middle_channel, 1, 1)
        self.postnet = WaveRecontructor(middle_channel, in_channel, kernels, strides, bias)

        self.masked_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.masking_mode = masking_mode

    def random_mask(self, input_tensor, padding_mask=None):
        bsz, seq_len, embed_dim = input_tensor.shape
        token_order = torch.randperm(seq_len)
        # shuffle
        shuffled_tokens = input_tensor[:, token_order, :]

        # take some tokens out as masked tokens
        shuffled_tokens = shuffled_tokens[:, :int(seq_len*self.masked_ratio), :]

        # deal with padding mask
        if padding_mask != None:
            shuffled_padding_mask = padding_mask[:, token_order]
            padding_mask = shuffled_padding_mask[:, :int(seq_len*self.masked_ratio)]

        return shuffled_tokens, token_order, padding_mask

    def uniform_mask(self, input_tensor, padding_mask):
        # uniform mask only deal with 25%, 50%, 75% mask ratio
        bsz, seq_len, embed_dim = input_tensor.shape

        # overall idxes of the whole sequence
        token_idxes = torch.arange(seq_len)

        # cut out the rest
        groups = seq_len // 4
        rest = seq_len % 4
        cutted_input_tensor = input_tensor[:, :seq_len - rest, :]
        rest_tensor = input_tensor[:, seq_len - rest:, :]
        cutted_token_idxes = token_idxes[:seq_len - rest]
        rest_token_idxes = token_idxes[seq_len - rest:]
        
        # four tokens as a group
        grouped_tensor = cutted_input_tensor.reshape(bsz, groups, 4, embed_dim)
        grouped_tensor_shape = grouped_tensor.shape
        cutted_token_idxes = cutted_token_idxes.reshape(groups, 4)
        cutted_token_idxes_shape = cutted_token_idxes.shape

        if padding_mask != None:
            cutted_padding_mask = padding_mask[:, :seq_len - rest]
            rest_padding_mask = padding_mask[:, seq_len - rest:]
            cutted_padding_mask = cutted_padding_mask.reshape(bsz, groups, 4)


        if self.masked_ratio == 0.75:
            # choose a token to preserve from every group with the same relative idx in every group
            idx = random.randint(0, 3)

            # mask the tensor
            select_mask = torch.zeros(grouped_tensor_shape).bool()
            select_mask[:, :, idx, :] = True
            grouped_tensor = torch.masked_select(grouped_tensor, select_mask)

            # mask the idxes
            select_mask = torch.zeros(cutted_token_idxes_shape).bool()
            select_mask[:, idx] = True
            cutted_token_idxes = torch.masked_select(cutted_token_idxes, select_mask)

            if padding_mask != None:
                select_mask = torch.zeros(cutted_padding_mask.shape).bool()
                select_mask[:, :, idx] = True
                cutted_padding_mask = torch.masked_select(cutted_padding_mask, select_mask)

        elif self.masked_ratio == 0.50:
            # choose 2 token to preserve from every group with the same relative idx in every group
            idx1 = random.randint(0, 3)
            idx2 = random.randint(0, 3)

            # mask the tesnor
            select_mask = torch.zeros(grouped_tensor_shape).bool()
            select_mask[:, :, idx1, :] = True
            select_mask[:, :, idx2, :] = True
            grouped_tensor = torch.masked_select(grouped_tensor, select_mask)

            # mask the idxes
            select_mask = torch.zeros(cutted_token_idxes_shape).bool()
            select_mask[:, idx1] = True
            select_mask[:, idx2] = True
            cutted_token_idxes = torch.masked_select(cutted_token_idxes, select_mask)

            if padding_mask != None:
                select_mask = torch.zeros(cutted_padding_mask.shape).bool()
                select_mask[:, :, idx1]= True
                select_mask[:, :, idx2]= True
                cutted_padding_mask = torch.masked_select(cutted_padding_mask, select_mask)

        elif self.masked_ratio == 0.25:
            # choose a token to discard from every group with the same relative idx in every group
            idx = random.randint(0, 3)

            # mask the tensor
            select_mask = torch.ones(grouped_tensor_shape).bool()
            select_mask[:, :, idx, :] = False
            grouped_tensor = torch.masked_select(grouped_tensor, select_mask)

            # mask the idxes
            select_mask = torch.ones(grouped_tensor_shape).bool()
            select_mask[:, idx] = False
            cutted_token_idxes = torch.masked_select(cutted_token_idxes, select_mask)

            if padding_mask != None:
                select_mask = torch.ones(cutted_padding_mask.shape).bool()
                select_mask[:, :, idx] = False
                cutted_padding_mask = torch.masked_select(cutted_padding_mask, select_mask)

        # reshape back
        masked_tokens = grouped_tensor.reshape(bsz, -1, embed_dim)
        cutted_token_idxes = cutted_token_idxes.reshape(-1)

        if padding_mask != None:
            cutted_padding_mask = cutted_padding_mask.reshape(bsz, -1)

        # concat the rest
        masked_tokens = torch.cat([masked_tokens, rest_tensor], dim=1)
        token_idxes = torch.cat([cutted_token_idxes, rest_token_idxes], dim=0)

        if padding_mask != None:
            cutted_padding_mask = torch.cat([cutted_padding_mask, rest_padding_mask], dim=1)
        else:
            cutted_padding_mask = None

        return masked_tokens, token_idxes, seq_len, cutted_padding_mask

    def add_masked_tokens_and_unshuffle(self, shuffled_tokens, token_order):
        bsz, seq_len, embed_dim = shuffled_tokens.shape
        original_seq_len = token_order.shape[0]

        # generate masked tokens
        masked_tokens = self.masked_token.expand(bsz, original_seq_len - seq_len, embed_dim)

        # concat with shuffled tokens
        shuffled_tokens = torch.cat([shuffled_tokens, masked_tokens], dim=1)

        # unshuffle
        reversed_idx = torch.sort(token_order).indices
        tokens = shuffled_tokens[:, reversed_idx, :]
        return tokens

    def uniform_add_masked_tokens_and_unshuffle(self, input_tensor, token_idxes, seq_len):
        # get full token idxes
        bsz, _, _ = input_tensor.shape
        full_token_idxes = torch.arange(seq_len)
        masked_token = self.masked_token.expand(bsz, -1, -1)

        # append token to input tensor if the original tensor is a mask
        for element in full_token_idxes:
            if not element in token_idxes:
                token_idxes = torch.cat([token_idxes, torch.tensor([element]).to(token_idxes.device)], dim=0)
                input_tensor = torch.cat([input_tensor, masked_token], dim=1)

        # unshuffle the tokens
        sorted_idxes = torch.sort(token_idxes).indices
        input_tensor = input_tensor[:, sorted_idxes, :]

        return input_tensor

    def forward(self, input_tensor, padding_mask=None):
        full_padding_mask = padding_mask
        hidden_states, padding_mask = self.prenet(input_tensor, padding_mask)
        hidden_states = self.in_feature_projection(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.pos_embeding(hidden_states)

        # shuffle
        if self.masking_mode == "random":
            shuffled_tokens, token_order, masked_padding_mask = self.random_mask(hidden_states, padding_mask)
        elif self.masking_mode == "uniform":
            shuffled_tokens, token_order, seq_len, masked_padding_mask = self.uniform_mask(hidden_states, padding_mask)
        elif self.masking_mode == "no_mask":
            shuffled_tokens = hidden_states

       
        # add cls token
        shuffled_tokens = torch.cat([self.cls_token.expand(shuffled_tokens.shape[0], 1, shuffled_tokens.shape[2]), shuffled_tokens], dim=1)
        if masked_padding_mask != None:
            masked_padding_mask = torch.cat([torch.ones((masked_padding_mask.shape[0], 1)).bool().to(masked_padding_mask.device), masked_padding_mask], dim=1)

        # encode
        shuffled_tokens = self.encoder(shuffled_tokens, padding_mask=masked_padding_mask)[:, 1:, :] # take out the cls token

        if self.masking_mode == "random":
            # append masked tokens and unshuffle
            tokens = self.add_masked_tokens_and_unshuffle(shuffled_tokens, token_order)
        elif self.masking_mode == "uniform":
            tokens = self.uniform_add_masked_tokens_and_unshuffle(shuffled_tokens, token_order, seq_len)
        elif self.masking_mode == "no_mask":
            tokens = shuffled_tokens

        
        # add cls token
        tokens = torch.cat([self.cls_token.expand(tokens.shape[0], 1, tokens.shape[2]), tokens], dim=1)
        if padding_mask != None:
            padding_mask = torch.cat([torch.ones((padding_mask.shape[0], 1)).bool().to(padding_mask.device), padding_mask], dim=1)

        # decode
        hidden_states = self.decoder(tokens, padding_mask=padding_mask)[:, 1:, :]
        hidden_states = hidden_states.transpose(1, 2)

        # out feature projection
        hidden_states = self.out_feature_projection(hidden_states)

        # convert token to wave
        wave = self.postnet(hidden_states)

        # mask out the padding part
        if full_padding_mask != None:
            full_padding_mask = full_padding_mask.unsqueeze(1)
            wave = wave.masked_fill(full_padding_mask == 0, 0)

        return wave



