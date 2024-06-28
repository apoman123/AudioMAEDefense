import torch
import torch.nn as nn
from wav2vec2_convolution import *
from wav2vec2_transformer import *
import random

KERNELS = [10,3,3,3,3,2,2]
STRIDES = [5,2,2,2,2,2,2]
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
        super().__init__()
        self.group_norm_conv = GroupNormConvLayer(in_cahnnel, out_channel, kernels[0], strides[0], bias)
        self.feature_extractor = nn.ModuleList([
            ConvLayer(out_channel, out_channel, kernel, stride, bias) for (kernel, stride) in (kernels, strides)
        ])

    def forward(self, hidden_states):
        hidden_states = self.group_norm_conv(hidden_states)
        for block in self.feature_extractor:
            hidden_states = block(hidden_states)

        return hidden_states
    
class Encoder(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, bias, depth) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout, bias) for _ in range(depth)
        ])
        
    def forward(self, hidden_states):
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return hidden_states
    
class Decoder(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, bias, depth) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout, bias) for _ in range(depth)
        ])
        
    def forward(self, hidden_states):
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return hidden_states


class WaveRecontructor(nn.Module):
    def __init__(self, in_channel, out_channel, kernels, strides, bias) -> None:
        super().__init__()
        self.output_conv = TransposeBatchNormConvLayer(in_channel, out_channel, kernels[0], strides[0], bias)
        kernels = kernels.pop(0)
        strides = strides.pop(0)
        self.wave_reconstructor = nn.ModuleList([
            TransposeBatchNormConvLayer(in_channel, in_channel, kernel, stride, bias) for (kernel, stride) in reversed(kernels, strides)
        ])
        
    def forward(self, hidden_states):
        for block in self.wave_reconstructor:
            hidden_states = block(hidden_states)
        wave = self.output_conv(hidden_states)
        return wave


class WaveMAE(nn.Module):
    def __init__(self, in_channel, middle_channel, embed_dim, num_heads, kernels, strides, bias, dropout, depth, masking_mode, masked_ratio=0.8) -> None:
        super().__init__()
        self.masked_ratio = masked_ratio
        self.prenet = FeatureEncoder(in_channel, middle_channel, kernels, strides, bias, embed_dim)
        self.pos_embeding = SinusoidalPositionalEmbedding(
            MAX_SPEECH_POSITIONS + PAD_TOKEN_ID + 1,
            embed_dim,
            PAD_TOKEN_ID
        )
        self.encoder = Encoder(embed_dim, num_heads, dropout, bias, depth)
        self.decoder = Decoder(embed_dim, num_heads, dropout, bias, depth)
        self.postnet = WaveRecontructor(in_channel, middle_channel, kernels, strides, bias)

        self.masked_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.masking_mode = masking_mode

    def random_mask(self, input_tensor):
        bsz, seq_len, embed_dim = input_tensor.shape
        token_order = torch.randperm(seq_len)

        # shuffle
        shuffled_tokens = input_tensor[:, token_order, :]

        # take some tokens out as masked tokens
        shuffled_tokens = shuffled_tokens[:, :int(seq_len*self.masked_ratio), :]
        return shuffled_tokens, token_order
    
    def uniform_mask(self, input_tensor):
        # uniform mask omly deal with 25%, 50%, 75% mask ratio 
        bsz, seq_len, embed_dim = input_tensor.shape
        
        # overall idxes of the whole sequence
        token_idxes = torch.arange(seq_len)

        # cut out the rest
        groups = seq_len // 4
        rest = seq_len % 4
        input_tensor = input_tensor[:, :seq_len - rest, :]
        rest_tensor = input_tensor[:, seq_len - rest:, :]
        cutted_token_idxes = token_idxes[:seq_len - rest]
        rest_token_idxes = token_idxes[seq_len - rest:]


        # four tokens as a group
        grouped_tensor = input_tensor.reshape(bsz, groups, 4, embed_dim)
        grouped_tensor_shape = grouped_tensor.shape
        cutted_token_idxes = cutted_token_idxes.reshape(groups, 4)
        cutted_token_idxes_shape = cutted_token_idxes.shape

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

        # reshape back
        masked_tokens = grouped_tensor.reshape(bsz, -1, embed_dim)
        cutted_token_idxes = cutted_token_idxes.reshape(-1)

        # concat the rest
        masked_tokens = torch.cat([masked_tokens, rest_tensor], dim=1)
        token_idxes = torch.cat([cutted_token_idxes, rest_token_idxes], dim=1)

        return masked_tokens, token_idxes, seq_len
    
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

        # append token to input tensor if the original tensor is a mask
        for element in full_token_idxes:
            if not element in token_idxes:
                token_idxes = torch.cat([token_idxes, element])
                input_tensor = torch.cat([input_tensor, self.masked_token], dim=1)

        # unshuffle the tokens
        sorted_idxes = torch.sort(token_idxes).indices
        input_tensor = input_tensor[:, sorted_idxes, :]

        return input_tensor
             
    def forward(self, input_tensor):
        hidden_states = self.prenet(input_tensor)
        hidden_states = hidden_states + self.pos_embeding(hidden_states)

        # shuffle
        if self.masking_mode == "random":
            shuffled_tokens, token_order = self.random_mask(hidden_states)
        elif self.masking_mode == "uniform":
            shuffled_tokens, token_order, seq_len = self.uniform_mask(hidden_states)

        # encode
        # add cls token
        shuffled_tokens = torch.cat([self.cls_token, shuffled_tokens], dim=1)
        shuffled_tokens = self.encoder(shuffled_tokens)[:, 1:, :] # take out the cls token

        if self.masking_mode == "random":
            # append masked tokens and unshuffle
            tokens = self.add_masked_tokens_and_unshuffle(shuffled_tokens, token_order)
        elif self.masking_mode == "uniform":
            tokens = self.uniform_add_masked_tokens_and_unshuffle(shuffled_tokens, token_order, seq_len)
            
        # decode
        # add cls token
        tokens = torch.cat([self.cls_token, shuffled_tokens], dim=1)
        hidden_states = self.decoder(tokens)[:, 1:, :]

        # convert token to wave
        wave = self.postnet(hidden_states)
        return wave




        
