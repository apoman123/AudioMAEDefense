import torch
import torch.nn as nn
from model.transformer_parts import Attention, TransformerBlock, SinusoidalPositionalEncoding
import random

IN_CHANNEL=128
OUT_CHANNEL=768
DROPOUT = 0.1
TRANSFORMER_BLOCK_DEPTH = 12
SAMPLING_RATE = 16000
MAX_SPEECH_POSITIONS = 16000 * 30
PAD_TOKEN_ID=1
BOS_TOKEN_ID=0
EOS_TOKEN_ID=2

class PatchToEmbedding(nn.Module):
    def __init__(self, in_channel, embed_dim, kernel=16, stride=16) -> None:
        super(PatchToEmbedding, self).__init__()

        self.conv = nn.Conv2d(in_channel, embed_dim, kernel, stride)

    def forward(self, input_tensor):
        embeddings = self.conv(input_tensor)
        return embeddings

class EmbeddingToPatch(nn.Module):
    def __init__(self, embed_dim, out_channel, kernel=16, stride=16) -> None:
        super(EmbeddingToPatch, self).__init__()

        self.linear = nn.Linear(embed_dim, kernel * stride * out_channel, bias=True)

    def forward(self, embeddings):
        result = self.linear(embeddings)
        return result


class Encoder(nn.Module):
    def __init__(self, embed_dim, num_heads, depth, dropout=0.1, bias=True) -> None:
        super(Encoder, self).__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout, bias) for _ in range(depth)
        ])
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, embeddings, padding_mask=None):
        for block in self.blocks:
            embeddings = block(embeddings, padding_mask)
        return self.layer_norm(embeddings)


class Decoder(nn.Module):
    def __init__(self, embed_dim, num_heads, depth, dropout=0.1, bias=True) -> None:
        super(Decoder, self).__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout, bias) for _ in range(depth)
        ])
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, embeddings, padding_mask=None):
        for block in self.blocks:
            embeddings = block(embeddings, padding_mask)

        return self.layer_norm(embeddings)


class SpectrogramMAE(nn.Module):
    def __init__(self, in_channel=1, embed_dim=768, num_heads=16,
                 depth=12, masking_mode="random", dropout=0.1,
                 bias=True, mask_ratio=0.8) -> None:
        super(SpectrogramMAE, self).__init__()
        self.masking_mode = masking_mode
        self.masked_ratio = mask_ratio
        self.patch_to_embedding = PatchToEmbedding(in_channel, embed_dim)
        self.pos_embeding = SinusoidalPositionalEncoding(embed_dim)
        self.encoder = Encoder(embed_dim, num_heads, depth, dropout, bias)
        self.decoder = Decoder(embed_dim, num_heads, depth, dropout, bias)
        self.embedding_to_patch = EmbeddingToPatch(embed_dim, in_channel)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.masked_token = nn.Parameter(torch.zeros(1, 1, embed_dim))


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
        input_tensor = input_tensor[:, :seq_len - rest, :]
        rest_tensor = input_tensor[:, seq_len - rest:, :]
        cutted_token_idxes = token_idxes[:seq_len - rest]
        rest_token_idxes = token_idxes[seq_len - rest:]

        # four tokens as a group
        grouped_tensor = input_tensor.reshape(bsz, groups, 4, embed_dim)
        grouped_tensor_shape = grouped_tensor.shape
        cutted_token_idxes = cutted_token_idxes.reshape(groups, 4)
        cutted_token_idxes_shape = cutted_token_idxes.shape

        if padding_mask != None:
            padding_mask = padding_mask[:, :seq_len - rest]
            rest_padding_mask = padding_mask[:, seq_len - rest:]

            padding_mask = padding_mask.reshape(bsz, groups, 4)


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
                select_mask = torch.zeros(padding_mask.shape).bool()
                select_mask[:, :, idx] = True
                padding_mask = torch.masked_select(padding_mask, select_mask)

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
                select_mask = torch.zeros(padding_mask.shape).bool()
                select_mask[:, :, idx1]= True
                select_mask[:, :, idx2]= True
                padding_mask = torch.masked_select(padding_mask, select_mask)

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
                select_mask = torch.ones(padding_mask.shape).bool()
                select_mask[:, :, idx] = False
                padding_mask = torch.masked_select(padding_mask, select_mask)

        # reshape back
        masked_tokens = grouped_tensor.reshape(bsz, -1, embed_dim)
        cutted_token_idxes = cutted_token_idxes.reshape(-1)
        if padding_mask != None:
            padding_mask = padding_mask.reshape(bsz, -1)

        # concat the rest
        masked_tokens = torch.cat([masked_tokens, rest_tensor], dim=1)
        token_idxes = torch.cat([cutted_token_idxes, rest_token_idxes], dim=1)
        if padding_mask != None:
            padding_mask = torch.cat([padding_mask, rest_padding_mask], dim=1)

        return masked_tokens, token_idxes, seq_len, padding_mask

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


    def forward(self, input_tensor, padding_masks=None, full_padding_masks=None):
        # patch to embedding
        bsz, channel, height, width = input_tensor.shape
        embeddings = self.patch_to_embedding(input_tensor)
        _, embed_dim, _, _ = embeddings.shape

        # reshape to (bsz, seq_len, embed_dim)
        embeddings = embeddings.flatten(2).transpose(1, 2)

        if self.masking_mode == "random":
            masked_tokens, token_order, masked_padding_masks = self.random_mask(embeddings, padding_masks)

        elif self.masking_mode == "uniform":
            masked_tokens, token_order, seq_len, masked_padding_masks = self.uniform_mask(embeddings, padding_masks)

        # add cls token
        masked_tokens = torch.cat([self.cls_token.expand(masked_tokens.shape[0], 1, masked_tokens.shape[2]), masked_tokens], dim=1)
        if padding_masks != None:
            bsz, seq_len = masked_padding_masks.shape
            masked_padding_masks = torch.cat([torch.zeros(bsz, 1).to(masked_padding_masks.device), masked_padding_masks], dim=1)


        # positional embedding
        masked_tokens = self.pos_embeding(masked_tokens)

        # encode
        masked_tokens = self.encoder(masked_tokens, padding_mask=masked_padding_masks)[:, 1:, :]

        if self.masking_mode == "random":
            tokens = self.add_masked_tokens_and_unshuffle(masked_tokens, token_order)
        elif self.masking_mode == "uniform":
            tokens = self.uniform_add_masked_tokens_and_unshuffle(masked_tokens, token_order, seq_len)

        # add cls token
        tokens = torch.cat([self.cls_token.expand(tokens.shape[0], 1, tokens.shape[2]), tokens], dim=1)
        if padding_masks != None:
            bsz, seq_len = padding_masks.shape
            padding_masks = torch.cat([torch.zeros(bsz, 1).to(padding_masks.device), padding_masks], dim=1)

        # decode
        embeddings = self.decoder(tokens, padding_mask=padding_masks)[:, 1:, :]

        # embedding to patch
        spectrograms = self.embedding_to_patch(embeddings)

        # reshape back to (bsz, embed_dim, height, width)
        spectrograms = spectrograms.reshape(bsz, channel, height, width)

        # mask out the padding part
        if full_padding_masks != None:
            spectrograms = spectrograms.masked_fill(full_padding_masks == 0, 0)

        return spectrograms

