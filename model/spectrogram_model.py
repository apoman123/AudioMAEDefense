import torch
import torch.nn as nn
from transformer_parts import Attention, TransformerBlock, SinusoidalPositionalEmbedding
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
        super().__init__()

        self.conv = nn.Conv1d(in_channel, embed_dim, kernel, stride)

    def forward(self, input_tensor):
        embeddings = self.conv(input_tensor)
        return embeddings
    
class EmbeddingToPatch(nn.Module):
    def __init__(self, embed_dim, out_channel, kernel=16, stride=16) -> None:
        super().__init__()
        self.deconv = nn.ConvTranspose1d(embed_dim, out_channel, kernel, stride)

    def forward(self, embeddings):
        result = self.deconv(embeddings)
        return result


class Encoder(nn.Module):
    def __init__(self, embed_dim, num_heads, depth, dropout=0.1, bias=True) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout, bias) for _ in range(depth)
        ])

    def forward(self, embeddings, padding_mask=None):
        for block in self.blocks:
            embeddings = block(embeddings, padding_mask)
        return embeddings


class Decoder(nn.Module):
    def __init__(self, embed_dim, num_heads, depth, dropout=0.1, bias=True) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout, bias) for _ in range(depth)
        ])
    
    def forward(self, embeddings, padding_mask=None):
        for block in self.blocks:
            embeddings = block(embeddings, padding_mask)
        
        return embeddings
    

class SpectrogramMAE(nn.Module):
    def __init__(self, in_channel=1, embed_dim=768, num_heads=16, 
                 depth=12, masking_mode="random", dropout=0.1, 
                 bias=True, mask_ratio=0.8) -> None:
        super().__init__()
        self.masking_mode = masking_mode
        self.mask_ratio = mask_ratio
        self.patch_to_embedding = PatchToEmbedding(in_channel, embed_dim)
        self.pos_embeding = SinusoidalPositionalEmbedding(
            MAX_SPEECH_POSITIONS + PAD_TOKEN_ID + 1,
            embed_dim,
            PAD_TOKEN_ID
        )
        self.encoder = Encoder(embed_dim, num_heads, depth, dropout, bias)
        self.decoder = Decoder(embed_dim, num_heads, depth, dropout, bias)
        self.embedding_to_patch = EmbeddingToPatch(embed_dim, in_channel)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 16, 16))


    def patchify(self, input_tensor):
        # 16*16 is a patch
        bsz, channels, seq_len = input_tensor.shape
        input_tensor = input_tensor.reshape(bsz, channels//16, 16, seq_len//16, 16)
        return input_tensor

    def unpatchify(self, input_tensor):
        bsz, channel_groups, _, seq_groups, _ = input_tensor.shape
        input_tensor = input_tensor.reshape(bsz, channel_groups*16, seq_groups*16)
        return input_tensor


    def random_mask(self, input_tensor):
        # patchify
        patches = self.patchify(input_tensor)
        bsz, channel_groups, _, seq_groups, _ = patches.shape

        # reshape the patches
        patches = patches.reshape(bsz, -1, 16, 16)
        bsz, patches_count, _, _ = patches.shape

        # get masked token count
        masked_token_count = int(patches_count * self.mask_ratio)

        # generate mask
        mask = torch.cat([torch.zeros(bsz, patches_count - masked_token_count, 16, 16), torch.ones(bsz, masked_token_count, 16, 16)], dim=0)
        idx = torch.randperm(mask.size(1))
        mask = mask[:, idx].bool()

        # fill masked token through the mask
        patches = torch.masked_fill(patches, mask, 0)

        # reshape back to patches
        patches = patches.reshape(bsz, channel_groups, 16, seq_groups, 16)

        # unpatchify
        patches = self.unpatchify(patches)

        return patches


    def uniform_mask(self, input_tensor):
        # patchify
        patches = self.patchify(input_tensor)
        bsz, channel_groups, _, seq_groups, _ = patches.shape

        # reshape to sequence of patches
        patches = input_tensor.reshape(bsz, -1, 16, 16)
        bsz, patches_count, _, _ = patches.shape

        # reshape patches, four patches a group
        rest_patches_count = patches_count % 4
        rest_patches = patches[:, rest_patches_count:, :, :]
        patches = patches[:, rest_patches_count:, :, :]

        patches = patches.reshape(bsz, patches_count//4, 4, 16, 16)


        if self.mask_ratio == 0.75:
            # choose a patch to keep in each group
            keep_idx = random.randint(0, 3)

            # generate mask
            mask = torch.ones(patches.shape).bool()
            mask[:, :, keep_idx, :, :] = False

            patches = torch.masked_fill(patches, mask, 0)

        elif self.mask_ratio == 0.5:
            # choose two patches in each group to mask out
            mask_idx1 = random.randint(0, 3)
            mask_idx2 = random.randint(0, 3)

            # generate mask
            mask = torch.zeros(patches.shape).bool()
            mask[:, :, mask_idx1, :, :] = True
            mask[:, :, mask_idx2, :, :] = True

            patches = torch.masked_fill(patches, mask, 0)

        elif self.mask_ratio == 0.25:
            # choose a patch in each group to mask out
            mask_idx = random.randint(0, 3)

            # generate mask
            mask = torch.zeros(patches.shape).bool()
            mask[:, :, keep_idx, :, :] = True

            patches = torch.masked_fill(patches, mask, 0)
        
        # reshape back to patches
        patches = torch.cat([patches, rest_patches], dim=1)
        patches = patches.reshape(bsz, channel_groups, 16, seq_groups, 16)

        # unpatchify
        patches = self.unpatchify(patches)

        return patches

    
    def forward(self, input_tensor, padding_mask=None):
        if self.masking_mode == "random":
            input_tensor = self.random_mask(input_tensor)

        elif self.masking_mode == "uniform":
            input_tensor = self.uniform_mask(input_tensor)

        embeddings = self.patch_to_embedding(input_tensor)
        embeddings = embeddings + self.pos_embeding(embeddings)

        # add cls token
        embeddings = torch.cat([self.cls_token, embeddings], dim=1)
        # encode
        embeddings = self.encoder(embeddings, padding_mask=padding_mask)[:, 1:, :]

        # add cls token
        embeddings = torch.cat([self.cls_token, embeddings], dim=1)
        embeddings = self.decoder(embeddings, padding_mask=padding_mask)

        embeddings = self.decoder(input_tensor)
        spectrograms = self.embedding_to_patch(embeddings)

        return spectrograms
