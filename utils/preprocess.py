from typing import List
import math
import numpy as np
import torch

def wave_padding(batch):
    lengths = [dic["audio"]["array"].shape[0] for dic in batch]
    max_length = max(lengths)
    additional_pads = 1600 - max_length % 1600 if max_length % 1600 != 0 else 0 # every sequence length that is the multiple of 1600 can be deconvoluted propoerly to original sequence length

    new_max_length = max_length + additional_pads

    waves = np.array([np.concatenate([dic["audio"]["array"], np.zeros(new_max_length - length)], axis=0) for dic, length in zip(batch, lengths)])
    padding_masks = np.array([np.concatenate([np.ones(length), np.zeros(new_max_length - length)]) for length in lengths])
    return {"input_values": torch.tensor(waves).unsqueeze(1).float(),
            "padding_masks": torch.tensor(padding_masks).float(),
            "full_padding_masks": None}

def spectrogram_padding(batch, num_mels=128): # spectrogram has 128 mels
    spectrograms = [np.squeeze(np.array(dic["spectrogram"]), axis=0) for dic in batch]
    spectrogram_lengths = [spec.shape[1] for spec in spectrograms]

    max_length = max(spectrogram_lengths)
    additional_pads = 16 - max_length % 16 if max_length % 16 != 0 else 0 # needs to pad the spectrogram to have sequence length of the multiple of 16
    new_max_length = max_length + additional_pads

    padded_spectrograms = np.expand_dims(
        np.array([np.concatenate([spec, np.zeros((num_mels, new_max_length - length))], axis=1) for spec, length in zip(spectrograms, spectrogram_lengths)])
        , axis=1
    )
    # generate padding masks
    valid_value_counts = [math.ceil(spec_length / 16) for spec_length in spectrogram_lengths]
    padding_masks = np.array([
        np.repeat(
            np.expand_dims(
                np.concatenate([
                    np.ones(valid_value_count),
                    np.zeros((new_max_length // 16 - valid_value_count))],
                    axis=0),
                axis=0
                ),
            num_mels//16,
            axis=0
        ).reshape(1, -1)
                     for valid_value_count in valid_value_counts])

    padding_masks = np.squeeze(padding_masks, axis=1)

    # full padding mask
    full_padding_mask = np.expand_dims(
        np.array([np.concatenate([np.ones(spec.shape), np.zeros((num_mels, new_max_length - length))], axis=1) for spec, length in zip(spectrograms, spectrogram_lengths)])
        , axis=1
    )

    return {"input_values": torch.tensor(padded_spectrograms).float(),
            "padding_masks": torch.tensor(padding_masks).float(),
            "full_padding_masks": torch.tensor(full_padding_mask).float()
            }

def get_noisy_input(input_values, sigma):
    noise = torch.rand_like(input_values) * sigma / 255
    noisy_inputs = noise + input_values
    return noisy_inputs