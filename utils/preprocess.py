from typing import List
import math
import numpy as np
import torch

def wave_padding(batch):
    lengths = [dic["audio"]["array"] for dic in batch]
    max_length = max(lengths)
    additional_pads = max_length % 8000 # every sequence length that is the multiple of 8000 can be deconvoluted propoerly to original sequence length
    new_max_length = max_length + additional_pads

    waves = [np.concatenate([dic["audio"]["array"], np.zeros(new_max_length - length)]) for dic, length in zip(batch, lengths)]
    padding_masks = [np.concatenate([np.ones(length), np.zeros(new_max_length - length)]) for length in lengths]
    return {"input_values": torch.tensor(waves), "padding_masks": torch.tensor(padding_masks)}

def spectrogram_padding(batch, num_mels=128): # spectrogram has 128 mels
    spectrograms = [np.array(dic["spectrogram"]) for dic in batch]
    spectrogram_lengths = [spec.shape[2] for spec in spectrograms]

    max_length = max(spectrogram_lengths)
    additional_pads = max_length % 16 # needs to pad the spectrogram to have sequence length of the multiple of 16
    new_max_length = max_length + additional_pads

    padded_spectrograms = [np.concatenate([spec, np.zeros(1, num_mels, new_max_length - length)]) for spec, length in zip(spectrograms, spectrogram_lengths)]

    # generate padding masks
    valid_value_counts = [math.ceil(spec_length // 16) for spec_length in spectrogram_lengths]
    padding_masks = [np.concatenate([np.ones(1, num_mels, valid_value_count),
                                     np.zeros(1, num_mels, new_max_length // 16 - valid_value_count)])
                     for valid_value_count in valid_value_counts]


    return {"input_values": torch.tensor(padded_spectrograms), "padding_masks": torch.tensor(padding_masks)}
