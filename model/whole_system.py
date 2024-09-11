import torch
import torch.nn as nn
from utils.preprocess import get_noisy_input

class WholeSystem(nn.Module):
    def __init__(self, defense_model, classifier, input_type, noise_level, defense_model_mode) -> None:
        super(WholeSystem, self).__init__()
        self.defense_model = defense_model
        self.classifier = classifier
        self.input_type = input_type
        self.noise_level = noise_level
        self.defense_model_mode = defense_model_mode

    def forward(self, input_data, padding_masks=None, full_padding_masks=None):
        if self.defense_model_mode == "nam" or self.defense_model_mode == "masked_nam":
            input_data = get_noisy_input(input_data, self.noise_level)

        reconstructed_input = self.defense_model(input_data)
        
        if self.input_type == "spectrogram":
            # make the reconstructed result back to original spectrogram
            mean = self.classifier.patch_to_embedding.batch_norm.running_mean.to()
            var = self.classifier.patch_to_embedding.batch_norm.running_var.to()
            reconstructed_input = reconstructed_input * var + mean
        elif self.input_type == "waveform":
            reconstructed_input = reconstructed_input.squeeze(1)

        classification_result = self.classifier(reconstructed_input)
        return classification_result