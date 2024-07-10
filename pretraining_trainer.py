from transformers import Trainer, TrainingArguments
from datasets import load_from_disk, load_dataset
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
import torch.nn as nn

from model.waveform_model import WaveMAE
from utils.preprocess import wave_padding

loss_fn = nn.MSELoss()

class WaveMAETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        print(inputs)
        reconstructed_wave = model(inputs["input_values"], inputs["padding_masks"])

        loss = loss_fn(reconstructed_wave, inputs["input_values"])

        return (loss, reconstructed_wave) if return_outputs else loss
        

dataset = load_dataset("agkphysics/AudioSet", "balanced")["train"]

training_args = TrainingArguments(
    output_dir="/home/apoman123/data/nas07/PersonalData/apoman123/pretrain_wavemae_checkpoints",
    do_eval=False,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=64,
    learning_rate=1e-5,
    num_train_epochs=100,
    lr_scheduler_type="linear",
    logging_dir="/home/apoman123/data/nas07/PersonalData/apoman123/pretrain_wavemae_checkpoints/logs",
    logging_strategy="epoch",
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    dataloader_num_workers=10,
    torch_compile=False,
    remove_unused_columns=False,
)


trainer = WaveMAETrainer(
    model=WaveMAE(),
    args=training_args,
    train_dataset=dataset,
    data_collator=wave_padding,
    optimizers=[Adam, LinearLR]
)

trainer.train()