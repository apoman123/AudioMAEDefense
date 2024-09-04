import os
import time
import sys

import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from datasets import load_dataset, load_from_disk, Audio
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from accelerate import Accelerator
from tqdm import tqdm
from transformers import SequenceFeatureExtractor
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.gamma import Gamma
from sklearn.metrics import accuracy_score

from model.waveform_model import WaveMAE
from model.spectrogram_model import SpectrogramMAE
from utils.preprocess import wave_padding, spectrogram_padding, get_noisy_input
from model.resnet_1d import ResNet50_1D
from model.resnet_2d import ResNet50_2D
from model.whole_system import WholeSystem
from attack.pgd import PGD
from attack.fakebob import FakeBob

# seed
torch.manual_seed(42)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="nccl", type=str)
    # dataset configuration
    parser.add_argument("--dataset", default="vctk", choices=["vctk", "speech_command", "esc50"])
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--pin_memory", default=True, type=bool)

    # model configuration
    parser.add_argument("--model_type", default="waveform", choices=["waveform", "spectrogram"], type=str)
    parser.add_argument("--embed_dim", default=768, type=int)
    parser.add_argument("--num_heads", default=16, type=int)
    parser.add_argument("--middle_channel", default=512, type=int)
    parser.add_argument("--depth", default=12, type=int)
    parser.add_argument("--masking_mode", default="random", choices=["random", "uniform"], type=str)
    parser.add_argument("--masked_ratio", default=0.8, type=float)

    # evaluating configuration
    parser.add_argument("--defense_model_task", default="masked_nam", choices=["nam", "uniform_mask", "random_mask", "masked_nam"])
    parser.add_argument("--noise_level", default=50, type=int)
    parser.add_argument("--defense_model_path", default=None, type=str)
    parser.add_argument("--classifier_path", default=None, type=str)

    # attack configuration
    parser.add_argument("--budget", default=50, type=int)
    parser.add_argument("--attack_type", default="pgd", choices=["pgd", "fakebob"], type=str)
    parser.add_argument("--epsilon", default=8/255, type=float)
    parser.add_argument("--norm", default="inf", choices=["inf", "l1", "l2"], type=str)

    return parser

def waveform_collate_fn(batch):
    padding_result = wave_padding(batch)
    padding_result["labels"] = [data["label"] for data in batch]
    return padding_result

def spectrogram_collate_fn(batch):
    padding_result = spectrogram_padding(batch)
    padding_result["labels"] = [data["label"] for data in batch]
    return padding_result

def main(args):
    # device
    device_id = "cuda:0"

    # dataset, need to implement for specific dataset
    if args.dataset == "vctk":
        whole_set = load_from_disk("/data/nas05/apoman123/vctk_less_than_15")
        labels = whole_set.unique("speaker_id")['VCTK']
        classes = len(labels)
        whole_set = whole_set.cast_column("audio", Audio(sampling_rate=16000))
        whole_set = whole_set.shuffle(seed=42).train_test_split(test_size=0.2)
        
        # rename speaker_id column to label
        whole_set = whole_set.rename_column("speaker_id", "label")

        # speaker to label dict
        label_dict = {}
        for id, speaker in enumerate(labels):
            label_dict[speaker] = id

    elif args.dataset == "speech_commands":
        whole_set = load_dataset("google/speech_commands", "v0.02")
        labels = whole_set['train'].unique("label")
        classes = len(labels)

    elif args.dataset == "esc50":
        whole_set = load_dataset("ashraq/esc50")
        whole_set = whole_set.rename_column("category", "label")
        classes = 50
        labels = whole_set["train"].unique("label")

        # category to label dict
        label_dict = {}
        for id, category in enumerate(labels):
            label_dict[category] = id

    
    evaluation_set = whole_set["validation"]
    evaluation_set_sampler = DistributedSampler(evaluation_set)
    if args.model_type == "waveform":
        eval_loader = DataLoader(evaluation_set, sampler=evaluation_set_sampler, batch_size=args.batch_size,
                                num_workers=args.num_workers, pin_memory=True,
                                collate_fn=waveform_collate_fn)
    elif args.model_type == "spectrogram":
        eval_loader = DataLoader(evaluation_set, sampler=evaluation_set_sampler, batch_size=args.batch_size,
                                num_workers=args.num_workers, pin_memory=True,
                                collate_fn=spectrogram_collate_fn)
        
    print(f"effective batch size is {args.batch_size}")

    # model
    if args.model_type == "waveform":
        defense_model = WaveMAE(middle_channel=args.middle_channel, embed_dim=args.embed_dim, num_heads=args.num_heads,
                        depth=args.depth, masking_mode=args.masking_mode, masked_ratio=args.masked_ratio)
        classifier = ResNet50_1D(num_classes=classes)

    elif args.model_type == "spectrogram":
        defense_model = SpectrogramMAE(embed_dim=args.embed_dim, num_heads=args.num_heads, depth=args.depth,
                            masking_mode=args.masking_mode, mask_ratio=args.masked_ratio)
        classifier = ResNet50_2D(num_classes=classes)

    # ddp_defense_model = DistributedDataParallel(defense_model, device_ids=[rank], output_device=rank, find_unused_parameters=args.find_unused_parameters)
    # ddp_classifier = DistributedDataParallel(classifier, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    print(f"Defense Model is: {defense_model}")
    print(f"Classifier is: {classifier}") 

    # load model
    defense_model_checkpoint = torch.load(args.defense_model_path, map_location=f"cuda:{device_id}")
    state_dict = {}
    for key in defense_model_checkpoint["model"]:
        new_key = key.replace("module.")
        state_dict[new_key] = defense_model_checkpoint["model"][key]
    defense_model.load_state_dict(state_dict)

    classifer_checkpoint = torch.load(args.classifier_path, map_location=f"cuda:{device_id}")
    state_dict = {}
    for key in classifer_checkpoint["model"]:
        new_key = key.replace("module.")
        state_dict[new_key] = defense_model_checkpoint["model"][key]
    classifier.load_state_dict(state_dict["model"])

    print(f"load defense model from {args.defense_model_path}")
    print(f"load classifier from {args.classifier_path}")

    wholesystem = WholeSystem(defense_model=defense_model, classifier=classifier, input_type=args.model_type, noise_level=args.noise_level, defense_model_mode=args.defense_model_task)
    # ddp_wholesystem = DistributedDataParallel(wholesystem, device_ids=[rank], output_device=rank, find_unused_parameters=args.find_unused_parameters)

    # adversarial attack
    if args.norm == "norm":
        norm = np.inf
    elif args.norm == "l1":
        norm = 1
    elif args.norm == "l2":
        norm = 2

    if args.attack_type == "pgd":  
        attack = PGD(
            model=wholesystem,
            eps=args.epsilon,
            steps=args.budget,
        )
    elif args.attack_type == "auto_attack":
        attack = FakeBob(
            model=wholesystem,
            lr=1e-5,
            steps=200,
            epsilon=0.002,
            norm_type=norm
        )

    # evaluation loop
    print(f"start evaluation!")
    wholesystem.eval()
    labels = np.array([])
    inference_result = np.array([])
    adv_inference_result = np.array([])
    with tqdm(total=len(eval_loader)) as eval_pbar:
        for step, data in enumerate(eval_loader):
            # convert data string label to numbers
            batch_labels = []
            for element in data['labels']:
                batch_labels.append(label_dict[element])
            data["labels"] = torch.tensor(batch_labels)

            # generate adversarial examples
            adv_input_data = attack(
                data['input_values'], 
                data['labels'], 
                data['padding_masks'], 
                data['full_padding_masks']
                )

            # input to the model
            adv_logits = wholesystem(adv_input_data, data['padding_masks'], data['full_padding_masks'])
            adv_classification_result = torch.argmax(adv_logits, dim=-1).cpu().numpy()
            adv_inference_result = np.concatenate([adv_inference_result, adv_classification_result])
            

            # benign examples
            logits = wholesystem(data['input_data'], data['padding_masks'], data['full_padding_masks'])
            classification_result = torch.argmax(logits, dim=-1).cpu().numpy()
            inference_result = np.concatenate([inference_result, classification_result])

            labels = np.concatenate([labels, data["labels"].cpu().numpy()])

            # progress bar
            eval_pbar.set_description(f"Step: {step+1}/{len(eval_loader)}")
            eval_pbar.update(1)
    
    adv_acc = accuracy_score(adv_inference_result, labels)
    acc = accuracy_score(inference_result, labels)

    print(f"the robust accuracy is: {adv_acc}")
    print(f"the standard accuracy is: {acc} ")

if __name__ == "__main__":
    parser = get_args()
    args = parser.parse_args()
    main(args)
