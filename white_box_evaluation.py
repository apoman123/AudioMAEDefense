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
from art.attacks.evasion import AutoAttack
from art.attacks.evasion.projected_gradient_descent import projected_gradient_descent


from model.waveform_model import WaveMAE
from model.spectrogram_model import SpectrogramMAE
from utils.preprocess import wave_padding, spectrogram_padding, get_noisy_input
from model.resnet_1d import ResNet50_1D
from model.resnet_2d import ResNet50_2D
from model.whole_system import WholeSystem

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
    parser.add_argument("--defense_model_path", default=None, type=str)
    parser.add_argument("--classifier_path", default=None, type=str)

    # attack configuration
    parser.add_argument("--budget", default=50, type=int)
    parser.add_argument("--attack_type", default="pgd", choices=["pgd", "auto_attack"], type=str)
    parser.add_argument("--epsilon", default=8/255, type=float)

    return parser


def collate_fn(batch):
    if args.model_type == "waveform":
        padding_result = wave_padding(batch)
    elif args.model_type == "spectrogram":
        padding_result = spectrogram_padding(batch)

    if args.finetuning_task == "nam" or args.finetuning_task == "masked_nam":
        sigma = gamma.sample()
        padding_result["noisy_input"] = get_noisy_input(padding_result["input_values"], sigma)

    return padding_result


def ddp_setup():
   """
   Args:
       rank: Unique identifier of each process
       world_size: Total number of processes
   """
   os.environ["MASTER_ADDR"] = "localhost"
   os.environ["MASTER_PORT"] = "12355"
   init_process_group(backend="nccl")

def main(args):
    # device
    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()

    # print nothing if rank != 0
    if rank != 0:
        f = open(os.devnull, 'w')
        sys.stdout = f

    # dataset, need to implement for specific dataset
    if args.dataset == "vctk":
        whole_set = load_from_disk("/data/nas05/apoman123/vctk_less_than_15")
        whole_set = whole_set.cast_column("audio", Audio(sampling_rate=16000))
        whole_set = whole_set.shuffle(seed=42).train_test_split(test_size=0.2)
        classes = 110

    elif args.dataset == "speech_commands":
        whole_set = load_dataset("google/speech_commands", "v0.02")
        classes = 34

    elif args.dataset == "esc50":
        whole_set = load_dataset("ashraq/esc50")
        classes = 50
    
    evaluation_set = whole_set["validation"]
    evaluation_set_sampler = DistributedSampler(evaluation_set)
    eval_loader = DataLoader(evaluation_set, sampler=evaluation_set_sampler, batch_size=args.batch_size,
                            num_workers=args.num_workers, pin_memory=True,
                            collate_fn=collate_fn)
        
    print(f"effective batch size is {args.batch_size * dist.get_world_size() * args.accum_steps}")

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
    classifier.load_state_dict(checkpoint["model"])

    print(f"load defense model from {args.defense_model_path}")
    print(f"load classifier from {args.classifier_path}")

    # ddp
    wholesystem = WholeSystem(defense_model=defense_model, classifier=classifier)
    ddp_wholesystem = DistributedDataParallel(wholesystem, device_ids=[rank], output_device=rank, find_unused_parameters=args.find_unused_parameters)

    # adversarial attack
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam()
    if args.attack_type == "pgd":  
        attack = projected_gradient_descent()
    elif args.attack_type == "auto_attack":
        attack = AutoAttack()

    # evaluation loop
    print(f"start evaluation!")
    ddp_wholesystem.eval()
    with tqdm(total=len(eval_loader)) as eval_pbar:
        for step, data in enumerate(eval_loader):
            if "noisy_input" in data:
                input_tensor = data["noisy_input"].to(device_id)
                padding_masks = data["padding_masks"].to(device_id)
                ground_truth = data["input_values"].to(device_id)

            else:
                input_tensor = data["input_values"].to(device_id)
                padding_masks = data["padding_masks"].to(device_id)
                ground_truth = input_tensor

            if args.model_type == "waveform":
                result = ddp_wholesystem(input_tensor, padding_masks)
            elif args.model_type == "spectrogram":
                full_padding_masks = data["full_padding_masks"].to(device_id)
                result = ddp_wholesystem(input_tensor, padding_masks, full_padding_masks)
            
            result.

            # progress bar
            eval_pbar.set_description(f"Step: {step+1}/{len(eval_loader)}")
            eval_pbar.update(1)
                

if __name__ == "__main__":
    parser = get_args()
    args = parser.parse_args()
    if args.finetuning_task == "nsm" or args.finetuning_task == "masked_nsm":
        gamma = Gamma(torch.tensor(25), torch.tensor(3)) # follow the implementation of NIM
    ddp_setup()
    main(args)
    destroy_process_group()
