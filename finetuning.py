import os
import time
import sys

import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from accelerate import Accelerator
from tqdm import tqdm
from transformers import SequenceFeatureExtractor
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.gamma import Gamma

from model.waveform_model import WaveMAE
from model.spectrogram_model import SpectrogramMAE
from utils.preprocess import wave_padding, spectrogram_padding, get_noisy_input

# seed
torch.manual_seed(42)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="nccl", type=str)
    # dataset configuration
    parser.add_argument("--dataset", default="vctk", choices=["vctk", "speech_command", "esc50"])
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--accum_steps", default=0, type=int)
    parser.add_argument("--pin_memory", default=True, type=bool)

    # model configuration
    parser.add_argument("--model_type", default="waveform", choices=["waveform", "spectrogram"], type=str)
    parser.add_argument("--embed_dim", default=768, type=int)
    parser.add_argument("--num_heads", default=16, type=int)
    parser.add_argument("--middle_channel", default=512, type=int)
    parser.add_argument("--depth", default=12, type=int)
    parser.add_argument("--masking_mode", default="random", choices=["random", "uniform"], type=str)
    parser.add_argument("--masked_ratio", default=0.8, type=float)

    # training configuration
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--finetuning_task", default="masked nsm", choices=["nsm", "uniform_mask", "random_mask", "masked_nsm"])
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--max_lr", default=2e-4, type=float)
    parser.add_argument("--save_epoch", default=1, type=int)
    parser.add_argument("--model_path", default=None, type=str)
    parser.add_argument("--resume", default=False, type=bool)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--log_dir", type=str)

    return parser


def collate_fn(batch):
    if args.model_type == "waveform":
        padding_result = wave_padding(batch)
    elif args.model_type == "spectrogram":
        padding_result = spectrogram_padding(batch)

    if args.finetuning_task == "nsm" or args.finetuning_task == "masked_nsm":
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
        whole_set = load_from_disk("/data/nas07/Dataset/Audio/VCTK/VCTK")
    elif args.dataset == "speech_commands":
        whole_set = load_dataset("google/speech_commands", "v0.02")
    elif args.dataset == "esc50":
        whole_set = load_dataset("ashraq/esc50")

    training_set = whole_set["train"]
    training_set_sampler = DistributedSampler(training_set, shuffle=True)
    train_loader = DataLoader(training_set, sampler=training_set_sampler, batch_size=args.batch_size,
                            num_workers=args.num_workers, pin_memory=True,
                            collate_fn=collate_fn) # add collate function if needed
    
    evaluation_set = whole_set["validation"]
    evaluation_set_sampler = DistributedSampler(evaluation_set)
    eval_loader = DataLoader(evaluation_set, sampler=evaluation_set_sampler, batch_size=args.batch_size,
                            num_workers=args.num_workers, pin_memory=True,
                            collate_fn=collate_fn)
        
    print(f"effective batch size is {args.batch_size * dist.get_world_size() * args.accum_steps}")

    # model
    if args.model_type == "waveform":
        model = WaveMAE(middle_channel=args.middle_channel, embed_dim=args.embed_dim, num_heads=args.num_heads,
                        depth=args.depth, masking_mode=args.masking_mode, masked_ratio=args.masked_ratio)
    elif args.model_type == "spectrogram":
        model = SpectrogramMAE(embed_dim=args.embed_dim, num_heads=args.num_heads, depth=args.depth,
                            masking_mode=args.masking_mode, mask_ratio=args.masked_ratio)
    model.to(device_id)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    ddp_model = DistributedDataParallel(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    print(f"Model is: {model}")

    # optimization
    lr = args.lr
    epochs = args.epochs
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)

    print(f"Loss function is: {loss_fn}")
    print(f"Optimizer is: {optimizer}")
    print(f"LR Scheduler is: {scheduler}")

    # resume
    if args.model_path != None:
        checkpoint = torch.load(args.model_path)
        start_epoch = checkpoint["epoch"]
        ddp_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print(f"load state dict from {args.model_path} and resume training from epoch {start_epoch+1}")
    else:
        start_epoch = 0

    # accelerate
    accelerator = Accelerator()
    ddp_model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    # summary writer
    if rank == 0:
        writer = SummaryWriter(log_dir=args.log_dir)

    # training loop
    ddp_model.train()
    print(f"start training model for {epochs}")
    with tqdm(total=len(train_loader) * epochs) as pbar:
        for epoch in range(start_epoch, epochs):
            if start_epoch != 0 and args.resume:
                args.resume = False
                pbar.update((start_epoch+1)*len(train_loader))

            total_train_loss = 0
            for step, data in enumerate(train_loader):
                if "noisy_input" in data:
                    input_tensor = data["noisy_input"].to(device_id)
                    padding_masks = data["padding_masks"].to(device_id)
                    ground_truth = data["input_values"].to(device_id)

                else:
                    input_tensor = data["input_values"].to(device_id)
                    padding_masks = data["padding_masks"].to(device_id)
                    ground_truth = input_tensor

                if args.model_type == "waveform":
                    result = ddp_model(input_tensor, padding_masks)
                elif args.model_type == "spectrogram":
                    full_padding_masks = data["full_padding_masks"].to(device_id)
                    result, normalized_input = ddp_model(input_tensor, padding_masks, full_padding_masks)
                    

                    # recover from normalized input to regular input
                    mean = ddp_model.module.patch_to_embedding.batch_norm.running_mean.to(device_id)
                    var = ddp_model.module.patch_to_embedding.batch_norm.running_var.to(device_id)
                    result = result * var + mean

                # calc the loss
                loss = loss_fn(result, ground_truth)
                total_loss += loss.item()

                # calc the gradient
                accelerator.backward(loss)

                # progress bar
                pbar.set_description(f"Epoch: {epoch+1}/{epochs} | Step: {step+1}/{len(train_loader)}")
                pbar.set_postfix(loss="{:.4f}".format(loss.item()), lr="{:.4f}".format(optimizer.param_groups[0]["lr"]))
                pbar.update(1)

                if (step+1) % args.accum_steps == 0:
                    # update the model
                    optimizer.step()
                    optimizer.zero_grad()
            
            total_eval_loss = 0
            ddp_model.eval()
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
                        result = ddp_model(input_tensor, padding_masks)
                    elif args.model_type == "spectrogram":
                        full_padding_masks = data["full_padding_masks"].to(device_id)
                        result, normalized_input = ddp_model(input_tensor, padding_masks, full_padding_masks)
                        

                        # recover from normalized input to regular input
                        mean = ddp_model.module.patch_to_embedding.batch_norm.running_mean.to(device_id)
                        var = ddp_model.module.patch_to_embedding.batch_norm.running_var.to(device_id)
                        result = result * var + mean

                    # calc the loss
                    loss = loss_fn(result, ground_truth)
                    total_eval_loss += loss.item()

                    # progress bar
                    eval_pbar.set_description(f"Step: {step+1}/{len(eval_loader)}")
                    eval_pbar.update(1)
                


            # step the scheduler
            scheduler.step()

            if (epoch+1) % args.save_epoch == 0:
                checkpoint = {
                    "epoch": epoch,
                    "model": ddp_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()
                }
                torch.save(checkpoint, args.save_path + f"/{model.__class__.__name__}_epoch_{epoch+1}.pth")

            # record the total loss
            mean_train_loss = total_train_loss / len(train_loader)
            mean_eval_loss = total_eval_loss / len(eval_loader)
            if rank == 0:
                writer.add_scalar("Training Loss", mean_train_loss, epoch+1)
                writer.add_scalar("Evaluation Loss", mean_eval_loss, epoch+1)

if __name__ == "__main__":
    parser = get_args()
    args = parser.parse_args()
    if args.finetuning_task == "nsm" or args.finetuning_task == "masked_nsm":
        gamma = Gamma(torch.tensor(25), torch.tensor(3)) # follow the implementation of NIM
    ddp_setup()
    main(args)
    destroy_process_group()
