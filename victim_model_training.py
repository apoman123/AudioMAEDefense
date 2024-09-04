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
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score

from utils.preprocess import wave_padding, spectrogram_padding, get_noisy_input
from model.resnet_1d import ResNet50_1D
from model.resnet_2d import ResNet50_2D

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

    # training configuration
    parser.add_argument("--mode_type", default="waveform", choices=["waveform", "spectrogram"], type=str)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--max_lr", default=2e-4, type=float)
    parser.add_argument("--save_epoch", default=1, type=int)
    parser.add_argument("--model_path", default=None, type=str)
    parser.add_argument("--resume", default=False, type=bool)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--log_dir", type=str)

    return parser


def waveform_collate_fn(batch):
    padding_result = wave_padding(batch)
    padding_result["labels"] = [data["label"] for data in batch]
    return padding_result

def spectrogram_collate_fn(batch):
    padding_result = spectrogram_padding(batch)
    padding_result["labels"] = [data["label"] for data in batch]
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

    training_set = whole_set["train"]
    training_set_sampler = DistributedSampler(training_set, shuffle=True)
    
    
    evaluation_set = whole_set["validation"]
    evaluation_set_sampler = DistributedSampler(evaluation_set)

    if args.model_type == "waveform":
        train_loader = DataLoader(training_set, sampler=training_set_sampler, batch_size=args.batch_size,
                                num_workers=args.num_workers, pin_memory=True,
                                collate_fn=waveform_collate_fn) # add collate function if needed
        eval_loader = DataLoader(evaluation_set, sampler=evaluation_set_sampler, batch_size=args.batch_size,
                                num_workers=args.num_workers, pin_memory=True,
                                collate_fn=waveform_collate_fn)
        
    elif args.model_type == "spectrogram":
        train_loader = DataLoader(training_set, sampler=training_set_sampler, batch_size=args.batch_size,
                                num_workers=args.num_workers, pin_memory=True,
                                collate_fn=spectrogram_collate_fn) # add collate function if needed
        eval_loader = DataLoader(evaluation_set, sampler=evaluation_set_sampler, batch_size=args.batch_size,
                                num_workers=args.num_workers, pin_memory=True,
                                collate_fn=spectrogram_collate_fn)
        
    print(f"effective batch size is {args.batch_size * dist.get_world_size() * args.accum_steps}")

    # model
    if args.model_type == "waveform":
        model = ResNet50_1D(num_classes=classes, channels=1)

    elif args.model_type == "spectrogram":
        model = ResNet50_2D(num_classes=classes, channels=1)

    model.to(device_id)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    ddp_model = DistributedDataParallel(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    print(f"Model is: {model}")

    # optimization
    lr = args.lr
    epochs = args.epochs
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)

    print(f"Loss function is: {loss_fn}")
    print(f"Optimizer is: {optimizer}")
    print(f"LR Scheduler is: {scheduler}")
    
    # resume
    if args.resume == True:
        checkpoint = torch.load(args.model_path, map_location=f"cuda:{device_id}")
        start_epoch = checkpoint["epoch"]
        ddp_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print(f"load state dict from {args.model_path} and resume training from epoch {start_epoch+1}")

    # accelerate
    accelerator = Accelerator(gradient_accumulation_steps=args.accum_steps)
    ddp_model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    # summary writer
    if rank == 0:
        writer = SummaryWriter(log_dir=args.log_dir)

    # training loop
    ddp_model.train()

    train_labels = torch.tensor([])
    eval_labels = torch.tensor([])
    train_inference_results = torch.tensor([])
    eval_inference_results = torch.tensor([])

    print(f"start training model for {epochs}")
    with tqdm(total=len(train_loader) * epochs) as pbar:
        for epoch in range(start_epoch, epochs):
            if start_epoch != 0 and args.resume:
                args.resume = False
                pbar.update((start_epoch+1)*len(train_loader))

            total_train_loss = 0
            for step, data in enumerate(train_loader):
                # convert data string label to numbers
                batch_labels = []
                for element in data['labels']:
                    batch_labels.append(label_dict[element])
                data["labels"] = torch.tensor(batch_labels).to(device_id)# convert labels

                # to devices
                input_tensor = data['input_values'].to(device_id)

                # input to the model
                result = ddp_model(input_tensor)

                # calc the loss
                loss = loss_fn(result, data["labels"])
                total_train_loss += loss.item()

                # get inference result
                inference_result = torch.argmax(result, dim=-1)
                train_inference_results = torch.cat([train_inference_results, inference_result], dim=0)
                train_labels = torch.cat([train_labels, data["labels"]], dim=0)

                # calc the gradient
                accelerator.backward(loss)

                # progress bar
                pbar.set_description(f"Epoch: {epoch+1}/{epochs} | Step: {step+1}/{len(train_loader)}")
                pbar.set_postfix(loss="{:.4f}".format(loss.item()), lr="{:.4f}".format(optimizer.param_groups[0]["lr"]))
                pbar.update(1)

                # if (step+1) % args.accum_steps == 0:
                #     # update the model
                #     optimizer.step()
                #     optimizer.zero_grad()
            
            total_eval_loss = 0
            ddp_model.eval()
            with tqdm(total=len(eval_loader)) as eval_pbar:
                for step, data in enumerate(eval_loader):
                    # convert data string label to numbers
                    batch_labels = []
                    for element in data['labels']:
                        batch_labels.append(label_dict[element])
                    data["labels"] = torch.tensor(batch_labels).to(device_id)# convert labels

                    # to devices
                    input_tensor = data['input_values'].to(device_id)

                    # input to the model
                    result = ddp_model(input_tensor)

                    # calc the loss
                    loss = loss_fn(result, data["labels"])
                    total_eval_loss += loss.item()

                    # get inference result
                    inference_result = torch.argmax(result, dim=-1)
                    eval_inference_results = torch.cat([eval_inference_results, inference_result], dim=0)
                    eval_labels = torch.cat([eval_labels, data["labels"]], dim=0)

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
            train_acc = accuracy_score(train_inference_results, train_labels)
            eval_acc = accuracy_score(eval_inference_results, eval_labels)

            if rank == 0:
                writer.add_scalar("Training Loss", mean_train_loss, epoch+1)
                writer.add_scalar("Evaluation Loss", mean_eval_loss, epoch+1)
                writer.add_scalar("Training Accuracy", train_acc, epoch+1)
                writer.add_scalar("Evaluation Accuracy", eval_acc, epoch+1)

if __name__ == "__main__":
    parser = get_args()
    args = parser.parse_args()
    ddp_setup()
    main(args)
    destroy_process_group()
