#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 20:22:26 2020

@author: krishna, Iuthing
"""


import sys
import os

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml
import logging
import json
import shutil
import time
from collections import OrderedDict
import datetime
from torchaudio import transforms as T

from modules.mfcc import MFCC_Delta
from modules.utils import speech_collate, count_parameters
from modules.waveform_dataset import WaveformDataset, FamilyWaveformDataset
from modules.contrastive_loss import SupConLoss


torch.multiprocessing.set_sharing_strategy('file_system')
# family = {'Chinese': {1, 5, 6}, 'European': {0, 2, 3, 4}}
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

if len(sys.argv) == 1:
    sys.argv.append("config.yaml")
with open(sys.argv[1], "r") as f:
    config = load_hyperpyyaml(f)
now = datetime.datetime.now()
savepath = os.path.join('ckpt', f'{now.strftime("%m%d_%H%M")}_{config["save_path"]}')
os.makedirs(savepath, exist_ok=True)
shutil.copy(os.path.abspath(sys.argv[1]), os.path.abspath(savepath))
writer = SummaryWriter(log_dir=f'{savepath}/log')

# Data related
transform_dict = OrderedDict(zip(config['transforms']['names'], config['transforms']['modules']))
transforms = nn.Sequential(transform_dict)

dataset_train = WaveformDataset(manifest=config["training_meta"], mode='train',
                                min_dur_sec=config["min_dur_sec"], wf_sec=config["sample_sec"], transforms=transforms)
dataset_val = WaveformDataset(manifest=config["validation_meta"], mode='train',
                              min_dur_sec=config["min_dur_sec"], wf_sec=config["sample_sec"], transforms=config['feature'])

dataloader_train = DataLoader(dataset_train, batch_size=config["train"]["batch_size"],
                              num_workers=config["train"]["num_workers"], shuffle=True,
                              collate_fn=speech_collate, pin_memory=True)
dataloader_val = DataLoader(dataset_val, batch_size=config["val"]["batch_size"],
                            num_workers=config["val"]["num_workers"], shuffle=False,
                            collate_fn=speech_collate, pin_memory=True)

# Model related
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
with open(config["class_ids"], "r") as f:
    class_ids = json.load(f)
    num_class = len(class_ids)
    logging.debug(f"num_class: {num_class}")
if isinstance(config['feature'], T.MelSpectrogram):
    input_dim = config["n_mels"]
elif isinstance(config['feature'], T.Spectrogram):
    input_dim = config["n_fft"]//2 + 1
elif isinstance(config['feature'], MFCC_Delta):
    input_dim = config["n_mfcc"] * 3
else:
    raise TypeError("config['feature'] must be one of Spectrogram, MelSpectrogram or MFCC_Delta")
logging.debug(f"input_dim: {input_dim}")
model = config['model'](input_dim, num_class)
logging.debug(model)
logging.info(f'Training model: {model.__class__.__name__}.')
logging.info(f'Model parameters: {count_parameters(model):,}')

# use multi-GPU if available
if torch.cuda.device_count() > 1:
    logging.info(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)
elif torch.cuda.device_count() == 1:
    logging.info(f"Using 1 GPU!")
else:
    logging.info(f'Using CPU!')
model.to(device)

optimizer = config["optimizer"](model.parameters())
celoss = nn.CrossEntropyLoss()
con_loss = SupConLoss()
# if config["contrastive_loss"]:
#     contrastive_loss = ContrastiveLoss()

# handle checkpoint
start_epoch = -1
start_step = -1
if config['checkpoint'] is not None:
    ckpt = torch.load(config['checkpoint'])
    try:
        if torch.cuda.device_count() > 1:
            model.module.load_state_dict(ckpt['model'])
        else:
            model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    except:
        # load without the last layer
        del ckpt['model']['output.weight']
        del ckpt['model']['output.bias']
        if torch.cuda.device_count() > 1:
            model.module.load_state_dict(ckpt['model'], strict=False)
        else:
            model.load_state_dict(ckpt['model'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer'])
    start_epoch = ckpt['epoch']
    start_step = ckpt['step']
    logging.info(f'Start training from epoch {start_epoch+1} with checkpoint "{config["checkpoint"]}".')
else:
    logging.info(f'Start training from scratch.')


def train(dataloader_train, epoch):
    train_loss_list = []
    full_preds = []
    full_gts = []
    model.train()
    start_time = time.time()
    pbar = tqdm(dataloader_train, dynamic_ncols=True)
    loss = 0.0
    for i, sample_batched in enumerate(pbar):
        pbar.set_description(desc=f"epoch {epoch}, loss={loss:.4f}")
        logging.debug(f"Taking {time.time() - start_time} seconds to load 1 batch")
        features = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sample_batched[0]])).float()
        labels = torch.from_numpy(np.asarray([torch_tensor[0].numpy() for torch_tensor in sample_batched[1]]))
        if len(sample_batched) == 3:
            families = torch.from_numpy(np.asarray([torch_tensor[0].numpy() for torch_tensor in sample_batched[2]]))
            families = families.to(device)
        elif len(sample_batched) == 4:
            features_orig = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sample_batched[2]])).float()
            families = torch.from_numpy(np.asarray([torch_tensor[0].numpy() for torch_tensor in sample_batched[3]]))
            features_orig, families = features_orig.to(device), families.to(device)
        features, labels = features.to(device), labels.to(device)
        features.requires_grad = True
        optimizer.zero_grad()
        pred_logits, x_vec = model(features)  # x_vec = B x Dim
        # _, x_vec_orig = model(features_orig)  # x_vec = B x Dim
        # CE loss
        loss = celoss(pred_logits, labels)
        # loss_1 = con_loss(torch.stack((x_vec, x_vec_orig), 1), families)
        # loss_1 = con_loss(torch.unsqueeze(x_vec, 1), families)
        # loss = loss_0 + 0.01*loss_1
        loss.backward()
        optimizer.step()
        train_loss_list.append(loss.item())
        # train_acc_list.append(accuracy)
        # if i_batch%10==0:
        #    logging.info('Loss {} after {} iteration'.format(np.mean(np.asarray(train_loss_list)),i_batch))

        predictions = np.argmax(pred_logits.detach().cpu().numpy(), axis=1)
        for pred in predictions:
            full_preds.append(pred)
        for lab in labels.detach().cpu().numpy():
            full_gts.append(lab)

        if config['save_step'] is not None:
            if ((i+1) % config["save_step"] == 0):
                mean_loss = np.mean(np.asarray(train_loss_list))
                model_save_path = os.path.join(savepath, f'ckpt_{epoch}_{i}_{mean_loss:.4}')
                save_checkpoint(model, optimizer, epoch, model_save_path, step=i)
        
        start_time = time.time()

    mean_acc = accuracy_score(full_gts, full_preds)
    mean_loss = np.mean(np.asarray(train_loss_list))
    writer.add_scalar("Loss/train", mean_loss, global_step=epoch)
    writer.add_scalar("Accuracy/train", mean_acc, global_step=epoch)
    logging.info(f'Total training loss {mean_loss:.4} and training accuracy {mean_acc:.4} after {epoch} epochs.')


def validation(dataloader_val, epoch, best_loss, old_best):
    model.eval()
    with torch.no_grad():
        val_loss_list = []
        full_preds = []
        full_gts = []
        for i_batch, sample_batched in enumerate(tqdm(dataloader_val, desc=f"epoch {epoch} val: ", dynamic_ncols=True)):
            # features = torch.from_numpy(np.asarray(
            #     [torch_tensor.numpy().T for torch_tensor in sample_batched[0]])).float()
            features = torch.from_numpy(np.asarray(
                [torch_tensor.numpy() for torch_tensor in sample_batched[0]])).float()
            labels = torch.from_numpy(np.asarray([torch_tensor[0].numpy() for torch_tensor in sample_batched[1]]))
            features, labels = features.to(device), labels.to(device)
            pred_logits, x_vec = model(features)
            # CE loss
            loss = celoss(pred_logits, labels)
            val_loss_list.append(loss.item())
            # train_acc_list.append(accuracy)
            predictions = np.argmax(pred_logits.detach().cpu().numpy(), axis=1)
            for pred in predictions:
                full_preds.append(pred)
            for lab in labels.detach().cpu().numpy():
                full_gts.append(lab)

        mean_acc = accuracy_score(full_gts, full_preds)
        mean_loss = np.mean(np.asarray(val_loss_list))
        writer.add_scalar("Loss/validation", mean_loss, global_step=epoch)
        writer.add_scalar("Accuracy/validation", mean_acc, global_step=epoch)
        logging.info(
            f'Total validation loss {mean_loss:.4} and validation accuracy {mean_acc:.4} after {epoch} epochs.')

        if config["save_epoch"] is not None and config["save_step"] is None:
            if ((epoch+1) % config["save_epoch"] == 0):
                model_save_path = os.path.join(savepath, f'ckpt_{epoch}_{mean_loss:.4}')
                save_checkpoint(model, optimizer, epoch, model_save_path)
                if mean_loss < best_loss:
                    if old_best is not None:
                        os.remove(os.path.join(savepath, old_best))
                    return mean_loss, None

        elif mean_loss < best_loss:
            filename = f'ckpt_best_{epoch}_{mean_loss:.4}'
            model_save_path = os.path.join(savepath, filename)
            save_checkpoint(model, optimizer, epoch, model_save_path)
            if old_best is not None:
                os.remove(os.path.join(savepath, old_best))
            new_best = filename
            return mean_loss, new_best

        return mean_loss, old_best


def save_checkpoint(model, opt, epoch, save_path, step=-1):
    logging.info(f'Saving model to {save_path}.')
    if step != -1:
        epoch -= 1
    if torch.cuda.device_count() > 1:
        state_dict = {'model': model.module.state_dict(), 'optimizer': opt.state_dict(), 'epoch': epoch, 'step': step}
    else:
        state_dict = {'model': model.state_dict(), 'optimizer': opt.state_dict(), 'epoch': epoch, 'step': step}
    torch.save(state_dict, save_path)


if __name__ == '__main__':
    best_val_loss = 100
    old_best = None
    for epoch in range(config["num_epochs"]):
        if epoch <= start_epoch:
            continue
        train(dataloader_train, epoch)
        val_loss, new_best = validation(dataloader_val, epoch, best_val_loss, old_best)
        old_best = new_best
        if val_loss < best_val_loss:
            best_val_loss = val_loss

writer.flush()
writer.close()
