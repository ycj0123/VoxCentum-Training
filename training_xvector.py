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
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml
import logging
import json
import shutil
import time

from modules.utils import speech_collate_pad
# from modules.contrastive_loss import ContrastiveLoss
from models.x_vector import X_vector
from modules.speech_dataset import SpeechDataset
from modules.feature_dataset import SpeechFeatureDataset


# torch.multiprocessing.set_sharing_strategy('file_system')
# family = {'Chinese': {1, 5, 6}, 'European': {0, 2, 3, 4}}
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

if len(sys.argv) == 1:
    sys.argv.append("config.yaml")
with open(sys.argv[1], "r") as f:
    config = load_hyperpyyaml(f)
os.makedirs(config["save_path"], exist_ok=True)
shutil.copy(os.path.abspath(sys.argv[1]), os.path.abspath(config["save_path"]))

### Data related
if config["extract_online"]:
    dataset_train = SpeechDataset(manifest=config["training_meta"],mode='train',
                                  spec_config=config['spectrogram'])
    dataset_val = SpeechDataset(manifest=config["validation_meta"],mode='train',
                                spec_config=config['spectrogram'])
else:
    dataset_train = SpeechFeatureDataset(manifest=config["training_feature"],mode='train')
    dataset_val = SpeechFeatureDataset(manifest=config["validation_feature"],mode='train')
# feature = config["feature"]()
# dataset_train = SpeechDataset(manifest=config["training_meta"], transforms=feature)
# dataset_val = SpeechDataset(manifest=config["validation_meta"], transforms=feature)

dataloader_train = DataLoader(dataset_train, batch_size = config["train"]["batch_size"],
                              num_workers = config["train"]["num_workers"], shuffle=True,
                              collate_fn=speech_collate_pad)
dataloader_val = DataLoader(dataset_val, batch_size = config["val"]["batch_size"],
                            num_workers = config["val"]["num_workers"], shuffle=False,
                            collate_fn=speech_collate_pad) 


## Model related
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
with open(config["class_ids"], "r") as f:
    class_ids = json.load(f)
    num_classes = len(class_ids)
    logging.debug(f"num_class: {num_classes}")
input_dim = config['spectrogram']["n_mels"] if config['spectrogram']['type'] == 'mel' \
        else config['spectrogram']["n_fft"]//2 + 1 
model = X_vector(input_dim, num_classes)

# use multi-GPU if available
if torch.cuda.device_count() > 1:
    logging.info(f"Using {torch.cuda.device_count()} GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)
else:
    logging.info(f"Using 1 GPU!")
model.to(device)

optimizer = config["optimizer"](model.parameters())
celoss = nn.CrossEntropyLoss()
# if config["contrastive_loss"]:
#     contrastive_loss = ContrastiveLoss()

# handle checkpoint
starting_epoch = -1
if config['checkpoint'] is not None:
    ckpt = torch.load(config['checkpoint'])
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    starting_epoch = ckpt['epoch']
    logging.info(f'Start training from epoch {starting_epoch+1} with checkpoint "{config["checkpoint"]}".')
else:
    logging.info(f'Start training from scratch.')


def train(dataloader_train,epoch):
    train_loss_list=[]
    full_preds=[]
    full_gts=[]
    model.train()
    start_time = time.time()
    for i_batch, sample_batched in enumerate(tqdm(dataloader_train, desc=f"epoch {epoch}: ")):
        logging.debug(f"Taking {time.time() - start_time} seconds to load 1 batch")
        features = torch.from_numpy(np.asarray([torch_tensor.numpy().T for torch_tensor in sample_batched[0]])).float()
        labels = torch.from_numpy(np.asarray([torch_tensor[0].numpy() for torch_tensor in sample_batched[1]]))
        features, labels = features.to(device), labels.to(device)
        features.requires_grad = True
        optimizer.zero_grad()
        pred_logits, x_vec = model(features) # x_vec = B x Dim
        #### CE loss
        loss_1 = celoss(pred_logits,labels)
        loss_2 = 0
        loss  = loss_1 + 0.75*loss_2
        loss.backward()
        optimizer.step()
        train_loss_list.append(loss.item())
        #train_acc_list.append(accuracy)
        #if i_batch%10==0:
        #    logging.info('Loss {} after {} iteration'.format(np.mean(np.asarray(train_loss_list)),i_batch))
        
        predictions = np.argmax(pred_logits.detach().cpu().numpy(),axis=1)
        for pred in predictions:
            full_preds.append(pred)
        for lab in labels.detach().cpu().numpy():
            full_gts.append(lab)
        
        start_time = time.time()
            
    mean_acc = accuracy_score(full_gts,full_preds)
    mean_loss = np.mean(np.asarray(train_loss_list))
    logging.info(f'Total training loss {mean_loss:.4} and training accuracy {mean_acc:.4} after {epoch} epochs.')
    

def validation(dataloader_val, epoch , best_loss, old_best):
    model.eval()
    with torch.no_grad():
        val_loss_list=[]
        full_preds=[]
        full_gts=[]
        for i_batch, sample_batched in enumerate(tqdm(dataloader_val, desc=f"epoch {epoch} val: ")):
            features = torch.from_numpy(np.asarray([torch_tensor.numpy().T for torch_tensor in sample_batched[0]])).float()
            labels = torch.from_numpy(np.asarray([torch_tensor[0].numpy() for torch_tensor in sample_batched[1]]))
            features, labels = features.to(device),labels.to(device)
            pred_logits,x_vec = model(features)
            #### CE loss
            loss = celoss(pred_logits,labels)
            val_loss_list.append(loss.item())
            #train_acc_list.append(accuracy)
            predictions = np.argmax(pred_logits.detach().cpu().numpy(),axis=1)
            for pred in predictions:
                full_preds.append(pred)
            for lab in labels.detach().cpu().numpy():
                full_gts.append(lab)
                
        mean_acc = accuracy_score(full_gts,full_preds)
        mean_loss = np.mean(np.asarray(val_loss_list))
        logging.info(f'Total validation loss {mean_loss:.4} and validation accuracy {mean_acc:.4} after {epoch} epochs.')
        

        if ((epoch+1) % config["save_epoch"] == 0):
            model_save_path = os.path.join(config["save_path"], f'ckpt_{epoch}_{mean_loss:.4}')
            logging.info(f'Saving model to {model_save_path}.')
            state_dict = {'model': model.state_dict(),'optimizer': optimizer.state_dict(),'epoch': epoch}
            torch.save(state_dict, model_save_path)
            if mean_loss < best_loss:
                if old_best is not None:
                    os.remove(os.path.join(config["save_path"], old_best))
                return mean_loss, None

        elif mean_loss < best_loss:
            filename = f'ckpt_best_{epoch}_{mean_loss:.4}'
            model_save_path = os.path.join(config["save_path"], filename)
            logging.info(f'Saving best model to {model_save_path}.')
            state_dict = {'model': model.state_dict(),'optimizer': optimizer.state_dict(),'epoch': epoch}
            torch.save(state_dict, model_save_path)
            if old_best is not None:
                os.remove(os.path.join(config["save_path"], old_best))
            new_best = filename
            return mean_loss, new_best

        return mean_loss, old_best
    

if __name__ == '__main__':
    best_val_loss = 100
    old_best = None
    for epoch in range(config["num_epochs"]):
        if epoch <= starting_epoch:
            continue
        train(dataloader_train, epoch)
        val_loss, new_best = validation(dataloader_val, epoch, best_val_loss, old_best)
        old_best = new_best
        if val_loss < best_val_loss:
            best_val_loss = val_loss

