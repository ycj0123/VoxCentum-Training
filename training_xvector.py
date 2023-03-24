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
import yaml

from modules.utils import speech_collate
# from modules.contrastive_loss import ContrastiveLoss
from models.x_vector_Indian_LID import X_vector
from modules.speech_dataset import SpeechDataset
from modules.feature_dataset import SpeechFeatureDataset


torch.multiprocessing.set_sharing_strategy('file_system')
# family = {'Chinese': {1, 5, 6}, 'European': {0, 2, 3, 4}}

# with open("config.yaml", "r") as f:
with open(sys.argv[1], "r") as f:
    config = yaml.safe_load(f)

### Data related
if config["extract_online"]:
    dataset_train = SpeechDataset(manifest=config["training_meta"],mode='train', spec_len_sec=config["sample_sec"])
    dataset_val = SpeechDataset(manifest=config["validation_meta"],mode='train', spec_len_sec=config["sample_sec"])
else:
    dataset_train = SpeechFeatureDataset(manifest=config["training_feature"],mode='train')
    dataset_val = SpeechFeatureDataset(manifest=config["validation_feature"],mode='train')

dataloader_train = DataLoader(dataset_train, batch_size=config["batch_size"],shuffle=True,collate_fn=speech_collate) 
dataloader_val = DataLoader(dataset_val, batch_size=config["batch_size"],shuffle=False,collate_fn=speech_collate) 


## Model related
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = X_vector(config["input_dim"], config["num_classes"])

# use multi-GPU if available
if torch.cuda.device_count() > 1:
  print("Using", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)

model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0, betas=(0.9, 0.98), eps=1e-9)
celoss = nn.CrossEntropyLoss()
# if config["contrastive_loss"]:
#     contrastive_loss = ContrastiveLoss()


def train(dataloader_train,epoch):
    train_loss_list=[]
    full_preds=[]
    full_gts=[]
    model.train()
    for i_batch, sample_batched in enumerate(tqdm(dataloader_train, desc=f"epoch {epoch}: ")):
    
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
        #    print('Loss {} after {} iteration'.format(np.mean(np.asarray(train_loss_list)),i_batch))
        
        predictions = np.argmax(pred_logits.detach().cpu().numpy(),axis=1)
        for pred in predictions:
            full_preds.append(pred)
        for lab in labels.detach().cpu().numpy():
            full_gts.append(lab)
            
    mean_acc = accuracy_score(full_gts,full_preds)
    mean_loss = np.mean(np.asarray(train_loss_list))
    print(f'Total training loss {mean_loss:.4} and training accuracy {mean_acc:.4} after {epoch} epochs')
    

def validation(dataloader_val,epoch):
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
        print(f'Total validation loss {mean_loss:.4} and validation accuracy {mean_acc:.4} after {epoch} epochs')
        
        if (epoch+1 % config["save_epoch"] == 0) or (epoch == config["num_epochs"]-1):
            model_save_path = os.path.join(config["save_path"], f'checkpoint_{epoch}_{mean_loss:.4}')
            os.makedirs(config["save_path"], exist_ok=True)
            state_dict = {'model': model.state_dict(),'optimizer': optimizer.state_dict(),'epoch': epoch}
            torch.save(state_dict, model_save_path)
    

if __name__ == '__main__':
    for epoch in range(config["num_epochs"]):
        train(dataloader_train,epoch)
        validation(dataloader_val,epoch)
