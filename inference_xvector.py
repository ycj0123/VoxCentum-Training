#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 20:22:26 2020

@author: krishna, Iuthing
"""
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader   
from sklearn.metrics import accuracy_score, f1_score

from modules.utils import speech_collate_pad
from models.x_vector import X_vector
from modules.speech_dataset import SpeechDataset


torch.multiprocessing.set_sharing_strategy('file_system')

########## Argument parser
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--model_path',type=str, default='saved_model/checkpoint_9_0.4685')
parser.add_argument('--testing_meta',type=str, default='manifest/testing.txt')

parser.add_argument('--input_dim', action="store_true", default=257) # n_fft // 2 + 1 or n_mel
parser.add_argument('--num_classes', action="store_true", default=7) # see manifest/class_ids.json
parser.add_argument('--batch_size', action="store_true", default=512)
args = parser.parse_args()

### Data related
dataset_test = SpeechDataset(manifest=args.testing_meta,mode='test')
dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size,shuffle=False,collate_fn=speech_collate_pad) 

## Model related
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = X_vector(args.input_dim, args.num_classes).to(device)
saved = torch.load(args.model_path)
model.load_state_dict(saved['model'])
celoss = nn.CrossEntropyLoss()

def inference(dataloader_val):
    model.eval()
    with torch.no_grad():
        val_loss_list=[]
        full_preds=[]
        full_gts=[]
        for i_batch, sample_batched in enumerate(dataloader_val):
            features = torch.from_numpy(np.stack([torch_tensor.numpy().T for torch_tensor in sample_batched[0]], axis=0)).float()
            labels = torch.from_numpy(np.asarray([torch_tensor[0].numpy() for torch_tensor in sample_batched[1]]))
            features, labels = features.to(device),labels.to(device)
            pred_logits,x_vec = model(features)
            #### CE loss
            # loss = celoss(pred_logits,labels)
            # val_loss_list.append(loss.item())
            # train_acc_list.append(accuracy)
            predictions = np.argmax(pred_logits.detach().cpu().numpy(),axis=1)
            for pred in predictions:
                full_preds.append(pred)
            for lab in labels.detach().cpu().numpy():
                full_gts.append(lab)
        df = pd.DataFrame(data={"Predictions": full_preds, "Ground Truth": full_gts})
        df.to_csv("output.csv")
        mean_acc = accuracy_score(full_gts,full_preds)
        f1s = f1_score(full_gts,full_preds, average=None)
        # mean_loss = np.mean(np.asarray(val_loss_list))
        print(f'Total testing accuracy: {mean_acc:.4}')
        print(f'F1 scores for each class: {f1s}')
    
if __name__ == '__main__':
    inference(dataloader_test)
