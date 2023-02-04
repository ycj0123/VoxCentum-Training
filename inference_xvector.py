#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 20:22:26 2020

@author: krishna
"""



import torch
import numpy as np
from torch.utils.data import DataLoader   
from SpeechDataGenerator_precomp_feats import SpeechDataGenerator_precomp_features
import torch.nn as nn
import os
import numpy as np
from torch import optim
import argparse
from models.x_vector_Indian_LID import X_vector
from sklearn.metrics import accuracy_score
from utils.utils import speech_collate_pad
import torch.nn.functional as F
from contrastive_loss import ContrastiveLoss
import pandas as pd
torch.multiprocessing.set_sharing_strategy('file_system')


########## Argument parser
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-model_path',type=str, default='0131_75/checkpoint_49_0.1850184296161369')
parser.add_argument('-testing_filepath',type=str, default='metatest/training_feat.txt')

parser.add_argument('-input_dim', action="store_true", default=257)
parser.add_argument('-num_classes', action="store_true", default=7)
parser.add_argument('-lamda_val', action="store_true", default=0.1)
parser.add_argument('-batch_size', action="store_true", default=256)
parser.add_argument('-use_gpu', action="store_true", default=True)
args = parser.parse_args()

### Data related
dataset_test = SpeechDataGenerator_precomp_features(manifest=args.testing_filepath,mode='test')
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
            loss = celoss(pred_logits,labels)
            val_loss_list.append(loss.item())
            #train_acc_list.append(accuracy)
            predictions = np.argmax(pred_logits.detach().cpu().numpy(),axis=1)
            for pred in predictions:
                full_preds.append(pred)
            for lab in labels.detach().cpu().numpy():
                full_gts.append(lab)
        df = pd.DataFrame(data={"Predictions": full_preds, "Ground Truth": full_gts})
        df.to_csv("output.csv")
        mean_acc = accuracy_score(full_gts,full_preds)
        mean_loss = np.mean(np.asarray(val_loss_list))
        print('Total testing loss {} and testing accuracy {}'.format(mean_loss,mean_acc))
    
if __name__ == '__main__':
    inference(dataloader_test)
