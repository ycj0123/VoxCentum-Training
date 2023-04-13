#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 20:22:26 2020

@author: krishna, Iuthing
"""
import os

import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import json
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm

from modules.utils import speech_collate_pad
from models.x_vector import X_vector
from modules.waveform_dataset import WaveformDataset


torch.multiprocessing.set_sharing_strategy('file_system')

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--training_dir', type=str, default='0411_2055_saved_model_conv_mfcc5e-4')
parser.add_argument('-m', '--model_path', type=str, default='0411_2055_saved_model_conv_mfcc5e-4/ckpt_best_46_0.1407')
parser.add_argument('-f', '--manifest_dir', type=str, default='manifest')
parser.add_argument('-o', '--output_path', type=str, default='output.csv')

parser.add_argument('-d', '--input_dim', action="store_true", default=39)  # (n_fft // 2 + 1) or n_mel or 39
parser.add_argument('-b', '--batch_size', action="store_true", default=512)
args = parser.parse_args()

# path related
test_meta = os.path.join(args.manifest_dir, 'testing.txt')
class_ids_path = os.path.join(args.manifest_dir, 'class_ids.json')
train_config = os.path.join(args.training_dir, 'config.yaml')
with open(train_config, "r") as f:
    config = load_hyperpyyaml(f)

# Data related
dataset_test = WaveformDataset(manifest=test_meta, mode='test', transforms=config['feature'])
dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, collate_fn=speech_collate_pad)

# Model related
with open(class_ids_path, "r") as f:
    class_ids = json.load(f)
    num_class = len(class_ids)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = config['model'](args.input_dim, num_class).to(device)
saved = torch.load(args.model_path)
model.load_state_dict(saved['model'])
celoss = nn.CrossEntropyLoss()


def inference(dataloader_val):
    model.eval()
    with torch.no_grad():
        val_loss_list = []
        full_preds = []
        full_gts = []
        for i_batch, sample_batched in enumerate(tqdm(dataloader_val, dynamic_ncols=True)):
            features = torch.from_numpy(
                np.stack([torch_tensor.numpy() for torch_tensor in sample_batched[0]], axis=0)).float()
            labels = torch.from_numpy(np.asarray([torch_tensor[0].numpy() for torch_tensor in sample_batched[1]]))
            features, labels = features.to(device), labels.to(device)
            pred_logits, x_vec = model(features)
            # CE loss
            # loss = celoss(pred_logits,labels)
            # val_loss_list.append(loss.item())
            # train_acc_list.append(accuracy)
            predictions = np.argmax(pred_logits.detach().cpu().numpy(), axis=1)
            for pred in predictions:
                full_preds.append(pred)
            for lab in labels.detach().cpu().numpy():
                full_gts.append(lab)
        df = pd.DataFrame(data={"Predictions": full_preds, "Ground Truth": full_gts})
        df.to_csv(args.output_path)
        mean_acc = accuracy_score(full_gts, full_preds)
        f1s = f1_score(full_gts, full_preds, average=None)
        # mean_loss = np.mean(np.asarray(val_loss_list))
        print(f'Total testing accuracy: {mean_acc:.4}')
        print(f'F1 scores for each class: {f1s}')


if __name__ == '__main__':
    inference(dataloader_test)
