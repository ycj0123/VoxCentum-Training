#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 20:22:26 2020

@author: krishna, Iuthing
"""
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

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
parser.add_argument('-t', '--training_dir', type=str, default='/home/ycj0123/x-vector-pytorch/ckpt/0605_0124_saved_model_new_all_ecapa/')
parser.add_argument('-m', '--model_path', type=str, default='/home/ycj0123/x-vector-pytorch/ckpt/0605_0124_saved_model_new_all_ecapa/ckpt_13_0.3387')
parser.add_argument('-f', '--manifest_dir', type=str, default='/home/ycj0123/x-vector-pytorch/manifest/new_all_alltrain')
parser.add_argument('-o', '--output_path', type=str, default='new_all_13_sorted.csv')

parser.add_argument('-d', '--input_dim', action="store_true", default=39)  # (n_fft // 2 + 1) or n_mel or 39
parser.add_argument('-b', '--batch_size', action="store_true", default=64)
parser.add_argument('-w', '--num_workers', action="store_true", default=12)
args = parser.parse_args()

# path related
test_meta = os.path.join(args.manifest_dir, 'training.txt')
class_ids_path = os.path.join(args.manifest_dir, 'class_ids.json')
train_config = os.path.join(args.training_dir, 'config.yaml')
with open(train_config, "r") as f:
    config = load_hyperpyyaml(f)

# Data related
dataset_test = WaveformDataset(manifest=test_meta, mode='test', transforms=config['feature'])
dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, collate_fn=speech_collate_pad, num_workers=args.num_workers)

# Model related
with open(class_ids_path, "r") as f:
    class_ids = json.load(f)
    num_class = len(class_ids)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = config['model'](args.input_dim, num_class).to(device)
saved = torch.load(args.model_path)
model.load_state_dict(saved['model'])
celoss = nn.CrossEntropyLoss(reduction='none')
id_classes = {v: k for k, v in class_ids.items()}
audio_links = [line.rstrip('\n').split(' ')[0] for line in open(test_meta)]


def inference(dataloader_test):
    model.eval()
    with torch.no_grad():
        full_preds = np.array([], dtype=int)
        full_pred_labels = []
        full_gts = np.array([], dtype=int)
        full_gt_labels = []
        full_losses = np.array([])
        for i_batch, sample_batched in enumerate(tqdm(dataloader_test, dynamic_ncols=True)):
            features = torch.stack(sample_batched[0])
            labels = torch.cat(sample_batched[1])
            features, labels = features.to(device), labels.to(device)
            pred_logits, x_vec = model(features)
            predictions = np.argmax(pred_logits.detach().cpu().numpy(), axis=1)
            losses = celoss(pred_logits, labels)

            full_preds = np.concatenate((full_preds, predictions))
            full_pred_labels += [id_classes[i] for i in predictions]
            full_gts = np.concatenate((full_gts, labels.detach().cpu().numpy()))
            full_gt_labels += [id_classes[i] for i in labels.detach().cpu().numpy()]
            full_losses = np.concatenate((full_losses, losses.detach().cpu().numpy()))
        # full_losses = np.around(full_losses, decimals=5)
        df = pd.DataFrame(data={"Ground Truth": full_gts, "Ground Truth Code": full_gt_labels,
                          "Predictions": full_preds, "Predictions Code": full_pred_labels, "Loss": full_losses,
                          "Path": audio_links})
        df.sort_values(by=['Ground Truth', 'Loss'], ascending=[True, False], inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.to_csv(args.output_path)
        mean_acc = accuracy_score(full_gts, full_preds)
        f1s = f1_score(full_gts, full_preds, average=None)
        print(f'Total testing accuracy: {mean_acc:.4}')
        print(f'F1 scores for each class: {f1s}')


if __name__ == '__main__':
    inference(dataloader_test)
