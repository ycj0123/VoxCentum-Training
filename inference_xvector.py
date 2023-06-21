#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 20:22:26 2020

@author: krishna, Iuthing
"""
import datetime
import os

import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import json
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report,  confusion_matrix
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sn
from pathlib import Path

from modules.utils import speech_collate_pad, find_all
from models.x_vector import X_vector
from modules.waveform_dataset import WaveformDataset


torch.multiprocessing.set_sharing_strategy('file_system')

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_path', type=str,
                    default='/home/ycj0123/x-vector-pytorch/ckpt/0603_2052_saved_model_new_random_ecapa/ckpt_19_0.3718')
parser.add_argument('-f', '--manifest_dir', type=str,
                    default='/home/ycj0123/x-vector-pytorch/manifest/new_random')
parser.add_argument('-o', '--output', type=str, default='output_rand_19')

parser.add_argument('-d', '--input_dim', action="store_true", default=39)  # (n_fft // 2 + 1) or n_mel or 39
parser.add_argument('-b', '--batch_size', action="store_true", default=128)
parser.add_argument('-w', '--num_workers', action="store_true", default=12)
args = parser.parse_args()

# path related
training_dir = Path(args.model_path).parent
test_meta = os.path.join(args.manifest_dir, 'testing.txt')
class_ids_path = os.path.join(args.manifest_dir, 'class_ids.json')
configs_path = find_all('config', training_dir)
assert len(configs_path) == 1, f'configs_path: {configs_path}. Must contain an only config inside the model directory.'
with open(configs_path[0], "r") as f:
    config = load_hyperpyyaml(f)
now = datetime.datetime.now()
savepath = os.path.join('outputs', f'{now.strftime("%m%d_%H%M")}_{args.output}')

# Data related
dataset_test = WaveformDataset(manifest=test_meta, mode='test', transforms=config['feature'])
dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, collate_fn=speech_collate_pad, num_workers=args.num_workers)

# Model related
with open(class_ids_path, "r") as f:
    class_ids = json.load(f)
    num_class = len(class_ids)
id_classes = [k for k in class_ids.keys()]
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = config['model'](args.input_dim, num_class).to(device)
saved = torch.load(args.model_path)
model.load_state_dict(saved['model'])
print(f"Inferencing with checkpoint {args.model_path}.")


def inference(dataloader_val):
    model.eval()
    with torch.no_grad():
        val_loss_list = []
        full_preds = []
        full_gts = []
        for i_batch, sample_batched in enumerate(tqdm(dataloader_val, dynamic_ncols=True)):
            features = torch.stack(sample_batched[0])
            labels = torch.cat(sample_batched[1])
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
    full_preds_code = [id_classes[i] for i in full_preds]
    full_gts_code = [id_classes[i] for i in full_gts]
    preds_df = pd.DataFrame(data={"Predictions": full_preds_code, "Ground Truth": full_gts_code})
    os.makedirs(savepath, exist_ok=True)
    preds_df.to_csv(os.path.join(savepath, "preds.csv"))
    mean_acc = accuracy_score(full_gts_code, full_preds_code)
    f1s = f1_score(full_gts_code, full_preds_code, average=None)
    report = classification_report(full_gts_code, full_preds_code, zero_division=0, output_dict=True)
    confusion = confusion_matrix(full_gts_code, full_preds_code, labels=id_classes)
    confusion_df = pd.DataFrame(confusion, index=id_classes, columns=id_classes)
    print(f'Total testing accuracy: {mean_acc:.4}')
    print(f'Total testing f1 macro: {np.mean(f1s):.4}')
    report_df = pd.DataFrame(data=report).T
    print(report_df)
    report_df.to_csv(os.path.join(savepath, "stats.csv"))
    confusion_df.to_csv(os.path.join(savepath, "confusion.csv"))
    plt.figure(figsize=(8,6))
    svm = sn.heatmap(confusion_df)#, cmap='coolwarm'
    plt.savefig(os.path.join(savepath, "heatmap.png"), dpi=400)


if __name__ == '__main__':
    inference(dataloader_test)
