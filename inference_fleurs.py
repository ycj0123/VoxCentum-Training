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
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report,  confusion_matrix
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sn
from pathlib import Path


from modules.utils import fleurs_collate_pad, find_all


torch.multiprocessing.set_sharing_strategy('file_system')

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_path', type=str,
                    default='/home/ycj0123/x-vector-pytorch/ckpt/0607_1613_saved_model_new_filtered_13_ecapa/ckpt_17_0.3341')
parser.add_argument('-f', '--manifest_dir', type=str,
                    default='/home/ycj0123/x-vector-pytorch/manifest/new_filtered_13')
parser.add_argument('-o', '--output', type=str, default='output_filtered13_17_fleurs')


parser.add_argument('-d', '--input_dim', action="store_true", default=39)  # (n_fft // 2 + 1) or n_mel or 39
parser.add_argument('-b', '--batch_size', action="store_true", default=64)
parser.add_argument('-w', '--num_workers', action="store_true", default=16)
args = parser.parse_args()

# path related
training_dir = Path(args.model_path).parent
class_ids_path = os.path.join(args.manifest_dir, 'class_ids.json')
configs_path = find_all('config', training_dir)
assert len(configs_path) == 1, f'configs_path: {configs_path}. Must contain an only config inside the model directory.'
with open(configs_path[0], "r") as f:
    config = load_hyperpyyaml(f)
now = datetime.datetime.now()
savepath = os.path.join('outputs', f'{now.strftime("%m%d_%H%M")}_{args.output}')

# Data related
dataset_test = load_dataset("google/fleurs", "all", split="test")
dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, collate_fn=fleurs_collate_pad, num_workers=args.num_workers)

# Model related
with open("manifest/fleurs/class_ids.json", "r") as f:
    fleurs_class_ids = json.load(f)
with open(class_ids_path, "r") as f:
    class_ids = json.load(f)
    num_class = len(class_ids)
id_classes = [k for k in class_ids.keys()]
fleurs_id_classes = [k for k in fleurs_class_ids.keys()]
diff_cls = set(class_ids) - (set(fleurs_class_ids))
diff_ids = [class_ids[l] for l in diff_cls]
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
            wfs = torch.stack(sample_batched[0]).float()
            labels = torch.cat(sample_batched[1])
            features = config['feature'](wfs)
            features, labels = features.to(device), labels.to(device)
            pred_logits, x_vec = model(features)
            pred_logits[:, diff_ids] = -np.inf
            predictions = np.argmax(pred_logits.detach().cpu().numpy(), axis=1)
            for pred in predictions:
                full_preds.append(pred)
            for lab in labels.detach().cpu().numpy():
                full_gts.append(lab)
    full_preds_code = [id_classes[i] for i in full_preds]
    full_gts_code = [fleurs_id_classes[i] for i in full_gts]
    preds_df = pd.DataFrame(data={"Predictions": full_preds_code, "Ground Truth": full_gts_code})
    os.makedirs(savepath, exist_ok=True)
    preds_df.to_csv(os.path.join(savepath, "preds.csv"))
    mean_acc = accuracy_score(full_gts_code, full_preds_code)
    # f1s = f1_score(full_gts_code, full_preds_code, average=None)
    report = classification_report(full_gts_code, full_preds_code, zero_division=0, output_dict=True, labels=fleurs_id_classes)
    confusion = confusion_matrix(full_gts_code, full_preds_code, labels=fleurs_id_classes)
    confusion_df = pd.DataFrame(confusion, index=fleurs_id_classes, columns=fleurs_id_classes)
    print(f'Total testing accuracy: {mean_acc:.4}')
    # print(f'Total testing f1 macro: {np.mean(f1s):.4}')
    report_df = pd.DataFrame(data=report).T
    print(report_df)
    report_df.to_csv(os.path.join(savepath, "stats.csv"))
    confusion_df.to_csv(os.path.join(savepath, "confusion.csv"))
    plt.figure(figsize=(8,6))
    svm = sn.heatmap(confusion_df)#, cmap='coolwarm'
    plt.savefig(os.path.join(savepath, "heatmap.png"), dpi=400)


if __name__ == '__main__':
    inference(dataloader_test)
