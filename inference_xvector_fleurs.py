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


from modules.utils import fleurs_collate_pad, speech_collate_pad
from models.x_vector import X_vector
from modules.waveform_dataset import WaveformDataset


torch.multiprocessing.set_sharing_strategy('file_system')

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--training_dir', type=str,
                    default='/home/itk0123/x-vector-pytorch/ckpt/0510_0125_saved_model_filtered_ecapa')
parser.add_argument('-m', '--model_path', type=str,
                    default='/home/itk0123/x-vector-pytorch/ckpt/0510_0125_saved_model_filtered_ecapa/ckpt_13_0.07809')
parser.add_argument('-f', '--manifest_dir', type=str,
                    default='/home/itk0123/x-vector-pytorch/manifest/manifest_filtered')
parser.add_argument('-o', '--output', type=str, default='output')


parser.add_argument('-d', '--input_dim', action="store_true", default=39)  # (n_fft // 2 + 1) or n_mel or 39
parser.add_argument('-b', '--batch_size', action="store_true", default=32)
args = parser.parse_args()

# path related
test_meta = os.path.join(args.manifest_dir, 'testing.txt')
class_ids_path = os.path.join(args.manifest_dir, 'class_ids.json')
train_config = os.path.join(args.training_dir, 'config.yaml')
with open(train_config, "r") as f:
    config = load_hyperpyyaml(f)
now = datetime.datetime.now()
savepath = os.path.join('outputs', f'{now.strftime("%m%d_%H%M")}_{args.output}')
os.makedirs(savepath, exist_ok=True)

# Data related
dataset_test = load_dataset("google/fleurs", "all", split="test")
dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, collate_fn=fleurs_collate_pad)

# Model related
with open("manifest/fleurs/class_ids.json", "r") as f:
    fleurs_class_ids = json.load(f)
with open(class_ids_path, "r") as f:
    class_ids = json.load(f)
    num_class = len(class_ids)
id_classes = [k for k in class_ids.keys()]
fleurs_id_classes = [k for k in fleurs_class_ids.keys()]
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
            wfs = torch.from_numpy(
                np.stack([torch_tensor.numpy() for torch_tensor in sample_batched[0]], axis=0)).float()
            labels = torch.from_numpy(np.asarray([torch_tensor[0].numpy() for torch_tensor in sample_batched[1]]))
            features = config['feature'](wfs)
            features, labels = features.to(device), labels.to(device)
            pred_logits, x_vec = model(features)
            predictions = np.argmax(pred_logits.detach().cpu().numpy(), axis=1)
            for pred in predictions:
                full_preds.append(pred)
            for lab in labels.detach().cpu().numpy():
                full_gts.append(lab)
        full_preds_code = [id_classes[i] for i in full_preds]
        full_gts_code = [fleurs_id_classes[i] for i in full_gts]
        preds_df = pd.DataFrame(data={"Predictions": full_preds_code, "Ground Truth": full_gts_code})
        preds_df.to_csv(args.preds)
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
