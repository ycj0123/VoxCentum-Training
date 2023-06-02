#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 14:09:31 2019

@author: Iuthing
"""
import numpy as np
import torch
import logging
from torch.utils.data import Dataset

from modules import utils

logger = logging.getLogger(__name__)


class WaveformDataset(Dataset):
    """Speech dataset."""

    def __init__(self, manifest, mode, min_dur_sec=None, wf_sec=None, transforms=None, feature=None):
        """
        Read the textfile and get the paths
        """
        self.audio_links = [line.rstrip('\n').split(' ')[0] for line in open(manifest)]
        self.labels = [int(line.rstrip('\n').split(' ')[1]) for line in open(manifest)]
        self.mode = mode
        if mode == 'train':
            self.min_dur_sec = min_dur_sec
            self.wf_sec = wf_sec
        self.transforms = transforms
        self.feature = feature

    def __len__(self):
        return len(self.audio_links)

    def __getitem__(self, idx):
        audio_link = self.audio_links[idx]
        class_id = self.labels[idx]
        if self.mode == 'train':
            waveform = utils.load_waveform(audio_link, mode='train',
                                           min_dur_sec=self.min_dur_sec, wf_sec=self.wf_sec)
        else:
            waveform = utils.load_waveform(audio_link, mode='test')
        waveform = torch.unsqueeze(waveform, 0).float()
        if self.transforms:
            wf_augmented = self.transforms(waveform)
        else:
            wf_augmented = waveform
        if self.feature:
            feat = self.feature(wf_augmented)
        else:
            feat = wf_augmented
        feat = torch.squeeze(feat)
        sample = (feat, torch.tensor([class_id]))
        return sample

class FamilyWaveformDataset(WaveformDataset):
    """Speech dataset."""

    def __init__(self, manifest, mode, min_dur_sec=None, wf_sec=None, feature=None, transforms=None):
        """
        Read the textfile and get the paths
        """
        super(FamilyWaveformDataset).__init__(manifest, mode, min_dur_sec, wf_sec, feature, transforms)
        self.families = [int(line.rstrip('\n').split(' ')[2]) for line in open(manifest)]

    def __getitem__(self, idx):
        audio_link = self.audio_links[idx]
        class_id = self.labels[idx]
        family_id = self.families[idx]
        if self.mode == 'train':
            waveform = utils.load_waveform(audio_link, mode='train',
                                           min_dur_sec=self.min_dur_sec, wf_sec=self.wf_sec)
        else:
            waveform = utils.load_waveform(audio_link, mode='test')
        waveform = torch.unsqueeze(waveform, 0).float()
        if self.transforms:
            wf_augmented = self.transforms(waveform)
        else:
            wf_augmented = waveform
        if self.feature:
            feat = self.feature(waveform)
            feat_augmented = self.feature(wf_augmented)
        else:
            feat = wf_augmented
        if self.transforms:
            feat = torch.squeeze(feat)
            feat_augmented = torch.squeeze(feat_augmented)
            sample = (feat, torch.tensor([class_id]), torch.tensor([family_id]), feat_augmented)
        else:
            feat = torch.squeeze(feat)
            sample = (feat, torch.tensor([class_id]), torch.tensor([family_id]))
        return sample