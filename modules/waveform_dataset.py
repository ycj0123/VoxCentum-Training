#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 14:09:31 2019

@author: Iuthing
"""
import numpy as np
import torch
import logging

from modules import utils

logger = logging.getLogger(__name__)


class WaveformDataset():
    """Speech dataset."""

    def __init__(self, manifest, mode, spec_config=None, transforms=None):
        """
        Read the textfile and get the paths
        """
        # self.mode=mode
        self.audio_links = [line.rstrip('\n').split(' ')[0] for line in open(manifest)]
        self.labels = [int(line.rstrip('\n').split(' ')[1]) for line in open(manifest)]
        self.mode = mode
        if mode == 'train':
            assert spec_config is not None, "Need spec_config in training mode."
            self.min_dur_sec = spec_config['min_dur_sec']
            self.wf_sec = spec_config['sample_sec']
        self.transforms = transforms

    def __len__(self):
        return len(self.audio_links)

    def __getitem__(self, idx):
        audio_link = self.audio_links[idx]
        class_id = self.labels[idx]
        # spec = utils.load_data(audio_link,mode='train', n_fft=512, spec_len=400)
        if self.mode == 'train':
            waveform = utils.load_waveform(audio_link, mode='train',
                                           min_dur_sec=self.min_dur_sec, wf_sec=self.wf_sec)
        else:
            waveform = utils.load_waveform(audio_link, mode='test')
        waveform = torch.unsqueeze(waveform, 0).float()
        if self.transforms:
            feat = self.transforms(waveform)
        else:
            feat = waveform
        feat = torch.squeeze(feat)
        sample = (feat, torch.from_numpy(np.ascontiguousarray(class_id)))
        return sample
