#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 14:09:31 2019

@author: krishna, Iuthing
"""
import numpy as np
import torch
import logging

from modules import utils

logger = logging.getLogger(__name__)

class SpeechDataset():
    """Speech dataset."""

    def __init__(self, manifest, mode, spec_config):
        """
        Read the textfile and get the paths
        """
        # self.mode=mode
        self.audio_links = [line.rstrip('\n').split(' ')[0] for line in open(manifest)]
        self.labels = [int(line.rstrip('\n').split(' ')[1]) for line in open(manifest)]
        self.mode = mode
        self.mel = True if spec_config['type'] == 'mel' else False
        self.n_fft = spec_config['n_fft']
        self.n_mels = spec_config['n_mels']
        self.win_length = spec_config['win_length']
        self.hop_length = spec_config['hop_length']
        self.min_dur_sec = spec_config['min_dur_sec']
        self.spec_len = spec_config['spec_sec'] * 16000 // self.hop_length


    def __len__(self):
        return len(self.audio_links)

    def __getitem__(self, idx):
        audio_link =self.audio_links[idx]
        class_id = self.labels[idx]
        #lang_label=lang_id[self.audio_links[idx].split('/')[-2]]
        spec = utils.load_data(audio_link, mode=self.mode, mel=self.mel, n_fft=self.n_fft,
                               spec_len=self.spec_len, win_length=self.win_length,
                               hop_length=self.hop_length, min_dur_sec=self.min_dur_sec,
                               n_mels=self.n_mels)
        sample = (torch.from_numpy(np.ascontiguousarray(spec)), torch.from_numpy(np.ascontiguousarray(class_id)))
        # spec = utils.load_data(audio_link,mode='train', n_fft=512, spec_len=400)
        # waveform, sr = torchaudio.load(audio_link)
        # waveform = torch.mean(waveform, dim=0)
        # logger.debug(waveform.shape)
        # feat = self.transforms(waveform)
        # sample = (feat, torch.from_numpy(np.ascontiguousarray(class_id)))
        return sample
        
    
