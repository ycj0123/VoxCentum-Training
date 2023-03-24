#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 11:15:47 2020

@author: krishna, Iuthing
"""

import argparse
import os
import numpy as np
from tqdm import tqdm

from modules import utils

def extract_features(audio_filepath, spec_len_sec):
    spec_len = 16000 * spec_len_sec // 160
    features = utils.feature_extraction(audio_filepath, spec_len=spec_len)
    return features
    
    

def FE_pipeline(feature_list, store_loc, mode, spec_len_sec):
    create_root = os.path.join(store_loc,mode)
    if not os.path.exists(create_root):
        os.makedirs(create_root)
    if mode=='train':
        fid = open('manifest/training_feat.txt','w')
    elif mode=='test':
        fid = open('manifest/testing_feat.txt','w')
    elif mode=='validation':
        fid = open('manifest/validation_feat.txt','w')
    else:
        print('Unknown mode')
    
    for row in tqdm(feature_list, desc=mode):
        filepath = row.split(' ')[0]
        lang_id = row.split(' ')[1]
        vid_folder = filepath.split('/')[-2]
        # lang_folder = filepath.split('/')[-3]
        filename = filepath.split('/')[-1]
        # create_folders = os.path.join(create_root,lang_folder,vid_folder)
        create_folders = os.path.join(create_root, vid_folder)
        if not os.path.exists(create_folders):
            os.makedirs(create_folders)
        extract_feats = extract_features(filepath, spec_len_sec)
        dest_filepath = create_folders+'/'+filename[:-4]+'.npy'
        np.save(dest_filepath,extract_feats)
        to_write = dest_filepath+' '+lang_id
        fid.write(to_write+'\n')
    fid.close()
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Configuration for data preparation")
    parser.add_argument("--raw_data", default="/mnt/metanas/VoxCentum_stage1", type=str,help='Dataset path')
    parser.add_argument("--meta_store_path", default="manifest", type=str, help='Save directory after processing')
    parser.add_argument("--spec_len_sec", default=10, type=int)
    config = parser.parse_args()

    store_loc = config.raw_data
    read_train = [line.rstrip('\n') for line in open(os.path.join(config.meta_store_path, 'training.txt'))]
    FE_pipeline(read_train, store_loc, mode='train', spec_len_sec = config.spec_len_sec)
    
    read_test = [line.rstrip('\n') for line in open(os.path.join(config.meta_store_path, 'testing.txt'))]
    FE_pipeline(read_test, store_loc, mode='test', spec_len_sec = config.spec_len_sec)
    
    read_val = [line.rstrip('\n') for line in open(os.path.join(config.meta_store_path, 'validation.txt'))]
    FE_pipeline(read_val, store_loc, mode='validation', spec_len_sec = config.spec_len_sec)
    
    
    
    
    
