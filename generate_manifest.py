#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 19:09:44 2020

@author: krishna, Iuthing
"""

import os
import random
import glob
import argparse
import json

from nlp2.file import get_files_from_dir
from tqdm import tqdm

def random_split(mylist, ratio):
    if ratio < 0.5:
        ratio = 1-ratio
    idx = int(ratio*len(mylist))
    random.shuffle(mylist)
    long = mylist[:idx]
    short = mylist[idx:]
    return long, short

def create_meta(files_list,store_loc,mode='train'):
    if not os.path.exists(store_loc):
        os.makedirs(store_loc)
    
    if mode=='train':
        meta_store = store_loc+'/training.txt'
        fid = open(meta_store,'w')
        for filepath in files_list:
            fid.write(filepath+'\n')
        fid.close()
    elif mode=='test':
        meta_store = store_loc+'/testing.txt'
        fid = open(meta_store,'w')
        for filepath in files_list:
            fid.write(filepath+'\n')
        fid.close()
    elif mode=='validation':
        meta_store = store_loc+'/validation.txt'
        fid = open(meta_store,'w')
        for filepath in files_list:
            fid.write(filepath+'\n')
        fid.close()
    else:
        print('Error in creating meta files')
    
def extract_split_files(folder_path, valid_frac, test_frac):
    all_lang_folders = sorted(glob.glob(folder_path+'/*/'))
    train_list = []
    val_list = []
    test_list = []

    for id, lang_folderpath in enumerate(tqdm(all_lang_folders, dynamic_ncols=True)):
        all_list = []
        for audio_path in get_files_from_dir(lang_folderpath, match='wav'):
            to_write = audio_path+' '+str(id)
            all_list.append(to_write)
        
        lang_train, other = random_split(all_list, valid_frac+test_frac)
        larger = max(valid_frac, test_frac)
        if larger > 0:
            lang_val, lang_test = random_split(other, larger/(valid_frac+test_frac))
        else:
            lang_val, lang_test = [], []
        train_list += lang_train
        val_list += lang_val
        test_list += lang_test
    class_ids = {l.split("/")[-2]: i for i, l in enumerate(all_lang_folders)}

    return train_list, test_list, val_list, class_ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Configuration for data preparation")
    parser.add_argument("-r", "--raw_data", default="/mnt/metanas/VoxCentum_stage1", type=str,help='Dataset path')
    parser.add_argument("-m", "--meta_store_path", default="manifest", type=str, help='Save directory after processing')
    parser.add_argument("-v", "--valid_frac", default="0.05", type=float, help="portion to split into valid set")
    parser.add_argument("-t", "--test_frac", default="0.05", type=float, help="portion to split into test set")
    config = parser.parse_args()
    train_list, test_list, val_lists, class_ids = extract_split_files(config.raw_data, config.valid_frac, config.test_frac)
    
    os.makedirs(config.meta_store_path, exist_ok=True)
    with open(f"{config.meta_store_path}/class_ids.json", "w+") as f:
        json.dump(class_ids, f, indent=4)
    
    create_meta(train_list, config.meta_store_path, mode='train')
    create_meta(test_list, config.meta_store_path, mode='test')
    create_meta(val_lists, config.meta_store_path, mode='validation')
    