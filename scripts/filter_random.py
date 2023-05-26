import pandas as pd
import os
from tqdm import tqdm


manifest_orig = pd.read_csv('/home/ycj0123/x-vector-pytorch/manifest/all_new_alltrain/training.txt', sep=' ')
output_dir = 'new_random'

first = True
for i in tqdm(range(137)):
    lang = manifest_orig[manifest_orig.iloc[:, 1] == i]
    if not lang.empty:
        if len(lang) > 46800:
            lang = lang.sample(39600)
            lang_train = lang.sample(36000)
            lang_valtest = lang.drop(lang_train.index)
            lang_val = lang_valtest.sample(1800)
            lang_test = lang_valtest.drop(lang_val.index)
        else:
            keep = int(len(lang) * 0.85)
            lang = lang.sample(keep)
            lang_train = lang.sample(frac=0.9)
            lang_valtest = lang.drop(lang_train.index)
            lang_val = lang_valtest.sample(frac=0.5)
            lang_test = lang_valtest.drop(lang_val.index)

        if first:
            manifefst_train = lang_train
            manifefst_val = lang_val
            manifefst_test = lang_test
            first = False
        else:
            manifefst_train = pd.concat([manifefst_train, lang_train], ignore_index=True)
            manifefst_val = pd.concat([manifefst_val, lang_val], ignore_index=True)
            manifefst_test = pd.concat([manifefst_test, lang_test], ignore_index=True)

new_manifest_train = pd.DataFrame({'Path': manifefst_train.iloc[:, 0], 'Gt': manifefst_train.iloc[:, 1]})
new_manifest_val = pd.DataFrame({'Path': manifefst_val.iloc[:, 0], 'Gt': manifefst_val.iloc[:, 1]})
new_manifest_test = pd.DataFrame({'Path': manifefst_test.iloc[:, 0], 'Gt': manifefst_test.iloc[:, 1]})

os.makedirs(output_dir, exist_ok=True)
new_manifest_train.to_csv(os.path.join(output_dir, 'training.txt'), header=False, sep=' ', index=False)
new_manifest_val.to_csv(os.path.join(output_dir, 'validation.txt'), header=False, sep=' ', index=False)
new_manifest_test.to_csv(os.path.join(output_dir, 'testing.txt'), header=False, sep=' ', index=False)