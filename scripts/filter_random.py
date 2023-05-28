import pandas as pd
import os
from tqdm import tqdm

manifest_dirs = ['/home/ycj0123/x-vector-pytorch/manifest/new_random/testing.txt',
                 '/home/ycj0123/x-vector-pytorch/manifest/new_random/training.txt',
                 '/home/ycj0123/x-vector-pytorch/manifest/new_random/validation.txt']
manifest_origs = [pd.read_csv(dir, sep=' ', names=['Path', 'Gt']) for dir in manifest_dirs]
manifest_orig = pd.concat(manifest_origs, axis=0, ignore_index=True)
print(manifest_orig)
output_dir = '/home/ycj0123/x-vector-pytorch/manifest/new_random_40_'
threshold_hour, train_hour, val_test_hour = 44, 40, 2
threshold_n, train_n, val_test_n = threshold_hour*360, train_hour*360, val_test_hour*360
filter_ratio, train_ratio, val_ratio = 1, 0.9, 0.5  # val = (1 - train_ratio) * val_ratio

first = True
for i in tqdm(range(137)):
    lang = manifest_orig[manifest_orig.iloc[:, 1] == i]
    if not lang.empty:
        if len(lang) > threshold_n:
            lang = lang.sample(train_n+(2*val_test_n))
            lang_train = lang.sample(train_n)
            lang_valtest = lang.drop(lang_train.index)
            lang_val = lang_valtest.sample(val_test_n)
            lang_test = lang_valtest.drop(lang_val.index)
        else:
            keep = int(len(lang) * filter_ratio)
            lang = lang.sample(keep)
            lang_train = lang.sample(frac=train_ratio)
            lang_valtest = lang.drop(lang_train.index)
            lang_val = lang_valtest.sample(frac=val_ratio)
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