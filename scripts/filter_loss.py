import pandas as pd
import os
from tqdm import tqdm


preds = pd.read_csv('new_all_13_sorted.csv', index_col=0)
output_dir = 'new_manifest_filtered'
threshold_hour, train_hour, val_test_hour = 110, 100, 5
threshold_n, train_n, val_test_n = threshold_hour*360, train_hour*360, val_test_hour*360
filter_ratio, train_ratio, val_ratio = 0.99, 0.9, 0.5  # val = (1 - train_ratio) * val_ratio

first = True
for i in tqdm(range(138)):
    lang = preds[preds['Ground Truth'] == i]
    if not lang.empty:
        keep = int(len(lang) * filter_ratio)
        lang = lang.iloc[-keep:]
        if len(lang) > threshold_n:
            lang_train = lang.sample(train_n)
            lang_valtest = lang.drop(lang_train.index)
            lang_val = lang_valtest.sample(val_test_n)
            lang_test = lang_valtest.drop(lang_val.index).sample(val_test_n)
        else:
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

new_manifest_train = pd.DataFrame({'Path': manifefst_train['Path'], 'Gt': manifefst_train['Ground Truth']})
new_manifest_val = pd.DataFrame({'Path': manifefst_val['Path'], 'Gt': manifefst_val['Ground Truth']})
new_manifest_test = pd.DataFrame({'Path': manifefst_test['Path'], 'Gt': manifefst_test['Ground Truth']})

os.makedirs(output_dir, exist_ok=True)
new_manifest_train.to_csv(os.path.join(output_dir, 'training.txt'), header=False, sep=' ', index=False)
new_manifest_val.to_csv(os.path.join(output_dir, 'validation.txt'), header=False, sep=' ', index=False)
new_manifest_test.to_csv(os.path.join(output_dir, 'testing.txt'), header=False, sep=' ', index=False)