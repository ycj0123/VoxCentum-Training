import pandas as pd
import os


preds = pd.read_csv('output.csv', index_col=0)
output_dir = 'manifest_filtered'

first = True
for i in range(138):
    lang = preds[preds['Ground Truth'] == i]
    if not lang.empty:
        if len(lang) > 46800:
            lang = lang.iloc[-39600:]
            lang_train = lang.sample(36000)
            lang_valtest = lang.drop(lang_train.index)
            lang_val = lang_valtest.sample(1800)
            lang_test = lang_valtest.drop(lang_val.index)
        else:
            keep = int(len(lang) * 0.85)
            lang = lang.iloc[-keep:]
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