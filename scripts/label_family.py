import json
import pandas as pd
from tqdm import tqdm
import os

input_dir = '/home/ycj0123/x-vector-pytorch/manifest/new_random_40_'
output_dir = '/home/ycj0123/x-vector-pytorch/manifest/new_random_40_family_'

with open("/home/ycj0123/x-vector-pytorch/iso_639-1.json", "r") as f:
    family_dict = json.load(f)

manifest_test = pd.read_csv(os.path.join(input_dir, "testing.txt"), sep=' ', header=None)
manifest_val = pd.read_csv(os.path.join(input_dir, "validation.txt"), sep=' ', header=None)
manifest_train = pd.read_csv(os.path.join(input_dir, "training.txt"), sep=' ', header=None)

with open(os.path.join(input_dir, "class_ids.json"), "r") as f:
    class_ids = json.load(f)
id_classes = [k for k in class_ids.keys()]

fam_idx = 0
families = {}
new_col = []
for i, r in tqdm(manifest_test.iterrows(), total=len(manifest_test)):
    lang_code = id_classes[int(r[1])].split("-")[0]
    family = family_dict[lang_code]['family']
    if not family in families:
        families[family] = fam_idx
        fam_idx += 1
    new_col.append(families[family])
manifest_test['family'] = new_col

new_col = []
for i, r in tqdm(manifest_train.iterrows(), total=len(manifest_train)):
    lang_code = id_classes[int(r[1])].split("-")[0]
    family = family_dict[lang_code]['family']
    new_col.append(families[family])
manifest_train['family'] = new_col

new_col = []
for i, r in tqdm(manifest_val.iterrows(), total=len(manifest_val)):
    lang_code = id_classes[int(r[1])].split("-")[0]
    family = family_dict[lang_code]['family']
    new_col.append(families[family])
manifest_val['family'] = new_col

os.makedirs(output_dir, exist_ok=True)
manifest_test.to_csv(os.path.join(output_dir, "testing.txt"), sep=' ', header=None, index=None)
manifest_val.to_csv(os.path.join(output_dir, "validation.txt"), sep=' ', header=None, index=None)
manifest_train.to_csv(os.path.join(output_dir, "training.txt"), sep=' ', header=None, index=None)
with open(os.path.join(output_dir, "family_ids.json"), "w+") as f:
    json.dump(families, f, indent=4)
with open(os.path.join(output_dir, "class_ids.json"), "w+") as f:
    json.dump(class_ids, f, indent=4)