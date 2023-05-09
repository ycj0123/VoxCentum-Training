import numpy as np
import json
import os

src_manifests = ['/home/itk0123/x-vector-pytorch/manifest_filtered']
tgt_cls_id = '/home/itk0123/x-vector-pytorch/manifest_filtered/class_ids.json'
output_folder = '/home/itk0123/x-vector-pytorch/manifest_filtered_relabel'
os.makedirs(output_folder, exist_ok=True)


def make_manifest(manifest_sep, mode='testing'):
    to_write = []
    for i, m in enumerate(manifest_sep):
        audio_links = [line.rstrip('\n').split(' ')[0] for line in open(f"{m}/{mode}.txt")]
        labels = [new_class_id[id_class[i][int(line.rstrip('\n').split(' ')[1])]] for line in open(f"{m}/{mode}.txt")]
        for j, (a, l) in enumerate(zip(audio_links, labels)):
            to_write.append(a+' '+str(l))
    with open(f"{output_folder}/{mode}.txt", "w+") as f:
        for filepath in to_write:
            f.write(filepath+'\n')


class_id = []
for m in src_manifests:
    with open(f"{m}/class_ids.json", "r") as f:
        class_id.append(json.load(f))

with open(tgt_cls_id, "r") as f:
    new_class_id = json.load(f)

id_class = [{v: k for k, v in c.items()} for c in class_id]

with open(f"{output_folder}/class_ids.json", "w+") as f:
    json.dump(new_class_id, f, indent=4)

make_manifest(src_manifests, 'training')
make_manifest(src_manifests, 'testing')
make_manifest(src_manifests, 'validation')
