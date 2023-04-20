import numpy as np
import json
import os

manifests = ['/home/ycj0123/x-vector-pytorch/manifest_stage1',
            '/home/ycj0123/x-vector-pytorch/manifest_stage2',
            '/home/ycj0123/x-vector-pytorch/manifest_stage2_home',
            '/home/ycj0123/x-vector-pytorch/manifest_stage3']
output_folder = '/home/ycj0123/x-vector-pytorch/manifest_all'
os.makedirs(output_folder, exist_ok=True)


def make_manifest(manifest_sep, mode='testing'):
    to_write = []
    for i, m in enumerate(manifest_sep):
        audio_links = [line.rstrip('\n').split(' ')[0] for line in open(f"{m}/{mode}.txt")]
        labels = [new_class_ids[id_class[i][int(line.rstrip('\n').split(' ')[1])]] for line in open(f"{m}/{mode}.txt")]
        for j, (a, l) in enumerate(zip(audio_links, labels)):
            to_write.append(a+' '+str(l))
    with open(f"{output_folder}/{mode}.txt", "w+") as f:
        for filepath in to_write:
            f.write(filepath+'\n')


class_id = []
for m in manifests:
    with open(f"{m}/class_ids.json", "r") as f:
        class_id.append(json.load(f))

classes = [d for c in class_id for d in c]
classes_unique = np.unique(classes)
assert len(classes) == len(classes_unique), "Repeated classes exsist!"

new_class_ids = {l: i for i, l in enumerate(classes)}
id_class = [{v: k for k, v in c.items()} for c in class_id]
# for c in class_ids:
#     id_class.append({v: k for k, v in c.items()})

with open(f"{output_folder}/class_ids.json", "w+") as f:
    json.dump(new_class_ids, f, indent=4)

make_manifest(manifests, 'training')
make_manifest(manifests, 'testing')
make_manifest(manifests, 'validation')
