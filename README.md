# x-vector-pytorch
This repo contains the implementation of the paper "Spoken Language Recognition using X-vectors" in Pytorch
Paper: https://danielpovey.com/files/2018_odyssey_xvector_lid.pdf

Tutorial : https://www.youtube.com/watch?v=8nZjiXEdMH0

Install required packges using requirements.txt
```bash
pip install -r requirements.txt
```

## Create manifest files for training and testing

This step creates training and testing files.

```bash=
python datasets.py --raw_data /path/to/raw_data --meta_store_path manifest 
```

Data should be structured as follows:

```bash=
├── /path/to/raw_data
    ├── language_x
        ...
    ├── language_y
        ...
    └── language_z
        ...
```

## Offline Fearture Extracting

You can choose to either extract features offline or do it while training (online).

```bash=
python feature_extraction.py  --raw_data /path/to/raw_data --meta_store_path manifest             
```

The extracted features will be stored as follows:

```bash=
├── /path/to/raw_data
    ├── train
        ...
    ├── validation
        ...
    └── test
        ...
```

## Training
This steps starts training the X-vector model for language identification 

```bash=
# offline
# remember to check in `training_xvector.py` the default of `--extract_online` is set to False
python training_xvector.py --training_feature manifest/training_feat.txt --validation_feature manifest/validation_feat.txt
                        --input_dim 257 --num_classes 14 --batch_size 256 --num_epochs 50 --save_epoch 10
                        --use_gpu

# online
python training_xvector.py --training_meta manifest/training.txt --validation_meta manifest/validation.txt
                        --input_dim 257 --num_classes 14 --batch_size 256 --num_epochs 50 --save_epoch 10
                        --use_gpu --extract_online
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
