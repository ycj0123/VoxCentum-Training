# x-vector-pytorch
This repo contains the implementation of the paper "Spoken Language Recognition using X-vectors" in Pytorch

Paper: https://danielpovey.com/files/2018_odyssey_xvector_lid.pdf

Tutorial : https://www.youtube.com/watch?v=8nZjiXEdMH0

Python version == `3.10.8` is recommended.

Install required packages using `requirements.txt`.
```bash=
conda create -n xvector python=3.10.8
conda activate xvector
conda install pip
pip install -r requirements.txt
```

## Create manifest files for training and testing

This step creates training and testing files.

```bash=
python generate_manifest.py --raw_data /path/to/raw_data --meta_store_path manifest 
```

Data should be structured as follows (having subfolders under each language is fine):

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

Note: Extracting offline doesn't seem to speed up training by a lot.

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
This step starts training the X-vector model for language identification. Remember to check `config.yaml` for hyperparameters.

```bash=
python training_xvector.py config.yaml
```

## Testing

```bash=
python inference_xvector.py --model_path path/to/ckpt --testing_meta path/to/manifest --num_classes same_as_training
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
