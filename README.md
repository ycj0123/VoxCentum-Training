# VoxCentum-Training

Training code for *Voxcentum: Spoken Language Identification for 100+ Languages Expanded to 100+ Hours*.

Python version == `3.10.8` is recommended.

Install required packages using `requirements.txt`.
```bash=
conda create -n voxcentum python=3.10.8
conda activate voxcentum
conda install pip
pip install -r requirements.txt
```

## Download the VoxCentum Dataset

TBD

## Create Manifest Files for Training and Testing

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

## Training
This step starts training the model for language identification. Remember to check `config.yaml` for hyperparameters.

```bash=
python training.py config.yaml
```

## Testing

```bash=
python inference.py --model_path /path/to/ckpt --manifest_dir /path/to/manifest --output /output/dir
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
