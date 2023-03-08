# x-vector-pytorch
This repo contains the implementation of the paper "Spoken Language Recognition using X-vectors" in Pytorch
Paper: https://danielpovey.com/files/2018_odyssey_xvector_lid.pdf

Tutorial : https://www.youtube.com/watch?v=8nZjiXEdMH0

Install required packges using requirements.txt
```bash
pip iinstall -r requirements.txt
```

## Create manifest files for training and testing

This step creates training and testing files.

```bash=
python datasets.py --raw_data /mnt/storage2t/crnn-lid_segmented --meta_store_path manifest 
```

Data should be structured as

```bash=
├── language_x
    ├── sub_folder_1
        ...
    ├── sub_folder_2
        ...
    └── sub_folder_3
        ...
```

## Offline Fearture Extracting

```bash=
python feature_extraction.py  --raw_data /mnt/storage2t/crnn-lid_segmented --meta_store_path manifest             
```

## Training
This steps starts training the X-vector model for language identification 
```bash=
python training_xvector.py --training_filepath meta/training.txt --testing_filepath meta/testing.txt
                        --validation_filepath meta/validation.txt --input_dim 40 --num_classes 8
                        --batch_size 32 --use_gpu True --num_epochs 100
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
