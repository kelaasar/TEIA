# TEIA
This repository contains code for ACL 2024 paper: [Transferable Embedding Inversion Attack: Uncovering Privacy Risks in Text Embeddings without Model Queries](https://aclanthology.org/2024.acl-long.230/)

## How to install dependency
1. Python version: 3.9
2. [Install poetry](https://python-poetry.org/docs/#installing-with-the-official-installer)
3. Install dependency (Need poetry)
    ```
    poetry install
    ```

## Prepare dataset
Run the following command to generate dataset
```
python generate_fulldataset.py --dataset_name=[dataset_name] --dataset_root=[path_to_save_dataset]
python generate_augdata.py --dataset_name=[dataset_name] --dataset_root=[path_to_save_aug_data]
```

## The migration is still in progress. You may encounter unfinished features or bugs. I am actively working on improvements and will release updates shortly. Thank you for your patience!