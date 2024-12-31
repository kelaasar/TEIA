# TEIA
This repository contains code for ACL 2024 paper: [Transferable Embedding Inversion Attack: Uncovering Privacy Risks in Text Embeddings without Model Queries](https://aclanthology.org/2024.acl-long.230/)

## Prerequisites
1. If you want to use LLM evaluation, you need to create .env file and add your openai api key
    ```
    OPENAI_API_KEY=[your_openai_api_key]
    ```

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

## How to run our code
1. Run our model (TEIA)
```
pyton main.py --project_name=[project_name] --exp_name=[exp_name] --dataset=[dataset_name]
```
2. Run baseline model (GEIA)
```
pyton main.py --geia --project_name=[project_name] --exp_name=[exp_name] --dataset=[dataset_name]
```
For more information, please check **research/config/base_config.py**

## The migration is still in progress. You may encounter unfinished features or bugs. I am actively working on improvements and will release updates shortly. Thank you for your patience!