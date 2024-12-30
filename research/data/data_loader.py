'''Load dataset'''
import json
import numpy as np

from pathlib import Path
from sklearn.datasets import fetch_20newsgroups
from datasets import load_dataset


def load_document(dataset: str = '20news'):
    '''Load dataset from different sources.'''
    if dataset == "20news":
        num_classes = 20
        raw_text, target = fetch_20newsgroups(data_home="./", subset='all', categories=None,
                                              shuffle=False, return_X_y=True)
        documents = list(raw_text)
        target = list(target)
    elif dataset == "qnli":
        num_classes = None
        data = load_dataset('glue', 'qnli', cache_dir="data/", split='all')
        documents, target = [], []
        for _, data_dict in enumerate(data):
            documents.append(data_dict['question'])
            documents.append(data_dict['sentence'])
    elif dataset == "personachat":
        num_classes = None
        documents, target = [], []
        data_dir = Path('research/data/personachat')
        data_split = ['train.txt', 'dev.txt', 'test.txt']
        for split in data_split:
            with open(data_dir / split, 'r', encoding='UTF-8') as file_data:
                data = json.load(file_data)
            for _, dic in enumerate(data):
                documents.extend(dic['conv'])
    elif dataset == "IMDB":
        data = load_dataset("imdb", split="all")
        raw_documents = data["text"]
        documents = []
        for doc in raw_documents:
            sentence = doc.split(".")[0]
            sentence += "."
            documents.append(sentence)
        target = data["label"]
        num_classes = 2
    elif dataset == "MIMC":
        data = load_dataset("Medilora/mimic_iii_diagnosis_anonymous", split="train")
        raw_documents = data["text"]
        documents = []
        for doc in raw_documents:
            sentence = doc.split(".")[0]
            sentence = sentence.split(":")[-1][1:]
            sentence += "."
            sentence_len = len(sentence.split())
            if not "year" in sentence:
                continue
            if sentence_len < 20:
                documents.append(sentence)
        target = []
        num_classes = 0
    elif dataset == "agnews":
        data = load_dataset("ag_news", split="all")
        raw_documents = data["text"]
        documents = []
        for doc in raw_documents:
            sentence = doc.split(".")[0]
            sentence += "."
            documents.append(sentence)
        target = data["label"]
        num_classes = 4
    elif dataset == "wiki":
        data = load_dataset("wikitext", 'wikitext-2-v1',
                            split="all")
        unprocessed_documents = data['text']
        documents = []
        for doc in unprocessed_documents:
            if len(doc) > 0:
                if doc[1] != '=':
                    documents.append(doc)
        target = []
        num_classes = 0
    elif dataset == "tweet":
        data = load_dataset("tweet_eval", "emotion", split="all")
        documents = data["text"]
        target = data["label"]
        num_classes = 4
    else:
        raise NotImplementedError

    return {"documents": documents, "target": np.array(target), "num_classes": num_classes}
