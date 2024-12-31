'''Data preparation for training and testing.'''
import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset
from research.data.data_loader import load_document


class DocDataset(Dataset):
    '''Personalized dataset for document embedding.'''

    def __init__(self, docs, doc_embs):
        self.docs = docs
        self.embs = doc_embs

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, index):
        return self.docs[index], self.embs[index]


class PivotDataset(Dataset):
    '''Pivot dataset'''

    def __init__(self, docs, emb1, emb2):
        self.docs = docs
        self.emb1 = emb1  # Private emb
        self.emb2 = emb2  # Surrogate emb

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, index):
        return self.docs[index], self.emb1[index], self.emb2[index]


class AdvDataset(Dataset):
    '''Personalized dataset for adv document embedding.'''

    def __init__(self, docs, doc_embs, surrogate_embs, domains):
        self.docs = docs
        self.embs = doc_embs
        self.surrogate_embs = surrogate_embs
        self.domains = domains

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, index):
        return self.docs[index], self.embs[index], self.surrogate_embs[index], self.domains[index]


def remove_empty_docs(dataset):
    '''Remove empty docs from dataset'''
    sent_list = []
    emb_list = []
    for data in dataset:
        if len(data[0]) == 0:
            continue
        sent_list.append(data[0])
        emb_list.append(data[1])

    return DocDataset(sent_list, np.array(emb_list))


def generate_data_split(config, dataset):
    '''
    Generate train, val, test split.
    '''
    train_size = int(0.8 * len(dataset))
    val_size = int((len(dataset) - train_size) / 8)
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size])

    # Training data ratio split
    remain_size = int(len(train_dataset) * config['train_ratio'])
    train_dataset, remain_dataset = torch.utils.data.random_split(
        train_dataset, [remain_size, len(train_dataset) - remain_size])

    return train_dataset, val_dataset, test_dataset, remain_dataset


def prepare_external_data(config):
    '''Prepare external data'''
    external_size = config['external_size']
    print(f"Start loading external dataset: {config['external_dataset']}")
    external_sents = load_document(
        dataset=config['external_dataset'])['documents']
    external_sents = external_sents[:external_size]
    # We don't need external embeddings
    external_embs = [0] * len(external_sents)
    external_dataset = DocDataset(external_sents, external_embs)

    return external_dataset


def load_prepared_dataset(config):
    '''Load prepared dataset contains original data'''
    dataset_path = os.path.join(
        config["dataset_root"], f"{config['dataset']}_FullDataset.pkl")
    with open(dataset_path, "rb") as dataset_f:
        full_dataset = pickle.load(dataset_f)

    # Get dataset
    sent_list, emb_list = [], []
    for sent, emb_dict in full_dataset.items():
        emb = emb_dict[config['blackbox_encoder']]
        emb = emb.astype(np.float32)
        sent_list.append(sent)
        emb_list.append(emb)

    # Pick first training size for training
    train_sent_list, train_emb_list = [], []
    for idx, (sent, emb) in enumerate(zip(sent_list, emb_list)):
        if idx >= config['training_size']:
            break
        train_sent_list.append(sent)
        train_emb_list.append(emb)

    train_dataset = DocDataset(train_sent_list, np.array(train_emb_list))

    # Pick final 2800 for validation
    validation_size = 2800 if not config['testing'] else 10
    val_sent_list = sent_list[len(sent_list) - validation_size:]
    val_embs_list = emb_list[len(emb_list) - validation_size:]
    val_dataset = DocDataset(val_sent_list, np.array(val_embs_list))
    print("Load prepared dataset done")

    return remove_empty_docs(train_dataset), val_dataset


def load_augmented_data(config, train_dataset):
    '''Load augmented data'''
    if config['option'] == 'None':
        print("No augmentation")
        return train_dataset

    print(f"Loading augmented data: {config['option']}")
    aug_dataset_path = os.path.join(
        config["dataset_root"], f"{config['dataset']}_FullDataset_aug.pkl")
    with open(aug_dataset_path, "rb") as aug_f:
        aug_dict = pickle.load(aug_f)

    sent_list, emb_list = [], []
    for sent, emb in train_dataset:
        sent_list.append(sent)
        emb_list.append(emb)
        for idx, aug_sent in enumerate(aug_dict[sent][config['option']]):
            if idx >= config['multiple']:
                break
            sent_list.append(aug_sent)
            emb_list.append(emb)

    return DocDataset(sent_list, np.array(emb_list))


def prepare_dataset(config, sent_list, doc_embs):
    '''Prepare train, val, test, and external dataset'''
    # Use half of the original dataset for training and half for external data
    data_size = int(len(sent_list) / 2)
    external_size = len(sent_list) - data_size
    dataset = DocDataset(sent_list[:data_size], doc_embs[:data_size])

    # Prepare external dataset
    if config['external_dataset'] != config['dataset']:
        external_dataset = prepare_external_data(config)
    else:
        print("Using half of the original dataset as external data")
        external_dataset = DocDataset(
            sent_list[data_size:data_size + external_size], doc_embs[data_size:data_size + external_size])

    train_dataset, val_dataset, test_dataset, _ = generate_data_split(
        config, dataset)

    return train_dataset, val_dataset, test_dataset, external_dataset


def prepare_pivot_data(train_dataset, surrogate):
    '''Prepare pivot data for training'''
    sent_list = []
    private_embs = []
    for sents, embs in train_dataset:
        sent_list.append(sents)
        private_embs.append(embs)
    private_embs = np.array(private_embs)
    surrogate_embs = surrogate.encode(sent_list, "private")

    return PivotDataset(sent_list, private_embs, surrogate_embs)


def prepare_adv_additional_data(pivot_dataset, additional_dataset, surrogate, config):
    '''Prepare additional data for adv training'''
    # Use external data to generate additional data
    sent_list = []
    for sents, _ in additional_dataset:
        sent_list.append(sents)
    additional_embs = surrogate.encode(sent_list, "external")

    # Prepare additional dataloader
    additional_dataset = DocDataset(sent_list, additional_embs)

    # Use weighted random sampler to sample train+additional data
    weights = [len(additional_dataset)] * len(pivot_dataset) + \
        [len(pivot_dataset)] * len(additional_dataset)

    # Prepare adv dataset
    adv_sents, adv_embs, surrogate_embs, adv_labels = [], [], [], []
    for sents, embs, s_embs in pivot_dataset:
        adv_sents.append(sents)
        adv_embs.append(embs)
        surrogate_embs.append(s_embs)
        adv_labels.append(0)
    for sents, embs in additional_dataset:
        adv_sents.append(sents)
        # This is just a fake one since additional data doesn't have private emb
        adv_embs.append(adv_embs[0])
        surrogate_embs.append(embs)
        adv_labels.append(1)

    adv_train_dataset = AdvDataset(
        adv_sents, adv_embs, surrogate_embs, adv_labels)

    sampler = WeightedRandomSampler(
        weights, len(adv_train_dataset), replacement=True)
    return DataLoader(dataset=adv_train_dataset,
                      batch_size=config['batch_size'],
                      sampler=sampler)


def prepare_geia_data(pivot_dataset, config):
    '''Prepare for geia training'''
    # Prepare geia dataset
    adv_sents, adv_embs, surrogate_embs, adv_labels = [], [], [], []
    for sents, embs, s_embs in pivot_dataset:
        adv_sents.append(sents)
        adv_embs.append(embs)
        surrogate_embs.append(s_embs)
        adv_labels.append(0)

    adv_train_dataset = AdvDataset(
        adv_sents, adv_embs, surrogate_embs, adv_labels)

    return DataLoader(dataset=adv_train_dataset,
                      batch_size=config['batch_size'])


def prepare_additional_data(train_dataset, additional_dataset, surrogate, config):
    '''Prepare additional data for training'''
    # Use validation data to generate additional data
    sent_list = []
    for sents, _ in additional_dataset:
        sent_list.append(sents)
    additional_embs = surrogate.encode(sent_list)

    # Prepare additional dataloader
    additional_dataset = DocDataset(sent_list, additional_embs)

    # Use weighted random sampler to sample train+additional data
    weights = [len(additional_dataset)] * len(train_dataset) + \
        [len(train_dataset)] * len(additional_dataset)
    train_dataset = ConcatDataset([train_dataset, additional_dataset])
    sampler = WeightedRandomSampler(
        weights, len(train_dataset), replacement=True)
    return DataLoader(dataset=train_dataset,
                      batch_size=config['batch_size'],
                      sampler=sampler)
