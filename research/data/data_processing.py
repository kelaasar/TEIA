'''Data processing utils.'''
import os
import json
from typing import List

import numpy as np
from tqdm.auto import tqdm
from scipy import sparse
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from research.utils.toolbox import get_free_gpu


def generate_tfidf_voc(preprocessed_docs: List[str], save_folder: str):
    '''Generate gensim_tfidf, vocabulary, and voc2idx mapping and save them to folder.'''
    tfidf_path = os.path.join(save_folder, 'TFIDF.npz')
    voc_path = os.path.join(save_folder, 'vocabulary.npy')
    map_path = os.path.join(save_folder, 'mapping.json')

    vectorizer = TfidfVectorizer(norm=None)
    tfidf_vector = vectorizer.fit_transform(preprocessed_docs)
    vocabulary = np.array(vectorizer.get_feature_names_out())
    mapping = vectorizer.vocabulary_

    sparse.save_npz(os.path.join(save_folder, "TFIDF.npz"), tfidf_vector)
    np.save(os.path.join(save_folder, "vocabulary.npy"), vocabulary)
    with open(os.path.join(save_folder, "mapping.json"), "w", encoding='UTF-8') as map_f:
        json.dump(mapping, map_f)

    # Load tfidf and voc from files.
    tfidf_vector = sparse.load_npz(tfidf_path).toarray()
    vocabulary = np.load(voc_path, allow_pickle=True)
    with open(map_path, 'r', encoding='UTF-8') as map_f:
        mapping = json.load(map_f)

    return tfidf_vector, vocabulary, mapping


def get_document_labels(preprocessed_docs: List[str],
                        data_folder: str = "./", dataset: str = "20news"):
    '''
    Generate target and put them under "data/precompute_target/data_name/" .
    args:
      preprocessed_docs: document after processing.
      data_folder: relative path for the data folder.

    Prepare all needed path.
    '''
    config_dir = os.path.join(data_folder, f"precompute_target/{dataset}")
    os.makedirs(config_dir, exist_ok=True)

    # Read precompute labels from files.
    # TFIDF && vocabulary
    tfidf_vector, vocabulary, _ = generate_tfidf_voc(
        preprocessed_docs, config_dir)

    labels = {'tf-idf': tfidf_vector}

    return labels, vocabulary


def get_document_embs(preprocessed_docs: list[str], encoder_type: str, device: str = None):
    '''
    Returns embeddings(input) for document decoder

            Parameters:
                    preprocessed_docs (list): 
                    model_name (str):
            Returns:
                    doc_embs (array):
                    model (class):
    '''
    print('Getting preprocess documents embeddings')
    if device is None:
        device = get_free_gpu()
    if encoder_type == 'average':
        model = SentenceTransformer(
            "average_word_embeddings_glove.840B.300d", device=device)
        doc_embs = np.array(model.encode(preprocessed_docs,
                            show_progress_bar=True, batch_size=16))
    elif encoder_type == 'doc2vec':
        doc_embs = []
        preprocessed_docs_split = [doc.split() for doc in preprocessed_docs]
        documents = [TaggedDocument(doc, [i])
                     for i, doc in enumerate(preprocessed_docs_split)]
        model = Doc2Vec(documents, vector_size=200, workers=4)
        for doc_split in preprocessed_docs_split:
            doc_embs.append(model.infer_vector(doc_split))
        doc_embs = np.array(doc_embs)
    elif encoder_type == 'sbert':
        model = SentenceTransformer(
            "all-mpnet-base-v2", device=device)
        doc_embs = np.array(model.encode(preprocessed_docs,
                            show_progress_bar=True, batch_size=16))
    elif encoder_type == 'mpnet':
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=device)
        doc_embs = np.array(model.encode(preprocessed_docs))
    elif encoder_type == 'st5':
        model = SentenceTransformer('sentence-transformers/sentence-t5-large', device=device)
        doc_embs = np.array(model.encode(preprocessed_docs))
    elif encoder_type == 'minilm':
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
        doc_embs = np.array(model.encode(preprocessed_docs))
    elif encoder_type == 'openai':
        from openai import OpenAI
        client = OpenAI()
        model="text-embedding-ada-002"
        doc_embs = []
        for batch_docs in tqdm(batch(preprocessed_docs, 2048)):
            output = client.embeddings.create(input = batch_docs, model=model)
            embs = np.array([i.embedding for i in output.data])
            doc_embs.append(embs)
        doc_embs = np.concatenate(doc_embs)
    else:
        print(f"Encoder type {encoder_type} not implemented.")
        raise NotImplementedError
        # TODO: Self define encoder model implement. Please implement encoder model in model folder.
        # if model is None:
        #     model = Black(device, encoder_type, num_classes)
        # doc_embs = model.encode_all(preprocessed_docs)

    del model
    return doc_embs


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
