from research.config.base_config import parse_argument
from research.data.data_prepare import load_document
from research.data.data_processing import get_document_embs
from research.utils.toolbox import same_seed
from collections import defaultdict
import random
import warnings
import pickle
import os

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # Initialize settings
    NUM_DOCS = 50000 # set a larger number for dedup
    EMBEDDING_MODELS = ["sbert", "st5"]
    same_seed(123)
    args = parse_argument()
    config = vars(args)

    print(config)
    documents = load_document(config["dataset"])["documents"]
    NUM_DOCS = min(NUM_DOCS, len(documents))
    assert len(documents) >= NUM_DOCS
    sampled_documents = random.sample(documents, NUM_DOCS)
    assert len(sampled_documents) == NUM_DOCS
    print(sampled_documents[:2])  # for sanity check

    # schema: {doc1: {emb1: [embedding1], emb2: [embedding2]}}
    outputs = defaultdict(dict)
    for encoder_name in EMBEDDING_MODELS:
        embedding = get_document_embs(sampled_documents, encoder_name)
        for doc, emb in zip(sampled_documents, embedding):
            outputs[doc][encoder_name] = emb
    outputs = dict(outputs)
    print("Number of documents:", len(outputs))

    # save results
    root = config["dataset_root"]
    os.makedirs(root, exist_ok=True, mode=777)
    path = os.path.join(root, f"{config['dataset']}_FullDataset.pkl")
    print("Saving processed file to:", path)
    with open(path, "wb") as f:
        pickle.dump(outputs, f)
