'''This file is used to evaluate the output sentences of the model'''
import os
import nltk
import string
import warnings
import evaluate
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from research.utils.toolbox import get_free_gpu
from sentence_transformers import SentenceTransformer, util
from tqdm import trange

load_dotenv()  # take environment variables from .env.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")  # Ignore all warnings globally


def remove_eos(pred: list[str]):
    '''Remove end of sentence token from the output of the model'''
    for i, sen in enumerate(pred):
        pred[i] = sen.replace('<|endoftext|>', '')


def get_rouge(pred: list[str], target: list[str]):
    '''Calculate the rouge score of the model output'''
    rouge = evaluate.load('rouge')
    return rouge.compute(predictions=pred, references=target)


def get_bleu(pred: list[str], target: list[str]):
    '''Calculate the bleu score of the model output'''
    cands_list_bleu = [sentence.split() for sentence in pred]
    refs_list_bleu = [[sentence.split()] for sentence in target]
    bleu_score = nltk.translate.bleu_score.corpus_bleu(
        refs_list_bleu, cands_list_bleu)
    bleu_score_1 = nltk.translate.bleu_score.corpus_bleu(
        refs_list_bleu, cands_list_bleu, weights=(1, 0, 0, 0))
    bleu_score_2 = nltk.translate.bleu_score.corpus_bleu(
        refs_list_bleu, cands_list_bleu, weights=(0.5, 0.5, 0, 0))

    return {'bleu1': bleu_score_1, 'bleu2': bleu_score_2, 'bleu': bleu_score}


def punctuation_remove(sent_list: list[str]):
    '''Remove punctuation from the model output'''
    # create a translation table that maps punctuation to None
    table = str.maketrans('', '', string.punctuation)
    # apply the translation table to each string in the list
    return [s.translate(table) for s in sent_list]


def exact_match(pred: list[str], target: list[str]):
    '''Calculate the exact match ratio of the model output'''
    assert len(pred) == len(target)
    count = 0
    for i, _ in enumerate(pred):
        gt_str = target[i]
        pred_str = pred[i]
        if gt_str == pred_str:
            count += 1
    ratio = count/len(target)
    return {'exact_match': ratio}


def batch(iterable, batch_size):
    '''Batch the sentences'''
    iterable = iter(iterable)
    while True:
        chunk = []
        for _ in range(batch_size):
            try:
                chunk.append(next(iterable))
            except StopIteration:
                yield chunk
                return
        yield chunk


def embed_similarity(pred: list[str], target: list[str], batch_size: int = 16):
    '''Calculate the average embedding similarity of the model output'''
    model = SentenceTransformer(
        'sentence-transformers/all-mpnet-base-v2', device=get_free_gpu())
    gt_batch = list(batch(target, batch_size))
    pred_batch = list(batch(pred, batch_size))
    cosine_scores_all = []
    for _, (gt_batch, pred_batch) in enumerate(zip(gt_batch, pred_batch)):
        if len(gt_batch) == 0 or len(pred_batch) == 0:
            continue
        embeddings1 = model.encode(gt_batch, convert_to_tensor=True)
        embeddings2 = model.encode(pred_batch, convert_to_tensor=True)
        cosine_scores = util.cos_sim(embeddings1, embeddings2)
        assert cosine_scores.size()[0] == cosine_scores.size()[1]
        score_list = [cosine_scores[k][k].item()
                      for k in range(cosine_scores.size()[0])]
        cosine_scores_all.extend(score_list)

    avg_score = np.mean(cosine_scores_all)
    return {'embed_similarity': avg_score}


def llm_eval(pred: list[str], target: list[str]):
    '''Evaluate the model output using the LLM evaluation'''
    client = OpenAI()
    result_score = []
    error_count = 0
    for i in trange(len(pred)):
        pred_str = pred[i]
        gt_str = target[i]
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Output a number between 0 and 1 describing the semantic similiarity, fluent ,and coherent between the following two sentences: please output the answer without any explaination."},
                    {"role": "user", "content": pred_str},
                    {"role": "user", "content": gt_str}
                ]
            )
            score = float(response.choices[0].message.content)
            if score < 0 or score > 1:
                raise ValueError
            result_score.append(score)
        except:
            error_count += 1
            continue

    print(f"Error count: {error_count}")
    avg_score = np.mean(result_score)
    return {'llm_eval': avg_score}


def calculate_metrics(pred: list[str], target: list[str]):
    '''Calculate the rouge, bleu and exact match ratio of the model output'''
    remove_eos(pred)
    result = {}
    result.update(embed_similarity(pred, target))
    result.update(get_rouge(pred, target))
    result.update(get_bleu(pred, target))
    result.update(exact_match(pred, target))
    return result


def eval_generation(pred: list[str], target: list[str]):
    '''LLM evaluation'''''
    remove_eos(pred)
    result = {}
    if os.getenv("OPENAI_API_KEY") is not None:
        result.update(llm_eval(pred, target))
    return result


if __name__ == '__main__':
    seten_1 = ["Hello world"]
    senten_2 = ["Hello world"]
    print(calculate_metrics(seten_1, senten_2))
    print(eval_generation(seten_1, senten_2))
