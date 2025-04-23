from research.config.base_config import parse_argument
from textattack.augmentation import WordNetAugmenter
from random_word import RandomWords
import os
import random
import numpy as np
import warnings
import pickle
from tqdm.auto import trange, tqdm
from openai import OpenAI
warnings.filterwarnings("ignore")

# EDA 
wordnet_aug = WordNetAugmenter()

def EDA(text, option, count):
    text = str(text)
    if option == "replace":
        print("replace.")
        for _ in range(count):
            text = wordnet_aug.augment(text)[0]
        return text
    if option == "delete":
        print("delete.")
        text_s = text.split()
        for _ in range(count):
            text_s.pop(random.randint(0, len(text_s)-1))
            if len(text_s) == 1:
                break
        return " ".join(text_s)
    if option == "swap":
        print("swap.")
        text_s = text.split()
        index_list = list(np.arange(len(text_s)))
        for _ in range(count):
            num = random.sample(index_list, 2)
            text_s[num[0]], text_s[num[1]] = text_s[num[1]], text_s[num[0]]
        return " ".join(text_s)
    if option == "insert":
        print("insert.")
        rw = RandomWords()
        text_s = text.split()
        for _ in range(count):
            words = rw.get_random_word()
            text_s.insert(random.randint(0, len(text_s)), words)
        return " ".join(text_s)
    

# LLMDA
def get_prompt(sentence: str) -> str:
    '''Format the instruction as a prompt for LLM.'''
    prompt = f'''Please rewrite the original sentence with synonyms within 2 words. 
please output 5 different new sentences.
please simply modify the original sentence without changing more than 2 words.

Example:
Original sentence:
What will IBM use to analyze weather and make predictions?
New sentence:
What will IBM employ to examine weather and make forecasts?
What will IBM use to evaluate weather and make forecasts?
How will IBM examine weather and make predictions?
What methods will IBM employ to analyze weather and provide predictions?
What will IBM use to analyze weather and forecast future conditions?

Original sentence:
The war continued until 1988, when the Iraqi army defeated the Iranian forces inside Iraq and pushed the remaining Iranian troops back across the border.
New sentence:
The war persisted until 1988, when the Iraqi military defeated the Iranian forces inside Iraq and pushed the remaining Iranian soldiers back across the border.
The conflict continued until 1988, when the Iraqi army triumphed over the Iranian forces inside Iraq and pushed the remaining Iranian troops back across the border.
The war continued until 1988, when the Iraqi army defeated the Iranian forces inside Iraq and forced the remaining Iranian soldiers back across the border.
The war endured until 1988, when the Iraqi army defeated the Iranian forces inside Iraq and forced the remaining Iranian soldiers back across the border.
The conflict continued until 1988, when the Iraqi army defeated the Iranian forces inside Iraq and forced the remaining Iranian soldiers back across the border.

Original sentence:{sentence}
New sentence:
    '''
    return prompt

def get_llm_aug(sentence):
    client = OpenAI()
    print("LLM.")

    prompt = get_prompt(sentence)

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0.8,
        messages=[
            {"role": "system", "content": "You are a useful assistant. "},
            {"role": "user", "content": prompt}
        ]
    )

    ans = completion.choices[0].message.content

    return ans

# sent = 'Since then, some brewers have used champagne yeasts to increase the alcohol content of their beers.'
# sent = 'Each byte is able to represent 256 different numbers (28 = 256); either from 0 to 255 or −128 to +127.'
# sent = "Eleven days after Orsini's assassination attempt in France, Victoria's eldest daughter married Prince Frederick William of Prussia in London."
# sent = 'What was again revised in May of 2010?'
# sent = "Having regained the south-east John split his forces, sending William Longespée to retake the north side of London and East Anglia, whilst John himself headed north via Nottingham to attack the estates of the northern barons."
# print(sent)
# print(get_llm_aug(sent))


# produce augment data
if __name__ == '__main__':
    args = parse_argument()
    config = vars(args)

    root = config["dataset_root"]
    fulldataset_path = os.path.join(root, f"{config['dataset']}_FullDataset.pkl")
    with open(fulldataset_path, 'rb') as f:
        FullDataset = pickle.load(f)

    # FullDataset_new = {}
    # for i, (k,v) in enumerate(FullDataset.items()):
    #     if i == 100:
    #         break
    #     FullDataset_new[k] = v

    FullDataset_aug = {}
    for sent in tqdm(FullDataset.keys()):
        FullDataset_aug[sent] = {"insert": [], "delete": [], "replace": [], "swap": [], "llm": []}
        try:
            text_LLM_all = get_llm_aug(sent)
            text_LLM_all = text_LLM_all.split("\n")
            for text_LLM in text_LLM_all:
                FullDataset_aug[sent]["llm"].append(text_LLM)
        except:
            pass
        for _ in range(1):
            try:
                text_insert = EDA(sent, "insert", 1)
                FullDataset_aug[sent]["insert"].append(text_insert)
            except:
                pass
            try:
                text_delete = EDA(sent, "delete", 1)
                FullDataset_aug[sent]["delete"].append(text_delete)
            except:
                pass
            try:
                text_replace = EDA(sent, "replace", 1)
                FullDataset_aug[sent]["replace"].append(text_replace)
            except:
                pass
            try:
                text_swap = EDA(sent, "swap", 1)
                FullDataset_aug[sent]["swap"].append(text_swap)
            except:
                pass

    save_path = os.path.join(root, f"{config['dataset']}_FullDataset_aug.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(FullDataset_aug, f)
    