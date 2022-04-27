import json
import numpy as np
from typing import List, Tuple

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

TARGET_TO_HS_TOK = {'DISABLED': '<|DISABLED_HS|>',
                    'JEWS': '<|JEWS_HS|>',
                    'LGBT+': '<|LGBT+_HS|>',
                    'MIGRANTS': '<|MIGRANTS_HS|>',
                    'MUSLIMS': '<|MUSLIMS_HS|>',
                    'POC': '<|POC_HS|>',
                    'WOMEN': '<|WOMEN_HS|>',
                    'other': '<|other_HS|>'}
TARGET_TO_CS_TOK = {'DISABLED': '<|DISABLED_CS|>',
                    'JEWS': '<|JEWS_CS|>',
                    'LGBT+': '<|LGBT+_CS|>',
                    'MIGRANTS': '<|MIGRANTS_CS|>',
                    'MUSLIMS': '<|MUSLIMS_CS|>',
                    'POC': '<|POC_CS|>',
                    'WOMEN': '<|WOMEN_CS|>',
                    'other': '<|other_CS|>'}

def process_train_data(filepath='../CounterSpeechGenerator/Data/Multitarget-CONAN.json'):
    with open(filepath, 'r') as f:
        multitarget_data = json.load(f)    
    
    hs_cs_pairs = []
    targets = []
    for example in multitarget_data:
        hate_speech = multitarget_data[example]['HATE_SPEECH']
        counter_speech = multitarget_data[example]['COUNTER_NARRATIVE']
        target = multitarget_data[example]['TARGET']

        hs_cs_pairs.append((hate_speech, counter_speech, target))
        targets.append(target)
    
    return hs_cs_pairs, targets

def generate_counterspeech(texts: List[str], labels: List[str], model_name='../CounterSpeechGenerator/Models/gpt2_medium') -> List[str]:
    
    # load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name + '_tokenizer')
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # don't generate special tokens
    bad_words = list(TARGET_TO_HS_TOK.values()) + list(TARGET_TO_CS_TOK.values())
    bad_word_ids = [tokenizer(bad_word).input_ids[0] for bad_word in bad_words]

    tokenized_prompts = []
    responses = []

    for text, label in zip(texts, labels):
        hs_tok = TARGET_TO_HS_TOK[label]
        cs_tok = TARGET_TO_CS_TOK[label]

        prompt = hs_tok + text + cs_tok

        tokenized_prompt = tokenizer(prompt, return_tensors="pt").input_ids #.cuda()
        tokenized_prompts.append(tokenized_prompts)
        
    max_len = max([len(p) for p in tokenized_prompts])
    
    for prompt in tokenized_prompts:
        # take best response
        output = model.generate(tokenized_prompt, do_sample=True, top_k=30, bad_word_ids=bad_word_ids,
                                max_length=max_len, top_p=0.95, temperature=1.9, num_return_sequences=1)[0] 
        decoded_output = tokenizer.decode(output, skip_special_tokens=True)
        
        _, response = decoded_output.split(cs_tok)
        responses.append(response)
    
    return responses
