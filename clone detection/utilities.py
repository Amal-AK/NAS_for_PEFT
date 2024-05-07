
import torch.nn as nn
from torch.utils.data.distributed import Dataset 
import torch
import json
import random
import logging
import os
import numpy as np

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger("name")





class InputFeatures ( object) : 

    def __init__(self,
                   code_ids,
                   label): 
        
        self.code_ids = code_ids
        self.label = label





def convert_examples_to_features(js,tokenizer,args):

    ids_args =  ((js['code1'], js['code2']))

    result = tokenizer(*ids_args, padding="max_length", max_length=args.code_length-2, truncation='longest_first')
 

    return InputFeatures( result['input_ids'] , js['label'])





class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None,is_test=None,lang=None):
        self.examples = []
       
        logger.info("Preparing the Dataset...\n")
        url_to_code = {}
        with open("./dataset/data.jsonl") as f:
            for line in f:
                line = line.strip()
                js = json.loads(line)
                code = ' '.join(js['func'].split())
                url_to_code[js['idx']] = code

        data = []
    
        with open(file_path) as f:
            for line in f:
                js = {}
                line = line.strip()
                url1, url2, label = line.split('\t')
                if url1 not in url_to_code or url2 not in url_to_code:
                    continue

                js['code1'] = url_to_code[url1]
                js['code2']= url_to_code[url2]
                js['label']= int(label)
                data.append(js)

        size =  int (args.train_data_rate * len(data))
        for js in data[:size]:
            
            self.examples.append(convert_examples_to_features(js,tokenizer,args))  


     



    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):   
        return (torch.tensor(self.examples[i].code_ids),torch.tensor(self.examples[i].label))








class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False




def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True




