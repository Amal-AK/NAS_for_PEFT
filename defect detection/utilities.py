
from torch.utils.data.distributed import Dataset 
import torch
import json
import random
import logging
import os
import numpy as np

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger("name")




def count_trainable_parameters (model): 
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        total_params += params
    print(f"===> Total Trainable Params: {total_params}")
    return total_params



class InputFeatures ( object) : 

    def __init__(self,
                   code_tokens,
                   code_ids,
                   label): 
        self.code_tokens = code_tokens
        self.code_ids =  code_ids
        self.label = label





def convert_examples_to_features(js,tokenizer,args):
  
    code=''.join(js['code_tokens'])
    code_tokens=tokenizer.tokenize(code)[:args.code_length-2]
    code_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    code_ids =  tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.code_length - len(code_ids)
    code_ids+=[tokenizer.pad_token_id]*padding_length
    return InputFeatures(code_tokens,code_ids,js['target'])






class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None,is_test=None,lang=None):
        self.examples = []
        self.len_list = []
        
        data=[]
            
        with open(file_path) as f:
            for line in f:
                line = json.loads(line.strip())
                js = {}
                code = ' '.join(line['func'].split())
                label = int(line['target'])
                js['code_tokens'] = code
                js['target'] = label
                data.append(js)

        size =  int (args.train_data_rate * len(data))
        for js in data[:size]:
            self.examples.append(convert_examples_to_features(js,tokenizer,args))
            self.len_list.append(len(data))
        
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


