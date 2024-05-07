
import sys
sys.path.append('.')
from OpenDelta.opendelta import AdapterModel
import argparse
import logging
import os
import torch
import numpy as np
from model import Model
from tqdm import tqdm
import torch.nn as nn
import transformers
from torch.nn.functional import binary_cross_entropy , binary_cross_entropy_with_logits
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler 
from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup, RobertaConfig, RobertaTokenizer , RobertaForSequenceClassification)
import torch.distributed as dis
from torch.nn.parallel import DistributedDataParallel as DDP
from utilities import *
from optimization import *
from sklearn.metrics import recall_score, precision_score, f1_score

transformers.utils.logging.set_verbosity_error()
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("name")





def train(args, model,  tokenizer ):
    """ Train the model """

    train_dataset=TextDataset(tokenizer, args, args.train_data_file)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=2 )
    
    eval_dataset = TextDataset(tokenizer, args,args.eval_data_file)
    if not args.do_optimization :
        test_dataset = TextDataset(tokenizer, args,args.test_data_file)

    optimizer =torch.optim.Adam(model.parameters(), lr=args.learning_rate )
    max_steps = len(train_dataloader) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max_steps*0.1, num_training_steps=max_steps)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)
    best_acc=  - np.inf
    model.zero_grad()
    loss_fn = nn.BCELoss()
    early_stopper = EarlyStopper(patience=3, min_delta=0.03)
    results =  {
        'train_loss' : [],
        'train_acc' : [],
        'eval_loss' : [],
        'eval_f1' : [] , 
        'eval_recall': [] ,
        'eval_precision': [] ,
        'test_f1' : [] , 
    }

    for idx in range(args.num_train_epochs): 

      
        LOSSes, ACCs =  [], []
        bar = tqdm(train_dataloader,total=len(train_dataloader))
        for step, batch in enumerate(bar):

            code_inputs = batch[0].to(args.device)  
            labels =  batch[1].to(args.device)  
            
            model.train()
            logits = model(code_inputs=code_inputs).to(args.device)
            labels= labels.unsqueeze(1).float().to(args.device)

            loss = loss_fn(logits,labels)
            LOSSes.append(loss.item())
  
            loss.backward()

            acc = (logits.cpu().detach().numpy().round() == labels.cpu().detach().numpy()).mean()
            ACCs.append(acc)
            bar.set_description("epoch {} loss {}  acc {}".format(idx, round(np.mean(LOSSes),3), round(np.mean(ACCs),3) ) )

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step() 

        
        results['train_loss'].append(round(np.mean(LOSSes),3))
        results['train_acc'] .append(round(np.mean(ACCs),3))

        eval_results = evaluate(args, model, tokenizer , eval_dataset)
        results['eval_loss'].append(round( eval_results['eval_loss'],3))
        results['eval_f1'].append(round( eval_results['f1'],3))
        results['eval_recall'].append(round( eval_results['recall'],3))
        results['eval_precision'].append(round( eval_results['precision'],3))

        for key, value in eval_results.items():
            logger.info("  %s = %s", key, round(value,4))  


        if eval_results['f1']>best_acc:
            best_acc=eval_results['f1']
            logger.info("\n "+"*"*30)  
            logger.info("  Best F1 score :%s",round(best_acc,4))
            logger.info("  "+"*"*30)   
        
            
        if not args.do_optimization : 
            test_result =   test(args, model, tokenizer , test_dataset)  
            results['test_f1'].append(test_result) 
    
        
        if early_stopper.early_stop(round(eval_results['eval_loss'],2)):             
            break
    
    return results  





def evaluate( args , model , tokenizer , eval_dataset) : 

    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)


    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset , sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=2,pin_memory=True)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[] 
    labels=[]
    loss_fn = nn.BCELoss()
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)        
        label=batch[1].to(args.device) 
        with torch.no_grad():
            logit = model(code_inputs = inputs)
            label= label.unsqueeze(1).float()
            lm_loss = loss_fn(logit,label)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
    logits=np.concatenate(logits,0)
    labels=np.concatenate(labels,0)
    preds = logits.round()
    eval_acc=np.mean(labels==preds)
    recall = recall_score(labels , preds)
    precision = precision_score(labels , preds)
    f1 = f1_score(labels , preds)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)
            
    result = {
        "eval_loss": float(perplexity),
        "eval_acc":round(eval_acc,4),
        "recall" : round (recall, 4) , 
        "precision" :  round(precision, 4),
        "f1" :  round(f1 , 4),

    }
    return result





def test(args, model, tokenizer, test_dataset):

    eval_sampler = SequentialSampler(test_dataset)
    eval_dataloader = DataLoader(test_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    logger.info("\n***** Running Test *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[]   
    labels=[]
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)        
        label=batch[1].to(args.device) 
        with torch.no_grad():
            logit = model(code_inputs = inputs)
            label= label.unsqueeze(1).float()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())

    logits=np.concatenate(logits,0)
    labels=np.concatenate(labels,0)
    preds= logits.round()
    test_acc = np.mean(labels==preds)
    recall = recall_score(labels , preds)
    precision = precision_score(labels , preds)
    f1 = f1_score(labels , preds)
    logger.info("  "+"*"*30)  
    logger.info("  Test accuracy :%s",round(test_acc,4))
    logger.info("  Test recall :%s",round(recall,4))
    logger.info("  Test precision :%s",round(precision,4))
    logger.info("  Test f1 :%s",round(f1,4))
    logger.info("  "+"*"*30)    

    return round(f1,4)
 




def main():



    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data_file", default="./dataset/train.txt", type=str, 
                        help="The input training data file (a json file).")
    parser.add_argument("--output_dir", default='./', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--num_classes", default=1, type=int,
                        help="The number of classes for the classification model")
    parser.add_argument("--eval_data_file", default="./dataset/valid.txt", type=str,
                        help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    parser.add_argument("--test_data_file", default="./dataset/test.txt", type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--codebase_file", default=None, type=str,
                        help="An optional input test data file to codebase (a jsonl file).")  
    
    parser.add_argument("--model_name_or_path", default='microsoft/codebert-base', type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")    
    parser.add_argument("--code_length", default=512, type=int,
                        help="Optional Code input sequence length after tokenization.") 
    parser.add_argument("--do_optimization", action='store_true',
                        help="Whether to run adapter optimization")  
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.") 
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--train_data_rate", default=0.1, type= float,
                        help="Data size for train")
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=15, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--local_rank', default=-1 ,type=int,
                        help="random seed for initialization")
    parser.add_argument('--population_size', default=20 ,type=int,
                        help="population size on the evolutionary optimization algorithm")
    parser.add_argument('--sample_size', default=10 ,type=int,
                        help="sample size on the evolutionary optimization algorithm")
    
    parser.add_argument('--cycles', default=20 ,type=int,
                        help="number of cycles on the evolutionary optimization algorithm")
    
    parser.add_argument('--optimization_history_file', default=None ,type=str,
                        help="saving the history of optimization")
    
    
    args = parser.parse_args()
    set_seed(seed=args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = [ 1 if torch.cuda.is_available() else 0][0]
    args.device = device
    logger.info("device: %s, n_gpu: %s",device, args.n_gpu)
    config = RobertaConfig.from_pretrained(args.model_name_or_path , num_labels = args.num_classes)


    if args.do_optimization: 
        history, population, best_of_all =  regularized_evolution(args, config)

    else : 


        tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
        model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path,config=config)  
        
        #delta_model = AdapterModel(backbone_model=model,modified_modules=['attention','[r](\d)+\.output'],bottleneck_dim=[32] )  
        #delta_model.freeze_module(exclude=["deltas", "classifier" ])
        #delta_model.log(delta_ratio=True, trainable_ratio=True, visualization=True)

        delta_parameters = [{'insert_modules': ('attention.self',), 'bottleneck_dim': (64,), 'non_linearity': 'gelu_new'},
                            {'insert_modules': ('intermediate', 'output', 'attention.self'), 'bottleneck_dim': (128, 64), 'non_linearity': 'gelu_new'},
                            {'insert_modules': ('intermediate',), 'bottleneck_dim': (16, 32), 'non_linearity': 'gelu_new'},
                            {'insert_modules': ('attention.self', 'intermediate'), 'bottleneck_dim': (64, 32), 'non_linearity': 'gelu_new'},
                            {'insert_modules': ('attention', 'attention.self', 'output'), 'bottleneck_dim': (128,), 'non_linearity': 'gelu_new'},
                            {'insert_modules': ('attention', 'attention.self', 'intermediate'), 'bottleneck_dim': (128,), 'non_linearity': 'relu'},
                            0, 0, 0, 0, 0, 0]

        model = get_delta_model(model , delta_parameters)

        model = Model( model)

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        model.to(args.device)

        
        if args.do_train:

            results = train(args , model ,tokenizer)
            print("train results", results)


        if args.do_eval:
                checkpoint_prefix = 'checkpoint-best-acc/model.bin'
                output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
                model.load_state_dict(torch.load(output_dir) , strict=False)      
                result=evaluate(args, model, tokenizer)
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(round(result[key],4)))
                    

        
        if args.do_test:
                checkpoint_prefix = 'checkpoint-best-acc/model.bin'
                output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
                model.load_state_dict(torch.load(output_dir),  strict=False)                  
                test(args, model, tokenizer)


if __name__ == "__main__":
    main()