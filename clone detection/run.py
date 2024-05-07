
import argparse
import logging
import os
import pickle
import torch
import numpy as np
from model import Model
from tqdm import tqdm
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy , binary_cross_entropy_with_logits
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup, RobertaConfig, RobertaModel, RobertaTokenizer , RobertaForSequenceClassification)
import torch.distributed as dis
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler , Sampler, Dataset 
from utilities import *
from optimization import *
import traceback
from torch.nn import CrossEntropyLoss, MSELoss

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("name")
logging.getLogger("transformers.modeling_utils").setLevel(logging.INFO)




def train(args, model,  tokenizer ):
    """ Train the model """


    #get training dataset
    train_dataset=TextDataset(tokenizer, args, args.train_data_file)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=2 )
    eval_dataset = TextDataset(tokenizer, args,args.eval_data_file)

    #get optimizer and scheduler
    optimizer =torch.optim.Adam(model.parameters(), lr=args.learning_rate )
    max_steps = len(train_dataloader) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max_steps*0.1, num_training_steps=max_steps)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)
    best_acc=  - np.inf
    model.zero_grad()
    early_stopper = EarlyStopper(patience=3, min_delta=0.05)
    results =  {
        'train_loss' : [],
        'train_acc' : [],
        'eval_loss' : [],
        'eval_acc': [], 
        'test_acc' : [],
    }

    for idx in range(args.num_train_epochs): 

      
        LOSSes, ACCs =  [], []
        bar = tqdm(train_dataloader,total=len(train_dataloader))
        for step, batch in enumerate(bar):

            #get inputs
            code_inputs1 = batch[0].to(args.device)  
            code_inputs2 = batch[1].to(args.device)  
            labels =  batch[2].to(args.device)  
            labels= labels.unsqueeze(1).float().to(args.device)
            #get predictions
            model.train()
            code_vec1 = model(code_inputs=code_inputs1).to(args.device)
            code_vec2 = model(code_inputs=code_inputs2).to(args.device)
        
            #labels= labels.unsqueeze(1).float().to(args.device)

            #calculate loss
            
            scores=torch.einsum("ab,cb->ac",code_vec1, code_vec2)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(scores, torch.arange(code_inputs1.size(0), device=scores.device))
            LOSSes.append(loss.item())
            loss.backward()
            predictions = (scores.argmax(dim=1) == labels.squeeze(1).long()).float()
            train_acc = predictions.mean().item()* 100
            
            ACCs.append(train_acc )

            bar.set_description("epoch {} loss {}  acc {}".format(idx, round(np.mean(LOSSes),3), round(np.mean(ACCs),3) ) )

            #optimizer step
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step() 

            '''

       
        #save train results 
        results['train_loss'].append(round(np.mean(LOSSes),3))
        results['train_acc'] .append(round(np.mean(ACCs),3))

         #run validation step
        eval_results = evaluate(args, model, tokenizer ,  eval_dataset)
        results['eval_loss'].append(round( eval_results['eval_loss'],3))
        results['eval_acc'].append(round(eval_results['eval_acc'],3))
        for key, value in eval_results.items():
            logger.info("  %s = %s", key, round(value,4))  

         
    
        # Save model checkpoint
        if eval_results['eval_acc']>best_acc:
            best_acc=eval_results['eval_acc']
            logger.info("\n "+"*"*30)  
            logger.info("  Best validation accuracy :%s",round(best_acc,4))
            logger.info("  "+"*"*30)   
    
        test_result =   test(args, model, tokenizer)  
        results['test_acc'].append(test_result)
    

 

        
        if early_stopper.early_stop(round(eval_results['eval_loss'],2)):             
            break
    '''
    return results  





def evaluate( args , model , tokenizer ,  eval_dataset) : 

    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)


    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset , sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=2,pin_memory=True)

    # Eval!
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
    eval_acc=np.mean(labels==logits.round())
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)
            
    result = {
        "eval_loss": float(perplexity),
        "eval_acc":round(eval_acc,4),
    }
    return result





def test(args, model, tokenizer):
    # Note that DistributedSampler samples randomly
    eval_dataset = TextDataset(tokenizer, args,args.test_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("\n***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
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
    logger.info("  "+"*"*30)  
    logger.info("  Test accuracy :%s",round(test_acc,4))
    logger.info("  "+"*"*30)    
    return round(test_acc,4)
 








def main():



    parser = argparse.ArgumentParser()

    ## Required parameters
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
    parser.add_argument("--train_data_rate", default=1.0, type= float,
                        help="Data size for train")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=6, type=int,
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

        model = RobertaModel.from_pretrained(args.model_name_or_path,config=config)  

        delta_model = AdapterModel(backbone_model=model,modified_modules=['attention','[r](\d)+\.output'],bottleneck_dim=[32] )  
        delta_model.freeze_module(exclude=["deltas"])
        delta_model.log(delta_ratio=True, trainable_ratio=True, visualization=True)
       
        model = Model( model)
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