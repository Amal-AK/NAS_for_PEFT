
import sys
sys.path.append('.')
from OpenDelta.opendelta import LoraModel
import argparse
import logging
import os
import torch
from model import Model
from adapter import train , test , evaluate
import transformers
from torch.nn.functional import binary_cross_entropy , binary_cross_entropy_with_logits
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup, RobertaConfig, RobertaTokenizer , RobertaForSequenceClassification)
from torch.nn.parallel import DistributedDataParallel as DDP
from utilities import *
from optimization import *
from sklearn.metrics import recall_score, precision_score, f1_score
transformers.utils.logging.set_verbosity_error()
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("name")





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
        delta_model = LoraModel(backbone_model=model, lora_r=16)
        delta_model.freeze_module(exclude=["deltas", "classifier" ])
        delta_model.log(delta_ratio=True, trainable_ratio=True, visualization=True)

       
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