
import collections 
from copy import deepcopy
import random
import numpy as np
from model import Model
from torch.nn.functional import binary_cross_entropy , binary_cross_entropy_with_logits
from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup, RobertaTokenizer , RobertaForSequenceClassification)
from OpenDelta.opendelta import AdapterModel
from transformers import RobertaTokenizer
import numpy as np 
import random 
import traceback
from adapter import train
from utilities import *





INSERT_LOCATIONS = ['attention', 'intermediate','output' , "attention.self"]

ACTIVATIONS = ['gelu_new', 'relu' ]

MAX_FNN_SIZE = 3

FNN_SIZES = [16 , 24 , 32 , 64 , 128 ]

FAIL_COMPILE = 0

CROSSOVER_RATE = 0.9




def dict_to_tuple (dic) : 

    if isinstance(dic , dict) : 
        for n , p in dic.items() : 
            if isinstance(p, list) :
                dic[n] = tuple(p)
    return tuple(dic)





def binary_list(size ) : 

    zero_count = random.randint(0, size-2)
    one_count = size - zero_count
    my_list = [0]*zero_count + [1]*one_count
    random.shuffle(my_list)

    return my_list



def random_fnn () : 
    fnn_deepth =  random.randint(1,MAX_FNN_SIZE)
    return random.choices(FNN_SIZES , k=fnn_deepth)
 

def random_insert_modules () : 
    nb_locations = random.randint(1,len(INSERT_LOCATIONS)-1)
    return random.sample(INSERT_LOCATIONS , k=nb_locations)


def random_configuration () : 

    insert_modules = random_insert_modules()
    fnn_sizes =  random_fnn()
    configuration = {
                    "insert_modules" : list(insert_modules),
                    "bottleneck_dim" : list(fnn_sizes),
            }
    return configuration



def random_adapter_parameters (config) : 
    ''''''

    adapter_layers = binary_list(config.num_hidden_layers)
    parameters = list()
    activation_fct =  random.choice(ACTIVATIONS)
    for layer_idx , insert_decision in enumerate(adapter_layers) : 
        if insert_decision==1 : 
            configuration = random_configuration()
            configuration["non_linearity"] =  activation_fct
            parameters.append(configuration)
        else : 
            parameters.append(0)

    return parameters 





def get_delta_model ( model, adapter_parameters:dict )  : 

    delta_model = None
    for layer_id, adapter_param in enumerate(adapter_parameters) :
        if adapter_param != 0 :  
            x =  [ ("layer." 
                    + str(layer_id)
                    + '.' 
                    + str(adapter_param['insert_modules'][idx]).strip()) for idx in range(len(adapter_param['insert_modules'] )) ]
            
            delta_model = AdapterModel(backbone_model=model,
                                    modified_modules= x,
                                    bottleneck_dim=adapter_param['bottleneck_dim'] ,
                                    non_linearity=adapter_param['non_linearity'])
           
    
    delta_model.freeze_module(exclude=["deltas", "classifier" ])
    delta_model.log(delta_ratio=True, trainable_ratio=True, visualization=False)

    return model




def regularized_evolution(args, config,  cycles=20 , population_size=20, sample_size=10, init_population=None, init_history=None):

    best_of_all = [[] , 0]
    # Initiate the population as a FILO queue
    population = collections.deque()
    history = []  
    if args.population_size : 
        population_size =  args.population_size
    if args.cycles : 
        cycles =  args.cycles
    if args.sample_size : 
        sample_size =  args.sample_size
    
    logger.info("Initialization of the population, may take a while ...")
    hash_pop = []
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    if init_population is None:
        while len(population) < population_size:
            model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path,config=config) 
            delta_parameters = random_adapter_parameters(config)
            print("\n ==> Delta parameters :  " , delta_parameters , "\n")
            model = get_delta_model(model , delta_parameters)
            adapter_params = tuple([dict_to_tuple(d)  if d!=0 else d for d in delta_parameters])
            delta_model = Model( model)
            
            if args.n_gpu > 1:
                delta_model = torch.nn.DataParallel(delta_model)
            delta_model.to(args.device)
            
            if hash(adapter_params) in hash_pop:
                continue

            hash_pop.append(hash(adapter_params))
            metric = fitness(delta_model,  args , tokenizer)
            print("\n ==> Fitness ( F1 score ) :  " , metric, "\n")
        
            population.append((delta_parameters, metric))
            history.append((delta_parameters, metric))
    else:
        population = init_population
        for individual in population:
            hash_pop.append(hash(tuple(individual[0])))
    if init_history is not None:
        history = init_history
    
    
    # Carry out evolution in cycles
    
    while len(history) - population_size < cycles:
        
            sample = []
        
            while len(sample) < sample_size:
                candidate = random.choice(list(population))
                sample.append(candidate)

            random_number =  random.random()
            if random_number < CROSSOVER_RATE:
                # do crossover 
                sample.sort(key=lambda i: i[1])
                parent1 , parent2 = sample[-1][0] , sample[-2][0]
                child =  crossover(parent1, parent2 , config)
                
            else : 
                # do mutation 

                # The parent is the best model in the sample
                parent = max(sample, key=lambda i: i[1])
                # Create the child model by mutating its architecture
                child = mutate(parent[0] , config)

            
            model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path,config=config) 
            child_model =   get_delta_model(model , child)
            child_model = Model( child_model )
            
            if args.n_gpu > 1:
                child_model= torch.nn.DataParallel(child_model)
            child_model.to(args.device)
            child_metric = fitness(child_model,  args , tokenizer)
         
            if child_metric == FAIL_COMPILE:
                  continue
            # If not failed, then add to the population
            population.append((child, child_metric))
            # and add to the history
            history.append((child, child_metric))
            # Remove the oldest model.
            population.popleft()
            # Best model in the current population
            best_candidate = max(list(population), key=lambda i: i[1])
            if best_candidate[1] > best_of_all[1]:
              # Best model during whole calculation

              best_of_all = best_candidate
            logger.info("Cycle {0}, the best candidate in pop has score {1}, and best in the run {2}"\
                  .format(len(history) - population_size, best_candidate[1], best_of_all[1]))
            population_mean = np.mean([x for _, x in list(population)])
            population_std = np.std([x for _, x in list(population)])
            logger.info("Mean {0} and standard deviation {1} of score in the population".format(population_mean, 
                                                                                          population_std))
    

    with open(os.path.join(args.output_dir, args.optimization_history_file ),'w') as f:
        for candidate in history:
            f.write(str(candidate)+'\n')
    print("Best of all candidates :" + str(best_of_all))
    return history, population, best_of_all






def mutate(adapter_structure , config) : 
    parent_hash = hash(tuple ([ dict_to_tuple(d)  if d!=0 else d for d in adapter_structure ]))
    model_len = config.num_hidden_layers-1
    new_structure = deepcopy(adapter_structure)
   
    while True:
        random_or_swap = random.randint(0, 1) # swap if 0 , random otherwise 
        
        if random_or_swap== 1 :
            layer_index = random.randint(0, model_len)
            new_element = random_configuration()
            new_element["non_linearity"] = [ d['non_linearity'] for d in adapter_structure if d!=0][0]
            new_structure[layer_index] = new_element
        else :
            while True : 
                first_layer = random.randint(0 ,model_len )
                second_layer = random.randint(0 , model_len )
                if first_layer == second_layer :
                    continue 
                else :
                    new_structure[second_layer] = adapter_structure[first_layer]
                    new_structure[first_layer] = adapter_structure[second_layer]
                    

                    break 
        
        if hash(tuple ([ dict_to_tuple(d) if d!=0 else d for d in new_structure ])) != parent_hash:
            break
        else:
            continue
    return new_structure







def fitness (model, args , tokenizer) : 
        fit = 0
        try : 
            results = train(args , model , tokenizer)
            acc = max (results['eval_f1'])
            fit = np.round(acc , 3) 
    
        except Exception as e:
            print(e)
            print(traceback.format_exc())
        return  fit




def crossover (parent1 , parent2 , config) : 
    #one point cross over 
    crossover_point = random.randint(1 , config.num_hidden_layers-2)
    child = []
    child[:crossover_point] = parent1[:crossover_point]
    child[crossover_point:] = parent2[crossover_point:]
    
    return child