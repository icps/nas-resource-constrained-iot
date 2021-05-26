import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import pickle
import numpy as np
import random as rnd
from pathlib import Path
from shutil import copyfile

from grammar import Grammar
from evaluator import Evaluator
from individual import Individual
from evolutionary import Evolutionary

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

rnd.seed(321)
np.random.seed(321)

#   Level | Level for Humans | Level Description                  
#  -------|------------------|------------------------------------ 
#   0     | DEBUG            | [Default] Print all messages       
#   1     | INFO             | Filter out INFO messages           
#   2     | WARNING          | Filter out INFO & WARNING messages 
#   3     | ERROR            | Filter out all messages  


def save_pop(population, save_path, run, gen):
    json_dump = []
    for ind in population:
        json_dump.append({'id': ind.id,
                          'phenotype': ind.phenotype,
                          'fitness': ind.fitness,
                          'metrics': ind.metrics,
                          'trainable_parameters': ind.trainable_parameters,
                          'num_epochs': ind.num_epochs,
                          'time': ind.time,
                          'train_time': ind.train_time})
        
    pathfile = os.path.join(f'{save_path}', f'run_{run}', f'gen_{gen}.csv')

    with open(pathfile, 'w') as f_json:
        f_json.write(json.dumps(json_dump, indent = 4))
        

def pickle_evaluator(evaluator, save_path, run):
    
    pathfile = os.path.join(f'{save_path}', f'run_{run}', 'evaluator.pkl')

    with open(pathfile, 'wb') as handle:
        pickle.dump(evaluator, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

def pickle_population(population, parent, save_path, run):
    with open('%s/run_%d/population.pkl' % (save_path, run), 'wb') as handle_pop:
        pickle.dump(population, handle_pop, protocol = pickle.HIGHEST_PROTOCOL)

    with open('%s/run_%d/parent.pkl' % (save_path, run), 'wb') as handle_pop:
        pickle.dump(parent, handle_pop, protocol = pickle.HIGHEST_PROTOCOL)

    with open('%s/run_%d/random.pkl' % (save_path, run), 'wb') as handle_random:
        pickle.dump(rnd.getstate(), handle_random, protocol = pickle.HIGHEST_PROTOCOL)

    with open('%s/run_%d/numpy.pkl' % (save_path, run), 'wb') as handle_numpy:
        pickle.dump(np.random.get_state(), handle_numpy, protocol = pickle.HIGHEST_PROTOCOL)
        
        
network_structure = [["features", 1, 10], ["classification", 1, 10]]
macro_structure   = ["learning"]
output_rule       = "softmax"

modules           = []
levels_back       = {"features": 1, "classification": 1}
reuse_layer       = 0.1
init_max          = {"features": [2,3,4], "classification": [1]}

path    = "grammar.txt"
grammar = Grammar(path)

default_train_time = 500
datagen            = None 
datagen_test       = None
dataset            = "cifar10"
save_path          = 'experiments'

num_generations    = 50

n_population       = 5

cnn_eval           = Evaluator(dataset)

run                = 0

last_gen           = 0
total_epochs       = 0
best_fitness       = None


mutation_parameters = {
    "add_layer"        : 0.2,
    "reuse_layer"      : 0.2,
    "remove_layer"     : 0.3,
    "add_connection"   : 0.2,
    "remove_connection": 0.2,
    "dsge_layer"       : 0.2,
    "macro_layer"      : 0.3,
    "train_longer"     : 0#0.2
}


for gen in range(last_gen, num_generations):
    
    if gen == 0:
        print('[%d] Creating the initial population' % (run))
        print('[%d] Performing generation: %d' % (run, gen))
        
        ### create first population
        population = []
        for _id_ in range(n_population):
            new_indv = Individual(network_structure, macro_structure, output_rule, _id_)
            new_indv = new_indv.create_individual(grammar, levels_back, reuse_layer, init_max)

            population.append(new_indv)
        
        ### evaluate first population
        population_fits = []
        for index, indv in enumerate(population):

            indv.current_time = 0
            indv.num_epochs   = 0
            indv.train_time   = default_train_time
            indv.id           = index
            
            pathfile      = os.path.join(save_path, f'run_{run}')
            where_to_save = os.path.join(pathfile, f'best_{gen}_{index}.hdf5')

            Path(os.path.join(pathfile, "")).mkdir(parents = True, exist_ok = True)

            indv_evaluation = indv.evaluate(grammar, cnn_eval, datagen, datagen_test, where_to_save)

            population_fits.append(indv_evaluation)
            
    else:
        print('[%d] Performing generation: %d' % (run, gen))
                
        population = []
        for _ in range(n_population):
            children = Evolutionary().mutation(parent, grammar, mutation_parameters, default_train_time)
            population.append(children)
        
        population.append(parent)
        
        # set elite variables to re-evaluation
        population[0].current_time = 0
        population[0].num_epochs   = 0
        parent_id                  = parent.id
        
        population_fits = []
        for index, indv in enumerate(population):
            indv.id         = index
            
            pathfile        = os.path.join(save_path, f'run_{run}')
            where_to_save   = os.path.join(pathfile, f'best_{gen}_{index}.hdf5')

            Path(os.path.join(pathfile, "")).mkdir(parents = True, exist_ok = True)
            
            parents_weights = os.path.join(pathfile, f'best_{gen-1}_{parent_id}.hdf5')
            indv_evaluation = indv.evaluate(grammar, cnn_eval, datagen, datagen_test, where_to_save, parents_weights)
            
            population_fits.append(indv_evaluation)


    parent = Evolutionary().select_fittest(population)

    if gen > 1:
        for x in range(len(population)):
            filename = os.path.join(f'{save_path}', f'run_{run}', f'best_{gen-2}_{x}')

            if os.path.isfile(filename + '.hdf5'):
                os.remove(filename + '.hdf5')
                
            if os.path.isfile(filename + '.h5'):
                os.remove(filename + '.h5')

    if best_fitness is None or parent.fitness > best_fitness:
        best_fitness    = parent.fitness
        best_individual = parent

        pathfile_hdf5   = os.path.join(f'{save_path}', f'run_{run}', f'best_{gen}_{parent.id}.hdf5')
        pathfile_h5     = os.path.join(f'{save_path}', f'run_{run}', f'best_{gen}_{parent.id}.h5')

        copyfile(pathfile_hdf5, os.path.join(f'{save_path}', f'run_{run}','best.hdf5'))
        copyfile(pathfile_h5, os.path.join(f'{save_path}', f'run_{run}','best.h5'))

        with open(os.path.join(f'{save_path}', f'run_{run}', 'best_parent.pkl'), 'wb') as handle:
                 pickle.dump(parent, handle, protocol = pickle.HIGHEST_PROTOCOL)

    print('[%d] Best fitness of generation %d: %f' % (run, gen, max(population_fits)))
    print('[%d] Best overall fitness: %f' % (run, best_fitness))


    #save population
    save_pop(population, save_path, run, gen)
    pickle_population(population, parent, save_path, run)

    total_epochs += sum([ind.num_epochs for ind in population])
