import tensorflow
import numpy as np
import random as rnd
from time import time
from multiprocessing import Pool

from evaluator import Evaluator as cnn_eval


def evaluate(args):
    cnn_eval, phenotype, load_prev_weights, weights_save_path,\
    parent_weights_path, train_time, num_epochs, datagen, datagen_test = args

    try:
        return cnn_eval.evaluate(phenotype, load_prev_weights, weights_save_path, parent_weights_path,\
                                 train_time, num_epochs, datagen, datagen_test)
    
    except tensorflow.errors.ResourceExhaustedError as e:
        return None



class Module:
    
    def __init__(self, module, min_expansions, max_expansions, levels_back):

        self.module         = module
        self.min_expansions = min_expansions
        self.max_expansions = max_expansions
        self.levels_back    = levels_back
        
        self.layers         = []
        self.connections    = {}
        
    
    def create_individual(self, grammar, reuse, init_max):
        
        num_expansions = rnd.choice(init_max[self.module])

        # Initialise layers
        for index in range(num_expansions):
            
            if index > 0 and rnd.random() <= reuse:
                r_idx = rnd.randint(0, index - 1)
                self.layers.append(self.layers[r_idx])
            
            else:
                self.layers.append(grammar.create_individual(self.module))

        # Initialise connections
        self.connections = {}
        for layer_idx in range(num_expansions):
            
            if layer_idx == 0:
                # the -1 layer is the input
                self.connections[layer_idx] = [-1,]
            
            else:
                connection_possibilities = list(range(max(0, layer_idx - self.levels_back), layer_idx - 1))
                
                if len(connection_possibilities) < self.levels_back - 1:
                    connection_possibilities.append(-1)

                sample_size = rnd.randint(0, len(connection_possibilities))
                
                self.connections[layer_idx] = [layer_idx - 1] 
                
                if sample_size > 0:
                    self.connections[layer_idx] += rnd.sample(connection_possibilities, sample_size)   
                    
                    
class Individual:
    
    def __init__(self, network_structure, macro_rules, output_rule, ind_id):
        
        self.network_structure    = network_structure
        self.output_rule          = output_rule
        self.macro_rules          = macro_rules
        self.id                   = ind_id
        
        self.modules              = []
        self.macro                = []
        self.output               = None
        self.phenotype            = None
        self.fitness              = None
        self.metrics              = None
        self.num_epochs           = None
        self.trainable_parameters = None
        self.time                 = None
        self.current_time         = 0
        self.train_time           = 0
        
                
    def create_individual(self, grammar, levels_back, reuse, init_max):
         
        for non_terminal, min_expansions, max_expansions in self.network_structure:
            new_module = Module(non_terminal, min_expansions, max_expansions, levels_back[non_terminal])
            new_module.create_individual(grammar, reuse, init_max)

            self.modules.append(new_module)

        # Initialise output
        self.output = grammar.create_individual(self.output_rule)

        # Initialise the macro structure: learning, data augmentation, etc.
        for rule in self.macro_rules:
            self.macro.append(grammar.create_individual(rule))

        return self
    
    
    def decode(self, grammar):

        phenotype     = ''
        offset        = 0
        layer_counter = 0
        
        for module in self.modules:
            offset = layer_counter
            
            for layer_idx, layer_genotype in enumerate(module.layers):
                layer_counter  += 1
                
                decoded_grammar = grammar.decode(module.module, layer_genotype)
                connections     = map(str, np.array(module.connections[layer_idx]) + offset)
                
                phenotype      += ' ' + decoded_grammar + ' input:' + ",".join(connections)

        decoded_grammar = grammar.decode(self.output_rule, self.output)
        phenotype      += ' ' + decoded_grammar + ' input:' + str(layer_counter - 1)

        for rule_idx, macro_rule in enumerate(self.macro_rules):
            phenotype += ' ' + grammar.decode(macro_rule, self.macro[rule_idx])

        self.phenotype = phenotype.rstrip().lstrip()
        
        return self.phenotype
    
    
    def evaluate(self, grammar, cnn_eval, datagen, datagen_test, weights_save_path, parent_weights_path = ''):
        
        phenotype = self.decode(grammar)
        start     = time()
        pool      = Pool(processes = 1)

        load_prev_weights = True
        if self.current_time == 0:
            load_prev_weights = False

        train_time = self.train_time - self.current_time
                
        params     = [(cnn_eval, phenotype, load_prev_weights, weights_save_path, parent_weights_path,\
                       train_time, self.num_epochs, datagen, datagen_test)]
        result     = pool.apply_async(evaluate, params)

        pool.close()
        pool.join()
        
        metrics = result.get()        
            
        if metrics is not None:
            self.metrics              = metrics
            self.num_epochs          += self.metrics['epochs']
            self.trainable_parameters = self.metrics['trainable_parameters']
            self.current_time        += (self.train_time - self.current_time)
            self.fitness              = self.metrics['accuracy_test']

        else:
            self.metrics              = None
            self.fitness              = -1
            self.num_epochs           = 0
            self.trainable_parameters = -1
            self.current_time         = 0


        self.time = time() - start

        return self.fitness