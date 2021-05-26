import numpy as np
import random as rnd
from copy import deepcopy

from grammar import Grammar
from individual import Individual
from evaluator import Evaluator

rnd.seed(321)
np.random.seed(321)



class Evolutionary:
    
    def calculate_crowding(self, scores):
    # https://pythonhealthcare.org/2018/10/06/95-when-too-many-multi-objective-solutions-exist-selecting-solutions-based-on-crowding-distances/
    
        # Crowding is based on a vector for each individual
        # All dimension is normalised between low and high. For any one dimension, all
        # solutions are sorted in order low to high. Crowding for chromsome x
        # for that score is the difference between the next highest and next
        # lowest score. Total crowding value sums all crowding for all scores

        population_size  = len(scores[:, 0])
        number_of_scores = len(scores[0, :])

        # create crowding matrix of population (row) and score (column)
        crowding_matrix = np.zeros((population_size, number_of_scores))

        # normalise scores (ptp is max-min)
        normed_scores = (scores - scores.min(0)) / scores.ptp(0)

        # calculate crowding distance for each score in turn
        for col in range(number_of_scores):
            crowding = np.zeros(population_size)

            # end points have maximum crowding
            crowding[0] = 1
            crowding[population_size - 1] = 1

            # Sort each score (to calculate crowding between adjacent scores)
            sorted_scores = np.sort(normed_scores[:, col])

            sorted_scores_index = np.argsort(normed_scores[:, col])

            # Calculate crowding distance for each individual
            crowding[1:population_size - 1] = (sorted_scores[2:population_size] - sorted_scores[0:population_size - 2])

            # resort to orginal order (two steps)
            re_sort_order   = np.argsort(sorted_scores_index)
            sorted_crowding = crowding[re_sort_order]

            # Record crowding distances
            crowding_matrix[:, col] = sorted_crowding

        # Sum crowding distances of each score
        crowding_distances = np.sum(crowding_matrix, axis=1)

        return crowding_distances


    def reduce_by_crowding(self, scores, number_to_select):
        # This function selects a number of solutions based on tournament of
        # crowding distances. Two members of the population are picked at
        # random. The one with the higher croding dostance is always picked

        population_ids        = np.arange(scores.shape[0])
        crowding_distances    = self.calculate_crowding(scores)
        picked_population_ids = np.zeros((number_to_select))

        picked_scores         = np.zeros((number_to_select, len(scores[0, :])))

        for i in range(number_to_select):

            population_size = population_ids.shape[0]

            fighter1ID      = rnd.randint(0, population_size - 1)
            fighter2ID      = rnd.randint(0, population_size - 1)


            if crowding_distances[fighter1ID] == crowding_distances[fighter2ID]:
                # get the best accuracy
                winner_id = np.argmax(scores[:, 1])


            # If fighter nº 1 is better
            elif crowding_distances[fighter1ID] > crowding_distances[fighter2ID]:
                winner_id = population_ids[fighter1ID]

            else:
                winner_id = population_ids[fighter2ID]


            # add solution to picked solutions array
            picked_population_ids[i] = winner_id

            # Add score to picked scores array
            picked_scores[i, :]      = scores[winner_id, :]

            # remove selected solution from available solutions
            population_ids           = np.delete(population_ids, (winner_id), axis=0)
            scores                   = np.delete(scores, (winner_id), axis=0)
            crowding_distances       = np.delete(crowding_distances, (winner_id),axis=0)


        # Convert to integer
        picked_population_ids = np.asarray(picked_population_ids, dtype=int)
        return (picked_population_ids)


    
    def pareto_frontier(self, population):
    # Dominância de Pareto
    ## uma solução S1 domina S2 se e somente se S1 não é pior que S2 em nenhum objetivo 
    ## e S1 é obrigatoriamente melhor que S2 em pelo menos um objetivo
    #
    ## S1 nunca pode ser pior que S2 em nenhum objetivo, no máximo igual
    ## o conjunto de soluções não-dominadas é chamado de fronte (ou fronteira) de Pareto
    
        # Count number of items
        population_size = len(population)

        scores = []

        for current_ind in population:
            
            if current_ind.fitness != -1:          
                parameters = current_ind.trainable_parameters
                accuracy   = current_ind.fitness

                scores.append([parameters, 1 - accuracy])

            else:
                scores.append([np.inf, 1 - 0])

        ## Assume that everyone is in the frontier (value 1), if not, remove it (value 0)
        pareto_front = [1] * len(population)

        for i in range(population_size):
            for j in range(population_size):

                param_i, acc_i = scores[i]
                param_j, acc_j = scores[j]

                pareto1 = all(jj <= ii for jj, ii in zip(scores[j], scores[i]))
                pareto2 = any(jj < ii for jj, ii in zip(scores[j], scores[i]))

                if pareto1 and pareto2:
                    pareto_front[i] = 0
                    break

        pareto_individuals = [population[i] for i in range(len(population)) if pareto_front[i] == 1]

        return pareto_individuals
    
    
    def select_fittest(self, population):
        pareto_population = self.pareto_frontier(population)
        
        if len(pareto_population) == 1:
            parent = pareto_population[0]
            
        else:
            scores    = [[x.trainable_parameters, x.fitness] for x in pareto_population]
            scores    = np.array(scores)
            
#             parent_id = np.argmax(scores)
            parent_id = self.reduce_by_crowding(scores, number_to_select = 1)
            
            parent    = pareto_population[int(parent_id)]
            
        return parent
    
    
    def mutation_dsge(self, layer, grammar):
        
        nt_keys = sorted(list(layer.keys()))
        nt_key  = rnd.choice(nt_keys)
        nt_idx  = rnd.randint(0, len(layer[nt_key]) - 1)

        sge_possibilities    = []
        random_possibilities = []
        if len(grammar.grammar[nt_key]) > 1:

            sge_possibilities = list(set(range(len(grammar.grammar[nt_key]))) - set([layer[nt_key][nt_idx]['ge']]))

            random_possibilities.append('ge')

        if layer[nt_key][nt_idx]['ga']:
            random_possibilities.extend(['ga', 'ga'])

        if random_possibilities:
            mt_type = rnd.choice(random_possibilities)

            if mt_type == 'ga':
                var_name = rnd.choice(sorted(list(layer[nt_key][nt_idx]['ga'].keys())))
                var_type, min_val, max_val, values = layer[nt_key][nt_idx]['ga'][var_name]
                value_idx = rnd.randint(0, len(values)-1)

                if var_type == 'int':
                    new_val = rnd.randint(min_val, max_val)

                elif var_type == 'float':
                    new_val = values[value_idx] + rnd.gauss(0, 0.15)
                    new_val = np.clip(new_val, min_val, max_val)

                layer[nt_key][nt_idx]['ga'][var_name][-1][value_idx] = new_val

            elif mt_type == 'ge':
                layer[nt_key][nt_idx]['ge'] = rnd.choice(sge_possibilities)

            else:
                return NotImplementedError


            
            
    def mutation_add_layer(self, module, add_layer, re_use_layer, grammar):
        
        # add-layer (duplicate or new)
        for _ in range(rnd.randint(1,2)): # add one or two layers
            if len(module.layers) < module.max_expansions and rnd.random() <= add_layer:

                if rnd.random() <= re_use_layer:
                    new_layer = rnd.choice(module.layers)

                else:
                    new_layer = grammar.create_individual(module.module)

                insert_pos = rnd.randint(0, len(module.layers)) # position to insert the layer

                # fix connections
                for _key_ in sorted(module.connections, reverse = True):

                    if _key_ >= insert_pos:

                        for value_idx, value in enumerate(module.connections[_key_]):

                            if value >= insert_pos - 1:
                                module.connections[_key_][value_idx] += 1

                        module.connections[_key_ + 1] = module.connections.pop(_key_)

                module.layers.insert(insert_pos, new_layer)

                # make the connections of the new layer
                if insert_pos == 0:
                    module.connections[insert_pos] = [-1]

                else:
                    connection_possibilities = list(range(max(0, insert_pos - module.levels_back), insert_pos - 1))

                    if len(connection_possibilities) < module.levels_back - 1:
                        connection_possibilities.append(-1)

                    sample_size = rnd.randint(0, len(connection_possibilities))

                    module.connections[insert_pos] = [insert_pos-1] 

                    if sample_size > 0:
                        module.connections[insert_pos] += rnd.sample(connection_possibilities, sample_size)
                        
                        
    
    def mutation_remove_layer(self, module, remove_layer):
        
        # remove-layer
        for _ in range(rnd.randint(1,2)):

            if len(module.layers) > module.min_expansions and rnd.random() <= remove_layer:
                remove_idx = rnd.randint(0, len(module.layers) - 1)
                del module.layers[remove_idx]

                # fix connections
                for _key_ in sorted(module.connections):
                    if _key_ > remove_idx:

                        if _key_ > remove_idx + 1 and remove_idx in module.connections[_key_]:
                            module.connections[_key_].remove(remove_idx)

                        for value_idx, value in enumerate(module.connections[_key_]):

                            if value >= remove_idx:
                                module.connections[_key_][value_idx] -= 1

                        module.connections[_key_-1] = list(set(module.connections.pop(_key_)))

                if remove_idx == 0:
                    module.connections[0] = [-1]
                   


    def mutation(self, parent, grammar, mutation_parameters, default_train_time):

        add_layer, re_use_layer, remove_layer, add_connection, \
        remove_connection, dsge_layer, macro_layer, train_longer = mutation_parameters.values()

        indv = deepcopy(parent)

        if rnd.random() <= train_longer:
            indv.train_time = indv.train_time + default_train_time

            return indv

        indv.current_time = 0
        indv.num_epochs   = 0
        indv.train_time   = default_train_time

        for module in indv.modules:

            self.mutation_add_layer(module, add_layer, re_use_layer, grammar)
            
            self.mutation_remove_layer(module, remove_layer)


            for layer_idx, layer in enumerate(module.layers):
                # dsge mutation
                if rnd.random() <= dsge_layer:
                    self.mutation_dsge(layer, grammar)

                # add connection
                if layer_idx != 0 and rnd.random() <= add_connection:
                    connection_possibilities = list(range(max(0, layer_idx - module.levels_back), layer_idx - 1))
                    connection_possibilities = list(set(connection_possibilities) - set(module.connections[layer_idx]))
                    
                    if len(connection_possibilities) > 0:
                        module.connections[layer_idx].append(rnd.choice(connection_possibilities))

                # remove connection
                r_value = rnd.random()
                if layer_idx != 0 and r_value <= remove_connection:
                    connection_possibilities = list(set(module.connections[layer_idx]) - set([layer_idx-1]))
                    
                    if len(connection_possibilities) > 0:
                        r_connection = rnd.choice(connection_possibilities)
                        module.connections[layer_idx].remove(r_connection)

        # macro level mutation
        for macro_idx, macro in enumerate(indv.macro): 
            if rnd.random() <= macro_layer:
                self.mutation_dsge(macro, grammar)   

        return indv

