from random import randint, uniform

class Grammar:
    
    def __init__(self, path):
        self.grammar = self.get_grammar(path)
        
    
    def get_grammar(self, path):
        
        raw_grammar = self.read_grammar_file(path)
        bnf_grammar = self.parse_grammar(raw_grammar)
        
        return bnf_grammar

    
    def read_grammar_file(self, path):
        with open(path, 'r') as fgrammar:
            raw_grammar = fgrammar.readlines()

        return raw_grammar


    def parse_grammar(self, raw_grammar):        
        grammar      = {}
        start_symbol = None

        for rule in raw_grammar:
            [raw_non_terminal, raw_rule_expansions] = rule.rstrip('\n').split('::=')

            rule_expansions = []
            for raw_production_rule in raw_rule_expansions.split('|'): 

                grammar_rule    = []
                production_rule = raw_production_rule.rstrip().lstrip().split(' ')
                for symbol in production_rule:

                    grammar_symbol = symbol.rstrip().lstrip().replace('<', '').replace('>', '')

                    if '<' in symbol:
                        symbol_type = 'non-terminal'
                    else:
                        symbol_type = 'terminal'

                    grammar_rule.append((grammar_symbol, symbol_type))

                rule_expansions.append(grammar_rule)

            non_terminal          = raw_non_terminal.rstrip().lstrip().replace('<', '').replace('>', '')
            grammar[non_terminal] = rule_expansions

            if start_symbol is None:
                start_symbol = non_terminal

        return grammar
        
    
    def __str__(self):
        """
        Prints the grammar in the BNF form
        """
        print_str = ''
        for _key_ in sorted(self.grammar):
            productions = ''
            
            for production in self.grammar[_key_]:
                
                for symbol, type_symbol in production:
                    
                    if type_symbol == 'non-terminal':
                        productions += ' <' + symbol + '>'
                    
                    else:
                        productions += ' '+ symbol
                
                productions += ' |'
            
            print_str += '<' + _key_ + '> ::=' + productions[:-2] + '\n'

        return print_str


    
    def create_individual(self, start_symbol):
        genotype = {}
        self.create_individual_recursive((start_symbol, 'non-terminal'), None, genotype)

        return genotype

    
    def create_individual_recursive(self, symbol, previous_nt, genotype):
        symbol, type_symbol = symbol

        if type_symbol == 'non-terminal':
            expansion_possibility = randint(0, len(self.grammar[symbol]) - 1)

            if symbol not in genotype:
                genotype[symbol] = [{'ge': expansion_possibility, 'ga': {}}]
            
            else:
                genotype[symbol].append({'ge': expansion_possibility, 'ga': {}})

            add_reals_idx = len(genotype[symbol]) - 1
            
            for sym in self.grammar[symbol][expansion_possibility]:
                self.create_individual_recursive(sym, (symbol, add_reals_idx), genotype)

        else:
            if '[' in symbol and ']' in symbol:
                genotype_key, genotype_idx = previous_nt
                
                terminal = symbol.replace('[', '').replace(']', '').split(',')
                [var_name, var_type, num_values, min_val, max_val] = terminal

                num_values       = int(num_values)
                min_val, max_val = float(min_val), float(max_val)

                if var_type == 'int':
                    values = [randint(min_val, max_val) for _ in range(num_values)]
                
                elif var_type == 'float':
                    values = [uniform(min_val, max_val) for _ in range(num_values)]

                genotype[genotype_key][genotype_idx]['ga'][var_name] = (var_type, min_val, max_val, values) 
                
                
    def decode(self, start_symbol, genotype):
        read_codons = dict.fromkeys(genotype.keys(), 0)        
        phenotype   = self.decode_recursive((start_symbol, 'non-terminal'), read_codons, genotype, '')

        return phenotype.lstrip().rstrip()


    def decode_recursive(self, symbol, read_integers, genotype, phenotype):
        
        symbol, type_symbol = symbol
        
        if type_symbol == 'non-terminal':
            
            if symbol not in read_integers:
                read_integers[symbol] = 0
                genotype[symbol]      = []

            if len(genotype[symbol]) <= read_integers[symbol]:
                ge_expansion_integer = randint(0, len(self.grammar[symbol]) - 1)
                genotype[symbol].append({'ge': ge_expansion_integer, 'ga': {}})

            current_nt             = read_integers[symbol]
            expansion_integer      = genotype[symbol][current_nt]['ge']
            read_integers[symbol] += 1
            expansion              = self.grammar[symbol][expansion_integer]

            used_terminals = []
            for sym in expansion:
                
                if sym[1] == 'non-terminal':
                    phenotype = self.decode_recursive(sym, read_integers, genotype, phenotype)
                
                else:
                    if '[' in sym[0] and ']' in sym[0]:
                        clean_symbol = sym[0].replace('[', '').replace(']', '').split(',')
                        [var_name, var_type, var_num_values, var_min, var_max] = clean_symbol
                        
                        if var_name not in genotype[symbol][current_nt]['ga']: 
                            var_num_values   = int(var_num_values)
                            var_min, var_max = float(var_min), float(var_max)
                            
                            if var_type == 'int':
                                values = [randint(var_min, var_max) for _ in range(var_num_values)]
                            
                            elif var_type == 'float':
                                values = [uniform(var_min, var_max) for _ in range(var_num_values)]

                            genotype[symbol][current_nt]['ga'][var_name] = (var_type, var_min, var_max, values)

                        values = genotype[symbol][current_nt]['ga'][var_name][-1]

                        phenotype += ' %s:%s' % (var_name, ','.join(map(str, values)))

                        used_terminals.append(var_name)
                    
                    else:
                        phenotype += ' ' + sym[0]
            
            other_terminals  = list(genotype[symbol][current_nt]['ga'].keys())
            unused_terminals = list(set(other_terminals) - set(used_terminals))
            
            if unused_terminals:
                for name in used_terminals:
                    del genotype[symbol][current_nt]['ga'][name]

        return phenotype