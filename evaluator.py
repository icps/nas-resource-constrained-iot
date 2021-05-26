import os
import numpy as np
from time import time

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import keras
import tensorflow
from keras import backend
from keras.utils.vis_utils import plot_model
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator


DEBUG = True


class TimedStopping(keras.callbacks.Callback):

    def __init__(self, seconds = None, verbose = 0):

        super(keras.callbacks.Callback, self).__init__()

        self.start_time = 0
        self.seconds    = seconds
        self.verbose    = verbose


    def on_train_begin(self, logs = {}):
        self.start_time = time()
        

    def on_epoch_end(self, epoch, logs = {}):
        
        if time() - self.start_time > self.seconds:
            self.model.stop_training = True
            
            if self.verbose:
                print('Stopping after %s seconds.' % self.seconds)           

                
class Evaluator:
    
    def __init__(self, dataset):
        self.dataset = self.load_dataset(dataset)   
        
    
    def load_dataset(self, dataset):
        # only cifar10 so far
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        
        x_train = x_train.astype('float32')
        x_test  = x_test.astype('float32')

        x_train = x_train.reshape((-1, 32, 32, 3))
        x_test  = x_test.reshape((-1, 32, 32, 3))

        evo_x_train, x_val, evo_y_train, y_val = train_test_split(x_train, y_train,
                                                                  test_size = 7000,
                                                                  stratify  = y_train)

        evo_x_val, evo_x_test, evo_y_val, evo_y_test = train_test_split(x_val, y_val,
                                                                        test_size = 3500,
                                                                        stratify  = y_val)

        n_classes   = 10
        evo_y_train = keras.utils.to_categorical(evo_y_train, n_classes)
        evo_y_val   = keras.utils.to_categorical(evo_y_val, n_classes)

        dataset     = {'evo_x_train': evo_x_train, 'evo_y_train': evo_y_train,
                       'evo_x_val': evo_x_val, 'evo_y_val': evo_y_val,
                       'evo_x_test': evo_x_test, 'evo_y_test': evo_y_test,
                       'x_test': x_test, 'y_test': y_test}
        
        return dataset
        
        
    def get_layers(self, phenotype):

        raw_phenotype = phenotype.split(' ')

        idx                 = 0
        first               = True
        node_type, node_val = raw_phenotype[idx].split(':')
        layers              = []

        while idx < len(raw_phenotype):
            if node_type == 'layer':
                
                if not first:
                    layers.append((layer_type, node_properties))
                else:
                    first = False
                    
                layer_type      = node_val
                node_properties = {}
            
            else:
                node_properties[node_type] = node_val.split(',')

            idx += 1
            
            if idx < len(raw_phenotype):
                node_type, node_val = raw_phenotype[idx].split(':')

        layers.append((layer_type, node_properties))

        return layers


    def get_learning(self, learning):
     
        raw_learning    = learning.split(' ')

        idx             = 0
        learning_params = {}
        
        while idx < len(raw_learning):
            param_name, param_value     = raw_learning[idx].split(':')
            learning_params[param_name] = param_value.split(',')
            
            idx += 1

        for _key_ in sorted(list(learning_params.keys())):
            
            if len(learning_params[_key_]) == 1:
                
                try:
                    learning_params[_key_] = eval(learning_params[_key_][0])
                
                except NameError:
                    learning_params[_key_] = learning_params[_key_][0]

        return learning_params

    

    def assemble_network(self, keras_layers, input_size):

        #input layer
        inputs = keras.layers.Input(shape = input_size)

        #Create layers -- ADD NEW LAYERS HERE
        layers = []
        for layer_type, layer_params in keras_layers:
            
            #convolutional layer
            if layer_type == 'conv':
                filters     = int(layer_params['num-filters'][0])
                kernel_size = (int(layer_params['filter-shape'][0]), int(layer_params['filter-shape'][0]))
                strides     = (int(layer_params['stride'][0]), int(layer_params['stride'][0]))
                padding     = layer_params['padding'][0]
                activation  = layer_params['act'][0]
                bias        = eval(layer_params['bias'][0])
                kernel_init = 'he_normal'
#                 kernel_init = 'glorot_uniform'
                kernel_reg  = keras.regularizers.l2(0.0005)
                                
                conv_layer  = keras.layers.Conv2D(filters            = filters, 
                                                  kernel_size        = kernel_size,
                                                  strides            = strides,
                                                  padding            = padding,
                                                  activation         = activation,
                                                  use_bias           = bias,
                                                  kernel_initializer = kernel_init,
                                                  kernel_regularizer = kernel_reg)
                layers.append(conv_layer)

            #batch-normalisation
            elif layer_type == 'batch-norm':
                batch_norm = keras.layers.BatchNormalization()
                layers.append(batch_norm)

            #average pooling layer
            elif layer_type == 'pool-avg':
                pool_size = (int(layer_params['kernel-size'][0]), int(layer_params['kernel-size'][0]))
                strides   = int(layer_params['stride'][0])
                padding   = layer_params['padding'][0]
                
                pool_avg  = keras.layers.AveragePooling2D(pool_size = pool_size,
                                                          strides   = strides,
                                                          padding   = padding)
                
                layers.append(pool_avg)

            #max pooling layer
            elif layer_type == 'pool-max':
                pool_size = (int(layer_params['kernel-size'][0]), int(layer_params['kernel-size'][0]))
                strides   = int(layer_params['stride'][0])
                padding   = layer_params['padding'][0]
                
                pool_max  = keras.layers.MaxPooling2D(pool_size = pool_size,
                                                      strides   = strides,
                                                      padding   = padding)
                layers.append(pool_max)

            #fully-connected layer
            elif layer_type == 'fc':
                
                num_units   = int(layer_params['num-units'][0])
                activation  = layer_params['act'][0]
                bias        = eval(layer_params['bias'][0])
                kernel_init = 'he_normal'
#                 kernel_init = 'glorot_uniform'
                kernel_reg  = keras.regularizers.l2(0.0005)
                
                fc = keras.layers.Dense(num_units,
                                        activation         = activation,
                                        use_bias           = bias,
                                        kernel_initializer = kernel_init,
                                        kernel_regularizer = kernel_reg)
                layers.append(fc)

            #dropout layer
            elif layer_type == 'dropout':
                dropout = keras.layers.Dropout(rate=float(layer_params['rate'][0]))
                layers.append(dropout)


            #END ADD NEW LAYERS


        #Connection between layers
        for layer in keras_layers:            
            layer[1]['input'] = list(map(int, layer[1]['input']))

        first_fc       = True
        data_layers    = []
        invalid_layers = []

        for layer_idx, layer in enumerate(layers):
            
            try:
                if len(keras_layers[layer_idx][1]['input']) == 1:
                    
                    if keras_layers[layer_idx][1]['input'][0] == -1:
                        data_layers.append(layer(inputs))
                    
                    else:
                        if keras_layers[layer_idx][0] == 'fc' and first_fc:
                            first_fc = False
                            flatten  = keras.layers.Flatten()(data_layers[keras_layers[layer_idx][1]['input'][0]])
                            data_layers.append(layer(flatten))
                            
                            continue

                        data_layers.append(layer(data_layers[keras_layers[layer_idx][1]['input'][0]]))

                else:
                    #Get minimum shape: when merging layers all the signals are converted to the minimum shape
                    minimum_shape = input_size[0]
                    
                    for input_idx in keras_layers[layer_idx][1]['input']:
                        
                        if input_idx != -1 and input_idx not in invalid_layers:
                            
                            if data_layers[input_idx].shape[-3:][0] < minimum_shape:
                                minimum_shape = int(data_layers[input_idx].shape[-3:][0])

                    #Reshape signals to the same shape
                    merge_signals = []
                    
                    for input_idx in keras_layers[layer_idx][1]['input']:
                        
                        if input_idx == -1:
                            
                            if inputs.shape[-3:][0] > minimum_shape:
                                actual_shape = int(inputs.shape[-3:][0])
                                new_size     = (actual_shape - (minimum_shape - 1), actual_shape - (minimum_shape - 1))
                                
                                layer        = keras.layers.MaxPooling2D(pool_size = new_size, strides = 1)
                                merge_signals.append(layer(inputs))
                            
                            else:
                                merge_signals.append(inputs)

                        elif input_idx not in invalid_layers:
                            
                            if data_layers[input_idx].shape[-3:][0] > minimum_shape:
                                actual_shape = int(data_layers[input_idx].shape[-3:][0])
                                new_size     = (actual_shape - (minimum_shape - 1), actual_shape - (minimum_shape - 1))
                                
                                layer        = keras.layers.MaxPooling2D(pool_size = new_size, strides = 1)
                                merge_signals.append(layer(data_layers[input_idx]))
                            
                            else:
                                merge_signals.append(data_layers[input_idx])

                    if len(merge_signals) == 1:
                        merged_signal = merge_signals[0]
                    
                    elif len(merge_signals) > 1:
                        merged_signal = keras.layers.concatenate(merge_signals)
                    
                    else:
                        merged_signal = data_layers[-1]

                    data_layers.append(layer(merged_signal))
                    
                                
            except ValueError as e:
                data_layers.append(data_layers[-1])
                invalid_layers.append(layer_idx)
                
                if DEBUG:
                    print(keras_layers[layer_idx][0])
                    print(e)

        model = keras.models.Model(inputs = inputs, outputs = data_layers[-1])
        
        if DEBUG:
            model.summary()

        return model


    def assemble_optimiser(self, learning):

        if learning['learning'] == 'rmsprop':
            return keras.optimizers.RMSprop(lr    = float(learning['lr']),
                                            rho   = float(learning['rho']),
                                            decay = float(learning['decay']))
        
        elif learning['learning'] == 'gradient-descent':
            return keras.optimizers.SGD(lr       = float(learning['lr']),
                                        momentum = float(learning['momentum']),
                                        decay    = float(learning['decay']),
                                        nesterov = bool(learning['nesterov']))

        elif learning['learning'] == 'adam':
            return keras.optimizers.Adam(lr     = float(learning['lr']),
                                         beta_1 = float(learning['beta1']),
                                         beta_2 = float(learning['beta2']),
                                         decay  = float(learning['decay']))

        

    def evaluate(self, phenotype, load_prev_weights, weights_save_path, parent_weights_path,\
                 train_time, num_epochs, datagen = None, datagen_test = None, input_size = (32, 32, 3)):

        model_phenotype, learning_phenotype = phenotype.split('learning:')
        learning_phenotype                  = 'learning:' + learning_phenotype.rstrip().lstrip()
        model_phenotype                     = model_phenotype.rstrip().lstrip().replace('  ', ' ')

        keras_layers   = self.get_layers(model_phenotype)
        keras_learning = self.get_learning(learning_phenotype)
        batch_size     = int(keras_learning['batch_size'])
        
        if load_prev_weights:
            model = keras.models.load_model(parent_weights_path.replace('.hdf5', '.h5'))
            model.build(input_shape = input_size)

        else:
            model = self.assemble_network(keras_layers, input_size)
            opt   = self.assemble_optimiser(keras_learning)
            model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])

        #early stopping
        early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = int(keras_learning['early_stop']))

        #time based stopping
        time_stop = TimedStopping(seconds = train_time, verbose = DEBUG)

        #save individual with the lowest validation loss
        monitor = ModelCheckpoint(weights_save_path, monitor = 'val_loss', verbose = DEBUG, 
                                  save_best_only = True, mode = 'auto')
       
        trainable_count = int(np.sum([backend.count_params(p) for p in model.trainable_weights]))
        
        if trainable_count > 500000:
        
            if DEBUG:

                print("------------------\n")
                print("Too much weight ({})! We gotta kill this model before it kill you too!".format(trainable_count))
                print("\n------------------")
            
            return None
        
        
        else: 
            step_epochs  = (self.dataset['evo_x_train'].shape[0] // batch_size)
            epochs       = int(keras_learning['epochs'])
            callback     = [early_stop, monitor, time_stop]

            x_train_data = self.dataset['evo_x_train']
            y_train_data = self.dataset['evo_y_train']

            x_val_data   = self.dataset['evo_x_val']
            y_val_data   = self.dataset['evo_y_val']

            if datagen is not None:                
                datagen      = eval(datagen)
                datagen_test = eval(datagen_test)

                score = model.fit(datagen.flow(x_train_data, y_train_data, batch_size = batch_size), 
                                  steps_per_epoch  = step_epochs,
                                  epochs           = epochs,
                                  validation_data  = datagen_test.flow(x_val_data, y_val_data, batch_size = batch_size),
                                  validation_steps = (x_val_data.shape[0] // batch_size),
                                  callbacks        = callback,
                                  initial_epoch    = num_epochs,
                                  verbose          = DEBUG)

            else:
                score = model.fit(x = x_train_data, y = y_train_data, batch_size = batch_size,
                                  epochs          = epochs,
                                  steps_per_epoch = step_epochs,
                                  validation_data = (x_val_data, y_val_data),
                                  callbacks       = callback,
                                  initial_epoch   = num_epochs,
                                  verbose         = DEBUG)


            #load weights with the lowest val loss
            if os.path.isfile(weights_save_path):
                model.load_weights(weights_save_path)

            #save final moodel to file
            model.save(weights_save_path.replace('.hdf5', '.h5'))

            #measure test performance
            if datagen_test is None:
                y_pred_test = model.predict(self.dataset['evo_x_test'], batch_size = batch_size, verbose = 0)

            else:
                datagen_test_flow = datagen_test.flow(self.dataset['evo_x_test'], batch_size = 100, shuffle = False)
                steps             = self.dataset['evo_x_test'].shape[0] // 100

                y_pred_test       = model.predict_generator(datagen_test_flow, steps = steps, verbose = DEBUG)

            y_pred_labels   = np.argmax(y_pred_test, axis = 1)        
            accuracy_test   = accuracy_score(self.dataset['evo_y_test'], y_pred_labels)

            if DEBUG:
                indv_id  = weights_save_path.split("best")[1].split("_")[-1].split(".")[0]
                filename = weights_save_path.split("best")[0] + 'model_plot_' + indv_id + '.png'
                
                plot_model(model, to_file = filename, show_shapes = True, show_layer_names = True)

            print("------------------\n")
            print("** Phenotype: {} \n** Accuracy test: {}".format(phenotype, accuracy_test))
            print("\n------------------")

            score.history['trainable_parameters'] = trainable_count
            score.history['accuracy_test']        = accuracy_test
            score.history['epochs']               = epochs

            return score.history


    def testing_performance(self, model_path):

        model    = keras.models.load_model(model_path)
        y_pred   = model.predict(self.dataset['x_test'])
        accuracy = self.fitness_metric(self.dataset['y_test'], y_pred)
        
        return accuracy