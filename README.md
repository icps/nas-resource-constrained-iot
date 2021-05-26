# Neural Architecture Search for Resource-constrained Devices on Internet of Things

## Abstract

The traditional process of extracting knowledge from Internet of Things (IoT) happens through Cloud Computing, by offloading the data generated in the IoT device to be processed in the cloud. 
However, this regime greatly increases data transmission and monetary costs, in addition, it is prone to privacy leakage issues. 
Therefore, it is of paramount importance to find solutions that not only achieve good results, but can be processed as close as possible to an IoT object. 
In this scenario, we developed a solution based on Neural Architecture Search (NAS) to generate models that are small enough to be deployed to IoT devices, without significantly losing inference performance. 
Our approach is based on Evolutionary Algorithms (AE), such as Grammatical Evolution, EA 1 + $\lambda$, and EA multi-objectives. 
Using model size and accuracy as fitness, our proposal was able to generate a Convolutional Neural Network (CNN) model with less than 2 MB, achieving an accuracy of about 81% in the CIFAR-10 dataset and 99% in MNIST, with only 150 thousand parameters approximately. 

This paper is available at: LINK

If you find this code useful, please consider citing our paper:

BIBTEX

Note that this code is heavily based on [Fast-DENSER](https://github.com/fillassuncao/fast-denser). 
Please, consider to cite them as well!

## About the Code

### Software requirements

This code is tested on Linux Ubuntu 18.04.03 and Debian 10. 
We use Python (version 3.7.3) with the Anaconda distribution (version 4.7.11). 
Prior to running the experiments, make sure to install the following required libraries:

- [scikit-learn](https://scikit-learn.org/stable/) (version 0.23.2)
- [Pandas](https://pandas.pydata.org/) (version 0.24.2)
- [Numpy](https://numpy.org/) (version 1.16.4)
- [Tensorflow](https://www.tensorflow.org/) (version 2.3.1)
- [Keras](https://keras.io/) (version 2.4.3)


### Project Structure and tips

- As the full experiment can take a few days to run completely, we recommend you to first execute a small experiment, changing the parameters in the ``main.py``. 
For instance, you can change the number of generations (variable ``num_generations``, line 88) from 50 to 10 or less.

- For each run (line 94), you must change the run parameter in the ``main.py`` to not overwrite the previous experiments.

- All the results will be saved in the folder ``experiments/``.

- We provide the notebook ``plots.ipynb`` to plot the results as presented in the article. 
It is important to pay attention to the paths when using this script, however.

- We provide as well the notebook ``tests.ipynb`` to extract the results from the best individual in the test set of CIFAR-10 and MNIST.

- In the folder ``experiments/`` we supply the best architecture found in our experiments (described on the paper).


## Support

Feel free to send any questions, comments, or suggestions (in english or portuguese) to Isadora Cardoso-Pereira (isadoracardoso@dcc.ufmg.br).
