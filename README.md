# Neural Architecture Search for Resource-constrained Devices on Internet of Things

## Abstract

The traditional process of extracting knowledge from Internet of Things (IoT) happens through Cloud Computing by offloading the data generated in the IoT device to be processed in the cloud. 
However, this regime significantly increases data transmission and monetary costs and may have privacy issues. 
Therefore, it is paramount to find solutions that achieve good results and can be processed as close as possible to an IoT object. 
In this scenario, we developed a Neural Architecture Search (NAS) solution to generate models small enough to be deployed to IoT devices without significantly losing inference performance. 
We based our approach on Evolutionary Algorithms, such as Grammatical Evolution and NSGA-II. 
Using model size and accuracy as fitness, our proposal generated a Convolutional Neural Network model with less than 2MB, achieving an accuracy of about 81% in the CIFAR-10 and 99% in MNIST, with only 150 thousand parameters approximately.


This work was presented at [IEEE Symposium on Computers and Communications (ISCC) 2021](https://iscc2021.unipi.gr/index.php). 

This paper is available at: #TODO

If you find this code useful, please consider citing our paper:

```
@inproceedings{Card2021:Neural,
author = "Isadora Cardoso-Pereira and Gisele Lobo-Pappa and Heitor S Ramos",
title  = "Neural Architecture Search for {Resource-Constrained} Internet of Things Devices",
booktitle = "2021 IEEE Symposium on Computers and Communications (ISCC) (IEEE ISCC 2021)",
address = virtual,
days = 4,
month = sep,
year = 2021,
keywords = "Neural Architecture Search; Internet of Things; Resource-constrained devices",
}
```

Note that this code is based on [Fast-DENSER](https://github.com/fillassuncao/fast-denser). 
Please, consider to cite them as well!

## About the Code

### Software requirements

We tested this code on Linux Ubuntu 18.04.03 and Debian 10. 
We use Python (version 3.7.3) with the Anaconda distribution (version 4.7.11). 
Prior to running the experiments, make sure to install the following required libraries:

- [scikit-learn](https://scikit-learn.org/stable/) (version 0.23.2)
- [Pandas](https://pandas.pydata.org/) (version 0.24.2)
- [Numpy](https://numpy.org/) (version 1.16.4)
- [Tensorflow](https://www.tensorflow.org/) (version 2.3.1)
- [Keras](https://keras.io/) (version 2.4.3)


### Project Structure and tips

- As the full experiment can take a few days to run completely, we recommend you to first execute a small experiment, changing the parameters in the ``main.py``. 
For instance, you can change the number of generations (line 88) from 50 to 10 or less.

- For each run (line 94), you must change the run parameter in the ``main.py`` to not overwrite the previous experiments.

- All the results will be saved in the folder ``experiments/``.

- We provide the notebook ``plots.ipynb`` to plot the results as presented in the article. 
It is important to pay attention to the paths when using this script, however.

- We provide as well the notebook ``tests.ipynb`` to extract the results from the best individual in the test set of CIFAR-10 and MNIST.

- In the folder ``experiments/`` we supply the best architecture found in our experiments (described on the paper).


## Support

Feel free to send any questions, comments, or suggestions (in english or portuguese) to Isadora Cardoso-Pereira (isadoracardoso@dcc.ufmg.br).
