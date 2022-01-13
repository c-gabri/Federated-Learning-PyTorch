# Federated-Learning (PyTorch)

Implementation of the vanilla federated learning paper : [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629).


Experiments are produced on MNIST, Fashion MNIST and CIFAR10 (both IID and non-IID). In case of non-IID, the data amongst the users can be split equally or unequally.

Since the purpose of these experiments are to illustrate the effectiveness of the federated learning paradigm, only simple models such as MLP and CNN are used.

## Requirments
Install all the packages from requirments.txt
* Python3
* Pytorch
* Torchvision

## Data
* Download train and test datasets manually or they will be automatically downloaded from torchvision datasets.
* Experiments are run on Mnist, Fashion Mnist and Cifar.
* To use your own dataset: Move your dataset to data directory and write a wrapper on pytorch dataset class.

## Running the experiments
The baseline experiment trains the model in the conventional way.

* To run the baseline experiment with MNIST on MLP using CPU:
```
python baseline_main.py --model=mlp --dataset=mnist --epochs=10
```
* Or to run it on GPU (eg: if gpu:0 is available):
```
python baseline_main.py --model=mlp --dataset=mnist --gpu=0 --epochs=10
```
-----

Federated experiment involves training a global model using many local models.

* To run the federated experiment with CIFAR on CNN (IID):
```
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --iid=1 --epochs=10
```
* To run the same experiment under non-IID condition:
```
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --iid=0 --epochs=10
```

You can change the default values of other parameters to simulate different conditions. Refer to the usage section.

## Usage
#### usage: main.py [ARGUMENTS]

#### general arguments:
* ```--centralized```         use centralized training (default: False)
* ```--epochs EPOCHS```       number of rounds of training (default: 10)
* ```--optimizer OPTIMIZER```
                        type of optimizer (default: sgd)
* ```--lr LR```               learning rate (default: 0.01)
* ```--momentum MOMENTUM```   SGD momentum (default: 0)
* ```--dataset DATASET```     name of dataset (default: cifar)
* ```--gpu GPU```             To use cuda, set to a specific GPU ID. Default set to
                        use CPU. (default: None)
* ```--model MODEL```         model name (default: cnn)
* ```--num_classes NUM_CLASSES```
                        number of classes (default: 10)
* ```--stopping_rounds STOPPING_ROUNDS```
                        rounds of early stopping (default: 10)
* ```--seed SEED```           random seed (default: 1)
* ```--verbose```, ```-v```         verbose (default: True)
* ```--help```, ```-h```            show this help message and exit (default: False)

#### federated arguments:
* ```--num_users NUM_USERS```, ```-K NUM_USERS```
                        number of clients (default: 100)
* ```--frac FRAC```, ```-C FRAC```  fraction of clients (default: 0.1)
* ```--local_ep LOCAL_EP```, ```-E LOCAL_EP```
                        number of local epochs (default: 10)
* ```--local_bs LOCAL_BS```, ```-B LOCAL_BS```
                        local batch size (default: 10)
* ```--server_lr SERVER_LR```
                        server learning rate (default: 1)
* ```--iid IID```             Default set to IID. Set to 0 for non```-IID```. (default: 1)
* ```--unequal UNEQUAL```     whether to use unequal data splits for non```-i```.i.d
                        setting (use 0 for equal splits) (default: 0)
* ```--hetero HETERO```       system heterogeneity (default: 0)
* ```--fedsgd```              use FedSGD algorithm (default: False)
* ```--fedavgm_momentum FEDAVGM_MOMENTUM```
                        use FedAvgM algorithm with specified server momentum
                        (default: 0)
* ```--fedir```               use FedIR algorithm (default: False)
* ```--fedvc_nvc FEDVC_NVC```
                        use FedVC algorithm with specified client size
                        (default: 0)
* ```--fedprox_mu FEDPROX_MU```
                        use FedProx algorithm with specified mu (default: 0)

#### model arguments:
* ```--kernel_num KERNEL_NUM```
                        number of each kind of kernel (default: 9)
* ```--kernel_sizes KERNEL_SIZES```
                        comma```-separated``` kernel size to use for convolution
                        (default: 3,4,5)
* ```--num_channels NUM_CHANNELS```
                        number of channels of imgs (default: 1)
* ```--norm NORM```           batch_norm, layer_norm, or None (default: batch_norm)
* ```--num_filters NUM_FILTERS```
                        number of filters for conv nets -- 32 for mini-
                        imagenet, 64 for omiglot. (default: 32)
* ```--max_pool MAX_POOL```   Whether use max pooling rather than strided
                        convolutions (default: True)

## Results on MNIST
#### Baseline Experiment:
The experiment involves training a single model in the conventional way.

Parameters: <br />
* ```Optimizer:```    : SGD 
* ```Learning Rate:``` 0.01

```Table 1:``` Test accuracy after training for 10 epochs:

| Model | Test Acc |
| ----- | -----    |
|  MLP  |  92.71%  |
|  CNN  |  98.42%  |

----

#### Federated Experiment:
The experiment involves training a global model in the federated setting.

Federated parameters (default values):
* ```Fraction of users (C)```: 0.1 
* ```Local Batch size  (B)```: 10 
* ```Local Epochs      (E)```: 10 
* ```Optimizer            ```: SGD 
* ```Learning Rate        ```: 0.01 <br />

```Table 2:``` Test accuracy after training for 10 global epochs with:

| Model |    IID   | Non-IID (equal)|
| ----- | -----    |----            |
|  MLP  |  88.38%  |     73.49%     |
|  CNN  |  97.28%  |     75.94%     |


## Further Readings
### Papers:
* [Federated Learning: Challenges, Methods, and Future Directions](https://arxiv.org/abs/1908.07873)
* [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)
* [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133)

### Blog Posts:
* [CMU MLD Blog Post: Federated Learning: Challenges, Methods, and Future Directions](https://blog.ml.cmu.edu/2019/11/12/federated-learning-challenges-methods-and-future-directions/)
* [Leaf: A Benchmark for Federated Settings (CMU)](https://leaf.cmu.edu/)
* [TensorFlow Federated](https://www.tensorflow.org/federated)
* [Google AI Blog Post](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html)
