# Federated Learning (PyTorch): FedSGD, FedAvg, FedAvgM, FedIR, FedVC, FedProx
PyTorch implementation of Federated Learning algorithms FedSGD, FedAvg, FedAvgM, FedIR, FedVC, FedProx

![federated_learning](https://1.bp.blogspot.com/-K65Ed68KGXk/WOa9jaRWC6I/AAAAAAAABsM/gglycD_anuQSp-i67fxER1FOlVTulvV2gCLcB/s1600/FederatedLearning_FinalFiles_Flow%2BChart1.png)

## Reference papers
* [1] [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629): FedSGD, FedAvg, MLP and CNN models
* [2] [Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification](https://arxiv.org/abs/1909.06335): FedAvgM, Synthetic Non-Identical Client Data
* [3] [Federated Visual Classification with Real-World Data Distribution](https://arxiv.org/abs/2003.08082): FedAvg, FedIR, FedVC
* [4] [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127): FedProx
* [5] [Gradient-Based Learning Applied to Document Recognition](https://axon.cs.byu.edu/~martinez/classes/678/Papers/Convolution_nets.pdf): LeNet5

## Requirements
* python 3.8.10
* matplotlib 3.4.3
* numpy 1.17.4
* tensorboardX 2.4.1
* torch 1.10.0
* torchvision 0.11.1

## Usage
```python main.py [ARGUMENTS]```

### General arguments:
* ```--centralized```:         use centralized training (default: ```False```)
* ```--epochs EPOCHS, -E EPOCHS```:
                        number of epochs (default: ```10```)
* ```--batch_size BATCH_SIZE, -B BATCH_SIZE```:
                        batch size (default: ```10```)
* ```--optimizer {sgd,adam}```:
                        optimizer name (default: ```sgd```)
* ```--lr LR```:               learning rate (default: ```0.01```)
* ```--momentum MOMENTUM```:   SGD momentum (default: ```0```)
* ```--dataset {cifar10,mnist}```:
                        dataset name (default: ```cifar10```)
* ```--gpu GPU```:             GPU ID (default: ```0```)
* ```--model {lenet5,resnet18,cnn,mlp}```:
                        model name (default: ```lenet5```)
* ```--num_classes NUM_CLASSES```:
                        number of classes (default: ```10```)
* ```--help, -h```:            show this help message and exit (default: ```False```)

### Federated arguments:
* ```--rounds ROUNDS, -T ROUNDS```:
                        communication rounds (default: ```10```)
* ```--num_users NUM_USERS, -K NUM_USERS```:
                        number of clients (default: ```100```)
* ```--frac FRAC, -C FRAC```:  fraction of clients (default: ```0.1```)
* ```--server_lr SERVER_LR```:
                        server learning rate (default: ```1```)
* ```--iid IID```:             Identicalness of class distributions (default: ```inf```)
* ```--balance BALANCE```:     Client balance (default: ```inf```)
* ```--hetero HETERO```:       system heterogeneity (default: ```0```)
* ```--fedsgd```:              use FedSGD algorithm (default: ```False```)
* ```--fedavgm_momentum FEDAVGM_MOMENTUM```:
                        use FedAvgM algorithm with specified server momentum
                        (default: ```0```)
* ```--fedir```:               use FedIR algorithm (default: ```False```)
* ```--fedvc_nvc FEDVC_NVC```:
                        use FedVC algorithm with specified client size
                        (default: ```0```)
* ```--fedprox_mu FEDPROX_MU```:
                        use FedProx algorithm with specified mu (default: ```0```)

### Model arguments:
* ```--kernel_num KERNEL_NUM```:
                        number of each kind of kernel (default: ```9```)
* ```--kernel_sizes KERNEL_SIZES```:
                        comma-separated kernel size to use for convolution
                        (default: ```3,4,5```)
* ```--num_channels NUM_CHANNELS```:
                        number of channels of imgs (default: ```1```)
* ```--norm NORM```:           batch_norm, layer_norm, or None (default: ```batch_norm```)
* ```--num_filters NUM_FILTERS```:
                        number of filters for conv nets -- 32 for mini-
                        imagenet, 64 for omiglot. (default: ```32```)
* ```--max_pool MAX_POOL```:   Whether use max pooling rather than strided
                        convolutions (default: ```True```)

### Output arguments:
* ```--quiet, -q```:           less verbose output (default: ```False```)
* ```--batch_print_interval BATCH_PRINT_INTERVAL```:
                        print stats every specified number of batches
                        (default: ```1```)
* ```--epoch_print_interval EPOCH_PRINT_INTERVAL```:
                        print stats every specified number of epochs (default:
                        1)

## Experiments
