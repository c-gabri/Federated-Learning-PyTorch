# Federated Visual Classification (PyTorch): FedSGD, FedAvg, FedAvgM, FedIR, FedVC, FedProx
PyTorch implementation of Federated Learning algorithms FedSGD, FedAvg, FedAvgM, FedIR, FedVC and FedProx, applied to visual classification. Client distributions are synthesized with arbitrary non-identicalness and imbalance (Dirichlet priors). Client systems cen be arbitrarily heterogeneous.

## Disclaimer
**Why Federated Learning is incompatible with privacy and why this is irrelevant for its success**

Federated Learning (FL) is often presented as a privacy-respecting framework for Machine Learning. While I wouldn't say this presentation is deceptive, I say it is narrow-minded. I claim that the overall effect of a widespread use of FL will be a net reduction in privacy. Precisely because data supposedly doesn't leave client devices, FL will no doubt encourage to play with private data that would have otherwise been left untouched. Since breaches in FL are still possible and trust in central servers is still needed (McMahan et al.), but possibly misplaced, more private data will inevitably mean less privacy, as it always does. The good news for FL proponents is: hardly anybody cares.

In fact, FL may be marketed today from a more honest angle, and perhaps a more effective one: _privacy is not that relevant when compared to the benefits that FL has to offer to its individual users and to society at large, if applied to this previously untapped wealth of data_. The last two years in particular have shown us how this angle is indeed viable, with most of the public willing to tolerate what would have been once considered outrageous privacy breaches, if the cause for such breaches is presented as valid enough and with enough insistence, and/or if the discomfort from opposing them is made high enough.

While some may still want to opt out of FL, I believe that governments, one of the primary stakeholders in this technology, will be able to change a lot of minds in the long run, mostly without the use of direct force, and to the benefit of private stakeholders too. If this data can indeed be used for "the public good" (say for "public health"), then why should "selfish" citizens unwilling to share it be treated the same as "generous" ones who do? What excuse do the selfish have when their data doesn't even need to leave their devices? Do the selfish want their country to be left behind, while other countries less concerned with privacy get ahead thanks to FL? And similar arguments.

To put it simply, the future of FL has never looked brighter since its introduction and, for the most part, its success will not depend on its commitment to privacy.

## Requirements
```
matplotlib==3.5.1
numpy==1.22.2
scikit_learn==1.0.2
timm==0.5.4
torch==1.10.2
torchinfo==1.6.3
torchvision==0.11.3
```

## Help
```
usage: python main.py [ARGUMENTS]

algorithm arguments:
  --rounds ROUNDS       number of communication rounds, or number of epochs if
                        --centralized (default: 200)
  --iters ITERS         number of iterations: the iterations of a round are
                        determined by the client with the largest number of
                        images (default: None)
  --num_clients NUM_CLIENTS, -K NUM_CLIENTS
                        number of clients (default: 100)
  --frac_clients FRAC_CLIENTS, -C FRAC_CLIENTS
                        fraction of clients selected at each round (default:
                        0.1)
  --train_bs TRAIN_BS, -B TRAIN_BS
                        client training batch size, 0 to use the whole
                        training set (default: 50)
  --epochs EPOCHS, -E EPOCHS
                        number of client epochs (default: 5)
  --hetero HETERO       probability of clients being stragglers, i.e. training
                        for less than EPOCHS epochs (default: 0)
  --drop_stragglers     drop stragglers (default: False)
  --server_lr SERVER_LR
                        server learning rate (default: 1)
  --server_momentum SERVER_MOMENTUM
                        server momentum for FedAvgM algorithm (default: 0)
  --mu MU               mu parameter for FedProx algorithm (default: 0)
  --centralized         use centralized algorithm (default: False)
  --fedsgd              use FedSGD algorithm (default: False)
  --fedir               use FedIR algorithm (default: False)
  --vc_size VC_SIZE     use FedVC algorithm with virtual client size VC_SIZE
                        (default: None)

dataset and split arguments:
  --dataset {cifar10,fmnist,mnist}
                        dataset, place yours in datasets.py (default: cifar10)
  --dataset_args DATASET_ARGS
                        dataset arguments (default: augment=True)
  --frac_valid FRAC_VALID
                        fraction of the training set to use for validation
                        (default: 0)
  --iid IID             identicalness of client distributions (default: inf)
  --balance BALANCE     balance of client distributions (default: inf)

model, optimizer and scheduler arguments:
  --model {cnn_cifar10,cnn_mnist,efficientnet,ghostnet,lenet5,lenet5_orig,mlp_mnist,mnasnet,mobilenet_v3}
                        model, place yours in models.py (default: lenet5)
  --model_args MODEL_ARGS
                        model arguments (default: ghost=True,norm=None)
  --optim {adam,sgd}    optimizer, place yours in optimizers.py (default: sgd)
  --optim_args OPTIM_ARGS
                        optimizer arguments (default:
                        lr=0.01,momentum=0,weight_decay=4e-4)
  --sched {const,fixed,plateau_loss,step}
                        scheduler, place yours in schedulers.py (default:
                        fixed)
  --sched_args SCHED_ARGS
                        scheduler arguments (default: None)

output arguments:
  --client_stats_every CLIENT_STATS_EVERY
                        compute and print client statistics every
                        CLIENT_STATS_EVERY batches, 0 for every epoch
                        (default: 0)
  --server_stats_every SERVER_STATS_EVERY
                        compute, print and log server statistics every
                        SERVER_STATS_EVERY rounds (default: 1)
  --name NAME           log to runs/NAME and save checkpoints to save/NAME,
                        None for YYYY-MM-DD_HH-MM-SS (default: None)
  --no_log              disable logging (default: False)
  --no_save             disable checkpoints (default: False)
  --quiet, -q           less verbose output (default: False)

other arguments:
  --test_bs TEST_BS     client test/validation batch size (default: 256)
  --seed SEED           random seed (default: 0)
  --device {cuda:0,cpu}
                        device to train/validate/test with (default: cuda:0)
  --resume              resume experiment from save/NAME checkpoint (default:
                        False)
  --help, -h            show this help message and exit (default: False)
```

## References
* [1] [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629): FedSGD, FedAvg, MLP MNIST, CNN MNIST, CNN CIFAR-10
* [2] [Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification](https://arxiv.org/abs/1909.06335): FedAvgM, Synthetic Non-Identical Client Data
* [3] [Federated Visual Classification with Real-World Data Distribution](https://arxiv.org/abs/2003.08082): FedIR, FedVC
* [4] [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127): FedProx
* [5] [TensorFlow Convolutional Neural Network Tutorial (2016)](https://github.com/tensorflow/models/blob/86ecc9730d751c1f72e3bfecac958166390f4125/tutorials/image/cifar10/cifar10.py): LeNet5
* [6] [Gradient-Based Learning Applied to Document Recognition](https://axon.cs.byu.edu/~martinez/classes/678/Papers/Convolution_nets.pdf): Original LeNet5
* [7] [GhostNet: More Features from Cheap Operations](https://arxiv.org/pdf/1911.11907.pdf): GhostNet, Ghost Module
* [8] [Model Rubik’s Cube: Twisting Resolution, Depth and Width for TinyNets](https://arxiv.org/pdf/2010.14819.pdf): TinyNet
* [9] [MnasNet: Platform-Aware Neural Architecture Search for Mobile](https://arxiv.org/pdf/1807.11626v3.pdf): MnasNet
* [10] [Searching for MobileNetV3](https://arxiv.org/pdf/1905.02244.pdf): MobileNetV3
* [11] [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/pdf/1905.11946.pdf): EfficientNet
* [12] [Group Normalization](https://arxiv.org/pdf/1803.08494.pdf): Group Normalization
* [13] [The Non-IID Data Quagmire of Decentralized Machine Learning](https://arxiv.org/pdf/1910.00189.pdf): Group Normalization for Federated Learning

## Screenshots
<img src="screenshots/tensorboard1.png?raw=true" width="100%" title="Tensorboard: scalars and images">
<p align=center>
  <img src="screenshots/tensorboard2.png?raw=true" width="32%" align="top" title="Tensorboard: model">
  <img src="screenshots/output1.png?raw=true" width="32%" align="top" title="Output: summary">
  <img src="screenshots/output2.png?raw=true" width="32%" align="top" title="Output: training">
</p>
