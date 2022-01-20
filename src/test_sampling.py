from sampling import cifar10_iid, cifar10_iid_noreimmission, cifar10_iid_unequal, cifar10_iid_unequal_noreimmission, cifar10_noniid, cifar10_noniid_unequal
from torchvision import datasets, transforms
import numpy as np
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import torch

def getClassIndexes(labels):
    class_id = labels[0]
    previous_id = 0
    labels_idx = {}
    for index,label in enumerate(labels):
        # print(index)
        if label != class_id:
            labels_idx[class_id] = [previous_id, index]
            class_id = label
            previous_id = index
    labels_idx[class_id] = [previous_id,len(labels)]

    return labels_idx

def emd_distance(distributions):
    emd = 0
    population = np.zeros(10)
    dists = [0,1,2,3,4,5,6,7,8,9]
    for user in distributions:
        population = np.add(population, distributions[user])
    # print(population)
    for user in distributions:
        # print(distributions[user])
        dist = stats.wasserstein_distance(dists, dists, population, distributions[user])
        if dist > 2:
            dist = 2
        emd += (np.sum(distributions[user])/np.sum(population))*dist

        # print(dist)
    print("EMD: " + str(emd))
    return emd

# takes as parameters: dataframe, number of users, number of items for each client (if not set it depends on db images/ num of users), 
# unbalance factor from 0 to 1 where 0 is a uniform distribution and 1 is distributed over a single class.
def distribution_iid(dataset, num_clients, unbalance_factor, fixed_items=0):
    available_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    probabilities = np.zeros(10)
    sum_prob = 0
    max_prob = 100
    # print(probabilities)
    dict_users, all_idxs = {i: np.array([]) for i in range(num_clients)}, [i for i in range(len(dataset))]
    idxs = np.arange(len(dataset))
    labels = np.array(dataset.targets)
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    # print(idxs_labels[1])
    class_idxs = getClassIndexes(idxs_labels[1])
    # print(class_idxs)
    idxs = idxs_labels[0, :]
    if fixed_items == 0:
        num_items = int(len(dataset)/num_clients)
    else:
        num_items = fixed_items
    # num_items = 10
    # num_clients = 1
    distributions = {}
    user_probabilities = {}
    for user in range(num_clients):
        random.shuffle(available_classes)
        probabilities[available_classes[0]] = max_prob
        curr_prob = max_prob
        i = 1
        while i<9:
            curr_prob -= curr_prob*unbalance_factor
            probabilities[available_classes[i]] = curr_prob
            probabilities[available_classes[i+1]] = curr_prob
            i += 2

        total_prob = np.sum(probabilities)
        distribution = np.zeros(10)
        for item in range(num_items):
            chosen = random.uniform(0, total_prob)
            cumulative = 0
            for index, probability in enumerate(probabilities):
                cumulative += probability
                if cumulative > chosen:
                    distribution[index] += 1
                    val = idxs[random.randint(class_idxs[index][0],class_idxs[index][1])-1]
                    dict_users[user] = np.append(dict_users[user],val)
                    break
        distributions[user] = distribution
        user_probabilities[user] = probabilities
    # drawHist(distribution,available_classes)
    # plt.show()
    return dict_users

def dirichlet():
    torch.distributions.Dirichlet(1/30-1 * torch.ones(30-1)).sample([10])


def drawHist(data,names):
    plt.figure()
    plt.bar(names,data)  # density=False would make counts
    plt.xticks(names)
    #plt.yticks(data) #This may be included or excluded as per need
    plt.xlabel('Names')
    plt.ylabel('Probability')



data_dir = '../data/cifar10/'
num_clients = 50
unbalance_factor = 0
apply_transform = transforms.Compose(
    [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                transform=apply_transform)

test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                transform=apply_transform)

# train_split = cifar10_noniid_unequal(train_dataset, num_clients)
# train_split = cifar10_noniid(train_dataset, num_clients)
# train_split = cifar10_iid(train_dataset, num_clients)
# train_split = cifar10_iid_unequal(train_dataset, num_clients)
train_split = distribution_iid(train_dataset, num_clients,unbalance_factor)

labels = np.array(train_dataset.targets)

print("Unbalance Factor: " + str(unbalance_factor))
distributions = {}
for user in train_split:
    # print("User: " + str(user))
    distribution = [0,0,0,0,0,0,0,0,0,0]
    for idx in train_split[user]:
        distribution[labels[int(idx)]] += 1
        # print(labels[int(idx)], end =',')
        # print(idx)
    # print(distribution)
    distributions[user] = distribution
    # print(distributions[user])
    # print(user_probabilities[user])

non_identicalness = emd_distance(distributions)
