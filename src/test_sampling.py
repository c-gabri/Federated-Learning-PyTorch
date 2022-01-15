from sampling import cifar_iid, cifar_iid_noreimmission, cifar_iid_unequal, cifar_iid_unequal_noreimmission, cifar_noniid, cifar_noniid_unequal
from torchvision import datasets, transforms
import numpy as np
import random
import matplotlib.pyplot as plt
import numpy as np
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


# takes as parameters: dataframe, number of users, number of items for each client (if not set it depends on db images/ num of users), 
# unbalance factor from 0 to 1 where 0 is a uniform distribution and 1 is distributed over a single class.
def distribution_iid(dataset, num_users, unbalance_factor, fixed_items=0):
    available_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 
    
    probabilities = np.zeros(10)
    sum_prob = 0
    max_prob = 100
    # print(probabilities)
    dict_users, all_idxs = {i: np.array([]) for i in range(num_users)}, [i for i in range(len(dataset))]
    idxs = np.arange(len(dataset))
    labels = np.array(dataset.targets)
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    # print(idxs_labels[1])
    class_idxs = getClassIndexes(idxs_labels[1])
    # print(class_idxs)
    idxs = idxs_labels[0, :]
    # print(idxs)
    # print(labels)
    if fixed_items == 0:
        num_items = int(len(dataset)/num_users)
    else:
        num_items = fixed_items
    # num_items = 10
    # num_users = 1
    distributions = {}
    user_probabilities = {}
    for user in range(num_users):
        random.shuffle(available_classes)
        probabilities[available_classes[0]] = max_prob
        curr_prob = max_prob
        i = 0
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
                # print(index)
                cumulative += probability
                if cumulative > chosen:
                    distribution[index] += 1
                    # print(class_idxs[index][0])
                    # print(class_idxs[index][1])
                    val = idxs[random.randint(class_idxs[index][0],class_idxs[index][1])-1]
                    dict_users[user] = np.append(dict_users[user],val)
                    break
        distributions[user] = distribution
        user_probabilities[user] = probabilities
    # drawHist(distribution,available_classes)
    # plt.show()
    
    #print(all_idxs)
    return dict_users, distributions, user_probabilities

def dirichlet():
    torch.distributions.Dirichlet(1/30-1 * torch.ones(30-1)).sample([10])  

def sampleFromClass():
    pass

def drawHist(data,names):
    plt.figure()
    plt.bar(names,data)  # density=False would make counts
    plt.xticks(names)
    #plt.yticks(data) #This may be included or excluded as per need
    plt.xlabel('Names')
    plt.ylabel('Probability')
    


data_dir = '../data/cifar/'
num_users = 50
apply_transform = transforms.Compose(
    [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                transform=apply_transform)

test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                transform=apply_transform)
# user_groups = cifar_noniid_unequal(train_dataset, num_users)
# user_groups = cifar_noniid(train_dataset, num_users)
# user_groups = cifar_iid(train_dataset, num_users)
# user_groups = cifar_iid_unequal(train_dataset, num_users)
user_groups, distributions, user_probabilities = distribution_iid(train_dataset, num_users,0.5)
labels = np.array(train_dataset.targets)



for user in user_groups:
    print("User: " + str(user))
    distribution = [0,0,0,0,0,0,0,0,0,0]
    for idx in user_groups[user]:
        distribution[labels[int(idx)]] += 1
        # print(labels[int(idx)], end =',')
        # print(idx)
    print(distribution)
    print(distributions[user])
    print(user_probabilities[user])
