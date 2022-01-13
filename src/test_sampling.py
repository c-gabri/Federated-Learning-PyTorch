from sampling import cifar10_iid, cifar10_iid_noreimmission, cifar10_iid_unequal, cifar10_iid_unequal_noreimmission, cifar10_noniid, cifar10_noniid_unequal
from torchvision import datasets, transforms
import numpy as np

data_dir = '../data/cifar10/'
num_users = 50
apply_transform = transforms.Compose(
    [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                transform=apply_transform)

test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                transform=apply_transform)
# user_groups = cifar10_noniid_unequal(train_dataset, num_users)
# user_groups = cifar10_noniid(train_dataset, num_users)
# user_groups = cifar10_iid(train_dataset, num_users)
user_groups = cifar10_iid_unequal(train_dataset, num_users)
labels = np.array(train_dataset.targets)



for user in user_groups:
    print("User: " + str(user))
    distribution = [0,0,0,0,0,0,0,0,0,0]
    for idx in user_groups[user]:
        distribution[labels[int(idx)]] += 1
        # print(labels[int(idx)], end =',')
        # print(idx)
    print(distribution)
# print(len(train_dataset))
# print(len(test_dataset))