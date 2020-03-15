from itertools import cycle
from time import time

import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F

from dragonfly.exd.experiment_caller import CPFunctionCaller
from dragonfly.opt import gp_bandit
from dragonfly.exd.cp_domain_utils import load_config_file

from multiprocessing import Pool

import signal
import sys

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    print(results)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


class Net(nn.Module):
    def __init__(self):
        # will probably do hyperparameter search here
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class Trainable_cifar10(object):
    """object implementing essential training methods"""
    def __init__(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                     download=True, transform=transform)
        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                    download=True, transform=transform)
        self.classes = ('plane', 'car', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.criterion = nn.CrossEntropyLoss()

    def reset(self, optimizer_type, log10_lr, log10_momentum, log10_beta2, batch_size):
        # print(optimizer_type, log10_lr, log10_momentum, log10_beta2, batch_size)
        self.model = Net().cuda()
        lr = 10 ** log10_lr
        momentum = 1 - 10 ** log10_momentum
        beta2 = 1 - 10 ** log10_beta2
        if optimizer_type == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(momentum, beta2))
        elif optimizer_type == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr,  momentum=momentum)
        else:
            raise
        workers = 0 if DEBUG else 6
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=int(batch_size),
                                                       shuffle=True, num_workers=workers)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=4,
                                                 shuffle=False, num_workers=workers)

    def train_and_eval(self):
        start = time()
        budget_seconds = 90
        running_loss = 0.0
        counter = 0
        for i, data in enumerate(cycle(self.trainloader), 0):
            inputs, labels = data
            self.optimizer.zero_grad()

            outputs = self.model(inputs.cuda())
            loss = self.criterion(outputs, labels.cuda())
            if torch.isnan(loss).any():
                #negative accuracy when something fails
                return -1.0
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            counter += 1
            if i % 500 == 499:
                # print('[%5d] loss: %.3f' % (i + 1, running_loss / counter))
                running_loss = 0.0
                counter = 0
            if time() - start > budget_seconds:
                # print('final -> [%5d] loss: %.3f' % (i + 1, running_loss / counter))
                break

        return self.eval()

    def eval(self):
        correct = 0.0
        total = 0.0
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                outputs = self.model(images.cuda())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.cuda()).sum().item()

        accuracy = 100 * correct / total
        return accuracy

def parallel_exec(draw):
    train_instance = Trainable_cifar10()
    train_instance.reset(*draw)
    return train_instance.train_and_eval()



if __name__ == "__main__":
    DEBUG = True
    config = load_config_file('cifar10-dom.json')
    domain, domain_orderings = config.domain, config.domain_orderings
    func_caller = CPFunctionCaller(None, domain, domain_orderings=domain_orderings)
    opt = gp_bandit.CPGPBandit(func_caller, ask_tell_mode=True)
    opt.initialise()

    parallel_jobs = 10
    train_instance = Trainable_cifar10()

    while True:
        results = {}
        if parallel_jobs > 0:
            draws = [opt.ask() for _ in range(parallel_jobs)]
            with Pool(parallel_jobs) as p:
                accuracies = p.map(parallel_exec, draws)
            # accuracies = [parallel_exec(d) for d in draws]
            [opt.tell([(d,a)]) for d,a in zip(draws, accuracies)]
            results.update({tuple(d):a for d,a in zip(draws, accuracies)})
            [print(d,' -> ', a) for d,a in zip(draws, accuracies)]
        else:
            draw = opt.ask()
            train_instance.reset(*draw)
            acc = train_instance.train_and_eval()
            print("input: ", draw)
            print("acc: ", acc)
            opt.tell([(draw, acc)])