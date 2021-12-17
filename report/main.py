import sys, os
import numpy as np
import torch
from torch.nn.functional import nll_loss
from torch.utils import data
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from models import *

# main function

# optimizer = optim.SGD(model.parameters(), lr=lr)
def train(model, train_loader):
    model.train()
    for (data, labels) in train_loader:
        model.optimizer.zero_grad()
        output = model(data)
        loss = nll_loss(output, labels)
        loss.backward()
        model.optimizer.step()
        
def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += nll_loss(output, target, reduction='sum').item()
            # test_loss += nll_loss(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).cpu().sum()
            
    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, correct.item()

def create_fig(model_string, epochs, test_loss, correct, test_correct, log_file):
    fig = plt.figure(figsize=(16,9))
    
    sub00 = fig.add_subplot(1, 2, 1)
    sub00.set_title(f'loss function per epoch - {model_string}')
    plt.plot(epochs, test_loss, 'ro')
    plt.xlabel('epochs')
    plt.ylabel('loss function')

    sub01 = fig.add_subplot(1, 2, 2)
    sub01.set_title(f'accuracy per epoch - {model_string}')
    plt.plot(epochs, correct, 'ro')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.savefig(os.path.join('research_data',f'{model_string}.png'), format='png')
    
    log_file.write(f'{model_string} - {test_correct}\n')
    
    

def main():
    if len(sys.argv) > 1:
        trainx_path, trainy_path = 'train_1000_x', 'train_1000_y' # I'll divide it 80:20 for train:test
    else:
        trainx_path, trainy_path = 'train_x', 'train_y' # I'll divide it 80:20 for train:test
    train_x_data = np.loadtxt(trainx_path) / 255
    train_y_data = np.loadtxt(trainy_path, dtype=int)

    l = len(train_x_data)
    train_dataset = TensorDataset(torch.from_numpy(
        train_x_data[0:int(0.8 * l)]).float(), torch.from_numpy(train_y_data[0:int(0.8 * l)]).long())
    test_dataset = TensorDataset(torch.from_numpy(
        train_x_data[-int(0.2*l):]).float(), torch.from_numpy(train_y_data[-int(0.2*l):]).long())

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=16)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=16)
    fmnist_test_loader = DataLoader(datasets.FashionMNIST('./data', train=False, download=True, transform=transforms.ToTensor()), batch_size=16)
    
    models = [ModelA(), ModelB(), ModelC(), ModelD(True), ModelD(False), ModelE(), ModelF()]
    strs = ['Model A', 'Model B', 'Model C', 'Model D batch-norm before activation', 'Model D batch-norm after activation', 'Model E', 'Model F']
    
    with open(os.path.join('research_data', 'test_correct.txt'), 'w') as log_file:
        data_per_epoch = dict()
        for model, s in zip(models, strs):
            print(s)
            for epoch in range(10):
                train(model, train_loader)
                # print('train: ', end='')
                # test(model, train_loader)
                # print('test: ', end='')
                loss, correct = test(model, test_loader)
                data_per_epoch[epoch] = (loss, correct / len(test_loader.dataset))
            
            
            _, test_correct = test(model, fmnist_test_loader)
            test_correct /= len(fmnist_test_loader.dataset)
            
            
            loss, correct = zip(*data_per_epoch.values())
            loss = list(loss)
            correct = list(correct)
            create_fig(s, range(1,11), loss, correct, test_correct, log_file)
            print(f'{s} - {test_correct}')
        
        



if __name__ == '__main__':
    main()
