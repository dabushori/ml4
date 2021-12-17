import sys
import numpy as np
import torch
from torch._C import dtype
from torch.nn.functional import nll_loss
from torch.utils.data import TensorDataset, DataLoader

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

def main():
    if len(sys.argv) > 1:
        trainx_path, trainy_path = 'train_1000_x', 'train_1000_y' # I'll divide it 80:20 for train:test
    else:
        trainx_path, trainy_path = 'train_x', 'train_y' # I'll divide it 80:20 for train:test
    train_x_data = np.loadtxt(trainx_path)
    train_y_data = np.loadtxt(trainy_path, dtype=int)

    l = len(train_x_data)
    train_dataset = TensorDataset(torch.from_numpy(
        train_x_data[0:int(0.8 * l)]).float(), torch.from_numpy(train_y_data[0:int(0.8 * l)]).long())
    test_dataset = TensorDataset(torch.from_numpy(
        train_x_data[-int(0.2*l):]).float(), torch.from_numpy(train_y_data[-int(0.2*l):]).long())

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=16)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=16)
    
    models = [ModelA(), ModelB(), ModelC(), ModelD(True), ModelD(False), ModelE(), ModelF()]
    strs = ['Model A', 'Model B', 'Model C', 'Model D batch-norm before activation', 'Model D batch-norm after activation', 'Model E', 'Model F']
    for model, s in zip(models, strs):
        print(s)
        for _ in range(10):
            train(model, train_loader)
            # print('train: ', end='')
            # test(model, train_loader)
            # print('test: ', end='')
            test(model, test_loader)
        print()



if __name__ == '__main__':
    main()
