import sys
import numpy as np
import torch
from torch.functional import Tensor
from torch.nn.functional import nll_loss
from torchvision import transforms, datasets
from torch.utils.data import TensorDataset, DataLoader

from models import Model1

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
            test_loss += nll_loss(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).cpu().sum()
            
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():

    trainx_path, trainy_path, testx_path, testy_path = 'train_x', 'train_y', 'test_x', 'test_y'
    train_x_data = np.loadtxt(trainx_path)
    train_y_data = np.loadtxt(trainy_path)

    l = len(train_x_data)
    train_dataset = TensorDataset(torch.from_numpy(
        train_x_data[0:int(0.8 * l)]), torch.from_numpy(train_y_data[0:int(0.8 * l)]))
    test_dataset = TensorDataset(torch.from_numpy(
        train_x_data[-int(0.2*l):]), torch.from_numpy(train_y_data[-int(0.2*l):]))

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=16)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=16)
    
    model = Model1()
    train(model, train_loader)
    test(model, test_loader)



if __name__ == '__main__':
    main()
