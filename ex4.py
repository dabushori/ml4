import sys
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F


class MyFashionMNISTModel(torch.nn.Module):
    def __init__(self, layers, lr, bn=None, dos=None, image_size=784):
        super(MyFashionMNISTModel, self).__init__()
        self.image_size = image_size
        self.fc0 = torch.nn.Linear(image_size, layers[0])  # hidden layer
        self.bn0 = torch.nn.BatchNorm1d(
            layers[0]) if bn else None  # batch normalization
        self.do0 = torch.nn.Dropout(dos[0]) if dos else None  # dropout layer
        self.fc1 = torch.nn.Linear(layers[0], layers[1])  # hidden layer
        self.bn1 = torch.nn.BatchNorm1d(
            layers[1]) if bn else None  # batch normalization
        self.do1 = torch.nn.Dropout(dos[1]) if dos else None  # dropout layer
        self.fc2 = torch.nn.Linear(layers[1], 10)  # output layer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = self.fc0(x)
        x = self.bn0(x) if self.bn0 else x
        x = F.relu(x)
        x = self.do0(x) if self.do0 else x
        x = self.fc1(x)
        x = self.bn1(x) if self.bn1 else x
        x = F.relu(x)
        x = self.do1(x) if self.do1 else x
        return F.log_softmax(self.fc2(x), dim=1)


def train(model, train_loader):
    model.train()
    for (data, labels) in train_loader:
        model.optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, labels)
        loss.backward()
        model.optimizer.step()


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # test_loss += nll_loss(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, correct.item()


def predict(model, x):
    return model(x).max(1, keepdim=True)[1]


def main():
    # regular code
    if len(sys.argv) < 4:
        print('Usgae: python ex4.py <path_to_train_x_file> <path_to_train_y_file> <path_to_test_x_file> <path_to_test_y_file>')
    trainx_path, trainy_path, testx_path, testy_path = sys.argv[1:5]
    train_x_data = np.loadtxt(trainx_path)
    train_y_data = np.loadtxt(trainy_path, dtype=int)
    test_x_data = np.loadtxt(testx_path)

    # # [0, 1] normalization
    # train_x_data = train_x_data / 255
    # test_x_data = test_x_data / 255

    # # [-0.5, 0.5] normalization
    # train_x_data = train_x_data / 255 - 0.5
    # test_x_data = test_x_data / 255 - 0.5

    # mean-std normalization
    data = torch.from_numpy(train_x_data)
    mean, std = data.mean().item(), data.std().item()
    train_x_data = (train_x_data - mean) / std
    test_x_data = (test_x_data - mean) / std

    train_dataset = TensorDataset(torch.from_numpy(
        train_x_data).float(), torch.from_numpy(train_y_data).long())
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=128)

    '''
    Interesting results:
    mean-std norm + 256,256 layers + 0.2, 0.5 dropouts + batch-norm + 50 epochs: 90.1% acc
    
    '''

    epochs = 50
    layers = [256, 256]
    lr = 0.01
    bn = True
    dos = [0.2, 0.5]
    model = MyFashionMNISTModel(layers, lr, bn, dos)

    # train
    for i in range(1, epochs+1):
        print(f'epoch {i}')
        train(model, train_loader)

    # test
    model.eval()
    with open(testy_path, 'w') as test_y:
        yhat = np.array([predict(model, torch.from_numpy(x).float())
                         for x in test_x_data], dtype=int)
        np.savetxt(test_y, yhat, fmt='%d')

    # for debug
    y = np.loadtxt('test_labels')
    acc = (y == yhat).sum()
    print(f'model accuracy - {acc}/{len(y)} ({acc/len(y)*100}%)')


'''
Results (80:20 on train):
128,64 - 9813/11000 (89.209%)
256,128 - 9827/11000 (89.336%)
128,64 + dropout - 9789/11000 (88.99%)
256,128 + dropout - 9834/11000 (89.4%)
128,64 + batch normalization - 9813/11000 (89.209%)
256,128 + batch normalization - 9867/11000 (89.7%)
128,64 + dropout + batch normalization - 9818/11000 (89.254%)
256,128 + dropout + batch normalization - 9848/11000 (89.527%)

Results (train + test):
128,64 - 4356/5000 (87.12%)
256,128 - 4440/5000 (88.8%)
128,64 + dropout - 4446/5000 (88.92%)
256,128 + dropout - 4448/5000 (88.96%)
128,64 + batch normalization - XXX
256,128 + batch normalization - XXX
128,64 + dropout + batch normalization - XXX
256,128 + dropout + batch normalization - XXX

256,128 + dropout 0.2 + batch norm + [-0.5,0.5] norm - 4502/5000 (90.03999999999999%)
'''


if __name__ == '__main__':
    main()
