import sys
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets
import torch.nn.functional as F


class MyFashionMNISTModel(torch.nn.Module):
    def __init__(self, image_size=784):
        super(MyFashionMNISTModel, self).__init__()
        self.image_size = image_size
        self.fc0 = torch.nn.Linear(image_size, 128) # hidden layer
        # self.do0 = torch.nn.Dropout(0.05) # dropout layer
        # self.bn0 = torch.nn.BatchNorm1d(128) # batch normalization
        self.fc1 = torch.nn.Linear(128, 64) # hidden layer
        # self.bn1 = torch.nn.BatchNorm1d(64) # batch normalization
        # self.do1 = torch.nn.Dropout(0.05) # dropout layer
        self.fc2 = torch.nn.Linear(64, 10) # output layer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = self.fc0(x)
        # x = self.bn0(x)
        x = F.relu(x)
        # x = self.do0(x)
        x = self.fc1(x)
        # x = self.bn1(x)
        x = F.relu(x)
        # x = self.do1(x)
        return F.log_softmax(self.fc2(x), dim=1)

# main function
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
    train_x_data = np.loadtxt(trainx_path) / 255
    train_y_data = np.loadtxt(trainy_path, dtype=int)
    test_x_data = np.loadtxt(testx_path) / 255

    # train_dataset = TensorDataset(torch.from_numpy(train_x_data).float(), torch.from_numpy(train_y_data).long())
    # train_loader = DataLoader(train_dataset, shuffle=True, batch_size=16)
    
    # model = MyFashionMNISTModel()
    # epochs = 50
    
    # # train
    # for _ in range(epochs):
    #     train(model, train_loader)

    # # test
    # with open(testy_path, 'w') as test_y:
    #     np.savetxt(test_y, np.array([predict(model, torch.from_numpy(x).float()) for x in test_x_data]), fmt='%d')
        
    
    # for debug
    l = len(train_x_data)
    train_dataset = TensorDataset(torch.from_numpy(
        train_x_data[0:int(0.8 * l)]).float(), torch.from_numpy(train_y_data[0:int(0.8 * l)]).long())
    test_dataset = TensorDataset(torch.from_numpy(
        train_x_data[-int(0.2*l):]).float(), torch.from_numpy(train_y_data[-int(0.2*l):]).long())

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=16)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=16)
    
    model = MyFashionMNISTModel()
    epochs = 50
    
    # train
    for i in range(epochs):
        print(f'epoch {i}')
        train(model, train_loader)
        test(model, test_loader)
        
'''
Results so far:
128,64 - 9813/11000
256,128 - 9827/11000
128,64 + dropout - 9789/11000
256,128 + dropout - 9834/11000
128,64 + batch normalization - 9813/11000
256,128 + batch normalization - 9867/11000
'''


if __name__ == '__main__':
    main()
