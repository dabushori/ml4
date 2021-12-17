import sys
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn.functional as F


class MyFashionMNISTModel(torch.nn.Module):
    def __init__(self, image_size=784):
        super(MyFashionMNISTModel, self).__init__()
        self.image_size = image_size
        self.fc0 = torch.nn.Linear(image_size, 128) # hidden layer
        # self.do0 = torch.nn.Dropout(0.05) # dropout layer
        self.fc1 = torch.nn.Linear(128, 64) # hidden layer
        # self.do1 = torch.nn.Dropout(0.05) # dropout layer
        self.fc2 = torch.nn.Linear(64, 10) # output layer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        # x = self.do0(x)
        x = F.relu(self.fc1(x))
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

def predict(model, x):
    return model(x).max(1, keepdim=True)[1]

def main():
    if len(sys.argv) < 4:
        print('Usgae: python ex4.py <path_to_train_x_file> <path_to_train_y_file> <path_to_test_x_file> <path_to_test_y_file>')
    trainx_path, trainy_path, testx_path, testy_path = sys.argv[1:5]
    train_x_data = np.loadtxt(trainx_path) / 255
    train_y_data = np.loadtxt(trainy_path, dtype=int)
    test_x_data = np.loadtxt(testx_path) / 255

    train_dataset = TensorDataset(torch.from_numpy(train_x_data).float(), torch.from_numpy(train_y_data).long())
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=16)
    
    model = MyFashionMNISTModel()
    epochs = 50
    
    # train
    for _ in range(epochs):
        train(model, train_loader)

    # test
    with open(testy_path, 'w') as test_y:
        np.savetxt(test_y, np.array([predict(model, torch.from_numpy(x).float()) for x in test_x_data]), fmt='%d')
    
        



if __name__ == '__main__':
    main()
