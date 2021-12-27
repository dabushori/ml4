import sys
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
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
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    # print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))
    return test_loss, correct.item()


def predict(model, x):
    return model(x).max(1, keepdim=True)[1]


def main():
    # regular code
    if len(sys.argv) < 4:
        print('Usgae: python ex4.py <path_to_train_x_file> <path_to_train_y_file> <path_to_test_x_file> <path_to_test_y_file>')
        exit(0)
    trainx_path, trainy_path, testx_path, testy_path = sys.argv[1:5]
    train_x_data = np.loadtxt(trainx_path)
    train_y_data = np.loadtxt(trainy_path, dtype=int)
    test_x_data = np.loadtxt(testx_path)

    state_path = f'{testy_path}_state_dict'

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

    val_len = int(len(train_x_data) / 10)
    val_x_data = train_x_data[:val_len]
    train_x_data = train_x_data[val_len:]
    val_y_data = train_y_data[:val_len]
    train_y_data = train_y_data[val_len:]

    train_dataset = TensorDataset(torch.from_numpy(
        train_x_data).float(), torch.from_numpy(train_y_data).long())
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=128)

    val_dataset = TensorDataset(torch.from_numpy(
        val_x_data).float(), torch.from_numpy(val_y_data).long())
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=128)

    epochs = 50
    layers = [256, 256]
    lr = 0.01
    bn = True
    dos = [0.2, 0.5]
    model = MyFashionMNISTModel(layers, lr, bn, dos)

    # train
    # min_loss = np.inf
    max_correct = -np.inf
    for i in range(1, epochs+1):
        print(f'epoch {i}')
        train(model, train_loader)
        loss, correct = test(model, val_loader)
        # if loss < min_loss:
        #     print(f'updating: {min_loss = }, {loss = }')
        #     min_loss = loss
        #     torch.save(model.state_dict(), state_path)
        if correct > max_correct:
            print(f'updating: {max_correct = }, {correct = }')
            max_correct = correct
            # save the model if upgraded
            torch.save(model.state_dict(), state_path)

    # load the best model
    model.load_state_dict(torch.load(state_path))

    # test
    model.eval()
    with open(testy_path, 'w') as test_y:
        yhat = np.array([predict(model, torch.from_numpy(x).float())
                         for x in test_x_data], dtype=int)
        np.savetxt(test_y, yhat, fmt='%d')

    # for debug
    # y = np.loadtxt('test_labels')
    # acc = (y == yhat).sum()
    # print(f'model accuracy - {acc}/{len(y)} ({acc/len(y)*100}%)')


if __name__ == '__main__':
    main()
