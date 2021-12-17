import torch
import torch.nn.functional as F

'''
Model A - Neural Network with two hidden layers, the first layer should
have a size of 100 and the second layer should have a size of 50, both
should be followed by ReLU activation function. Train this model with
SGD optimizer.
'''
class ModelA(torch.nn.Module):
    def __init__(self, image_size=784):
        super(ModelA, self).__init__()
        self.image_size = image_size
        self.fc0 = torch.nn.Linear(image_size, 100)
        self.fc1 = torch.nn.Linear(100, 50)
        self.fc2 = torch.nn.Linear(50, 10)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)
    
'''
Model B - Neural Network with two hidden layers, the first layer should
have a size of 100 and the second layer should have a size of 50, both
should be followed by ReLU activation function, train this model with
ADAM optimizer.
'''
class ModelB(torch.nn.Module):
    def __init__(self, image_size=784):
        super(ModelB, self).__init__()
        self.image_size = image_size
        self.fc0 = torch.nn.Linear(image_size, 100)
        self.fc1 = torch.nn.Linear(100, 50)
        self.fc2 = torch.nn.Linear(50, 10)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)

'''
Model C - Dropout - add dropout layers to model B. You should place
the dropout on the output of the hidden layers.
'''
class ModelC(torch.nn.Module):
    def __init__(self, image_size=784):
        super(ModelC, self).__init__()
        self.image_size = image_size
        self.fc0 = torch.nn.Linear(image_size, 100)
        self.fc1 = torch.nn.Linear(100, 50)
        self.dropout0 = torch.nn.Dropout(0.05)
        self.fc2 = torch.nn.Linear(50, 10)
        self.dropout1 = torch.nn.Dropout(0.05)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = self.dropout0(x)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        return F.log_softmax(self.fc2(x), dim=1)
    
'''
Model D - Batch Normalization - add Batch Normalization layers to
model B. Where should you place the Batch Normalization layer? before the activation functions or after? try both cases and report the
results in the report file.
'''
class ModelD(torch.nn.Module):
    def __init__(self, bn_before_activation, image_size=784):
        super(ModelD, self).__init__()
        self.bn_before_activation = bn_before_activation
        self.image_size = image_size
        self.fc0 = torch.nn.Linear(image_size, 100)
        self.bn0 = torch.nn.BatchNorm1d(100)
        self.fc1 = torch.nn.Linear(100, 50)
        self.bn1 = torch.nn.BatchNorm1d(50)
        self.fc2 = torch.nn.Linear(50, 10)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.bn0(self.fc0(x))) if self.bn_before_activation else self.bn0(F.relu(self.fc0(x)))
        x = F.relu(self.bn1(self.fc1(x))) if self.bn_before_activation else self.bn1(F.relu(self.fc1(x)))
        return F.log_softmax(self.fc2(x), dim=1)
    
'''
Model E - Neural Network with five hidden layers:[128,64,10,10,10] using ReLU .
'''
class ModelE(torch.nn.Module):
    def __init__(self, image_size=784):
        super(ModelE, self).__init__()
        self.image_size = image_size
        self.fc0 = torch.nn.Linear(image_size, 128)
        self.fc1 = torch.nn.Linear(128, 64)
        self.fc2 = torch.nn.Linear(64, 10)
        self.fc3 = torch.nn.Linear(10, 10)
        self.fc4 = torch.nn.Linear(10, 10)
        self.fc5 = torch.nn.Linear(10, 10)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return F.log_softmax(self.fc5(x), dim=1)
    
'''
Model F - Neural Network with five hidden layers:[128,64,10,10,10] using Sigmoid.
'''
class ModelF(torch.nn.Module):
    def __init__(self, image_size=784):
        super(ModelF, self).__init__()
        self.image_size = image_size
        self.fc0 = torch.nn.Linear(image_size, 128)
        self.fc1 = torch.nn.Linear(128, 64)
        self.fc2 = torch.nn.Linear(64, 10)
        self.fc3 = torch.nn.Linear(10, 10)
        self.fc4 = torch.nn.Linear(10, 10)
        self.fc5 = torch.nn.Linear(10, 10)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = torch.sigmoid(self.fc0(x))
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return F.log_softmax(self.fc5(x), dim=1)