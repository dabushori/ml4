import torch

'''
Model A - Neural Network with two hidden layers, the first layer should
have a size of 100 and the second layer should have a size of 50, both
should be followed by ReLU activation function. Train this model with
SGD optimizer.
'''
class Model1(torch.nn.Module):
    def __init__(self, image_size=784):
        super(Model1, self).__init__()
        self.image_size = image_size
        self.fc0 = torch.nn.Linear(image_size, 100)
        self.fc1 = torch.nn.Linear(100, 50)
        self.fc2 = torch.nn.Linear(50, 10)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = torch.nn.functional.relu(self.fc0(x))
        x = torch.nn.functional.relu(self.fc1(x))
        return torch.nn.functional.log_softmax(self.fc2(x), dim=1)
    
    
    