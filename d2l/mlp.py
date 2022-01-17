import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

num_inputs, num_hiddens, num_outputs = 784, 256, 10
W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens)*0.01, requires_grad=True)
b1 = nn.Parameter(torch.zeros(num_hiddens), requires_grad=True)
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs)*0.01, requires_grad=True)
b2 = nn.Parameter(torch.zeros(num_outputs), requires_grad=True)
params = [W1, b1, W2, b2]


def net(X):
    X = X.reshape(-1, num_inputs)
    H = relu(X@W1 + b1)
    return (H@W2 + b2)


def relu(x):
    return torch.max(x, torch.zeros_like(x))


loss = nn.CrossEntropyLoss(reduction='none')
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params=params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)


