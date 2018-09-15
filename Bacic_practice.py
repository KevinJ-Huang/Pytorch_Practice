import torch

# x = torch.Tensor(2,3,4)
# print x
# print x.size()

# a = torch.rand(2,3,4)
# b = torch.rand(2,3,4)
# _=torch.add(a,b, out=x)
# print a
# print b
# print x

from torch.autograd import Variable
# x = torch.rand(5)
# x = Variable(x,requires_grad = True)
# y = x*2
# grads = torch.FloatTensor([1,2,3,4,5])
# y.backward(grads)
# print(x.grad)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(-1,self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self,x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
# print(net)
# print(len(list(net.parameters())))
input = Variable(torch.randn(1,1,32,32))

# print(out)
optimizer = optim.SGD(net.parameters(),lr = 0.01)

for i in range(1000):
    optimizer.zero_grad()
    output = net(input)
    loss = criterion(output,target)
    loss.backward()
    optimizer.step()
