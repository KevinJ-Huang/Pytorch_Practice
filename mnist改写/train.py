import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import argparse
import torchvision.transforms as transforms
import torchvision.datasets as dsets

train_dataset = dsets.MNIST(root = '../../data_sets/mnist',
                           train = True,
                           transform = transforms.ToTensor(),
                           download = True)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5,stride=1,padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5,stride=1,padding=2)
        self.fc1 = nn.Linear(7*7*64, 1024)
        self.fc2 = nn.Linear(1024,10)

    def forward(self, x, target = None):
        x = F.max_pool2d(F.relu(self.conv1(x)),kernel_size=2,stride=2,padding=0)
        x = F.max_pool2d(F.relu(self.conv2(x)),kernel_size=2,stride=2,padding=0)
        x = x.view(-1,7*7*64)
        x = F.relu(self.fc1(x))
        x = F.dropout(x,training=True)
        x = self.fc2(x)
        x = F.log_softmax(x,dim=1)
        return x

model = Net()

def main(args):
    if args.cuda:
        model.cuda()
    criterion = nn.CrossEntropyLoss()
    training_accuracy = 0
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = args.batch_size,
                                           shuffle = True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    model.train()
    for epochs in range(1,20000):
        for i,(images,labels) in enumerate(train_loader):
            if args.cuda:
                images, labels = images.cuda(), labels.cuda()
            images = Variable(images.view(-1,1,28,28))
            labels = Variable(labels)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            if i%100 == 0:
                pred = outputs.data.max(1)[1]
                training_accuracy += pred.eq(labels.data[1]).sum()
                print('epochs:%d,Loss:%5f,Acuuracy:%g'%(epochs,loss,training_accuracy))
                training_accuracy = 0


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='learning rate for the stochastic gradient update.')
    parser.add_argument('--batch_size', type=int, default=50, help='input batch size for training (default: 64)')
    parser.add_argument('--cuda', action='store_true', default=True,help='Enable CUDA training')
    args = parser.parse_args()
    main(args)




