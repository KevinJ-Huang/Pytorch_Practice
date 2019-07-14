1.双路网络
```cpp
class ConvBlock(nn.Module):

    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.instance_norm1 = nn.InstanceNorm2d(16, affine=True)
        self.instance_norm2 = nn.InstanceNorm2d(16, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.relu(self.instance_norm1(self.conv1(x)))
        y = self.relu(self.instance_norm2(self.conv2(y))) + x
        return y





class Encoder(nn.Module):
    def __init__(self,decoder1,decoder2):
        super(Encoder, self).__init__()
        self.decoder1 = decoder1
        self.decoder2 = decoder2
        self.conv1 = nn.Conv2d(3, 16, 9, padding=4)
        self.blocks = nn.Sequential(
            ConvBlock(),
            ConvBlock(),
            ConvBlock(),
            ConvBlock(),
        )

    def forward(self, input1, input2):
        x = self.conv1(input1)
        x = self.blocks(x)
        x = self.decoder1(input1, x)

        x2 = self.conv1(input2)
        x2 = self.blocks(x2)
        x2 = self.decoder2(input2, x2)

        return x,x2

    def forward_test(self,input):
        x = self.conv1(input)
        x = self.blocks(x)
        x = self.decoder2(input, x)
        return x


class UNet_pre(nn.Module):
    def __init__(self):
        super(UNet_pre, self).__init__()
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 3, 9, padding=4)

    def forward(self, input, x):
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return (x * input).clamp(-0.05, 1.05)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 3, 9, padding=4)

    def forward(self, input, x):
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return (x * input).clamp(-0.05, 1.05)

```
