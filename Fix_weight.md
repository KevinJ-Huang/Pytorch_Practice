网络里固定一部分权重
```cpp
class UNet(nn.Module):

    def __init__(self):
        super(UNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 9, padding=4)
        self.blocks = nn.Sequential(
            ConvBlock(),
            ConvBlock(),
            ConvBlock(),
            ConvBlock(),
        )
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 3, 9, padding=4)
        for p in self.parameters():
            p.requires_grad = False
        self.conv_trans = nn.Conv2d(3, 16, 3, padding=1)
```
训练时，要有
```cpp
optimizer = optim.Adam(filter(lambda p: p.requires_grad,config.model.parameters()))
```
