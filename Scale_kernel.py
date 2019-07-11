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
        for p in self.parameters():
            p.requires_grad = False
        self.trans1 = nn.Conv2d(3, 16, 3, padding=1)
        self.trans2 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 3, 9, padding=4)

    def forward(self, x):
        input = x
        x_trans1 = torch.mean(torch.mean(torch.mean(self.trans1(x),3),2),1)
        x_trans2 = torch.mean(torch.mean(torch.mean(self.trans2(x),3),2),1)
        x_out = []
        for i in range(1):
            x = input.narrow(0, i, 1)
            x = F.relu(self.conv1(x))
            x = self.blocks(x)

            weight1 = self.conv2.weight * x_trans1[i]
            weight2 = self.conv3.weight * x_trans2[i]

            x = F.relu(F.conv2d(x, weight1, bias = self.conv2.bias,stride=1, padding=1))
            x = F.relu(F.conv2d(x, weight2, bias = self.conv3.bias,stride=1, padding=1))
            x = self.conv4(x)
            x_out.append(x)
        out = torch.cat(x_out, dim=0)
        return (out*input).clamp(-0.05,1.05)
