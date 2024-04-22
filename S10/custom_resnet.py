class DavidResNet(nn.Module):
    def __init__(self):
        super(DavidResNet, self).__init__()
        # CONVOLUTION BLOCK 1
        self.prep_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv1 = self.conv_block(64, 128)
        self.resnet_b1 = self.resnet_block(128)
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv2 = self.conv_block(256, 512)
        self.resnet_b2 = self.resnet_block(512)
        self.pool1 = nn.MaxPool2d(4,4)
        self.fc = nn.Linear(in_features=512, out_features=10)

    def conv_block(self, n_in, n_out):
        X = nn.Sequential(
            nn.Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(n_out),
            nn.ReLU()
        )
        return X

    def resnet_block(self, n_out):
      res_block = nn.Sequential(
          nn.Conv2d(in_channels=n_out, out_channels=n_out, kernel_size=(3, 3), padding=1, bias=False),
          nn.BatchNorm2d(n_out),
          nn.ReLU(),
          nn.Conv2d(in_channels=n_out, out_channels=n_out, kernel_size=(3, 3), padding=1, bias=False),
          nn.BatchNorm2d(n_out),
          nn.ReLU()
      )
      return res_block



    def forward(self, x):
        x = self.prep_layer(x)
        x1 = self.conv1(x)
        x2 = self.resnet_b1(x1)
        x = x1 + x2
        x = self.layer2(x)
        x1 = self.conv2(x)
        x2 = self.resnet_b2(x1)
        x = x1 + x2
        x = self.pool1(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)
