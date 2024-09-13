import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


from torchviz import make_dot

# https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html - documetnation on how to make a pytorch model

# So this is the triple convolution, chat gpt says we should use normalization dont know if we should keep it
class TripleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TripleConv, self).__init__()
        self.triple_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.triple_conv(x)
# the down module is what the unet uses durint hte firlst half 
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.conv_pool = nn.Sequential(
            
            TripleConv(in_channels, out_channels),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.conv_pool(x)        


class UpConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConvTranspose, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = TripleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # adding from the down section
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
class UpBilinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBilinear, self).__init__()
        # the bilinear is provided by the nn module, we set the mode to bilinear here
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = TripleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Padding if necessary to handle odd-sized inputs
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
# this is teh final convolution that is in the unet with a 1x1 kernel
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Unet(nn.Module):
    def __init__(self, n_channels, n_classes, variant='convtranspose'):
        super(Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.variant = variant

        self.inc = TripleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        
        if variant == 'convtranspose':
            self.up1 = UpConvTranspose(1024, 512)
            self.up2 = UpConvTranspose(512, 256)
            self.up3 = UpConvTranspose(256, 128)
            self.up4 = UpConvTranspose(128, 64)
        elif variant == 'bilinear':
            self.up1 = UpBilinear(1024, 512)
            self.up2 = UpBilinear(512, 256)
            self.up3 = UpBilinear(256, 128)
            self.up4 = UpBilinear(128, 64)
        
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        return logits
    


model = Unet(n_channels=3, n_classes=2, variant='convtranspose')  # Example: 3 input channels, 2 output classes

x = torch.randn(1, 3, 128, 128)  # Batch size of 1, 3 channels (RGB), 128x128 image size
y = model(x)

writer = SummaryWriter()
writer.add_graph(model, x)
writer.close()


# run tensorboard --logdir=runs to see network
