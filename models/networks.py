import torch
from torch import nn
from torchvision import models
import sys
sys.path.insert(1, '../Multi-Scale-Attention-master/src')
print(sys.path)
from models.my_stacked_danet import DAF_stack
# sys.path.insert(1, '../Swin-Unet-main/')
# from networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
def conv3x3d(in_, out):
    return nn.Conv3d(in_, out, 3, padding=1)

class ConvRelu3d(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3d(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class DecoderBlock3d(nn.Module):      #upscaling block
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvRelu3d(in_channels, middle_channels),
            nn.ConvTranspose3d(middle_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),        #upscaling layer
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.block(x)
    
"""VGG11 is
Sequential(
  (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): ReLU(inplace=True)
  (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (4): ReLU(inplace=True)
  (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (6): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (7): ReLU(inplace=True)
  (8): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (9): ReLU(inplace=True)
  (10): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (11): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (12): ReLU(inplace=True)
  (13): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (14): ReLU(inplace=True)
  (15): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (16): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (17): ReLU(inplace=True)
  (18): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (19): ReLU(inplace=True)
  (20): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
)
"""

class UNet113d(nn.Module):          #3D U-Net based on VGG11
    def __init__(self, num_filters=32, num_in = 1, num_out = 1):

        super().__init__()

        self.pool = nn.MaxPool3d(2, 2)      #downscaling layer
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(num_in, num_filters * 2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(num_filters * 2, num_filters * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv3s = nn.Conv3d(num_filters * 4, num_filters * 8, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv3 = nn.Conv3d(num_filters * 8, num_filters * 8, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv4s = nn.Conv3d(num_filters * 8, num_filters * 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv4 = nn.Conv3d(num_filters * 16, num_filters * 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv5s = nn.Conv3d(num_filters * 16, num_filters * 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv5 = nn.Conv3d(num_filters * 16, num_filters * 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.center = DecoderBlock3d(num_filters * 8 * 2, num_filters * 8 * 2, num_filters * 8)
        self.dec5 = DecoderBlock3d(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 8)
        self.dec4 = DecoderBlock3d(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 4)
        self.dec3 = DecoderBlock3d(num_filters * (8 + 4), num_filters * 4 * 2, num_filters * 2)
        self.dec2 = DecoderBlock3d(num_filters * (4 + 2), num_filters * 2 * 2, num_filters)
        self.dec1 = ConvRelu3d(num_filters * (2 + 1), num_filters)
        self.final = nn.Conv3d(num_filters, num_out, kernel_size=1)

    def forward(self, x):                                   # (batch,1,x,y)
        conv1 = self.relu(self.conv1(x))                    # (batch,64,x,y)
        conv2 = self.relu(self.conv2(self.pool(conv1)))     # (batch,128,x/2,y/2)
        conv3s = self.relu(self.conv3s(self.pool(conv2)))   # (batch,256,x/4,y/4)
        conv3 = self.relu(self.conv3(conv3s))               # (batch,256,x/4,y/4)
        conv4s = self.relu(self.conv4s(self.pool(conv3)))   # (batch,512,x/8,y/8)
        conv4 = self.relu(self.conv4(conv4s))               # (batch,512,x/8,y/8)
        conv5s = self.relu(self.conv5s(self.pool(conv4)))   # (batch,512,x/16,y/16)
        conv5 = self.relu(self.conv5(conv5s))               # (batch,512,x/16,y/16)

        center = self.center(self.pool(conv5))              # (batch,256,x/16,y/16)
        
        dec5 = self.dec5(torch.cat([center, conv5], 1))     # (batch,256,x/8,y/8)
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))       # (batch,128,x/4,y/4)
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))       # (batch,64,x/2,y/2)
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))       # (batch,32,x,y)
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))       # (batch,32,x,y)
        return self.final(dec1)                             # (batch,1,x,y)


class UNetSim3d(nn.Module):                 #Simplified 3D U-Net
    def __init__(self, num_filters=32, num_in = 1, num_out = 1):
        super().__init__()

        self.pool = nn.MaxPool3d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(num_in, num_filters * 2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(num_filters * 2, num_filters * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.center = DecoderBlock3d(num_filters * 4, num_filters * 2, num_filters * 1)
        self.dec2 = DecoderBlock3d(num_filters * (4 + 1), num_filters * 1, num_filters * 1)       #in,middle,out
        self.dec1 = ConvRelu3d(num_filters * (2 + 1), num_filters//2)
        self.final = nn.Conv3d(num_filters//2, num_out, kernel_size=1)

    def forward(self, x):                                   # (batch,1,x,y) - (batch,filters,x,y)
        conv1 = self.relu(self.conv1(x))                    # (batch,64,x,y)
        conv2 = self.relu(self.conv2(self.pool(conv1)))     # (batch,128,x/2,y/2)
        
        center = self.center(self.pool(conv2))              # (batch,32,x/2,y/2)
        
        dec2 = self.dec2(torch.cat([center, conv2], 1))     # (batch,32,x,y)
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))       # (batch,16,x,y)
        out = self.final(dec1)                              # (batch,1,x,y)
        return out
    
    
class UNetSim3dSRx2(nn.Module):                 #Simplified 3D 2x upscaling U-Net
    def __init__(self, num_filters=32, num_in = 1, num_out =1):
        super().__init__()
        self.pool = nn.MaxPool3d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(num_in, num_filters * 2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(num_filters * 2, num_filters * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.center = DecoderBlock3d(num_filters * 4, num_filters * 2, num_filters * 1)
        self.dec2 = DecoderBlock3d(num_filters * (4 + 1), num_filters * 2, num_filters * 2)       #in,middle,out
        self.dec1 = DecoderBlock3d(num_filters * 4, num_filters * 1, num_filters // 2)
        self.final = nn.Conv3d(num_filters//2, num_out, kernel_size=1)

    def forward(self, x):                                   # (batch,1,x,y) - (batch,filters,x,y)
        conv1 = self.relu(self.conv1(x))                    # (batch,64,x,y)
        conv2 = self.relu(self.conv2(self.pool(conv1)))     # (batch,128,x/2,y/2)
        
        center = self.center(self.pool(conv2))              # (batch,32,x/2,y/2)
        
        dec2 = self.dec2(torch.cat([center, conv2], 1))     # (batch,64,x,y)
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))       # (batch,16,x*2,y*2)
        out = self.final(dec1)                              # (batch,3,x*2,y*2)
        return out
    
class UNetSim3dSRd2(nn.Module):                 #Simplified 3D 2x downscaling U-Net
    def __init__(self, num_filters=32, num_in = 1, num_out = 1):
        super().__init__()
        self.pool = nn.MaxPool3d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(num_in, num_filters * 2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(num_filters * 2, num_filters * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv3s = nn.Conv3d(num_filters * 4, num_filters * 8, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv3 = nn.Conv3d(num_filters * 8, num_filters * 8, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.center = DecoderBlock3d(num_filters * 4 * 2, num_filters * 2, num_filters * 1)
        self.dec2 = DecoderBlock3d(num_filters * (4 * 2 + 1), num_filters * 1, num_filters * 1)       #in,middle,out
        self.dec1 = ConvRelu3d(num_filters * (2 * 2 + 1), num_filters//2)
        self.final = nn.Conv3d(num_filters//2, num_out, kernel_size=1)

    def forward(self, x):                                   # (batch,1,x,y) - (batch,filters,x,y)
        conv1 = self.relu(self.conv1(x))                    # (batch,64,x,y)
        conv2 = self.relu(self.conv2(self.pool(conv1)))     # (batch,128,x/2,y/2)
        conv3s = self.relu(self.conv3s(self.pool(conv2)))   # (batch,256,x/4,y/4)
        conv3 = self.relu(self.conv3(conv3s))               # (batch,256,x/4,y/4)
        
        center = self.center(self.pool(conv3))              # (batch,32,x/4,y/4)
        
        dec2 = self.dec2(torch.cat([center, conv3], 1))     # (batch,32,x/2,y/2)
        dec1 = self.dec1(torch.cat([dec2, conv2], 1))       # (batch,16,x/2,y/2)
        out = self.final(dec1)                              # (batch,1,x/2,y/2)
        return out
        
    
class UNetSim3dSRx4(nn.Module):                 #Simplified 3D 4x upscaling U-Net
    def __init__(self, num_filters=32, num_in = 1, num_out = 1):
        super().__init__()

        self.pool = nn.MaxPool3d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(num_in, num_filters * 2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(num_filters * 2, num_filters * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.center = DecoderBlock3d(num_filters * 4, num_filters * 2, num_filters * 1)
        self.dec2 = DecoderBlock3d(num_filters * (4 + 1), num_filters * 2, num_filters * 2)       #in,middle,out
        self.dec2s = DecoderBlock3d(num_filters * (2+1), num_filters * 1, num_filters * 2)       #in,middle,out
        self.dec1 = DecoderBlock3d(num_filters * 4, num_filters * 1, num_filters // 2)
        self.final = nn.Conv3d(num_filters//2, num_out, kernel_size=1)

    def forward(self, x):                                   # (batch,num_in,x,y)
        conv1 = self.relu(self.conv1(x))                    # (batch,64,x,y)
        conv2 = self.relu(self.conv2(conv1))                # (batch,128,x,y)
        
        center = self.center(self.pool(conv2))              # (batch,32,x,y)
        
        dec2 = self.dec2(torch.cat([center, conv2], 1))     # (batch,64,x*2,y*2)
        dec2s = self.dec2s(torch.cat([center,conv1],1))     # (batch,64,x*2,y*2)
        dec1 = self.dec1(torch.cat([dec2, dec2s], 1))       # (batch,16,x*4,y*4)
        out = self.final(dec1)                              # (batch,3,x*4,y*4)
        return out
    
    
class UNetSim3dSRd4(nn.Module):                 #Simplified 3D 4x downscaling U-Net
    def __init__(self, num_filters=32, num_in = 1, num_out =1):
        super().__init__()
        
        self.pool = nn.MaxPool3d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        #self.conv0 = ConvRelu3d(num_in, 8)
        self.conv1 = nn.Conv3d(num_in, num_filters * 2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(num_filters * 2, num_filters * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv3s = nn.Conv3d(num_filters * 4, num_filters * 8, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv3 = nn.Conv3d(num_filters * 8, num_filters * 8, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.center = DecoderBlock3d(num_filters * 4 * 2, num_filters * 2, num_filters * 1)
        self.dec2 = DecoderBlock3d(num_filters * (4 * 2 + 1), num_filters * 1, num_filters * 1)       #in,middle,out
        self.dec1 = ConvRelu3d(num_filters * (2 * 2 + 1), num_filters//2)
        self.final = nn.Conv3d(num_filters//2, num_out, kernel_size=1)

    def forward(self, x):                                   # (batch,3,x,y)
        #conv0 = self.conv0(x)                              # (batch,8,x,y)
        conv1 = self.relu(self.conv1(self.pool(x)))         # (batch,64,x/2,y/2)
        conv2 = self.relu(self.conv2(self.pool(conv1)))     # (batch,128,x/4,y/4)
        conv3s = self.relu(self.conv3s(self.pool(conv2)))   # (batch,256,x/8,y/8)
        conv3 = self.relu(self.conv3(conv3s))               # (batch,256,x/8,y/8)
        
        center = self.center(self.pool(conv3))              # (batch,32,x/8,y/8)
        
        dec2 = self.dec2(torch.cat([center, conv3], 1))     # (batch,32,x/4,y/4)
        dec1 = self.dec1(torch.cat([dec2, conv2], 1))       # (batch,16,x/4,y/4)
        out = self.final(dec1)                              # (batch,1,x/4,y/4)
        return out

