import sys

sys.path.append(r"C:\Users\wyyaa123\Desktop\IMUdeblur")

import torch
import torch.nn as nn
from model.mobilenet_v2 import MobileNetV2
from mmcv.ops import ModulatedDeformConv2d, ModulatedDeformConv2dPack, DeformConv2d, DeformConv2dPack

class FPNHead(nn.Module):
    def __init__(self, num_in, num_mid, num_out):
        super().__init__()

        self.block0 = nn.Conv2d(num_in, num_mid, kernel_size=3, padding=1, bias=False)
        self.block1 = nn.Conv2d(num_mid, num_out, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = nn.functional.relu(self.block0(x), inplace=True)
        x = nn.functional.relu(self.block1(x), inplace=True)
        return x

class FPN(nn.Module):

    def __init__(self, norm_layer, num_filters=128, pretrained=True):
        """Creates an `FPN` instance for feature extraction.
        Args:
          num_filters: the number of filters in each output pyramid level
          pretrained: use ImageNet pre-trained backbone feature extractor
        """

        super().__init__()
        net = MobileNetV2(n_class=1000)

        if pretrained:
            #Load weights into the project directory
            state_dict = torch.load('./mobilenet_v2.pth.tar', map_location='cpu') # add map_location='cpu' if no gpu
            net.load_state_dict(state_dict)
        self.features = net.features

        self.enc0 = nn.Sequential(*self.features[0:2])
        self.enc1 = nn.Sequential(*self.features[2:4])
        self.enc2 = nn.Sequential(*self.features[4:7])
        self.enc3 = nn.Sequential(*self.features[7:11])
        self.enc4 = nn.Sequential(*self.features[11:16])

        self.td1 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                 norm_layer(num_filters),
                                 nn.ReLU(inplace=True))
        self.td2 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                 norm_layer(num_filters),
                                 nn.ReLU(inplace=True))
        self.td3 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                 norm_layer(num_filters),
                                 nn.ReLU(inplace=True))
        
        self.lateral4 = nn.Conv2d(160, num_filters, kernel_size=1, bias=False)
        self.lateral3 = nn.Conv2d(64, num_filters, kernel_size=1, bias=False)
        self.lateral2 = nn.Conv2d(32, num_filters, kernel_size=1, bias=False)

        self.lateral1 = nn.Sequential(nn.Conv2d(24, num_filters, kernel_size=3, padding=2, dilation=2), 
                                      nn.Conv2d(num_filters, num_filters, kernel_size=1, bias=False))
        
        self.lateral0 = nn.Sequential(nn.Conv2d(16, num_filters // 2, kernel_size=3, padding=3, dilation=3), 
                                      nn.Conv2d(num_filters // 2, num_filters // 2, kernel_size=1, bias=False))

        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.features.parameters():
            param.requires_grad = True


    def forward(self, x):

        # Bottom-up pathway, from MobileNet
        enc0 = self.enc0(x) #outp: 16

        enc1 = self.enc1(enc0) # outp: 24

        enc2 = self.enc2(enc1) # outp: 32

        enc3 = self.enc3(enc2) # outp: 64

        enc4 = self.enc4(enc3) # outp: 160

        # Lateral connections

        lateral4 = self.lateral4(enc4)
        lateral3 = self.lateral3(enc3)
        lateral2 = self.lateral2(enc2)
        lateral1 = self.lateral1(enc1)
        lateral0 = self.lateral0(enc0)

        # Top-down pathway
        map4 = lateral4
        map3 = self.td1(lateral3 + nn.functional.upsample(map4, scale_factor=2, mode="nearest"))
        map2 = self.td2(lateral2 + nn.functional.upsample(map3, scale_factor=2, mode="nearest"))
        map1 = self.td3(lateral1 + nn.functional.upsample(map2, scale_factor=2, mode="nearest"))
        return lateral0, map1, map2, map3, map4


if __name__ == "__main__":

    # net = FPN(nn.BatchNorm2d)
    net = MobileNetV2(1000)

    input = torch.randn((1, 3, 224, 224))
    net = nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1, dilation=2)

    # outps = net(input)

    print (*net.features[2: 4])
