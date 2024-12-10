import torch
import torch.nn as nn
import torch.nn.functional as F

def Split(x):
    c = int(x.size()[1])
    c1 = round(c * 0.5)
    x1 = x[:, :c1, :, :].contiguous()
    x2 = x[:, c1:, :, :].contiguous()
    return x1, x2

class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_prelu(output)

        return output

class conv3x3_resume(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()
        self.conv3x3 = Conv(nIn // 2, nIn // 2, kSize, 1, padding=1, bn_acti=True)
        self.conv1x1_resume = Conv(nIn // 2, nIn, 1, 1, padding=0, bn_acti=False)

    def forward(self, input):
        output = self.conv3x3(input)
        output = self.conv1x1_resume(output)
        return output

class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output

class Init_Block(nn.Module):
    def __init__(self):
        super(Init_Block, self).__init__()
        self.init_conv = nn.Sequential(
            Conv(3, 32, 3, 2, padding=1, bn_acti=True),
            Conv(32, 32, 3, 1, padding=1, bn_acti=True),
            Conv(32, 32, 3, 1, padding=1, bn_acti=True),
        )

    def forward(self, x):
        o = self.init_conv(x)
        return o

def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

class ConvBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class MDPCBlocks(nn.Module):
    def __init__(self, nIn, d=1, kSize=3, dkSize=3):
        super().__init__()

        self.conv3x3 = Conv(nIn, nIn // 2, kSize, 1, padding=1, bn_acti=True)
        self.dconv_left = Conv(nIn // 4, nIn // 4, (dkSize, dkSize), 1,
                               padding=(1, 1), groups=nIn // 4, bn_acti=True)
        self.dconv_right = Conv(nIn // 4, nIn // 4, (dkSize, dkSize), 1,
                                padding=(1 * d, 1 * d), groups=nIn // 4, dilation=(d, d), bn_acti=True)
        self.bn_relu_1 = BNPReLU(nIn)
        self.conv3x3_resume = conv3x3_resume(nIn , nIn , (dkSize, dkSize), 1,
                                padding=(1 , 1 ),  bn_acti=True)

    def forward(self, input):

        output = self.conv3x3(input)
        x1, x2 = Split(output)
        letf = self.dconv_left(x1)
        right = self.dconv_right(x2)
        output = torch.cat((letf, right), 1)
        output = self.conv3x3_resume(output)
        return self.bn_relu_1(output + input)

class MultiMDPC(nn.Module):
    def __init__(self, in_channels, scales=[1,2,4]):
        super().__init__()
        self.MDPC_modules = nn.ModuleList([MDPCBlocks(in_channels, d=s) for s in scales])

    def forward(self, x):
        outputs = [MDPC(x) for MDPC in self.MDPC_modules]
        return sum(outputs)

class FMWABlocks(nn.Module):
    def __init__(self, in_channels, feature_channels):
        super(FMWABlocks, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.attention_map = nn.Sequential(
            nn.Conv2d(feature_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        attention = self.attention_map(features)
        return x * attention + x  # Feature Reweighting and Fusion

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)
        self.mmdpc = MultiMDPC(out_channels)
        self.fmwa = FMWABlocks(out_channels, out_channels // 2)

    def forward(self, x):
        out = self.maxpool(x)
        out = self.nConvs(out)
        out = self.mmdpc(out)
        out = self.fmwa(out)
        return out

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, (2, 2), 2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        out = self.up(x)
        x = torch.cat([out, skip_x], dim=1)
        x = self.nConvs(x)
        return x

class MRSA_Net(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        if n_classes != 1:
            self.n_classes = n_classes + 1

        in_channels = 64
        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.down1 = DownBlock(in_channels, in_channels * 2, nb_Conv=2)
        self.down2 = DownBlock(in_channels * 2, in_channels * 4, nb_Conv=2)
        self.down3 = DownBlock(in_channels * 4, in_channels * 8, nb_Conv=2)
        self.down4 = DownBlock(in_channels * 8, in_channels * 8, nb_Conv=2)
        self.up4 = UpBlock(in_channels * 16, in_channels * 4, nb_Conv=2)
        self.up3 = UpBlock(in_channels * 8, in_channels * 2, nb_Conv=2)
        self.up2 = UpBlock(in_channels * 4, in_channels, nb_Conv=2)
        self.up1 = UpBlock(in_channels * 2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, self.n_classes, kernel_size=(1, 1))

        if n_classes == 1:
            self.last_activation = nn.Sigmoid()
        else:
            self.last_activation = None

    def forward(self, x):
        x = x.float()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)

        if self.last_activation is not None:
            logits = self.last_activation(self.outc(x))
        else:
            logits = self.outc(x)

        return logits