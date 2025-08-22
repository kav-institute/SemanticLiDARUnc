import torch
import torch.nn as nn
from baselines.SalsaNext import adf

# Use a single, consistent keeper everywhere
keep_variance_fn = adf.keep_variance_fn

class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super().__init__()
        self.conv1 = adf.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=1,
                                keep_variance_fn=keep_variance_fn)
        self.act1 = adf.LeakyReLU(keep_variance_fn=keep_variance_fn)

        self.conv2 = adf.Conv2d(out_filters, out_filters, (3,3), padding=1,
                                keep_variance_fn=keep_variance_fn)
        self.act2 = adf.LeakyReLU(keep_variance_fn=keep_variance_fn)
        self.bn1 = adf.BatchNorm2d(out_filters, keep_variance_fn=keep_variance_fn)

        self.conv3 = adf.Conv2d(out_filters, out_filters, (3,3), dilation=1, padding=1,
                                keep_variance_fn=keep_variance_fn)
        self.act3 = adf.LeakyReLU(keep_variance_fn=keep_variance_fn)
        self.bn2 = adf.BatchNorm2d(out_filters, keep_variance_fn=keep_variance_fn)

    def forward(self, x):
        shortcut = self.conv1(*x)
        shortcut = self.act1(*shortcut)

        resA = self.conv2(*shortcut)
        resA = self.act2(*resA)
        resA1 = self.bn1(*resA)

        resA = self.conv3(*resA1)
        resA = self.act3(*resA)
        resA2 = self.bn2(*resA)

        output = shortcut[0] + resA2[0], shortcut[1] + resA2[1]
        output = output[0], keep_variance_fn(output[1])
        return output

class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3), stride=1,
                 pooling=True, drop_out=True, p=0.2):
        super().__init__()
        self.pooling = pooling
        self.drop_out = drop_out
        self.p = p

        self.conv1 = adf.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=stride,
                                keep_variance_fn=keep_variance_fn)
        self.act1 = adf.LeakyReLU(keep_variance_fn=keep_variance_fn)

        self.conv2 = adf.Conv2d(in_filters, out_filters, kernel_size=(3,3), padding=1,
                                keep_variance_fn=keep_variance_fn)
        self.act2 = adf.LeakyReLU(keep_variance_fn=keep_variance_fn)
        self.bn1 = adf.BatchNorm2d(out_filters, keep_variance_fn=keep_variance_fn)

        self.conv3 = adf.Conv2d(out_filters, out_filters, kernel_size=(3,3), dilation=2, padding=2,
                                keep_variance_fn=keep_variance_fn)
        self.act3 = adf.LeakyReLU(keep_variance_fn=keep_variance_fn)
        self.bn2 = adf.BatchNorm2d(out_filters, keep_variance_fn=keep_variance_fn)

        self.conv4 = adf.Conv2d(out_filters, out_filters, kernel_size=(2, 2), dilation=2, padding=1,
                                keep_variance_fn=keep_variance_fn)
        self.act4 = adf.LeakyReLU(keep_variance_fn=keep_variance_fn)
        self.bn3 = adf.BatchNorm2d(out_filters, keep_variance_fn=keep_variance_fn)

        self.conv5 = adf.Conv2d(out_filters*3, out_filters, kernel_size=(1, 1),
                                keep_variance_fn=keep_variance_fn)
        self.act5 = adf.LeakyReLU(keep_variance_fn=keep_variance_fn)
        self.bn4 = adf.BatchNorm2d(out_filters, keep_variance_fn=keep_variance_fn)

        if pooling:
            self.dropout = adf.Dropout(p=self.p, keep_variance_fn=keep_variance_fn)
            # keep original behavior: stride=2, padding=1 with given kernel_size
            self.pool = adf.AvgPool2d(keep_variance_fn, kernel_size=kernel_size, stride=2, padding=1)
        else:
            self.dropout = adf.Dropout(p=self.p, keep_variance_fn=keep_variance_fn)

    def forward(self, x):
        shortcut = self.conv1(*x)
        shortcut = self.act1(*shortcut)

        resA = self.conv2(*x)
        resA = self.act2(*resA)
        resA1 = self.bn1(*resA)

        resA = self.conv3(*resA1)
        resA = self.act3(*resA)
        resA2 = self.bn2(*resA)

        resA = self.conv4(*resA2)
        resA = self.act4(*resA)
        resA3 = self.bn3(*resA)

        concat_mean = torch.cat((resA1[0], resA2[0], resA3[0]), dim=1)
        concat_var  = torch.cat((resA1[1], resA2[1], resA3[1]), dim=1)
        concat = concat_mean, keep_variance_fn(concat_var)

        resA = self.conv5(*concat)
        resA = self.act5(*resA)
        resA = self.bn4(*resA)
        resA = shortcut[0] + resA[0], shortcut[1] + resA[1]
        resA = resA[0], keep_variance_fn(resA[1])

        if self.pooling:
            resB = self.dropout(*resA) if self.drop_out else resA
            resB = self.pool(*resB)
            resB = resB[0], keep_variance_fn(resB[1])
            return resB, resA
        else:
            resB = self.dropout(*resA) if self.drop_out else resA
            resB = resB[0], keep_variance_fn(resB[1])
            return resB

class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, drop_out=True, p=0.2):
        super().__init__()
        self.drop_out = drop_out
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.p = p

        self.dropout1 = adf.Dropout(p=self.p, keep_variance_fn=keep_variance_fn)
        self.dropout2 = adf.Dropout(p=self.p, keep_variance_fn=keep_variance_fn)

        self.conv1 = adf.Conv2d(in_filters//4 + 2*out_filters, out_filters, (3,3), padding=1,
                                keep_variance_fn=keep_variance_fn)
        self.act1 = adf.LeakyReLU(keep_variance_fn=keep_variance_fn)
        self.bn1 = adf.BatchNorm2d(out_filters, keep_variance_fn=keep_variance_fn)

        self.conv2 = adf.Conv2d(out_filters, out_filters, (3,3), dilation=2, padding=2,
                                keep_variance_fn=keep_variance_fn)
        self.act2 = adf.LeakyReLU(keep_variance_fn=keep_variance_fn)
        self.bn2 = adf.BatchNorm2d(out_filters, keep_variance_fn=keep_variance_fn)

        self.conv3 = adf.Conv2d(out_filters, out_filters, (2,2), dilation=2, padding=1,
                                keep_variance_fn=keep_variance_fn)
        self.act3 = adf.LeakyReLU(keep_variance_fn=keep_variance_fn)
        self.bn3 = adf.BatchNorm2d(out_filters, keep_variance_fn=keep_variance_fn)

        self.conv4 = adf.Conv2d(out_filters*3, out_filters, kernel_size=(1,1),
                                keep_variance_fn=keep_variance_fn)
        self.act4 = adf.LeakyReLU(keep_variance_fn=keep_variance_fn)
        self.bn4 = adf.BatchNorm2d(out_filters, keep_variance_fn=keep_variance_fn)

        self.dropout3 = adf.Dropout(p=self.p, keep_variance_fn=keep_variance_fn)
        self.dropout4 = adf.Dropout(p=self.p, keep_variance_fn=keep_variance_fn)

    def forward(self, x, skip):
        mean, var = x
        # PixelShuffle requires channels divisible by 4
        assert mean.shape[1] % 4 == 0 and var.shape[1] % 4 == 0, \
            f"PixelShuffle expects channels % 4 == 0, got mean {mean.shape}, var {var.shape}"
        upA_mean = nn.PixelShuffle(2)(mean)
        upA_var  = nn.PixelShuffle(2)(var)
        upA = upA_mean, keep_variance_fn(upA_var)

        if self.drop_out:
            upA = self.dropout1(*upA)

        upB_mean = torch.cat((upA[0], skip[0]), dim=1)
        upB_var  = torch.cat((upA[1], skip[1]), dim=1)
        upB = upB_mean, keep_variance_fn(upB_var)

        if self.drop_out:
            upB = self.dropout2(*upB)

        upE = self.conv1(*upB)
        upE = self.act1(*upE)
        upE1 = self.bn1(*upE)

        upE = self.conv2(*upE1)
        upE = self.act2(*upE)
        upE2 = self.bn2(*upE)

        upE = self.conv3(*upE2)
        upE = self.act3(*upE)
        upE3 = self.bn3(*upE)

        concat_mean = torch.cat((upE1[0], upE2[0], upE3[0]), dim=1)
        concat_var  = torch.cat((upE1[1], upE2[1], upE3[1]), dim=1)
        concat = concat_mean, keep_variance_fn(concat_var)

        if self.drop_out:
            concat = self.dropout3(*concat)

        upE = self.conv4(*concat)
        upE = self.act4(*upE)
        upE = self.bn4(*upE)

        if self.drop_out:
            upE = self.dropout4(*upE)

        upE = upE[0], keep_variance_fn(upE[1])
        return upE

class SalsaNextUncertainty(nn.Module):
    def __init__(self, nclasses, p=0.2, nchannels=5):
        super().__init__()
        self.nclasses = nclasses
        self.p = p
        self.nchannels = nchannels

        self.downCntx  = ResContextBlock(self.nchannels, 32)
        self.downCntx2 = ResContextBlock(32, 32)
        self.downCntx3 = ResContextBlock(32, 32)

        self.resBlock1 = ResBlock(32, 2 * 32, pooling=True, drop_out=False, p=self.p)
        self.resBlock2 = ResBlock(2 * 32, 4 * 32, pooling=True, p=self.p)
        self.resBlock3 = ResBlock(4 * 32, 8 * 32, pooling=True, p=self.p)
        self.resBlock4 = ResBlock(8 * 32, 8 * 32, pooling=True, p=self.p)
        self.resBlock5 = ResBlock(8 * 32, 8 * 32, pooling=False, p=self.p)

        self.upBlock1 = UpBlock(8 * 32, 4 * 32, p=self.p)
        self.upBlock2 = UpBlock(4 * 32, 4 * 32, p=self.p)
        self.upBlock3 = UpBlock(4 * 32, 2 * 32, p=self.p)
        self.upBlock4 = UpBlock(2 * 32, 32, drop_out=False, p=self.p)

        self.logits = adf.Conv2d(32, nclasses, kernel_size=(1, 1), keep_variance_fn=keep_variance_fn)

    def forward(self, x):
        inputs_mean = x
        inputs_variance = torch.zeros_like(inputs_mean) + 2e-7
        x = inputs_mean, keep_variance_fn(inputs_variance)

        downCntx = self.downCntx(x)
        downCntx = self.downCntx2(downCntx)
        downCntx = self.downCntx3(downCntx)

        down0c, down0b = self.resBlock1(downCntx)
        down1c, down1b = self.resBlock2(down0c)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        down5c = self.resBlock5(down3c)

        up4e = self.upBlock1(down5c, down3b)
        up3e = self.upBlock2(up4e, down2b)
        up2e = self.upBlock3(up3e, down1b)
        up1  = self.upBlock4(up2e, down0b)

        logits = self.logits(*up1)
        logits = logits[0], keep_variance_fn(logits[1])
        return logits

if __name__ == "__main__":
    import numpy as np
    model = SalsaNextUncertainty(20).cuda()
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters: ", pytorch_total_params / 1e6, "M")
    # quick smoke test
    model.eval()
    with torch.no_grad():
        inputs = torch.randn(1, 5, 128, 2048, device="cuda")
        out_mean, out_var = model(inputs)
        print("Output mean/var shapes:", out_mean.shape, out_var.shape)
        print("Finite check:", torch.isfinite(out_mean).all().item(), torch.isfinite(out_var).all().item())
