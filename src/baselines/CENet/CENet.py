import torch.nn as nn
import torch
from torch.nn import functional as F
from CENet_HardDNet import HarDNet
from CENet_ResNet34 import ResNet_34

class CENet(nn.Module):
    def __init__(self, nclasses, aux=True, model="HarDNet"):
        super(CENet, self).__init__()
        self.aux = aux
        if model=="HarDNet":
            self.model = HarDNet(nclasses, aux=aux)
        elif model=="ResNet_34":
            self.model = ResNet_34(nclasses, aux=aux)
        
    def forward(self, x):
        
        if self.aux:
            out, res_2, res_3, res_4 = self.model.forward(x)
            return [out, res_2, res_3, res_4]
        else:
            out = self.model.forward(x)
            return out

if __name__ == "__main__":
    import time
    import numpy as np
    model = CENet(20, aux=True,model="HarDNet").cuda()
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters: ", pytorch_total_params / 1000000, "M")
    # Timer
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    inference_times = []
    model.eval()
    with torch.no_grad():
        for i in range(100):
            inputs = torch.randn(1, 5, 128, 2048).cuda()
            start.record()
            outputs = model(inputs)
            end.record()
            torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
            print("inference took {} ms".format(start.elapsed_time(end)))
            
            inference_times.append(start.elapsed_time(end))
            #time.sleep(0.15)
    print("inference mean {} ms".format(np.mean(inference_times)))