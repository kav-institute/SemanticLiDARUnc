from ResNet import *

backbone="ResNet34_aspp_1"

if backbone=="ResNet34_aspp_1":
	Backend=resnet34_aspp_1(if_BN=True,if_remission=True,if_range=True)
	S_H=SemanticHead(20,1152)

if backbone=="ResNet34_aspp_2":
	Backend=resnet34_aspp_2(if_BN=True,if_remission=True,if_range=True)
	S_H=SemanticHead(20,128*13)


if backbone=="ResNet34_point":
	Backend=resnet34_point(if_BN=True,if_remission=True,if_range=True,with_normal=False)
	S_H=SemanticHead(20,1024)

model=Final_Model(Backend,S_H)


class FIDNet(nn.Module):
    def __init__(self, nclasses, backbone="ResNet34_aspp_1",with_normal=False):
        super(FIDNet, self).__init__()
        if backbone=="ResNet34_aspp_1":
            Backend=resnet34_aspp_1(if_BN=True,if_remission=True,if_range=True)
            S_H=SemanticHead(nclasses,1152)

        if backbone=="ResNet34_aspp_2":
            Backend=resnet34_aspp_2(if_BN=True,if_remission=True,if_range=True)
            S_H=SemanticHead(nclasses,128*13)


        if backbone=="ResNet34_point":
            Backend=resnet34_point(if_BN=True,if_remission=True,if_range=True,with_normal=with_normal)
            S_H=SemanticHead(nclasses,1024)
        self.model=Final_Model(Backend,S_H)
        
        
    def forward(self, x):
        
        out = self.model(x)
        return out

if __name__ == "__main__":
    import time
    import numpy as np
    model = FIDNet(20, "ResNet34_point").cuda()
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