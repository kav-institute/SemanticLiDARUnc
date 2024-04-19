import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch

def remove_batchnorm(model):
    # Create a new Sequential module to reconstruct the model architecture without batchnorm
    new_model = nn.Sequential()
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            # Skip batchnorm layers
            continue
        elif isinstance(module, nn.Sequential):
            # If the module is Sequential, recursively remove batchnorm from its children
            new_model.add_module(name, remove_batchnorm(module))
        elif name in ["0","1","2"]:
            new_model.add_module(name, remove_batchnorm(module))
        else:
            # Add other layers to the new model
            new_model.add_module(name, module)
            
    return new_model

class AttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Define convolutional layers for query, key, and value
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # Define convolutional layer for attention scores
        self.attention_conv = nn.Conv2d(out_channels, 1, kernel_size=1)
        
        # Softmax layer
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, features):
        # Query, key, and value transformations
        query = self.query_conv(features)
        key = self.key_conv(features)
        value = self.value_conv(features)
        
        # Compute attention scores
        attention_scores = self.attention_conv(torch.tanh(query + key))
        
        # Apply softmax to get attention weights
        attention_weights = self.softmax(attention_scores)
        
        # Apply attention to the value
        attended_features = value * attention_weights
        
        return attended_features

class SemanticNetworkWithFPN(nn.Module):#
    """
    Semantic Segmentation Network with Feature Pyramid Network (FPN) using a ResNet backbone.

    Args:
        resnet_type (str): Type of ResNet model to use. Supported types: 'resnet18', 'resnet34', 'resnet50', 'resnet101'.
        meta_channel_dim (int): Number meta channels used in the FPN
        interpolation_mode (str): Interpolation mode used to resize the meta channels (default='nearest'). Supported types: 'nearest', 'bilinear', 'bicubic'.
        num_classes (int): Number of semantic classes
        no_bn (bool): Option to disable batchnorm (not recommended)
        attention (bool): Option to use a attention mechanism in the FPN (see: https://arxiv.org/pdf/1706.03762.pdf)
        
    """
    def __init__(self, resnet_type='resnet18', meta_channel_dim=3, interpolation_mode = 'nearest', num_classes = 3, no_bn=False, attention=True):
        super(SemanticNetworkWithFPN, self).__init__()

        self.interpolation_mode = interpolation_mode
        self.num_classes = num_classes
        self.no_bn = no_bn
        self.attention = attention
        # Load pre-trained ResNet model
        if resnet_type == 'resnet18':
            self.resnet = models.resnet18(pretrained=True)
            base_channels = 512  # Number of channels in the last layer of ResNet18
        elif resnet_type == 'resnet34':
            self.resnet = models.resnet34(pretrained=True)
            base_channels = 512  # Number of channels in the last layer of ResNet34
        elif resnet_type == 'resnet50':
            self.resnet = models.resnet50(pretrained=True)
            base_channels = 2048  # Number of channels in the last layer of ResNet50
        elif resnet_type == 'resnet101':
            self.resnet = models.resnet101(pretrained=True)
            base_channels = 2048  # Number of channels in the last layer of ResNet101

        else:
            raise ValueError("Invalid ResNet type. Supported types: 'resnet18', 'resnet34', 'resnet50', 'resnet101'.")

        # remove all batchnorm layers
        if self.no_bn:
            self.resnet = remove_batchnorm(self.resnet)
        
        # Modify the first convolution layer to take 1+meta_channel_dim channels
        self.meta_channel_dim = meta_channel_dim
        self.resnet.conv1 = nn.Conv2d(2 + meta_channel_dim, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # Extract feature maps from different layers of ResNet
        self.layer1 = nn.Sequential(self.resnet.conv1, self.resnet.relu, self.resnet.maxpool, self.resnet.layer1)
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4
        
        # Attention blocks
        self.attention4 = AttentionModule(base_channels // 2, base_channels // 2)
        self.attention3 = AttentionModule(base_channels // 4, base_channels // 4)
        self.attention2 = AttentionModule(base_channels // 8, base_channels // 8)
        self.attention1 = AttentionModule(base_channels // 16, base_channels // 16)
        

        # FPN blocks
        self.fpn_block4 = self._make_fpn_block(base_channels, base_channels // 2)
        self.fpn_block3 = self._make_fpn_block(base_channels // 2, base_channels // 4)
        self.fpn_block2 = self._make_fpn_block(base_channels // 4, base_channels // 8)
        self.fpn_block1 = self._make_fpn_block(base_channels // 8, base_channels // 16)

        # upsamle layers
        self.upsample_layer_x4 = nn.ConvTranspose2d(in_channels=base_channels // 2, out_channels=base_channels // 2, kernel_size=8, stride=8, padding=0)
        self.upsample_layer_x3 = nn.ConvTranspose2d(in_channels=base_channels // 4, out_channels=base_channels // 4, kernel_size=4, stride=4, padding=0)
        self.upsample_layer_x2 = nn.ConvTranspose2d(in_channels=base_channels // 8, out_channels=base_channels // 8, kernel_size=2, stride=2, padding=0)
        
        if self.no_bn:
            self.decoder_semantic = nn.Sequential(
                nn.Conv2d(base_channels // 2 + base_channels // 4 + base_channels // 8 + base_channels // 16, base_channels // 16, kernel_size=3, stride=1, padding=1),
                #nn.Conv2d(base_channels // 8 + base_channels // 16, base_channels // 16, kernel_size=3, stride=1, padding=1),
                #nn.BatchNorm2d(base_channels // 16),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels // 16, base_channels // 16, kernel_size=3, stride=1, padding=1),
                #nn.BatchNorm2d(base_channels // 16),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(base_channels // 16, self.num_classes, kernel_size=4, stride=2, padding=1),
                nn.ELU(inplace=True)
            )
        else:
            self.decoder_semantic = nn.Sequential(
                nn.Conv2d(base_channels // 2 + base_channels // 4 + base_channels // 8 + base_channels // 16, base_channels // 16, kernel_size=3, stride=1, padding=1),
                #nn.Conv2d(base_channels // 8 + base_channels // 16, base_channels // 16, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(base_channels // 16),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels // 16, base_channels // 16, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(base_channels // 16),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(base_channels // 16, self.num_classes, kernel_size=4, stride=2, padding=1),
                nn.ELU(inplace=True)
            )
    def _make_fpn_block(self, in_channels, out_channels):
        """
        Create an FPN block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.

        Returns:
            nn.Module: FPN block.
        """
        if self.no_bn:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                #nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            
    def forward(self, x, meta_channel):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor representing depth prediction.
        """
        
        
        # Inject meta channel before ResNet layers
        if self.meta_channel_dim > 0:
            # Resize Meta Channels
            # Downsample the meta channel
            meta_channel1 = F.interpolate(meta_channel, scale_factor=1/2, mode=self.interpolation_mode)
            meta_channel2 = F.interpolate(meta_channel, scale_factor=1/4, mode=self.interpolation_mode)
            meta_channel3 = F.interpolate(meta_channel, scale_factor=1/8, mode=self.interpolation_mode)
            x = torch.cat([x, meta_channel], dim=1)
            x1 = self.layer1(x)
            x = torch.cat([x1[:,0:-self.meta_channel_dim,...], meta_channel1], dim=1)
            x2 = self.layer2(x)
            x = torch.cat([x2[:,0:-self.meta_channel_dim,...], meta_channel2], dim=1)
            x3 = self.layer3(x)
            x = torch.cat([x3[:,0:-self.meta_channel_dim,...], meta_channel3], dim=1)
            x4 = self.layer4(x)

        else:
        
            # Encoder (ResNet)
            x1 = self.layer1(x)
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            x4 = self.layer4(x3)

        # FPN
        x4 = self.fpn_block4(x4)
        x3 = self.fpn_block3(x3)
        x2 = self.fpn_block2(x2)
        x1 = self.fpn_block1(x1)
        
        # Attention
        if self.attention:
            x4 = self.attention4(x4)
            x3 = self.attention3(x3)
            x2 = self.attention2(x2)
            x1 = self.attention1(x1)

        x4 = self.upsample_layer_x4(x4)
        x3 = self.upsample_layer_x3(x3)
        x2 = self.upsample_layer_x2(x2)

        # Concatenate feature maps
        x = torch.cat([x1, x2, x3, x4], dim=1)


        # Decoder
        x_semantics = self.decoder_semantic(x) + 1 # offset of 1 to shift elu to ]0,inf[
        
        return x_semantics
    