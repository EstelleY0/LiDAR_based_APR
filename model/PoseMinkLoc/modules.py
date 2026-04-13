import torch
import torch.nn as nn
import MinkowskiEngine as ME

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x is a Minkowski SparseTensor
        # We need to compute global pooling over all points in each batch
        # SparseTensor.F is [N, C], SparseTensor.C is [N, 4] (batch_idx, x, y, z)
        
        # This is a bit tricky with ME. For simplicity, we can use ME.MinkowskiGlobalAvgPooling 
        # or implement it manually using batch indices.
        
        # A simple way to do GeM in ME is to raise features to power p, 
        # then use GlobalAvgPooling, then take p-th root.
        
        features = x.F
        features = torch.clamp(features, min=self.eps)
        features = features.pow(self.p)
        
        # Create a new SparseTensor with these powered features
        x_p = ME.SparseTensor(features=features, coordinate_map_key=x.coordinate_map_key, 
                               coordinate_manager=x.coordinate_manager)
        
        avg_pool = ME.MinkowskiGlobalAvgPooling()
        x_p_avg = avg_pool(x_p)
        
        # Features are now [Batch, C]
        res = x_p_avg.F.pow(1./self.p)
        return res

class MinkResNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1024):
        super(MinkResNet, self).__init__()
        
        # Initial convolution
        self.conv1 = ME.MinkowskiConvolution(in_channels, 64, kernel_size=3, stride=2, dimension=3)
        self.bn1 = ME.MinkowskiBatchNorm(64)
        self.relu = ME.MinkowskiReLU(inplace=True)
        
        # Basic Blocks (simplified ResNet-style)
        self.block1 = self._make_layer(64, 128, stride=2)
        self.block2 = self._make_layer(128, 256, stride=2)
        self.block3 = self._make_layer(256, 512, stride=2)
        self.block4 = self._make_layer(512, out_channels, stride=2)
        
        self.gem = GeM()

    def _make_layer(self, in_planes, out_planes, stride):
        layers = []
        layers.append(ME.MinkowskiConvolution(in_planes, out_planes, kernel_size=3, stride=stride, dimension=3))
        layers.append(ME.MinkowskiBatchNorm(out_planes))
        layers.append(ME.MinkowskiReLU(inplace=True))
        layers.append(ME.MinkowskiConvolution(out_planes, out_planes, kernel_size=3, stride=1, dimension=3))
        layers.append(ME.MinkowskiBatchNorm(out_planes))
        layers.append(ME.MinkowskiReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gem(x) # Returns [Batch, out_channels]
        return x
