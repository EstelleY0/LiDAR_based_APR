import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from model.HypLiLoc.modules import SphericalProjection, SAGALayer, HyperbolicFusion
from model.utils import MARegressor

class HypLiLoc(nn.Module):
    def __init__(self, hidden_units=512, freeze_backbone=False):
        super(HypLiLoc, self).__init__()
        
        # 3D Branch (Simplified SAGA)
        # Oxford/RobotCar point clouds are often ~4096 points
        self.saga1 = SAGALayer(npoint=1024, radius=0.2, nsample=32, in_channel=3, mlp=[32, 32, 64])
        self.saga2 = SAGALayer(npoint=256, radius=0.4, nsample=32, in_channel=64+3, mlp=[64, 64, 128])
        self.saga3 = SAGALayer(npoint=64, radius=0.8, nsample=16, in_channel=128+3, mlp=[128, 256, 512])
        
        # 2D Branch (ResNet18)
        self.projection = SphericalProjection(h=64, w=900)
        self.resnet = resnet18(pretrained=True)
        # Adapt first conv for 5 channels (x, y, z, depth, intensity)
        self.resnet.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.proj_2d_feat = nn.Linear(512, 512)
        
        # Fusion
        self.fusion = HyperbolicFusion(dim=512)
        
        if freeze_backbone:
            for param in self.saga1.parameters(): param.requires_grad = False
            for param in self.saga2.parameters(): param.requires_grad = False
            for param in self.saga3.parameters(): param.requires_grad = False
            for param in self.resnet.parameters(): param.requires_grad = False

        # Pose Regressor
        self.regressor = MARegressor(in_channel=512, hidden_units=hidden_units)

    def forward(self, pc):
        # pc: [B, N, 3] or [B, 3, N]
        if pc.shape[1] == 3:
            xyz = pc
            pc_t = pc.transpose(1, 2)
        else:
            xyz = pc.transpose(1, 2)
            pc_t = pc

        # 3D Branch
        s_xyz, f_3d = self.saga1(xyz, None)
        s_xyz, f_3d = self.saga2(s_xyz, f_3d)
        _, f_3d = self.saga3(s_xyz, f_3d)
        f_3d = torch.max(f_3d, dim=-1)[0] # Global Max Pooling [B, 512]
        
        # 2D Branch
        img = self.projection(pc_t) # [B, 5, 64, 900]
        x = self.resnet.conv1(img)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        f_2d = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1) # [B, 512]
        f_2d = self.proj_2d_feat(f_2d)
        
        # Fusion
        f_fused = self.fusion(f_3d, f_2d)
        
        # Pose Regression
        pose = self.regressor(f_fused)
        return pose
