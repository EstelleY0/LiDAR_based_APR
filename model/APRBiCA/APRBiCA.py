import torch
import torch.nn as nn
import torch.nn.functional as F
from model.APRBiCA.modules import PointNet2Encoder, BiCA, GatingUnit
from model.utils import MARegressor

class APRBiCA(nn.Module):
    def __init__(self, in_channel=3, hidden_units=512):
        super(APRBiCA, self).__init__()
        self.encoder = PointNet2Encoder(in_channel=in_channel)
        
        # Feature dimensions from PointNet2Encoder:
        # l1: 128, l2: 256, l3: 512, l4: 1024
        
        # BiCA typically fuses high-level semantic and low-level geometric features.
        # We'll use l3 (64 points, 512 dim) and l4 (1 point, 1024 dim).
        
        self.proj_l4 = nn.Conv1d(1024, 512, 1)
        
        self.bica = BiCA(dim=512)
        self.gate3 = GatingUnit(dim=512)
        self.gate4 = GatingUnit(dim=512)
        
        # After fusion, we aggregate to a global feature
        self.regressor = MARegressor(in_channel=512 * 2, hidden_units=hidden_units)

    def forward(self, pc):
        # pc: [B, N, 3] or [B, 3, N]
        # PointNet++ expects [B, 3, N] for xyz and [B, D, N] for points
        if pc.shape[1] == 3:
            pc_t = pc
        else:
            pc_t = pc.transpose(1, 2).contiguous()
            
        l1, l2, l3, l4 = self.encoder(pc_t, None)
        
        # l3: [B, 512, 64]
        # l4: [B, 1024, 1]
        
        l4_proj = self.proj_l4(l4) # [B, 512, 1]
        
        # Bidirectional Cross-Attention
        att3, att4 = self.bica(l3, l4_proj)
        # att3: [B, 512, 64]
        # att4: [B, 512, 1]
        
        # Gating Fusion
        f3 = self.gate3(l3, att3) # [B, 512, 64]
        f4 = self.gate4(l4_proj, att4) # [B, 512, 1]
        
        # Global Aggregation
        # Max pool f3 to get global feature
        g3 = torch.max(f3, dim=-1)[0] # [B, 512]
        g4 = f4.squeeze(-1) # [B, 512]
        
        fused = torch.cat([g3, g4], dim=-1) # [B, 1024]
        
        # Pose Regression
        pose = self.regressor(fused)
        
        return pose
