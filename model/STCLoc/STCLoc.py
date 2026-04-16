# Modified from PSYZ1234/STCLoc (https://github.com/PSYZ1234/STCLoc)

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.STCLoc.modules import STCLocEncoder, STCLocDecoder


class STCLoc(nn.Module):
    def __init__(self, steps=1, num_class_loc=10, num_class_ori=10, freeze_backbone=False):
        super(STCLoc, self).__init__()
        self.encoder         = STCLocEncoder(steps=steps, feature_correlation=(steps > 1))
        
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
                
        # SRML Modules
        self.regressor       = STCLocDecoder(1024, [1024, 1024, 1024])
        self.classifier_t    = STCLocDecoder(1024, [1024, 1024])
        self.classifier_q    = STCLocDecoder(1024, [1024, 1024])
        
        self.fc_position     = nn.Linear(1024, 3)
        self.fc_orientation  = nn.Linear(1024, 3)
        
        # Classification heads are part of the original STCLoc, but we might not supervise them in single-frame simple training 
        self.fc_cls_loc      = nn.Linear(1024, num_class_loc)
        self.fc_cls_ori      = nn.Linear(1024, num_class_ori)
        
        self.fc_finall_pose  = nn.Linear(1024, 1024)
        self.bn_finall_pose  = nn.BatchNorm1d(1024)

    def forward(self, input):
        # input is [B, N, 3] or [B, 3, N]
        if input.shape[-1] == 3:
            input = input.transpose(1, 2)
            
        x        = self.encoder(input)  # [B*T, D]
        y        = self.regressor(x)  # [B*T, D]
        loc      = self.classifier_t(x)  # [B*T, D]
        ori      = self.classifier_q(x)  # [B*T, D]
        
        loc_norm = F.normalize(loc, dim=1)  # [B*T, D]
        ori_norm = F.normalize(ori, dim=1)  # [B*T, D]
        
        z        = y * loc_norm * ori_norm  # [B*T, D]
        z        = F.relu(self.bn_finall_pose(self.fc_finall_pose(z)))  # [B*T, D]
        
        t        = self.fc_position(z)  # [B*T, 3]
        q        = self.fc_orientation(z)  # [B*T, 3]
        
        cls_loc  = self.fc_cls_loc(loc)
        cls_ori  = self.fc_cls_ori(ori)
        cls_loc  = F.log_softmax(cls_loc, dim=1)
        cls_ori  = F.log_softmax(cls_ori, dim=1)
        
        pose = torch.cat([t, q], dim=-1)
        return pose, cls_loc, cls_ori
