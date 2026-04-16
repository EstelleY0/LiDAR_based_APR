import torch
import torch.nn as nn


class AtLocCriterion(nn.Module):
    def __init__(self, t_loss_fn=nn.L1Loss(), q_loss_fn=nn.L1Loss(), sax=0.0, saq=-3.0, learn_beta=True):
        super(AtLocCriterion, self).__init__()
        self.t_loss_fn = t_loss_fn
        self.q_loss_fn = q_loss_fn
        self.sax = nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)
        self.saq = nn.Parameter(torch.Tensor([saq]), requires_grad=learn_beta)

    def forward(self, pred, targ):
        loss = (torch.exp(-self.sax) * self.t_loss_fn(pred[:, :3], targ[:, :3]) + self.sax
                + torch.exp(-self.saq) * self.q_loss_fn(pred[:, 3:], targ[:, 3:]) + self.saq)
        return loss

class STCLocCriterion(nn.Module):
    def __init__(self, t_loss_fn=nn.L1Loss(), q_loss_fn=nn.L1Loss()):
        super(STCLocCriterion, self).__init__()
        self.t_loss_fn = t_loss_fn
        self.q_loss_fn = q_loss_fn
        self.loc_loss = nn.NLLLoss()
        self.ori_loss = nn.NLLLoss()

    def forward(self, pred_pose, pred_loc, pred_ori, gt_pose, gt_loc, gt_ori):
        loss_pose = 1.0 * self.t_loss_fn(pred_pose[:, :3], gt_pose[:, :3]) + 10.0 * self.q_loss_fn(pred_pose[:, 3:], gt_pose[:, 3:])
        loss_cls = 1.0 * self.loc_loss(pred_loc, gt_loc.long()) + 1.0 * self.ori_loss(pred_ori, gt_ori.long())
        
        final_loss = 1.5 * loss_pose + 1.0 * loss_cls
        return final_loss, loss_pose * 1.5, loss_cls
