import torch
import torch.nn as nn
import torch.nn.functional as F

def quaternion_logarithm(q):
    """
    Computes the logarithmic map of unit quaternions.
    Input q: [Batch, 4] (w, x, y, z)
    Returns: [Batch, 3] (vector part of log_q)
    """
    # Normalize to ensure unit quaternion
    q = F.normalize(q, p=2, dim=1)

    q_w = q[:, 0:1]    # Scalar part (w)
    q_vec = q[:, 1:]   # Vector part (x, y, z)

    v_norm = torch.norm(q_vec, p=2, dim=1, keepdim=True)

    eps = 1e-7
    q_w_clamped = torch.clamp(q_w, -1.0 + eps, 1.0 - eps)
    theta = torch.acos(q_w_clamped)
    log_q = torch.where(v_norm > eps, (theta / v_norm) * q_vec, q_vec)

    return log_q

class MARegressor(nn.Module):
    def __init__(self, in_channel=1024, hidden_units=512):
        super(MARegressor, self).__init__()

        self.trans = nn.Sequential(
            nn.Linear(in_channel, hidden_units),
            nn.BatchNorm1d(hidden_units),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_units, hidden_units // 2),
            nn.BatchNorm1d(hidden_units // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_units // 2, 3)
        )

        self.logq = nn.Sequential(
            nn.Linear(in_channel, hidden_units),
            nn.BatchNorm1d(hidden_units),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_units, hidden_units // 2),
            nn.BatchNorm1d(hidden_units // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_units // 2, 3)
        )

    def forward(self, x):
        t = self.trans(x)
        r = self.logq(x)

        return torch.cat([t, r], dim=-1)
