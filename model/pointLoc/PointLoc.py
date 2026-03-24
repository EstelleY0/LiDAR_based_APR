import torch.nn as nn

from model.pointLoc.modules import PointCloudEncoder, SelfAttnModule, GroupAllLayer, PoseRegressor


class PointLoc(nn.Module):
    def __init__(self):
        super(PointLoc, self).__init__()
        self.point_cloud_encoder = PointCloudEncoder()
        self.self_attention_module = SelfAttnModule()
        self.group_all_layers_module = GroupAllLayer()
        self.pose_regressor = PoseRegressor()

    def forward(self, input):
        xyz, feature = self.point_cloud_encoder(input)

        feature = feature.transpose(1, 2)
        weighted_feature = self.self_attention_module(feature)

        weighted_feature = weighted_feature.transpose(1, 2)
        feature_vectors = self.group_all_layers_module(weighted_feature)

        pose_reg = self.pose_regressor(feature_vectors)

        return pose_reg
