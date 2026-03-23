# SGMNet
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from sceneGraphEncodingNet.non_local_dot_product import NONLocalBlock2D, NONLocalBlock2D_soft


class CSMG(nn.Module):
    """
    Cluster Similarity Masking Graph module
    Used for semi-global feature learning and scene graph encoding

    Args:
    - input_channel: input feature channels (default: 256)
    - num_clusters: number of clusters (default: 4)
    - alpha: sharpness of clustering (default: 1.0)
    """

    def __init__(self, input_channel=256, output_channel=256, num_clusters=4, alpha=1.0):
        super(CSMG, self).__init__()
        self.num_clusters = num_clusters

        self.nl_conv = nn.Sequential(
            # input: [B, input_channel, H, W]
            # output: [B, output_channel, H, W]
            nn.Conv2d(input_channel, output_channel, 3, 1, 1),

            # non-local block for long-range dependency
            # input/output: [B, output_channel, H, W]
            NONLocalBlock2D(output_channel),

            # downsample
            # output: [B, output_channel, H/2, W/2]
            nn.MaxPool2d(2, 2),
        )

        self.alpha = alpha

        # node conv
        # output: [B, num_clusters, H/2, W/2]
        self.conv_node = nn.Conv2d(output_channel, num_clusters, 1, bias=True)

        # learnable centroids [K, D]
        self.centroids = nn.Parameter(torch.rand(num_clusters, output_channel))

        # weight: [K, D, 1, 1]
        self.conv_node.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )

        # bias: [K]
        self.conv_node.bias = nn.Parameter(
            -self.alpha * self.centroids.norm(dim=1)
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass

        Args:
            x: [B, input_channel, H, W]

        Returns:
            sim_scores: [B, K, (H/2)*(W/2)]
            d: [B, K, D]
            d_flatten: [B, K*D]
        """

        # +----+
        # | 01 | feature extraction
        # +----+
        # extract semi-global features via non-local block
        x_nl = self.nl_conv(x)
        B, D = x_nl.shape[:2]

        # channel normalization
        x_nl = F.normalize(x_nl, p=2, dim=1)

        # flatten
        x_flatten = x_nl.view(B, D, -1)

        # expand for similarity
        x_expand = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3)

        # +----+
        # | 02 | clustering
        # +----+
        soft_assign = self.conv_node(x_nl).view(B, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        # +----+
        # | 03 | feature reconstruction
        # +----+
        x_rebuild = x_expand * soft_assign.unsqueeze(2)

        # +----+
        # | 04 | cluster center
        # +----+
        centroids_norm = F.normalize(self.centroids, p=2, dim=1)
        cluster_c = centroids_norm.unsqueeze(0).expand(B, -1, -1)

        # +----+
        # | 05 | similarity score
        # +----+
        sim_scores = torch.bmm(cluster_c, x_flatten)
        sim_scores = self.relu(sim_scores)

        sim_scores_mask = sim_scores.expand(D, -1, -1, -1).permute(1, 2, 0, 3)

        # +----+
        # | 06 | similarity-guided reconstruction
        # +----+
        X_star = x_rebuild * sim_scores_mask

        # +----+
        # | 07 | feature fusion
        # +----+
        d = X_star.sum(dim=-1)
        d = F.normalize(d, p=2, dim=2)

        d_flatten = d.view(B, -1)
        d_flatten = F.normalize(d_flatten, p=2, dim=1)

        return sim_scores, d, d_flatten, x_nl


class CSMG_soft(CSMG):
    """
    CSMG with softmax-normalized non-local block
    """

    def __init__(self, input_channel=256, output_channel=256, num_clusters=4, alpha=1.0):
        super(CSMG_soft, self).__init__(input_channel, output_channel, num_clusters, alpha)

        self.nl_conv = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 3, 1, 1),

            # softmax non-local
            NONLocalBlock2D_soft(output_channel),

            nn.MaxPool2d(2, 2),
        )


class JointNet(nn.Module):
    def __init__(self, backbone, CSMG):
        super(JointNet, self).__init__()
        self.backbone = backbone
        self.module = CSMG
        self.ref_feat = None

    def forward(self, x):

        # x: [B, 3, H, W]
        x = self.backbone(x)  # [B, 256, H/16, W/16]

        sim_scores, d, d_flatten, x_nl = self.module(x)
        B, C, HW = sim_scores.shape

        feature_map_size = int(HW ** 0.5)
        PATCH_SIZE = 256

        # coordinate scaling
        scale_factor = PATCH_SIZE / feature_map_size

        sim_scores = F.normalize(sim_scores, p=1, dim=1)

        sim_scores_np = sim_scores.cpu().detach().numpy()
        p_list = []

        for nb in range(B):

            p = []

            for nc in range(C):
                sim_score_by_cluster = sim_scores_np[nb, nc]

                idx_k = np.where(sim_score_by_cluster > 0)
                scores = sim_score_by_cluster[idx_k]

                idx_k = np.asarray(idx_k)
                scores = np.asarray(scores)

                if scores.sum() == 0:
                    p.append((0.0, 0.0))
                    continue

                # grid coords
                point_x = idx_k // feature_map_size
                point_y = idx_k % feature_map_size

                point_x = point_x @ scores / scores.sum()
                point_y = point_y @ scores / scores.sum()

                final_x = point_x[0] * scale_factor
                final_y = point_y[0] * scale_factor

                p.append((final_x, final_y))

            p_list.append(p)

        output = {
            'nl_feat': x_nl,
            'descriptor': d,
            'descriptor_flatten': d_flatten,
            'position': p_list,
        }

        return output


class JointNet_soft(JointNet):
    """
    JointNet with CSMG_soft
    """

    def __init__(self, backbone, CSMG_soft):
        super(JointNet_soft, self).__init__(backbone, CSMG_soft)
        self.module = CSMG_soft