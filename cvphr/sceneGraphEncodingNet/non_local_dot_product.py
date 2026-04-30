# Non-Local Block
import torch
from torch import nn


class _NonLocalBlockND(nn.Module):
    """
    Non-local block:

    1. Idea: capture long-range dependencies via pairwise relations.
    2. Structure:
        theta (query), phi (key): similarity
        g (value): feature embedding
        W: output projection (init as zero)
    3. Steps:
        compute pairwise similarity
        normalize attention
        weighted aggregation
        residual connection
    4. Shapes:
        input: [B, C, T, H, W]
        inter: [B, C/2, T, H, W]
        affinity: [B, THW, THW]
        output: [B, C, T, H, W]
    5. Sub-sampling:
        reduces computation (e.g. THW -> THW/4)
    """

    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]  # ensure valid dimension

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # default inter_channels = C/2
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # select ops by dimension
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d((1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d((2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(2)
            bn = nn.BatchNorm1d

        # g: value embedding (C -> C/2)
        self.g = conv_nd(in_channels, self.inter_channels, 1, 1, 0)

        # W: projection back to C
        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(self.inter_channels, self.in_channels, 1, 1, 0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(self.inter_channels, self.in_channels, 1, 1, 0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        # theta: query
        self.theta = conv_nd(self.in_channels, self.inter_channels, 1, 1, 0)

        # phi: key
        self.phi = conv_nd(self.in_channels, self.inter_channels, 1, 1, 0)

        # optional downsampling
        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=False):
        """
        Args:
            x: (b, c, t, h, w)
            return_nl_map: return attention map

        Examples:
            1D: (b, c, t)
            2D: (b, c, h, w)
            3D: (b, c, t, h, w)
        """

        batch_size = x.size(0)

        # g: [b, c, t,h,w] -> [b, N/4, C]
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # theta: [b, N, C]
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        # phi: [b, C, N]
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        # affinity
        f = torch.matmul(theta_x, phi_x)

        # normalize (divide by N)
        N = f.size(-1)
        f_div_C = f / N

        # alternative:
        # f_div_C = torch.softmax(f, dim=-1)

        """
        Why divide by N:
        - stability (avoid large values)
        - consistent scaling across resolutions
        - similar to softmax but keeps raw distribution
        """

        # attention aggregation
        y = torch.matmul(f_div_C, g_x)

        # reshape back
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])

        # residual
        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(
            in_channels,
            inter_channels=inter_channels,
            dimension=2,
            sub_sample=sub_sample,
            bn_layer=bn_layer
        )


class NONLocalBlock2D_soft(_NonLocalBlockND):
    """
    2D non-local block with softmax normalization
    """

    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D_soft, self).__init__(
            in_channels,
            inter_channels=inter_channels,
            dimension=2,
            sub_sample=sub_sample,
            bn_layer=bn_layer
        )

    def forward(self, x, return_nl_map=False):
        """
        forward with softmax normalization
        """

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)

        # softmax normalization
        f_div_C = torch.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)

        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])

        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z


if __name__ == '__main__':
    pass