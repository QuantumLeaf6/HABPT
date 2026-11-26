import torch
import torch.nn as nn
import torch.nn.functional as F

from util.func import expmap0, poincare_distance


class HyperbolicConv2D(nn.Module):
    """
    Projection layer (paper Eq. 2).

    It maps the CNN feature map onto the Poincare ball with the exponential
    map at the origin and computes, for every latent prototype, the Poincare
    distance to each spatial location (paper Eq. 3).
    """

    def __init__(self, num_prototypes, num_features, w_1, h_1, input_channels,
                 prototype_vectors, curvature=6.0):
        super(HyperbolicConv2D, self).__init__()
        self.num_prototypes = num_prototypes
        self.curvature = curvature
        self.prototype_vectors = prototype_vectors
        self.conv1x1 = nn.Conv2d(input_channels, num_features, kernel_size=1, bias=False)

    def check_nan(self, tensor, tensor_name):
        if torch.isnan(tensor).any():
            raise ValueError(f'NaN values found in {tensor_name}')

    def forward(self, xs):
        self.check_nan(xs, 'input to forward')
        self.check_nan(self.conv1x1.weight, 'conv1x1 weights')

        # 1x1 convolution projects the feature map into the prototype space
        xs = self.conv1x1(xs)
        xs = torch.clamp(xs, -10.0, 10.0)
        self.check_nan(xs, 'xs before embedding')

        bs, num_features, w_in, h_in = xs.size()

        # Embed feature vectors and prototypes into the Poincare ball (Eq. 2)
        xs = xs.permute(0, 2, 3, 1).reshape(bs, w_in * h_in, num_features)
        xs_hyp = expmap0(xs, c=self.curvature)

        num_prototypes = self.num_prototypes
        prototypes = self.prototype_vectors.view(num_prototypes, num_features)
        prototypes_hyp = expmap0(prototypes, c=self.curvature)

        # Pairwise Poincare distance between every patch and every prototype
        # x: (bs, N, 1, D), y: (1, 1, P, D) -> (bs, N, P)
        x_exp = xs_hyp.unsqueeze(2)
        y_exp = prototypes_hyp.unsqueeze(0).unsqueeze(0)
        distance = poincare_distance(x_exp, y_exp, c=self.curvature)  # (bs, N, P)

        distance = distance.permute(0, 2, 1).view(bs, num_prototypes, w_in, h_in)
        return distance
