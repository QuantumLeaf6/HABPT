import math

import torch
import torch.nn.functional as F


def expmap0(v, c=6.0, eps=1e-7):
    """
    Exponential map from the origin of the Poincare ball (paper Eq. 2).

    For a point v in the tangent space at the origin, the embedding on the
    Poincare ball of curvature -c (c > 0) is
        exp_0(v) = tanh(sqrt(c) * ||v|| / 2) * v / (sqrt(c) * ||v||).
    This maps an arbitrary Euclidean vector into the ball so that the norm of
    the result is always below 1 / sqrt(c).

    :param v: tensor of arbitrary shape, the last dimension is the feature dim
    :param c: magnitude of the negative curvature (paper: c = -6)
    :return: tensor of the same shape, mapped onto the Poincare ball
    """
    if c == 0:
        return v
    sqrt_c = math.sqrt(c)
    norm = v.norm(dim=-1, keepdim=True)
    return torch.tanh(sqrt_c * norm / 2.0) * v / (sqrt_c * norm + eps)


def poincare_distance(x, y, c=6.0, eps=1e-7):
    """
    Poincare geodesic distance (paper Eq. 3).

    For curvature -c (c > 0):
        d(x, y) = (1/sqrt(c)) * acosh(1 + 2c ||x-y||^2 /
                                        ((1 - c||x||^2) (1 - c||y||^2)))
    When c = 0 this reduces to the Euclidean distance (control setting).

    x and y must be points inside the ball (c||x||^2 < 1), which is guaranteed
    by embedding them with :func:`expmap0`.

    :param x: tensor of arbitrary shape, last dim is the feature dim
    :param y: tensor broadcastable with x, last dim is the feature dim
    :param c: magnitude of the negative curvature (paper: c = -6)
    :return: tensor of shape broadcast(x, y)[..., :-1]
    """
    x_norm_sq = (x * x).sum(dim=-1, keepdim=True)
    y_norm_sq = (y * y).sum(dim=-1, keepdim=True)
    diff_norm_sq = ((x - y) * (x - y)).sum(dim=-1, keepdim=True)

    if c == 0:
        return diff_norm_sq.sqrt().squeeze(-1)

    denominator = (1.0 - c * x_norm_sq) * (1.0 - c * y_norm_sq)
    denominator = torch.clamp(denominator, min=eps)

    acosh_arg = 1.0 + 2.0 * c * diff_norm_sq / denominator
    acosh_arg = torch.clamp(acosh_arg, min=1.0 + eps)

    return torch.acosh(acosh_arg).squeeze(-1) / math.sqrt(c)


def hyperbolic_min_pool2d(distances, kernel_size, stride=None, padding=0):
    """
    Min pooling over a prototype distance map (paper Eq. 10).

    ``distances`` is already a map of Poincare distances of shape
    (batch_size, num_prototypes, H, W). This function keeps, for every
    prototype, the smallest distance over the spatial window, locating the
    most activated prototypical part of the input.
    """
    if stride is None:
        stride = kernel_size

    if padding > 0:
        distances = F.pad(distances, (padding, padding, padding, padding),
                          mode='constant', value=float('inf'))

    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, kernel_size)
    if not isinstance(stride, tuple):
        stride = (stride, stride)

    # Min pooling is implemented as max pooling over the negated distances
    return -F.max_pool2d(-distances, kernel_size=kernel_size, stride=stride)
