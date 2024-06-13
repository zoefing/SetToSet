# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from PIL import ImageFilter
import random
import torch
   
class TwoSetsTransform:
    """Take two sets (each containing 10 images) of random crops of one image as the query and key sets."""

    def __init__(self, base_transform, set_size=10):
        self.base_transform = base_transform
        self.set_size = set_size

    def __call__(self, x):
        set1 = torch.stack([self.base_transform(x) for _ in range(self.set_size)])
        set2 = torch.stack([self.base_transform(x) for _ in range(self.set_size)])
        return [set1, set2]

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
