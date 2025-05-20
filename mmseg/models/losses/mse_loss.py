"""
No@
August 26th, 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES

# only applicable to regression outputs
@LOSSES.register_module()
class MSELossWithIgnoreIndex(nn.Module):
    def __init__(self, 
                 loss_name='loss_mse',
                 loss_weight = 1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self, cls_score, label, ignore_index=255, weight=None):
        mask = (label != ignore_index).type_as(cls_score)
        diff = cls_score.squeeze(1) - label
        diff = diff * mask
        loss = torch.sum(diff ** 2) / mask.sum()
        return self.loss_weight * loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
