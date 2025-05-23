"""
No@
"""
# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Tuple, Union

import torch
from mmcls.models import VisionTransformer  # From mmselfsup

from mmseg.registry import MODELS
# from ..utils import build_2d_sincos_position_embedding
from mmselfsup.models.utils.position_embedding import build_2d_sincos_position_embedding


@MODELS.register_module()
class MAEViT_CCH(VisionTransformer):
    """Vision Transformer for MAE pre-training.
        MODIFIED by No@ FROM mae_vit.py
            IT INCLUDES:
                - Customized in_channels

    A PyTorch implement of: `An Image is Worth 16x16 Words: Transformers
    for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.
    This module implements the patch masking in MAE and initialize the
    position embedding with sine-cosine position embedding.

    Args:
        arch (str | dict): Vision Transformer architecture
            Default: 'b'
        img_size (int | tuple): Input image size
        patch_size (int | tuple): The patch size
        in_channels (int): The num of input channels. Defaults to 3.    
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Defaults to True.
        output_cls_token (bool): Whether output the cls_token. If set True,
            `with_cls_token` must be True. Defaults to True.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Defaults to "bicubic".
        patch_cfg (dict): Configs of patch embeding. Defaults to an empty dict.
        layer_cfgs (Sequence | dict): Configs of each transformer layer in
            encoder. Defaults to an empty dict.
        mask_ratio (bool): The ratio of total number of patches to be masked.
            Defaults to 0.75.
        init_cfg (Union[List[dict], dict], optional): Initialization config
            dict. Defaults to None.
    """

    def __init__(self,
                 arch: Union[str, dict] = 'b',
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 out_indices: Union[Sequence, int] = -1,
                 drop_rate: float = 0,
                 drop_path_rate: float = 0,
                 norm_cfg: dict = dict(type='LN', eps=1e-6),
                 final_norm: bool = True,
                 output_cls_token: bool = True,
                 interpolate_mode: str = 'bicubic',
                 patch_cfg: dict = dict(),
                 layer_cfgs: dict = dict(),
                 mask_ratio: float = 0.75,
                 frozen_stages: int = -1,
                 init_cfg: Optional[Union[List[dict], dict]] = None) -> None:
        super().__init__(
            arch=arch,
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            out_indices=out_indices,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            norm_cfg=norm_cfg,
            final_norm=final_norm,
            frozen_stages=frozen_stages,
            output_cls_token=output_cls_token,
            interpolate_mode=interpolate_mode,
            patch_cfg=patch_cfg,
            layer_cfgs=layer_cfgs,
            init_cfg=init_cfg)

        # position embedding is not learnable during pretraining
        self.pos_embed.requires_grad = False
        self.mask_ratio = mask_ratio
        self.num_patches = self.patch_resolution[0] * self.patch_resolution[1]

        # freeze stages only when self.frozen_stages > 0
        if self.frozen_stages > 0:
            if self.frozen_stages > self.num_layers:
                self.frozen_stages = self.num_layers 
            self._freeze_stages()


    def init_weights(self) -> None:
        """Initialize position embedding, patch embedding and cls token."""
        super().init_weights()
        pos_embed = build_2d_sincos_position_embedding(
            int(self.num_patches**.5),
            self.pos_embed.shape[-1],
            cls_token=True)
        self.pos_embed.data.copy_(pos_embed.float())

        w = self.patch_embed.projection.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=.02)

    # def random_masking(
    #     self,
    #     x: torch.Tensor,
    #     mask_ratio: float = 0.75
    # ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     """Generate the mask for MAE Pre-training.

    #     Args:
    #         x (torch.Tensor): Image with data augmentation applied, which is
    #             of shape B x L x C.
    #         mask_ratio (float): The mask ratio of total patches.
    #             Defaults to 0.75.

    #     Returns:
    #         Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #             masked image, mask and the ids to restore original image.
    #               - x_masked (torch.Tensor): masked image.
    #               - mask (torch.Tensor): mask used to mask image.
    #               - ids_restore (torch.Tensor): ids to restore original image.
    #     """
    #     N, L, D = x.shape  # batch, length, dim
    #     len_keep = int(L * (1 - mask_ratio))

    #     noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

    #     # sort noise for each sample
    #     ids_shuffle = torch.argsort(
    #         noise, dim=1)  # ascend: small is keep, large is remove
    #     ids_restore = torch.argsort(ids_shuffle, dim=1)

    #     # keep the first subset
    #     ids_keep = ids_shuffle[:, :len_keep]
    #     x_masked = torch.gather(
    #         x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    #     # generate the binary mask: 0 is keep, 1 is remove
    #     mask = torch.ones([N, L], device=x.device)
    #     mask[:, :len_keep] = 0
    #     # unshuffle to get the binary mask
    #     mask = torch.gather(mask, dim=1, index=ids_restore)

    #     return x_masked, mask, ids_restore

    # def forward(
    #         self, x: torch.Tensor
    # ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     """Generate features for masked images.

    #     This function generates mask and masks some patches randomly and get
    #     the hidden features for visible patches.

    #     Args:
    #         x (torch.Tensor): Input images, which is of shape B x C x H x W.

    #     Returns:
    #         Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    #         Hidden features, mask and the ids to restore original image.

    #             - x (torch.Tensor): hidden features, which is of shape
    #               B x (L * mask_ratio) x C.
    #             - mask (torch.Tensor): mask used to mask image.
    #             - ids_restore (torch.Tensor): ids to restore original image.
    #     """
    #     B = x.shape[0]
    #     x = self.patch_embed(x)[0]
    #     # add pos embed w/o cls token
    #     x = x + self.pos_embed[:, 1:, :]

    #     # masking: length -> length * mask_ratio
    #     x, mask, ids_restore = self.random_masking(x, self.mask_ratio)

    #     # append cls token
    #     cls_token = self.cls_token + self.pos_embed[:, :1, :]
    #     cls_tokens = cls_token.expand(B, -1, -1)
    #     x = torch.cat((cls_tokens, x), dim=1)

    #     for _, layer in enumerate(self.layers):
    #         x = layer(x)
    #     # Use final norm
    #     x = self.norm1(x)

    #     return (x, mask, ids_restore)
