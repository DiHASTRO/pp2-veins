"""Attention and dense skip fusion adapted from RAU-Net++ (Tang et al.).

Primary source: https://doi.org/10.3390/app14010251, sections 3.2.2 and 3.3.2.
RAFF follows the prose description of the three branches, rather than the
inconsistent ordering in equation (6). See README.md for adaptation choices.
"""

from contextlib import contextmanager, nullcontext

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.checkpoint import checkpoint


ATTENTION_REDUCTION = 16
BOTTLENECK_REDUCTION = 4


@contextmanager
def _temporary_batchnorm_buffers(module):
    """Recompute activations without updating BatchNorm running stats twice.

    Use temporary buffer objects, so tensors saved by autograd are never modified
    in place when restoring the running statistics from the original forward.
    """
    originals = []
    try:
        for layer in module.modules():
            if isinstance(layer, nn.BatchNorm2d) and layer.track_running_stats:
                buffers = (layer.running_mean, layer.running_var, layer.num_batches_tracked)
                originals.append((layer, buffers))
                layer.running_mean = buffers[0].clone()
                layer.running_var = buffers[1].clone()
                layer.num_batches_tracked = buffers[2].clone()
        yield
    finally:
        for layer, buffers in originals:
            layer.running_mean, layer.running_var, layer.num_batches_tracked = buffers


class ChannelSpatialAttention(nn.Module):
    """CBAM: shared channel MLP followed by a spatial attention gate."""

    def __init__(self, channels, reduction=ATTENTION_REDUCTION):
        super().__init__()
        hidden_channels = max(1, channels // reduction)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
        )
        self.average_pool = nn.AdaptiveAvgPool2d(1)
        self.maximum_pool = nn.AdaptiveMaxPool2d(1)
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, features):
        channel_gate = torch.sigmoid(
            self.channel_mlp(self.average_pool(features))
            + self.channel_mlp(self.maximum_pool(features))
        )
        attended = features * channel_gate
        spatial_features = torch.cat(
            [attended.mean(dim=1, keepdim=True), attended.amax(dim=1, keepdim=True)],
            dim=1,
        )
        return attended * torch.sigmoid(self.spatial_conv(spatial_features))


class ResidualAttentionFeatureFusion(nn.Module):
    """Shape-preserving RAFF: identity + local pooling + gated bottleneck."""

    def __init__(
        self,
        channels,
        bottleneck_reduction=BOTTLENECK_REDUCTION,
        attention_reduction=ATTENTION_REDUCTION,
    ):
        super().__init__()
        bottleneck_channels = max(1, channels // bottleneck_reduction)
        attention_channels = max(1, channels // attention_reduction)
        self.pool_branch = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.bottleneck_branch = nn.Sequential(
            nn.Conv2d(channels, bottleneck_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        # No BatchNorm on the pooled [B,C,1,1] tensor: the final batch may be 1.
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, attention_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(attention_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, features):
        return (
            features
            + self.pool_branch(features)
            + self.bottleneck_branch(features) * self.channel_gate(features)
        )


class RiverAttentionEncoder(nn.Module):
    """Keep encoder features intact except for CBAM at the deepest level."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.attention = ChannelSpatialAttention(encoder.out_channels[-1])

    @property
    def out_channels(self):
        return self.encoder.out_channels

    @property
    def output_stride(self):
        return self.encoder.output_stride

    def _checkpoint_contexts(self):
        return nullcontext(), _temporary_batchnorm_buffers(self.encoder)

    def forward(self, images):
        if self.training and torch.is_grad_enabled():
            # EfficientNet uses stochastic drop-connect. Preserve its RNG state
            # so the recomputed activations use the original masks.
            features = checkpoint(
                self.encoder,
                images,
                use_reentrant=False,
                context_fn=self._checkpoint_contexts,
                preserve_rng_state=True,
            )
        else:
            features = self.encoder(images)
        features = list(features)
        features[-1] = self.attention(features[-1])
        return features


class RiverSkipDecoderBlock(nn.Module):
    """Apply separate RAFF modules to incoming dense skips before concatenation.

    SMP concatenates equal-width skip features before calling DecoderBlock.
    Splitting them here preserves that decoder's topology and interpolation.
    """

    def __init__(self, block, feature_channels, skip_count):
        super().__init__()
        self.block = block
        self.feature_channels = feature_channels
        self.fusion = nn.ModuleList(
            [ResidualAttentionFeatureFusion(feature_channels) for _ in range(skip_count)]
        )

    def _decode(self, features, skip):
        skip_features = torch.split(skip, self.feature_channels, dim=1)
        refined_skip = torch.cat(
            [fusion(part) for fusion, part in zip(self.fusion, skip_features)], dim=1
        )
        return self.block(features, refined_skip)

    def _checkpoint_contexts(self):
        return nullcontext(), _temporary_batchnorm_buffers(self)

    def forward(self, features, skip=None):
        expected_channels = self.feature_channels * len(self.fusion)
        if skip is None or skip.shape[1] != expected_channels:
            raise ValueError(f"Expected a dense skip with {expected_channels} channels")
        if self.training and torch.is_grad_enabled():
            # These decoder blocks contain no stochastic layers. The encoder's
            # original drop-connect is outside this recomputed region.
            return checkpoint(
                self._decode,
                features,
                skip,
                use_reentrant=False,
                context_fn=self._checkpoint_contexts,
                preserve_rng_state=False,
            )
        return self._decode(features, skip)


class RiverUNetPlusPlus(nn.Module):
    """The baseline SMP architecture with encoder CBAM and per-edge RAFF."""

    def __init__(self, num_classes, encoder_weights="imagenet"):
        super().__init__()
        self.unet = smp.UnetPlusPlus(
            encoder_name="efficientnet-b3",
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
        )
        self.unet.encoder = RiverAttentionEncoder(self.unet.encoder)

        # Use SMP's existing blocks and forward; do not rebuild its dense graph.
        decoder = self.unet.decoder
        for layer_idx in range(decoder.depth):
            for depth_idx in range(layer_idx + 1):
                block_name = f"x_{depth_idx}_{layer_idx}"
                decoder.blocks[block_name] = RiverSkipDecoderBlock(
                    decoder.blocks[block_name],
                    feature_channels=decoder.skip_channels[layer_idx],
                    skip_count=layer_idx + 1 - depth_idx,
                )
        # The final full-resolution block has no skip and remains unchanged.

    def forward(self, images):
        return self.unet(images)
