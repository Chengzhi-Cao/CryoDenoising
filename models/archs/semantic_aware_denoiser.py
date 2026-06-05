import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvBlock, self).__init__()
        padding = kernel_size // 2
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding),
            nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        return self.body(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1))

    def forward(self, x):
        return x + self.body(x)


class ResidualStack(nn.Module):
    def __init__(self, channels, num_blocks):
        super(ResidualStack, self).__init__()
        self.body = nn.Sequential(*[ResidualBlock(channels)
                                    for _ in range(num_blocks)])

    def forward(self, x):
        return self.body(x)


class FrequencyBranch(nn.Module):
    """Amplitude processing in Fourier space with a spatial fallback."""

    def __init__(self, channels):
        super(FrequencyBranch, self).__init__()
        self.pre = nn.Conv2d(channels, channels, 1, 1, 0)
        self.amp = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 1, 1, 0))
        self.post = nn.Conv2d(channels, channels, 1, 1, 0)

    def _fft_new(self, x):
        freq = torch.fft.rfft2(x, norm='ortho')
        amp = torch.sqrt(freq.real * freq.real + freq.imag * freq.imag + 1e-8)
        phase = torch.atan2(freq.imag, freq.real)
        amp = self.amp(amp)
        real = amp * torch.cos(phase)
        imag = amp * torch.sin(phase)
        freq = torch.complex(real, imag)
        return torch.fft.irfft2(freq, s=x.shape[-2:], norm='ortho')

    def _fft_old(self, x):
        freq = torch.rfft(x, signal_ndim=2, normalized=True, onesided=True)
        real = freq[..., 0]
        imag = freq[..., 1]
        amp = torch.sqrt(real * real + imag * imag + 1e-8)
        phase = torch.atan2(imag, real)
        amp = self.amp(amp)
        real = amp * torch.cos(phase)
        imag = amp * torch.sin(phase)
        freq = torch.stack([real, imag], dim=-1)
        return torch.irfft(
            freq, signal_ndim=2, normalized=True, onesided=True,
            signal_sizes=x.shape[-2:])

    def forward(self, x):
        x = self.pre(x)
        if hasattr(torch, 'fft') and hasattr(torch.fft, 'rfft2'):
            out = self._fft_new(x)
        elif hasattr(torch, 'rfft') and hasattr(torch, 'irfft'):
            out = self._fft_old(x)
        else:
            out = self.amp(x)
        return self.post(out)


class CrossInteractionBlock(nn.Module):
    def __init__(self, channels):
        super(CrossInteractionBlock, self).__init__()
        self.spatial_to_freq = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.Sigmoid())
        self.freq_to_spatial = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.Sigmoid())
        self.fuse_f = nn.Conv2d(channels * 2, channels, 1, 1, 0)
        self.fuse_s = nn.Conv2d(channels * 2, channels, 1, 1, 0)

    def forward(self, freq_feat, spatial_feat):
        f_gate = self.spatial_to_freq(spatial_feat)
        s_gate = self.freq_to_spatial(freq_feat)
        freq_out = self.fuse_f(torch.cat(
            [freq_feat, freq_feat * f_gate + spatial_feat], dim=1))
        spatial_out = self.fuse_s(torch.cat(
            [spatial_feat, spatial_feat * s_gate + freq_feat], dim=1))
        return freq_out, spatial_out


class SpatialFrequencyBlock(nn.Module):
    def __init__(self, channels, num_blocks):
        super(SpatialFrequencyBlock, self).__init__()
        self.freq = FrequencyBranch(channels)
        self.spatial = ResidualStack(channels, num_blocks)
        self.cross = CrossInteractionBlock(channels)

    def forward(self, x):
        freq_feat = x
        spatial_feat = x
        freq_feat = self.freq(freq_feat)
        spatial_feat = self.spatial(spatial_feat)
        freq_feat, spatial_feat = self.cross(freq_feat, spatial_feat)
        return freq_feat + spatial_feat


class CrossLocalRelationshipBlock(nn.Module):
    """Graph-style attention over pooled local semantic regions."""

    def __init__(self, channels, patch_grid=8):
        super(CrossLocalRelationshipBlock, self).__init__()
        self.patch_grid = patch_grid
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.proj = nn.Linear(channels, channels)
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True))
        self.out = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        b, c, h, w = x.size()
        grid_h = min(self.patch_grid, h)
        grid_w = min(self.patch_grid, w)
        local_feat = self.local(x)
        nodes = F.adaptive_avg_pool2d(local_feat, (grid_h, grid_w))
        nodes = nodes.flatten(2).transpose(1, 2)

        q = self.query(nodes)
        k = self.key(nodes)
        v = self.value(nodes)
        attn = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(float(c))
        attn = F.softmax(attn, dim=-1)
        related = torch.bmm(attn, v)
        related = self.proj(related).transpose(1, 2)
        related = related.contiguous().view(b, c, grid_h, grid_w)
        related = F.interpolate(related, size=(h, w), mode='bilinear',
                                align_corners=False)
        return x + self.out(related)


class MultiScaleFusionBlock(nn.Module):
    def __init__(self, channels):
        super(MultiScaleFusionBlock, self).__init__()
        self.weight_semantic = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.Sigmoid())
        self.weight_noise = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.Sigmoid())
        self.refine = ResidualBlock(channels)

    def forward(self, semantic_feat, noise_feat):
        shared = semantic_feat + noise_feat
        semantic_weight = self.weight_semantic(shared)
        noise_weight = self.weight_noise(shared)
        semantic_out = semantic_feat + semantic_weight * noise_feat
        noise_out = noise_feat + noise_weight * semantic_feat
        return self.refine(semantic_out + noise_out)


class SemanticAwareDenoiser(nn.Module):
    """Semantic-aware micrograph denoising network.

    The network follows the paper pipeline at implementation level: semantic
    masks split backbone features into semantic and non-semantic components,
    the semantic branch models cross-region relationships, the non-semantic
    branch combines spatial and Fourier-domain processing, and both branches
    are fused at multiple scales for Noise2Noise-style restoration.
    """

    def __init__(self, in_nc=3, mask_nc=1, out_nc=3, nf=48, num_blocks=4,
                 num_sf_blocks=3, patch_grid=8):
        super(SemanticAwareDenoiser, self).__init__()
        self.mask_nc = mask_nc
        self.stem = ConvBlock(in_nc, nf)

        self.encoder1 = ResidualStack(nf, num_blocks)
        self.down1 = nn.Conv2d(nf, nf * 2, 3, 2, 1)
        self.encoder2 = ResidualStack(nf * 2, num_blocks)
        self.down2 = nn.Conv2d(nf * 2, nf * 4, 3, 2, 1)
        self.bottleneck = ResidualStack(nf * 4, num_blocks)

        self.semantic1 = CrossLocalRelationshipBlock(nf, patch_grid)
        self.semantic2 = CrossLocalRelationshipBlock(nf * 2, patch_grid)
        self.semantic3 = CrossLocalRelationshipBlock(nf * 4, patch_grid)

        self.noise1 = SpatialFrequencyBlock(nf, num_sf_blocks)
        self.noise2 = SpatialFrequencyBlock(nf * 2, num_sf_blocks)
        self.noise3 = SpatialFrequencyBlock(nf * 4, num_sf_blocks)

        self.fuse1 = MultiScaleFusionBlock(nf)
        self.fuse2 = MultiScaleFusionBlock(nf * 2)
        self.fuse3 = MultiScaleFusionBlock(nf * 4)

        self.up2 = nn.Conv2d(nf * 4, nf * 2, 3, 1, 1)
        self.dec2 = ResidualStack(nf * 2, num_blocks)
        self.up1 = nn.Conv2d(nf * 2, nf, 3, 1, 1)
        self.dec1 = ResidualStack(nf, num_blocks)
        self.out = nn.Conv2d(nf, out_nc, 3, 1, 1)

    @staticmethod
    def _resize_mask(mask, feat):
        if mask is None:
            return feat.new_zeros(feat.size(0), 1, feat.size(2), feat.size(3))
        if mask.size(1) > 1:
            mask = mask.mean(dim=1, keepdim=True)
        if mask.shape[-2:] != feat.shape[-2:]:
            mask = F.interpolate(mask, size=feat.shape[-2:], mode='bilinear',
                                 align_corners=False)
        return mask.clamp(0, 1)

    def _split_features(self, feat, mask):
        mask = self._resize_mask(mask, feat)
        semantic_feat = feat * mask
        noise_feat = feat * (1 - mask)
        return semantic_feat, noise_feat

    def _process_scale(self, feat, mask, semantic_branch, noise_branch,
                       fuse_branch):
        semantic_feat, noise_feat = self._split_features(feat, mask)
        semantic_feat = semantic_branch(semantic_feat)
        noise_feat = noise_branch(noise_feat)
        return fuse_branch(semantic_feat, noise_feat)

    def forward(self, x, semantic_mask=None):
        residual = x
        feat1 = self.encoder1(self.stem(x))
        fused1 = self._process_scale(
            feat1, semantic_mask, self.semantic1, self.noise1, self.fuse1)

        feat2 = self.encoder2(F.leaky_relu(self.down1(fused1), 0.1,
                                           inplace=True))
        fused2 = self._process_scale(
            feat2, semantic_mask, self.semantic2, self.noise2, self.fuse2)

        feat3 = self.bottleneck(F.leaky_relu(self.down2(fused2), 0.1,
                                             inplace=True))
        fused3 = self._process_scale(
            feat3, semantic_mask, self.semantic3, self.noise3, self.fuse3)

        up2 = F.interpolate(fused3, size=fused2.shape[-2:], mode='bilinear',
                            align_corners=False)
        up2 = self.dec2(self.up2(up2) + fused2)
        up1 = F.interpolate(up2, size=fused1.shape[-2:], mode='bilinear',
                            align_corners=False)
        up1 = self.dec1(self.up1(up1) + fused1)
        out = self.out(up1)
        if out.shape == residual.shape:
            out = out + residual
        return out
