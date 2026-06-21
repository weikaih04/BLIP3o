"""2nd-stage discrete tokenizer for TRELLIS.2 SLAT latents (Kyvo-style, adapted to 3D).

Pipeline (per modality, shape OR tex — each its own tokenizer):
    sparse SLAT {coords(N,3)∈[0,32), feats(N,32)}
      → densify → dense (32+1, 32,32,32)  [32 feat channels + 1 occupancy mask]
      → Encoder3D (2× stride-2)  → (z_ch, 8,8,8)
      → VQ (codebook n_e)        → 8³ = 512 discrete tokens
      → Decoder3D (2× up)        → (32+1, 32,32,32)  [feat recon + occupancy logit]

The frozen TRELLIS.2 VAE is NOT touched — this compresses its latent `z` into tokens and
back to `ẑ`. Reference: VQGAN (CompVis/taming-transformers) straight-through VQ, made 3D.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- densify
def densify(coords: torch.Tensor, feats: torch.Tensor, res: int = 32) -> torch.Tensor:
    """Scatter sparse (coords, feats) into a dense grid + occupancy channel.

    coords: (N,3) integer in [0,res).   feats: (N,C) float.
    returns: (C+1, res, res, res)  — last channel is the binary occupancy mask.
    """
    C = feats.shape[1]
    grid = feats.new_zeros(C + 1, res, res, res)
    x, y, z = coords[:, 0].long(), coords[:, 1].long(), coords[:, 2].long()
    grid[:C, x, y, z] = feats.t().to(grid.dtype)   # features at active voxels
    grid[C, x, y, z] = 1.0                          # occupancy = 1 at active voxels
    return grid


# ----------------------------------------------------------------------------- VQ
class VectorQuantizer(nn.Module):
    """Standard straight-through VQ (taming-transformers formulation), N-D safe.

    Input z: (B, e_dim, *spatial). Returns (z_q same shape, vq_loss, indices (B,*spatial))."""
    def __init__(self, n_e: int, e_dim: int, beta: float = 0.25):
        super().__init__()
        self.n_e, self.e_dim, self.beta = n_e, e_dim, beta
        self.embedding = nn.Embedding(n_e, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)

    def forward(self, z: torch.Tensor):
        # (B, C, *spatial) -> (B, *spatial, C)
        perm = (0,) + tuple(range(2, z.dim())) + (1,)
        z_p = z.permute(*perm).contiguous()
        flat = z_p.view(-1, self.e_dim)
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2 z·e
        d = (flat.pow(2).sum(1, keepdim=True)
             + self.embedding.weight.pow(2).sum(1)
             - 2 * flat @ self.embedding.weight.t())
        idx = d.argmin(1)
        z_q = self.embedding(idx).view(z_p.shape)
        # codebook + commitment loss
        vq_loss = F.mse_loss(z_q.detach(), z_p) + self.beta * F.mse_loss(z_q, z_p.detach())
        # straight-through
        z_q = z_p + (z_q - z_p).detach()
        # back to (B, C, *spatial)
        inv = (0, z.dim() - 1) + tuple(range(1, z.dim() - 1))
        z_q = z_q.permute(*inv).contiguous()
        idx = idx.view(z_p.shape[:-1])  # (B, *spatial)
        return z_q, vq_loss, idx


class FSQ(nn.Module):
    """Finite Scalar Quantization (Mentzer et al. 2023) — codebook-free, no collapse.

    Input z: (B, d, *spatial) with d = len(levels). Implicit codebook = prod(levels).
    Returns (z_q same shape, zero_loss, indices (B,*spatial))."""
    def __init__(self, levels=(8, 8, 8, 4, 4)):
        super().__init__()
        self.register_buffer("_levels", torch.tensor(levels, dtype=torch.float32))
        basis = torch.cumprod(torch.tensor((1,) + tuple(levels[:-1]), dtype=torch.long), 0)
        self.register_buffer("_basis", basis)
        self.d = len(levels)
        self.codebook_size = int(torch.prod(self._levels).item())

    @staticmethod
    def _round_ste(z):
        return z + (z.round() - z).detach()

    def _bound(self, z, eps=1e-3):
        half_l = (self._levels - 1) * (1 - eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return torch.tanh(z + shift) * half_l - offset

    def forward(self, z):
        perm = (0,) + tuple(range(2, z.dim())) + (1,)
        z_p = z.permute(*perm).contiguous()                  # (B,*spatial,d)
        codes = self._round_ste(self._bound(z_p))            # integers centered at 0
        half_w = (self._levels // 2)
        z_q = codes / half_w                                 # normalize to ~[-1,1]
        # integer index per token (for AR / usage stats)
        idx_levels = (codes + half_w).long()                 # in [0, level)
        idx = (idx_levels * self._basis).sum(-1)             # (B,*spatial)
        inv = (0, z.dim() - 1) + tuple(range(1, z.dim() - 1))
        z_q = z_q.permute(*inv).contiguous()
        return z_q, z.new_zeros(()), idx


# ----------------------------------------------------------------------------- conv blocks
def _gn(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(32, c), c)


class ResBlock3D(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.n1 = _gn(c_in); self.c1 = nn.Conv3d(c_in, c_out, 3, 1, 1)
        self.n2 = _gn(c_out); self.c2 = nn.Conv3d(c_out, c_out, 3, 1, 1)
        self.skip = nn.Conv3d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class Encoder3D(nn.Module):
    """(B, in_ch, R,R,R) -> (B, z_ch, R/4, R/4, R/4)  via 2 stride-2 downsamples."""
    def __init__(self, in_ch: int, z_ch: int, base: int = 64, mults=(1, 2, 4)):
        super().__init__()
        self.stem = nn.Conv3d(in_ch, base, 3, 1, 1)
        chs = [base * m for m in mults]
        blocks = []
        c = base
        for i, ch in enumerate(chs):
            blocks.append(ResBlock3D(c, ch))
            if i < len(chs) - 1:                       # downsample after all but last
                blocks.append(nn.Conv3d(ch, ch, 3, 2, 1))
            c = ch
        self.blocks = nn.ModuleList(blocks)
        self.out = nn.Sequential(_gn(c), nn.SiLU(), nn.Conv3d(c, z_ch, 1))

    def forward(self, x):
        x = self.stem(x)
        for b in self.blocks:
            x = b(x)
        return self.out(x)


class Decoder3D(nn.Module):
    """(B, z_ch, r,r,r) -> (B, out_ch, 4r,4r,4r)  via 2 upsamples."""
    def __init__(self, z_ch: int, out_ch: int, base: int = 64, mults=(4, 2, 1)):
        super().__init__()
        chs = [base * m for m in mults]
        self.inp = nn.Conv3d(z_ch, chs[0], 3, 1, 1)
        blocks = []
        c = chs[0]
        for i, ch in enumerate(chs):
            blocks.append(ResBlock3D(c, ch))
            if i < len(chs) - 1:
                blocks.append(nn.Upsample(scale_factor=2, mode="nearest"))
                blocks.append(nn.Conv3d(ch, ch, 3, 1, 1))
            c = ch
        self.blocks = nn.ModuleList(blocks)
        self.out = nn.Sequential(_gn(c), nn.SiLU(), nn.Conv3d(c, out_ch, 1))

    def forward(self, x):
        x = self.inp(x)
        for b in self.blocks:
            x = b(x)
        return self.out(x)


# ----------------------------------------------------------------------------- tokenizer
class SlatTokenizer(nn.Module):
    """3D quantized autoencoder over a densified SLAT latent. feat_ch = SLAT channels (32).

    quant: "fsq" (codebook-free, no collapse — default) or "vq" (plain VQ, collapse-prone)."""
    def __init__(self, feat_ch: int = 32, quant: str = "fsq", z_ch: int = 8, n_e: int = 8192,
                 fsq_levels=(8, 8, 8, 4, 4), base: int = 64):
        super().__init__()
        self.feat_ch, self.quant = feat_ch, quant
        if quant == "fsq":
            self.q = FSQ(fsq_levels); z_dim = self.q.d
            self.codebook_size = self.q.codebook_size
        else:
            self.q = VectorQuantizer(n_e, z_ch); z_dim = z_ch
            self.codebook_size = n_e
        self.encoder = Encoder3D(feat_ch + 1, z_dim, base)     # +1 occupancy input
        self.decoder = Decoder3D(z_dim, feat_ch + 1, base)     # +1 occupancy logit out

    def forward(self, dense: torch.Tensor):
        """dense: (B, feat_ch+1, R,R,R). returns (recon (B,feat_ch+1,R,R,R), q_loss, idx)."""
        z = self.encoder(dense)
        z_q, q_loss, idx = self.q(z)
        recon = self.decoder(z_q)
        return recon, q_loss, idx


def tokenizer_losses(recon: torch.Tensor, dense_gt: torch.Tensor, feat_ch: int = 32):
    """Returns dict(feat_l1, occ_bce, occ_iou). feat L1 only on GT-active voxels."""
    feat_pred, occ_logit = recon[:, :feat_ch], recon[:, feat_ch:feat_ch + 1]
    feat_gt, occ_gt = dense_gt[:, :feat_ch], dense_gt[:, feat_ch:feat_ch + 1]
    # occupancy is ~5% positive (sparse) → weight the positive class or BCE collapses to all-empty.
    pos = occ_gt.sum().clamp(min=1.0)
    pos_weight = ((occ_gt.numel() - pos) / pos).detach()
    occ_bce = F.binary_cross_entropy_with_logits(occ_logit, occ_gt, pos_weight=pos_weight)
    mask = occ_gt > 0.5
    feat_l1 = F.l1_loss(feat_pred[mask.expand_as(feat_pred)], feat_gt[mask.expand_as(feat_gt)])
    with torch.no_grad():
        pred_occ = occ_logit > 0
        inter = (pred_occ & mask).sum().float()
        union = (pred_occ | mask).sum().float().clamp(min=1)
        occ_iou = inter / union
    return {"feat_l1": feat_l1, "occ_bce": occ_bce, "occ_iou": occ_iou}
