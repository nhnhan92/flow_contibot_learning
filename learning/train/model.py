
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
import torchvision

# ─────────────────────────────────────────────────────────────────────────────
# Shared primitives
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings for diffusion timestep"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Conv1dBlock(nn.Module):
    """
    1D Convolution block: Conv1d → GroupNorm → Mish

    padding is auto-set to kernel_size // 2 so sequence length is preserved.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      padding=kernel_size // 2),
            nn.GroupNorm(groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class SpatialSoftmax(nn.Module):
    """
    Spatial Softmax pooling — Stanford-style vision pooling for manipulation.

    For each of the K selected channels in the (B, C, H, W) feature map,
    computes a spatial probability distribution via softmax and returns the
    *expected (x, y) position* — effectively learning 2D keypoint locations
    rather than averaging features spatially.

    Why it matters for manipulation:
        GlobalAveragePool answers "what features are present?"
        SpatialSoftmax answers "WHERE are those features?" — the (x,y)
        coordinates are directly useful for spatial reasoning (reaching,
        grasping, tracking an object).

    Output:
        (B, num_keypoints * 2)  — interleaved [x0, y0, x1, y1, ...]
        Coordinates are normalised to [-1, 1].

    Args:
        num_keypoints  : K channels to turn into keypoints (default 32).
                         Selects the first K channels of the feature map.
        temperature    : Softmax sharpness.  Lower → more uniform (softer);
                         higher → more peaked (more localized).  Default 1.0.
        learnable_temp : Make temperature a learned parameter.

    """

    def __init__(self, num_keypoints=32, temperature=1.0, learnable_temp=False):
        super().__init__()
        self.num_keypoints = num_keypoints
        if learnable_temp:
            self.temperature = nn.Parameter(torch.ones(1) * temperature)
        else:
            self.register_buffer('temperature', torch.tensor(float(temperature)))

    def forward(self, x):
        """
        Args:
            x : (B, C, H, W)  — backbone feature map
        Returns:
            keypoints : (B, num_keypoints * 2)
        """
        B, C, H, W = x.shape
        K = min(self.num_keypoints, C)

        # Select first K channels
        x = x[:, :K, :, :]                         # (B, K, H, W)

        # Flatten spatial dims; temperature-scaled softmax over H×W
        x_flat = x.reshape(B, K, H * W)            # (B, K, H*W)
        attn    = torch.softmax(x_flat / self.temperature, dim=-1)
        attn    = attn.reshape(B, K, H, W)          # (B, K, H, W)

        # Normalised coordinate grids in [-1, 1]
        device  = x.device
        xs = torch.linspace(-1.0, 1.0, W, device=device)  # (W,)
        ys = torch.linspace(-1.0, 1.0, H, device=device)  # (H,)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # (H, W)

        # Expected x and y positions for each keypoint
        exp_x = (attn * grid_x[None, None]).sum(dim=(-2, -1))  # (B, K)
        exp_y = (attn * grid_y[None, None]).sum(dim=(-2, -1))  # (B, K)

        # Interleave into (B, K*2)
        keypoints = torch.stack([exp_x, exp_y], dim=-1)  # (B, K, 2)
        return keypoints.reshape(B, K * 2)               # (B, K*2)

# ─────────────────────────────────────────────────────────────────────────────
# v2 — Stanford-style FiLM UNet (ConditionalResidualBlock1D + ConditionalUNet1D)
# ─────────────────────────────────────────────────────────────────────────────

class Downsample1D(nn.Module):
    """Learnable 2× downsampling via strided convolution (Stanford-style)"""
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample1D(nn.Module):
    """Learnable 2× upsampling via transposed convolution (Stanford-style)"""
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class ConditionalResidualBlock1D(nn.Module):
    """
    Residual block with FiLM conditioning (Feature-wise Linear Modulation).

    Closely follows Stanford diffusion_policy ConditionalResidualBlock1D:

        x ──► Conv1dBlock ──► FiLM(cond) ──► Conv1dBlock ──► + ──► out
        │                                                      │
        └──────────── residual_proj (1×1 conv if needed) ─────┘

    FiLM injects the global condition (obs features + timestep embedding)
    multiplicatively (scale) and additively (bias) between the two convolutions,
    making the conditioning far more expressive than simple additive injection.

    Args:
        in_channels  : Input channel count
        out_channels : Output channel count
        cond_dim     : Global condition dimension (obs + timestep concat)
        kernel_size  : Conv kernel size (default 5, same as Stanford)
        n_groups     : GroupNorm groups (default 8)
    """

    def __init__(self, in_channels, out_channels, cond_dim,
                 kernel_size=5, n_groups=8):
        super().__init__()
        self.out_channels = out_channels

        # Two convolution blocks
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels,  out_channels, kernel_size=kernel_size, groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size=kernel_size, groups=n_groups),
        ])

        # FiLM projection: cond_dim → (scale, bias) for out_channels
        # Applied between the two Conv1dBlocks.
        # Output: 2 * out_channels → first half = scale, second half = bias
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_channels * 2),
        )

        # Residual projection: 1×1 conv when in_channels ≠ out_channels
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, cond):
        """
        Args:
            x    : (B, in_channels, T)
            cond : (B, cond_dim)  — global FiLM condition

        Returns:
            out  : (B, out_channels, T)
        """
        # ── First conv ────────────────────────────────────────────────────────
        out = self.blocks[0](x)             # (B, out_channels, T)

        # ── FiLM modulation ───────────────────────────────────────────────────
        film = self.cond_encoder(cond)       # (B, out_channels * 2)
        film = film.unsqueeze(-1)            # (B, out_channels * 2, 1)
        scale, bias = film.chunk(2, dim=1)  # each (B, out_channels, 1)

        # scale * out + bias  (broadcast over time dimension T)
        out = scale * out + bias

        # ── Second conv ───────────────────────────────────────────────────────
        out = self.blocks[1](out)            # (B, out_channels, T)

        # ── Residual connection ───────────────────────────────────────────────
        return out + self.residual_conv(x)


class ConditionalUNet1D(nn.Module):
    """
    1D Temporal U-Net with FiLM conditioning — Stanford-style (v2).

    Key differences from DiffusionUNet1D (v1):
      • ConditionalResidualBlock1D at every level with true residual connections
      • FiLM: scale + bias injected per-block (not additive at input only)
      • Diffusion timestep embedding is concatenated with obs features and
        passed as a single global FiLM condition into every ResBlock
      • Learnable Downsample1D (strided conv) / Upsample1D (transposed conv)
        instead of MaxPool1d / bilinear Upsample
      • Kernel size 5 (vs 3 in v1)
      • Timestep embed dim 256 (vs 128 in v1)
      • input_proj / final_conv isolate the UNet body from the raw action_dim,
        keeping all GroupNorm operations on channel counts ≥ start_dim=256

    Architecture (down_dims=[256, 512, 1024], pred_horizon=16):

        [B, action_dim=9, 16]
            ↓ input_proj (Conv1d 9→256)
        [B, 256, 16]
            ↓ Level-0 ResBlocks (256→256)  → skip0[B, 256, 16]
            ↓ Downsample                   → [B, 256, 8]
            ↓ Level-1 ResBlocks (256→512)  → skip1[B, 512, 8]
            ↓ Downsample                   → [B, 512, 4]
            ↓ Level-2 ResBlocks (512→1024) → skip2[B, 1024, 4]
              (no downsample — is_last)     → [B, 1024, 4]
            ↓ Bottleneck ResBlocks
        [B, 1024, 4]
            ↑ cat(skip2)=[2048] → ResBlocks(2048→512) → Upsample → [B, 512, 8]
            ↑ cat(skip1)=[1024] → ResBlocks(1024→256) → Upsample → [B, 256, 16]
            ↑ cat(skip0)=[512]  → ResBlocks(512→256)  → Identity → [B, 256, 16]
            ↓ final_conv (Conv1dBlock 256→256, Conv1d 256→9)
        [B, action_dim=9, 16]

    Args:
        input_dim               : Action dimension (e.g. 9)
        global_cond_dim         : Obs feature dimension (vision + state)
        diffusion_step_embed_dim: Sinusoidal timestep embed dim (default 256)
        down_dims               : Channel widths per encoder level
        kernel_size             : Conv kernel (default 5)
        n_groups                : GroupNorm groups (default 8)
    """

    def __init__(
        self,
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
    ):
        super().__init__()

        start_dim = down_dims[0]   # 256

        # ── Timestep embedding (inside UNet, like Stanford) ───────────────────
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )

        # Global FiLM condition = obs_features + timestep_embed (concatenated)
        cond_dim = global_cond_dim + diffusion_step_embed_dim

        # ── Input projection ──────────────────────────────────────────────────
        # Lifts action_dim → start_dim so all UNet channels are ≥ start_dim.
        # This avoids GroupNorm(8, action_dim=9) which would fail.
        self.input_proj = nn.Conv1d(input_dim, start_dim, kernel_size=1)

        # ── Build encoder / decoder dims ──────────────────────────────────────
        # all_dims = [start_dim=256] + down_dims = [256, 256, 512, 1024]
        # in_out   = [(256,256), (256,512), (512,1024)]
        all_dims = [start_dim] + list(down_dims)
        in_out   = list(zip(all_dims[:-1], all_dims[1:]))

        # ── Encoder (downsampling path) ───────────────────────────────────────
        self.down_modules = nn.ModuleList()
        for idx, (dim_in, dim_out) in enumerate(in_out):
            is_last = idx == len(in_out) - 1
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_in,  dim_out, cond_dim, kernel_size, n_groups),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim, kernel_size, n_groups),
                # Downsample at every level except the deepest
                Downsample1D(dim_out) if not is_last else nn.Identity(),
            ]))

        # ── Bottleneck ────────────────────────────────────────────────────────
        mid_dim = all_dims[-1]   # 1024
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups),
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups),
        ])

        # ── Decoder (upsampling path) ─────────────────────────────────────────
        # reversed(in_out) = [(512,1024), (256,512), (256,256)]
        # Skip connections double the input channels at each level.
        self.up_modules = nn.ModuleList()
        for idx, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = idx == len(in_out) - 1
            self.up_modules.append(nn.ModuleList([
                # dim_out * 2 because of skip-connection concatenation
                ConditionalResidualBlock1D(dim_out * 2, dim_in, cond_dim, kernel_size, n_groups),
                ConditionalResidualBlock1D(dim_in,      dim_in, cond_dim, kernel_size, n_groups),
                # Upsample at every level except the shallowest
                Upsample1D(dim_in) if not is_last else nn.Identity(),
            ]))

        # ── Output head ───────────────────────────────────────────────────────
        # Projects start_dim → input_dim (mirrors input_proj)
        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size, groups=n_groups),
            nn.Conv1d(start_dim, input_dim, kernel_size=1),
        )

    def forward(self, sample, timestep, global_cond):
        """
        Args:
            sample      : Noisy action sequence (B, action_dim, pred_horizon)
            timestep    : Diffusion timestep (B,)
            global_cond : Observation features (B, global_cond_dim)

        Returns:
            Predicted noise (B, action_dim, pred_horizon)
        """
        # ── Timestep embedding ────────────────────────────────────────────────
        t_emb = self.diffusion_step_encoder(timestep)   # (B, step_embed_dim)

        # ── Global FiLM condition = obs + timestep ────────────────────────────
        film_cond = torch.cat([global_cond, t_emb], dim=-1)  # (B, cond_dim)

        # ── Input projection ──────────────────────────────────────────────────
        x = self.input_proj(sample)    # (B, start_dim, T)

        # ── Encoder ───────────────────────────────────────────────────────────
        skips = []
        for resnet1, resnet2, downsample in self.down_modules:
            x = resnet1(x, film_cond)
            x = resnet2(x, film_cond)
            skips.append(x)
            x = downsample(x)

        # ── Bottleneck ────────────────────────────────────────────────────────
        for mid in self.mid_modules:
            x = mid(x, film_cond)

        # ── Decoder ───────────────────────────────────────────────────────────
        for resnet1, resnet2, upsample in self.up_modules:
            skip = skips.pop()                           # LIFO skip connection
            x = torch.cat([x, skip], dim=1)             # concat along channels
            x = resnet1(x, film_cond)
            x = resnet2(x, film_cond)
            x = upsample(x)

        # ── Output ────────────────────────────────────────────────────────────
        return self.final_conv(x)


# ─────────────────────────────────────────────────────────────────────────────
# Vision encoders
# ─────────────────────────────────────────────────────────────────────────────

class VisionEncoder(nn.Module):
    """Simple CNN encoder for images (legacy, use ResNet18VisionEncoder instead)"""

    def __init__(self, obs_horizon=2, output_dim=256):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),   # 96→48
            nn.GroupNorm(8, 32), nn.Mish(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 48→24
            nn.GroupNorm(8, 64), nn.Mish(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 24→12
            nn.GroupNorm(8, 128), nn.Mish(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),# 12→6
            nn.GroupNorm(8, 256), nn.Mish(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.combine = nn.Sequential(
            nn.Linear(256 * obs_horizon, output_dim),
            nn.Mish(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        x = self.conv(x)
        x = x.reshape(B, T * 256)
        return self.combine(x)


class ResNet18VisionEncoder(nn.Module):
    """
    ResNet18-based vision encoder with switchable spatial pooling.

    Args:
        obs_horizon          : Number of observation frames (default 2)
        output_dim           : Output feature dimension (default 256)
        pretrained           : Use ImageNet pretrained ResNet18 weights
        use_spatial_softmax  : True → SpatialSoftmax,  False → AvgPool
        num_keypoints        : Keypoints for SpatialSoftmax (default 32 → 64-D/frame)
        crop_pad             : Random crop padding in pixels (0 = disabled)
    """

    def __init__(
        self,
        obs_horizon=2,
        output_dim=256,
        pretrained=True,
        use_spatial_softmax=False,
        num_keypoints=32,
        crop_pad=0,
    ):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.use_spatial_softmax = use_spatial_softmax
        self.crop_pad = crop_pad

        # ── ResNet18 backbone (up to layer4, 512 channels) ────────────────────
        resnet = torchvision.models.resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )

        # ── Spatial pooling ───────────────────────────────────────────────────
        if use_spatial_softmax:
            self.pool = SpatialSoftmax(num_keypoints=num_keypoints)
            feat_per_frame = num_keypoints * 2   # e.g. 32 kp → 64-D
        else:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            feat_per_frame = 512

        # ── Temporal projection ───────────────────────────────────────────────
        # Flattens all obs_horizon frames then projects to output_dim
        self.combine = nn.Sequential(
            nn.Linear(feat_per_frame * obs_horizon, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def _apply_random_crop(self, x):
        """
        Per-image independent random crop augmentation (training only).
        Args:
            x : (N, C, H, W)  — N = B * obs_horizon flattened images
        Returns:
            (N, C, H, W)  — same spatial dimensions
        """
        N, C, H, W = x.shape
        pad = self.crop_pad
        x = F.pad(x, (pad, pad, pad, pad), mode='reflect')   # (N, C, H+2p, W+2p)
        pH, pW = H + 2 * pad, W + 2 * pad
        tops  = torch.randint(0, pH - H + 1, (N,), device=x.device)
        lefts = torch.randint(0, pW - W + 1, (N,), device=x.device)
        return torch.stack([
            x[i, :, tops[i]:tops[i] + H, lefts[i]:lefts[i] + W]
            for i in range(N)
        ])

    def forward(self, x):
        """
        Args:
            x : (B, obs_horizon, C, H, W)
        Returns:
            (B, output_dim)
        """
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)

        # Random crop augmentation — training only, disabled at eval time
        if self.training and self.crop_pad > 0:
            x = self._apply_random_crop(x)

        x = self.backbone(x)          # (B*T, 512, H', W')

        if self.use_spatial_softmax:
            x = self.pool(x)          # (B*T, num_keypoints * 2)
        else:
            x = self.pool(x).flatten(1)  # (B*T, 512)

        feat_per_frame = x.shape[-1]
        x = x.reshape(B, T * feat_per_frame)   # (B, T * feat_per_frame)
        return self.combine(x)                  # (B, output_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Top-level policy
# ─────────────────────────────────────────────────────────────────────────────

class DiffusionPolicy(nn.Module):
    """
    Complete Diffusion Policy model.

    Combines:
      • Vision encoder  (ResNet18 or custom CNN)
      • State encoder   (MLP for proprioceptive state)
      • Diffusion U-Net (v1 simple additive OR v2 Stanford-style FiLM)

    Training:  DDPM with 100 steps → MSE loss on predicted noise
    Inference: DDIM with 16 steps  → 6× faster than DDPM

    Args:
        use_film_unet : True  → ConditionalUNet1D (FiLM, Stanford-style, recommended)
                        False → DiffusionUNet1D   (simple additive, legacy)
    """

    def __init__(
        self,
        obs_horizon=2,
        pred_horizon=16,
        action_dim=9,               # UR5e TCP pose (6D) + flowbot PWM (3D)
        state_dim=9,                # UR5e TCP pose (6D) + flowbot PWM (3D)
        vision_feature_dim=256,
        state_feature_dim=64,
        num_diffusion_iters=100,
        num_inference_steps=16,
        use_resnet=True,
        use_film_unet=True,         # True = FiLM UNet (v2), False = simple (v1)
        film_step_embed_dim=256,    # Timestep embed dim for FiLM UNet
        film_down_dims=None,        # Channel dims for FiLM UNet (default [256,512,1024])
        film_kernel_size=5,         # Conv kernel for FiLM UNet
        use_spatial_softmax=None,   # None → auto (True if use_film_unet else False)
        num_keypoints=32,           # SpatialSoftmax keypoints (Stanford default: 32)
        crop_pad=0,                 # Random crop padding pixels (0 = disabled)
    ):
        super().__init__()

        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.num_diffusion_iters = num_diffusion_iters
        self.num_inference_steps = num_inference_steps
        self.use_film_unet = use_film_unet

        # Auto-enable SpatialSoftmax when using the FiLM UNet (Stanford-style)
        if use_spatial_softmax is None:
            use_spatial_softmax = use_film_unet

        # ── Vision encoder ────────────────────────────────────────────────────
        if use_resnet:
            self.vision_encoder = ResNet18VisionEncoder(
                obs_horizon=obs_horizon,
                output_dim=vision_feature_dim,
                pretrained=True,
                use_spatial_softmax=use_spatial_softmax,
                num_keypoints=num_keypoints,
                crop_pad=crop_pad,
            )
        else:
            self.vision_encoder = VisionEncoder(obs_horizon, vision_feature_dim)

        # ── State encoder (MLP on flattened obs window) ───────────────────────
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim * obs_horizon, state_feature_dim),
            nn.Mish(),
            nn.Linear(state_feature_dim, state_feature_dim),
        )

        # ── Diffusion U-Net ───────────────────────────────────────────────────
        cond_dim = vision_feature_dim + state_feature_dim

        _down_dims = film_down_dims or [256, 512, 1024]
        self.diffusion_model = ConditionalUNet1D(
            input_dim=action_dim,
            global_cond_dim=cond_dim,
            diffusion_step_embed_dim=film_step_embed_dim,
            down_dims=_down_dims,
            kernel_size=film_kernel_size,
            n_groups=8,
        )


        # ── Noise schedulers ──────────────────────────────────────────────────
        self.noise_scheduler_train = DDPMScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon',
        )
        self.noise_scheduler_infer = DDIMScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon',
        )

    def forward(self, obs_state, obs_image, actions=None, train=True):
        """
        Args:
            obs_state : (B, obs_horizon, state_dim)
            obs_image : (B, obs_horizon, C, H, W)
            actions   : (B, pred_horizon, action_dim)  — training only
            train     : True → return loss,  False → return predicted actions

        Returns:
            train=True  : scalar loss
            train=False : (B, pred_horizon, action_dim)
        """
        B = obs_state.shape[0]

        # ── Encode observations ───────────────────────────────────────────────
        vision_features = self.vision_encoder(obs_image)              # (B, vision_dim)
        state_flat      = obs_state.reshape(B, -1)                    # (B, T*state_dim)
        state_features  = self.state_encoder(state_flat)              # (B, state_feat_dim)
        cond = torch.cat([vision_features, state_features], dim=-1)   # (B, cond_dim)

        if train:
            assert actions is not None

            # Actions: (B, T, action_dim) → (B, action_dim, T) for 1D conv
            actions = actions.transpose(1, 2)

            noise     = torch.randn_like(actions)
            timesteps = torch.randint(
                0, self.num_diffusion_iters,
                (B,), device=actions.device
            ).long()

            noisy_actions = self.noise_scheduler_train.add_noise(
                actions, noise, timesteps
            )

            noise_pred = self.diffusion_model(noisy_actions, timesteps, cond)

            return nn.functional.mse_loss(noise_pred, noise)

        else:
            # Inference: DDIM denoising from pure noise
            noisy_action = torch.randn(
                (B, self.action_dim, self.pred_horizon),
                device=obs_state.device
            )
            self.noise_scheduler_infer.set_timesteps(self.num_inference_steps)

            for t in self.noise_scheduler_infer.timesteps:
                noise_pred = self.diffusion_model(
                    noisy_action,
                    t.unsqueeze(0).expand(B).to(obs_state.device),
                    cond,
                )
                noisy_action = self.noise_scheduler_infer.step(
                    noise_pred, t, noisy_action, eta=0.0
                ).prev_sample

            # Back to (B, pred_horizon, action_dim)
            return noisy_action.transpose(1, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Noise schedulers (hand-rolled, equivalent to diffusers DDPMScheduler /
# DDIMScheduler with squaredcos_cap_v2 + prediction_type='epsilon')
# ─────────────────────────────────────────────────────────────────────────────

class DDPMScheduler:
    """DDPM noise scheduler (training)"""

    def __init__(self, num_train_timesteps=100, beta_schedule='squaredcos_cap_v2',
                 clip_sample=True, prediction_type='epsilon'):
        self.num_train_timesteps = num_train_timesteps
        self.clip_sample = clip_sample
        self.prediction_type = prediction_type

        if beta_schedule == 'linear':
            self.betas = torch.linspace(1e-4, 0.02, num_train_timesteps)
        elif beta_schedule == 'squaredcos_cap_v2':
            steps = num_train_timesteps + 1
            x = torch.linspace(0, num_train_timesteps, steps)
            alphas_cumprod = torch.cos(
                ((x / num_train_timesteps) + 0.008) / 1.008 * np.pi * 0.5
            ) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, original_samples, noise, timesteps):
        alphas_cumprod = self.alphas_cumprod.to(original_samples.device)
        sqrt_alpha = alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus = (1 - alphas_cumprod[timesteps]) ** 0.5

        # Broadcast to sample shape
        while len(sqrt_alpha.shape) < len(original_samples.shape):
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
        while len(sqrt_one_minus.shape) < len(original_samples.shape):
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)

        return sqrt_alpha * original_samples + sqrt_one_minus * noise

    def set_timesteps(self, num_inference_steps):
        self.num_inference_steps = num_inference_steps
        self.timesteps = torch.linspace(
            self.num_train_timesteps - 1, 0, num_inference_steps
        ).long()

    def step(self, model_output, timestep, sample):
        t      = timestep
        prev_t = t - self.num_train_timesteps // self.num_inference_steps

        acp_t      = self.alphas_cumprod[t].to(sample.device)
        acp_t_prev = (self.alphas_cumprod[prev_t].to(sample.device)
                      if prev_t >= 0 else torch.tensor(1.0).to(sample.device))

        beta_t      = 1 - acp_t
        beta_t_prev = 1 - acp_t_prev

        if self.prediction_type == 'epsilon':
            pred_x0 = (sample - beta_t ** 0.5 * model_output) / acp_t ** 0.5
        else:
            pred_x0 = model_output

        if self.clip_sample:
            pred_x0 = torch.clamp(pred_x0, -1, 1)

        coeff_x0   = (acp_t_prev ** 0.5 * self.betas[t].to(sample.device)) / beta_t
        coeff_x    = self.alphas[t].to(sample.device) ** 0.5 * beta_t_prev / beta_t
        pred_prev  = coeff_x0 * pred_x0 + coeff_x * sample

        if t > 0:
            noise     = torch.randn_like(model_output)
            variance  = (self.betas[t].to(sample.device) * beta_t_prev / beta_t) ** 0.5
            pred_prev = pred_prev + variance * noise

        return type('obj', (object,), {'prev_sample': pred_prev})()


class DDIMScheduler:
    """
    DDIM scheduler for fast deterministic inference.
    """

    def __init__(self, num_train_timesteps=100, beta_schedule='squaredcos_cap_v2',
                 clip_sample=True, prediction_type='epsilon'):
        self.num_train_timesteps = num_train_timesteps
        self.clip_sample = clip_sample
        self.prediction_type = prediction_type

        if beta_schedule == 'linear':
            self.betas = torch.linspace(1e-4, 0.02, num_train_timesteps)
        elif beta_schedule == 'squaredcos_cap_v2':
            steps = num_train_timesteps + 1
            x = torch.linspace(0, num_train_timesteps, steps)
            alphas_cumprod = torch.cos(
                ((x / num_train_timesteps) + 0.008) / 1.008 * np.pi * 0.5
            ) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, original_samples, noise, timesteps):
        alphas_cumprod = self.alphas_cumprod.to(original_samples.device)
        sqrt_alpha = alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus = (1 - alphas_cumprod[timesteps]) ** 0.5
        while len(sqrt_alpha.shape) < len(original_samples.shape):
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
        while len(sqrt_one_minus.shape) < len(original_samples.shape):
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)
        return sqrt_alpha * original_samples + sqrt_one_minus * noise

    def set_timesteps(self, num_inference_steps):
        self.num_inference_steps = num_inference_steps
        step_ratio   = self.num_train_timesteps // num_inference_steps
        self.timesteps = torch.flip(
            torch.arange(0, num_inference_steps) * step_ratio, [0]
        ).long()

    def step(self, model_output, timestep, sample, eta=0.0):
        """
        DDIM denoising step.
        eta=0 → deterministic, eta=1 → stochastic (= DDPM).
        """
        alpha_prod_t = self.alphas_cumprod[timestep].to(sample.device)
        prev_t       = timestep - self.num_train_timesteps // self.num_inference_steps
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_t].to(sample.device)
            if prev_t >= 0 else torch.tensor(1.0).to(sample.device)
        )

        beta_prod_t      = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        if self.prediction_type == 'epsilon':
            pred_x0 = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        else:
            pred_x0 = model_output

        if self.clip_sample:
            pred_x0 = torch.clamp(pred_x0, -1, 1)

        variance = 0
        if eta > 0:
            variance = (beta_prod_t_prev / beta_prod_t) * (
                1 - alpha_prod_t / alpha_prod_t_prev
            )
            variance = torch.clamp(variance, min=1e-20)

        pred_dir    = (1 - alpha_prod_t_prev - eta * variance) ** 0.5 * model_output
        pred_prev   = alpha_prod_t_prev ** 0.5 * pred_x0 + pred_dir

        if eta > 0 and timestep > 0:
            noise     = torch.randn_like(model_output)
            pred_prev = pred_prev + (eta * variance) ** 0.5 * noise

        return type('obj', (object,), {'prev_sample': pred_prev})()
