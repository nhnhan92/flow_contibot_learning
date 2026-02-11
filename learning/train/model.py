"""
Diffusion Policy Model for Pick-Place
Based on Diffusion Policy paper: https://diffusion-policy.cs.columbia.edu/
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional
import torchvision


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
    """1D Convolution block with group norm"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.GroupNorm(groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class DiffusionUNet1D(nn.Module):
    """
    1D U-Net for action sequence diffusion

    Args:
        action_dim: Dimension of action (6 for robot pose)
        cond_dim: Dimension of conditioning (from vision + state encoder)
        diffusion_step_embed_dim: Embedding dimension for diffusion timestep
    """

    def __init__(
        self,
        action_dim=6,
        pred_horizon=16,
        cond_dim=256,
        diffusion_step_embed_dim=128,
        down_dims=[256, 512, 1024],
    ):
        super().__init__()

        self.action_dim = action_dim
        self.pred_horizon = pred_horizon

        # Timestep embedding
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )

        # Input projection
        self.input_proj = Conv1dBlock(action_dim, down_dims[0])

        # Condition projection
        self.cond_proj = nn.Linear(cond_dim, down_dims[0])

        # Downsample blocks
        self.down_modules = nn.ModuleList([])
        in_dim = down_dims[0]
        for out_dim in down_dims[1:]:
            self.down_modules.append(nn.ModuleList([
                Conv1dBlock(in_dim + diffusion_step_embed_dim, out_dim),
                Conv1dBlock(out_dim, out_dim),
                nn.MaxPool1d(2),
            ]))
            in_dim = out_dim

        # Middle
        self.mid_modules = nn.ModuleList([
            Conv1dBlock(in_dim + diffusion_step_embed_dim, in_dim),
            Conv1dBlock(in_dim + diffusion_step_embed_dim, in_dim),
        ])

        # Upsample blocks
        self.up_modules = nn.ModuleList([])
        for i, out_dim in enumerate(reversed(down_dims[:-1])):
            # Skip connection comes from corresponding down block
            # reversed(h) gives us [down_dims[-1], down_dims[-2], ...]
            skip_dim = list(reversed(down_dims))[i]
            self.up_modules.append(nn.ModuleList([
                nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
                Conv1dBlock(in_dim + skip_dim + diffusion_step_embed_dim, out_dim),
                Conv1dBlock(out_dim, out_dim),
            ]))
            in_dim = out_dim

        # Output
        self.final_conv = nn.Conv1d(in_dim, action_dim, 1)

    def forward(self, x, timestep, cond):
        """
        Args:
            x: Noisy actions (B, action_dim, pred_horizon)
            timestep: Diffusion timestep (B,)
            cond: Conditioning from encoder (B, cond_dim)

        Returns:
            Predicted noise (B, action_dim, pred_horizon)
        """
        # Timestep embedding (B, embed_dim)
        timestep_embed = self.diffusion_step_encoder(timestep)

        # Input projection
        x = self.input_proj(x)  # (B, down_dims[0], T)

        # Add conditioning
        cond_embed = self.cond_proj(cond)  # (B, down_dims[0])
        cond_embed = cond_embed[:, :, None]  # (B, down_dims[0], 1)
        x = x + cond_embed

        # Downsample
        h = []
        for down_conv1, down_conv2, pool in self.down_modules:
            # Add timestep
            t = timestep_embed[:, :, None].expand(-1, -1, x.shape[-1])
            x = torch.cat([x, t], dim=1)
            x = down_conv1(x)
            x = down_conv2(x)
            h.append(x)
            x = pool(x)

        # Middle
        for mid_conv in self.mid_modules:
            t = timestep_embed[:, :, None].expand(-1, -1, x.shape[-1])
            x = torch.cat([x, t], dim=1)
            x = mid_conv(x)

        # Upsample
        for (upsample, up_conv1, up_conv2), skip in zip(self.up_modules, reversed(h)):
            x = upsample(x)
            x = torch.cat([x, skip], dim=1)
            t = timestep_embed[:, :, None].expand(-1, -1, x.shape[-1])
            x = torch.cat([x, t], dim=1)
            x = up_conv1(x)
            x = up_conv2(x)

        # Output
        x = self.final_conv(x)

        return x


class VisionEncoder(nn.Module):
    """Simple CNN encoder for images (legacy)"""

    def __init__(self, obs_horizon=2, output_dim=256):
        super().__init__()

        self.obs_horizon = obs_horizon

        # Per-image encoder
        self.conv = nn.Sequential(
            # 96x96
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 48x48
            nn.GroupNorm(8, 32),
            nn.Mish(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 24x24
            nn.GroupNorm(8, 64),
            nn.Mish(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 12x12
            nn.GroupNorm(8, 128),
            nn.Mish(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 6x6
            nn.GroupNorm(8, 256),
            nn.Mish(),
            nn.AdaptiveAvgPool2d(1),  # 1x1
        )

        # Combine observations
        self.combine = nn.Sequential(
            nn.Linear(256 * obs_horizon, output_dim),
            nn.Mish(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x):
        """
        Args:
            x: (B, obs_horizon, C, H, W)

        Returns:
            (B, output_dim)
        """
        B, T, C, H, W = x.shape

        # Reshape to process all images
        x = x.reshape(B * T, C, H, W)

        # Encode
        x = self.conv(x)  # (B*T, 256, 1, 1)
        x = x.reshape(B, T * 256)

        # Combine
        x = self.combine(x)

        return x


class ResNet18VisionEncoder(nn.Module):
    """
    ResNet18-based vision encoder (matches original diffusion_policy)

    Uses pretrained ResNet18 as backbone for better visual features.
    Processes each frame independently then combines across observation horizon.
    """

    def __init__(self, obs_horizon=2, output_dim=256, pretrained=True):
        super().__init__()

        self.obs_horizon = obs_horizon

        # Load pretrained ResNet18
        resnet = torchvision.models.resnet18(pretrained=pretrained)

        # Remove the final FC layer and avgpool
        # ResNet18 outputs 512 features from layer4
        self.backbone = nn.Sequential(
            resnet.conv1,      # 7x7 conv, stride 2
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,    # 3x3 maxpool, stride 2
            resnet.layer1,     # 64 channels
            resnet.layer2,     # 128 channels
            resnet.layer3,     # 256 channels
            resnet.layer4,     # 512 channels
        )

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Projection head to combine temporal features
        # ResNet18 outputs 512 features per frame
        self.combine = nn.Sequential(
            nn.Linear(512 * obs_horizon, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x):
        """
        Args:
            x: (B, obs_horizon, C, H, W) - images in [-1, 1] range

        Returns:
            (B, output_dim) - visual features
        """
        B, T, C, H, W = x.shape

        # Reshape to process all frames
        x = x.reshape(B * T, C, H, W)

        # Extract features with ResNet backbone
        x = self.backbone(x)  # (B*T, 512, H', W')

        # Global average pooling
        x = self.avgpool(x)  # (B*T, 512, 1, 1)
        x = x.flatten(1)  # (B*T, 512)

        # Reshape to separate batch and time
        x = x.reshape(B, T * 512)  # (B, obs_horizon * 512)

        # Combine temporal features
        x = self.combine(x)  # (B, output_dim)

        return x


class DiffusionPolicy(nn.Module):
    """
    Complete Diffusion Policy model

    Combines vision encoder, state encoder, and diffusion U-Net.
    Uses DDPM (100 steps) for training and DDIM (16 steps) for fast inference.
    """

    def __init__(
        self,
        obs_horizon=2,
        pred_horizon=16,
        action_dim=6,
        state_dim=7,  # 6 (robot pose) + 1 (gripper)
        vision_feature_dim=256,
        state_feature_dim=64,
        num_diffusion_iters=100,
        num_inference_steps=16,  # DDIM inference steps (6x faster than DDPM)
        use_resnet=True,  # Use ResNet18 (True) or custom CNN (False)
    ):
        super().__init__()

        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.num_diffusion_iters = num_diffusion_iters
        self.num_inference_steps = num_inference_steps

        # Vision encoder: ResNet18 (better) or custom CNN (legacy)
        if use_resnet:
            self.vision_encoder = ResNet18VisionEncoder(obs_horizon, vision_feature_dim, pretrained=True)
        else:
            self.vision_encoder = VisionEncoder(obs_horizon, vision_feature_dim)

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim * obs_horizon, state_feature_dim),
            nn.Mish(),
            nn.Linear(state_feature_dim, state_feature_dim),
        )

        # Diffusion model
        cond_dim = vision_feature_dim + state_feature_dim
        self.diffusion_model = DiffusionUNet1D(
            action_dim=action_dim,
            pred_horizon=pred_horizon,
            cond_dim=cond_dim,
        )

        # Noise schedulers
        # DDPM for training (100 steps)
        self.noise_scheduler_train = DDPMScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon',
        )

        # DDIM for inference (16 steps - 6x faster)
        self.noise_scheduler_infer = DDIMScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon',
        )

    def forward(self, obs_state, obs_image, actions=None, train=True):
        """
        Args:
            obs_state: (B, obs_horizon, state_dim)
            obs_image: (B, obs_horizon, C, H, W)
            actions: (B, pred_horizon, action_dim) - only for training
            train: Whether in training mode

        Returns:
            If train: loss
            If not train: predicted actions
        """
        B = obs_state.shape[0]

        # Encode observations
        vision_features = self.vision_encoder(obs_image)  # (B, vision_dim)

        state_flat = obs_state.reshape(B, -1)  # (B, obs_horizon * state_dim)
        state_features = self.state_encoder(state_flat)  # (B, state_dim)

        # Combine conditioning
        cond = torch.cat([vision_features, state_features], dim=-1)  # (B, cond_dim)

        if train:
            # Training: Use DDPM with 100 steps
            assert actions is not None

            # Transpose actions to (B, action_dim, pred_horizon)
            actions = actions.transpose(1, 2)

            # Sample noise
            noise = torch.randn_like(actions)

            # Sample timestep uniformly from [0, num_diffusion_iters)
            timesteps = torch.randint(
                0, self.num_diffusion_iters,
                (B,), device=actions.device
            ).long()

            # Add noise using DDPM scheduler
            noisy_actions = self.noise_scheduler_train.add_noise(
                actions, noise, timesteps
            )

            # Predict noise
            noise_pred = self.diffusion_model(noisy_actions, timesteps, cond)

            # MSE loss
            loss = nn.functional.mse_loss(noise_pred, noise)

            return loss

        else:
            # Inference: Use DDIM with 16 steps (6x faster than DDPM)
            # Start from random noise
            noisy_action = torch.randn(
                (B, self.action_dim, self.pred_horizon),
                device=obs_state.device
            )

            # Set DDIM timesteps (16 steps uniformly spaced)
            self.noise_scheduler_infer.set_timesteps(self.num_inference_steps)

            # DDIM denoising loop (deterministic, eta=0)
            for t in self.noise_scheduler_infer.timesteps:
                # Predict noise
                noise_pred = self.diffusion_model(
                    noisy_action,
                    t.unsqueeze(0).expand(B).to(obs_state.device),
                    cond
                )

                # DDIM denoising step (deterministic with eta=0)
                noisy_action = self.noise_scheduler_infer.step(
                    noise_pred, t, noisy_action, eta=0.0
                ).prev_sample

            # Transpose back to (B, pred_horizon, action_dim)
            actions = noisy_action.transpose(1, 2)

            return actions


class DDPMScheduler:
    """Simple DDPM noise scheduler"""

    def __init__(self, num_train_timesteps=1000, beta_schedule='linear',
                 clip_sample=True, prediction_type='epsilon'):
        self.num_train_timesteps = num_train_timesteps
        self.clip_sample = clip_sample
        self.prediction_type = prediction_type

        # Beta schedule
        if beta_schedule == 'linear':
            self.betas = torch.linspace(1e-4, 0.02, num_train_timesteps)
        elif beta_schedule == 'squaredcos_cap_v2':
            # Cosine schedule
            steps = num_train_timesteps + 1
            x = torch.linspace(0, num_train_timesteps, steps)
            alphas_cumprod = torch.cos(((x / num_train_timesteps) + 0.008) / 1.008 * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, original_samples, noise, timesteps):
        """Add noise to samples"""
        alphas_cumprod = self.alphas_cumprod.to(original_samples.device)
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def set_timesteps(self, num_inference_steps):
        """Set timesteps for inference"""
        self.num_inference_steps = num_inference_steps
        self.timesteps = torch.linspace(
            self.num_train_timesteps - 1, 0, num_inference_steps
        ).long()

    def step(self, model_output, timestep, sample):
        """Denoise step"""
        t = timestep
        prev_t = t - self.num_train_timesteps // self.num_inference_steps

        alpha_prod_t = self.alphas_cumprod[t].to(sample.device)
        alpha_prod_t_prev = self.alphas_cumprod[prev_t].to(sample.device) if prev_t >= 0 else torch.tensor(1.0).to(sample.device)

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # Compute predicted original sample
        if self.prediction_type == 'epsilon':
            pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        else:
            pred_original_sample = model_output

        # Clip
        if self.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        # Compute coefficients for pred_original_sample and current sample
        pred_original_sample_coeff = (alpha_prod_t_prev ** 0.5 * self.betas[t].to(sample.device)) / beta_prod_t
        current_sample_coeff = self.alphas[t].to(sample.device) ** 0.5 * beta_prod_t_prev / beta_prod_t

        # Compute predicted previous sample
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

        # Add noise
        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output)
            variance = (self.betas[t].to(sample.device) * beta_prod_t_prev / beta_prod_t) ** 0.5 * noise

        pred_prev_sample = pred_prev_sample + variance

        return type('obj', (object,), {'prev_sample': pred_prev_sample})()


class DDIMScheduler:
    """
    DDIM (Denoising Diffusion Implicit Models) scheduler for fast inference

    DDIM is a deterministic sampler that allows fewer inference steps (e.g., 16)
    while maintaining quality. This is crucial for real-time robot control.

    Reference: Song et al. "Denoising Diffusion Implicit Models" (ICLR 2021)
    """

    def __init__(self, num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2',
                 clip_sample=True, prediction_type='epsilon'):
        self.num_train_timesteps = num_train_timesteps
        self.clip_sample = clip_sample
        self.prediction_type = prediction_type

        # Beta schedule (same as DDPM)
        if beta_schedule == 'linear':
            self.betas = torch.linspace(1e-4, 0.02, num_train_timesteps)
        elif beta_schedule == 'squaredcos_cap_v2':
            # Cosine schedule
            steps = num_train_timesteps + 1
            x = torch.linspace(0, num_train_timesteps, steps)
            alphas_cumprod = torch.cos(((x / num_train_timesteps) + 0.008) / 1.008 * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, original_samples, noise, timesteps):
        """Add noise to samples (same as DDPM)"""
        alphas_cumprod = self.alphas_cumprod.to(original_samples.device)
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def set_timesteps(self, num_inference_steps):
        """Set timesteps for inference (uniform spacing)"""
        self.num_inference_steps = num_inference_steps

        # Linearly spaced timesteps
        step_ratio = self.num_train_timesteps // num_inference_steps
        self.timesteps = torch.arange(0, num_inference_steps) * step_ratio
        self.timesteps = torch.flip(self.timesteps, [0]).long()

    def step(self, model_output, timestep, sample, eta=0.0):
        """
        DDIM denoising step

        Args:
            model_output: Predicted noise (epsilon)
            timestep: Current timestep
            sample: Current noisy sample
            eta: Stochasticity parameter (0 = deterministic DDIM, 1 = DDPM)

        Returns:
            Object with prev_sample attribute
        """
        # Get current and previous alpha_cumprod
        alpha_prod_t = self.alphas_cumprod[timestep].to(sample.device)

        # Get previous timestep
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep].to(sample.device) if prev_timestep >= 0 else torch.tensor(1.0).to(sample.device)

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # Compute predicted original sample from predicted noise
        if self.prediction_type == 'epsilon':
            pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        else:
            pred_original_sample = model_output

        # Clip predicted original sample
        if self.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        # Compute variance for stochastic sampling
        variance = 0
        if eta > 0:
            variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
            variance = torch.clamp(variance, min=1e-20)

        # Compute predicted previous sample (deterministic DDIM update)
        pred_sample_direction = (1 - alpha_prod_t_prev - eta * variance) ** 0.5 * model_output
        pred_prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction

        # Add variance (stochastic component)
        if eta > 0 and timestep > 0:
            noise = torch.randn_like(model_output)
            pred_prev_sample = pred_prev_sample + (eta * variance) ** 0.5 * noise

        return type('obj', (object,), {'prev_sample': pred_prev_sample})()
