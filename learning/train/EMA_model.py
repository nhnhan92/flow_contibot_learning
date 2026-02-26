

class EMAModel:
    """
    Exponential Moving Average — scoped to the UNet denoiser only.

    Matches Stanford diffusion_policy behaviour:
      • Applied to noise_pred_net (UNet) ONLY, not the vision/state encoders.
        This lets the encoders adapt quickly via normal gradient updates while
        the denoiser stays stable.
      • Adaptive decay:  decay = min(max_value, (1 + step) / (10 + step))
        - step 0      → decay ≈ 0.10   (fast updates early in training)
        - step 1 000  → decay ≈ 0.99
        - step 10 000 → decay ≈ 0.999
        - step → ∞    → decay → max_value (= 0.9999)
        This warm-start prevents the EMA from being poisoned by bad early
        weights — the shadow tracks the live model closely at first, then
        smooths more aggressively once training has stabilised.

    Args:
        unet      : The UNet submodule (model.diffusion_model)
        max_value : Upper bound on decay (Stanford default: 0.9999)
    """

    def __init__(self, unet, max_value=0.9999):
        self.model     = unet
        self.max_value = max_value
        self.num_updates = 0
        self.shadow    = {}
        self.backup    = {}

        for name, param in unet.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @property
    def decay(self):
        """Adaptive decay: low early (fast tracking), high later (stable)."""
        s = self.num_updates
        return min(self.max_value, (1.0 + s) / (10.0 + s))

    def update(self):
        d = self.decay
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (1.0 - d) * param.data + d * self.shadow[name]
        self.num_updates += 1

    def apply_shadow(self):
        """Swap in EMA weights (call before validation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore live weights (call after validation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self):
        return {'shadow': self.shadow, 'num_updates': self.num_updates}

    def load_state_dict(self, state):
        self.shadow      = state['shadow']
        self.num_updates = state.get('num_updates', 0)
