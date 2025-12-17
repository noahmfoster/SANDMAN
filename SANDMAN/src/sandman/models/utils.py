import torch
import torch.nn as nn
import math
from typing import Protocol, Callable, Tuple, Optional, Dict, Any


# ============================================================
# MaskingPolicy (UPDATED: neuron-level masks)
# ============================================================

# MaskingPolicy returns three neuron-level boolean masks:
#   corrupt_mask [B, T, N] : where inputs are corrupted (noise / zeroing / etc.)
#   hidden_mask  [B, T, N] : where information is hidden at the REGION-token level
#                            (implementation will hide an entire region token at (B,T)
#                             if ANY neuron in that region is hidden at that time)
#   target_mask  [B, T, N] : where loss is evaluated
MaskingPolicy = Callable[
    [Dict[str, Any]],
    Tuple[
        torch.Tensor,  # corrupt_mask [B,T,N]
        torch.Tensor,  # hidden_mask  [B,T,N]
        torch.Tensor,  # target_mask  [B,T,N]
    ],
]


# ============================================================
# TargetSpec protocol
# ============================================================

class TargetSpec(Protocol):
    """
    Defines what tensor is the target, and how to compute loss.
    """
    name: str

    def get_target(self, batch: dict) -> torch.Tensor:
        """Return target tensor [B, T, N]"""

    def loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss over masked entries"""


# ============================================================
# Targets (neuron-level likelihoods)
# ============================================================

class _PoissonTargetBase:
    def __init__(self, name: str):
        self.name = name
        self.softplus = nn.Softplus()
        self.loss_fn = nn.PoissonNLLLoss(
            log_input=False,
            full=True,
            reduction="none",
        )

    def loss(
        self,
        pred: torch.Tensor,    # [B,T,N]
        target: torch.Tensor,  # [B,T,N]
        mask: torch.Tensor,    # [B,T,N] bool
    ) -> torch.Tensor:
        rate = self.softplus(pred)
        nll = self.loss_fn(rate, target)
        mask_f = mask.to(nll.dtype)
        return (nll * mask_f).sum() / mask_f.sum().clamp(min=1.0)


class _MSETargetBase:
    def __init__(self, name: str):
        self.name = name
        self.loss_fn = nn.MSELoss(reduction="none")

    def loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        mse = self.loss_fn(pred, target)
        mask_f = mask.to(mse.dtype)
        return (mse * mask_f).sum() / mask_f.sum().clamp(min=1.0)


class SpikeCountPoissonTarget(_PoissonTargetBase):
    """
    Target = spike counts per bin.
    """
    def __init__(self, key: str = "spikes_data"):
        super().__init__(name="spike_counts_poisson")
        self.key = key

    def get_target(self, batch: dict) -> torch.Tensor:
        return batch[self.key]  # [B,T,N]


class FiringRatePoissonTarget(_PoissonTargetBase):
    """
    Target = firing rates (Poisson λ).
    Optionally scaled by bin width.
    """
    def __init__(
        self,
        key: str = "fr",
        bin_width_seconds: Optional[float] = None,
    ):
        super().__init__(name="firing_rates_poisson")
        self.key = key
        self.bin_width_seconds = bin_width_seconds

    def get_target(self, batch: dict) -> torch.Tensor:
        fr = batch[self.key]
        if self.bin_width_seconds is not None:
            fr = fr * self.bin_width_seconds
        return fr


class SpikeCountMSETarget(_MSETargetBase):
    """
    Target = spike counts per bin (MSE).
    """
    def __init__(self, key: str = "spikes_data"):
        super().__init__(name="spike_counts_mse")
        self.key = key

    def get_target(self, batch: dict) -> torch.Tensor:
        return batch[self.key]


# ============================================================
# Neuron-mask utilities (NEW)
# ============================================================

def normalize_neuron_mask_output(
    out,
    *,
    B: int,
    T: int,
    N: int,
):
    """
    Validate masking policy output.

    Returns:
      corrupt_mask [B,T,N]
      hidden_mask  [B,T,N]
      target_mask  [B,T,N]
    """
    if not isinstance(out, tuple) or len(out) != 3:
        raise ValueError(
            "masking_policy must return (corrupt_mask, hidden_mask, target_mask) "
            "with each mask shaped [B,T,N]"
        )

    corrupt_nm, hidden_nm, target_nm = out

    for name, m in [("corrupt_mask", corrupt_nm), ("hidden_mask", hidden_nm), ("target_mask", target_nm)]:
        if not torch.is_tensor(m):
            raise ValueError(f"{name} must be a torch.Tensor")
        if m.dtype != torch.bool:
            raise ValueError(f"{name} must be dtype=bool, got {m.dtype}")
        if m.shape != (B, T, N):
            raise ValueError(f"{name} must be [B,T,N]=({B},{T},{N}), got {m.shape}")

    return corrupt_nm, hidden_nm, target_nm


def neuron_mask_to_region_time_mask(
    hidden_nm: torch.Tensor,      # [B,T,N]
    neuron_regions: torch.Tensor  # [B,N]
) -> torch.Tensor:
    """
    Convert hidden neuron mask -> region-time mask by OR-reducing within each region.

    Returns:
      hidden_rm [B, T, R] where R = # unique regions in neuron_regions[0]
      and region ordering is torch.unique(neuron_regions[0]) (sorted).
    """
    if hidden_nm.dtype != torch.bool:
        raise ValueError("hidden_nm must be bool")
    if neuron_regions.ndim != 2:
        raise ValueError("neuron_regions must be [B,N]")

    B, T, N = hidden_nm.shape
    unique_regions = torch.unique(neuron_regions[0])  # sorted
    R = len(unique_regions)

    device = hidden_nm.device
    hidden_rm = torch.zeros((B, T, R), dtype=torch.bool, device=device)

    for r_idx, rid in enumerate(unique_regions):
        idxs = torch.where(neuron_regions[0] == rid)[0]  # [N_r]
        # OR over neurons in region at each (B,T)
        hidden_rm[:, :, r_idx] = hidden_nm.index_select(dim=2, index=idxs).any(dim=2)

    return hidden_rm


# ============================================================
# Example neuron-level masking policies (UPDATED)
# ============================================================

def mask_one_region_some_times(
    batch: dict,
    p_time: float = 0.5,
):
    """
    Masks a single region for a random subset of time steps, expressed at neuron level.
    (Internally the model will convert hidden_nm -> hidden_rm by OR over neurons in region.)
    Target mask is constant across time.
    """
    spikes = batch["spikes_data"]  # [B,T,N]
    regions = batch["neuron_regions"]

    B, T, N = spikes.shape
    device = spikes.device
    neuron_regions = regions
    unique_regions = torch.unique(neuron_regions[0])
    rid = unique_regions[torch.randint(len(unique_regions), (1,), device=device)].item()

    # time mask shared across batch (simple + stable)
    time_mask = (torch.rand(T, device=device) < p_time)[None, :, None].expand(B, -1, -1)  # [B,T,1]

    # neuron mask for chosen region
    region_neuron_mask = (neuron_regions == rid)[:, None, :].expand(B, T, N)  # [B,T,N]

    hidden_nm = region_neuron_mask & time_mask
    corrupt_nm = hidden_nm.clone()          # default: corrupt where hidden
    target_nm = region_neuron_mask.clone()  # default: evaluate loss everywhere

    return corrupt_nm, hidden_nm, target_nm

def mask_one_region(batch: dict):
    """
    Masks a single region for all time steps.
    """
    return mask_one_region_some_times(batch, p_time=1.0)


def no_mask_policy(batch: dict):
    """
    No corruption, no hiding, no loss.
    (Useful for debugging; for training, you probably want a non-empty target mask.)
    """
    spikes = batch["spikes_data"]
    B, T, N = spikes.shape
    device = spikes.device
    z = torch.zeros((B, T, N), dtype=torch.bool, device=device)
    o = torch.ones((B, T, N), dtype=torch.bool, device=device)
    return o, z, o


# ============================================================
# Device handling
# ============================================================

def move_scheduler_to_device(scheduler, device):
    """Move all scheduler tensors to the specified device"""
    for key in ["betas", "alphas", "alphas_cumprod", "alphas_cumprod_prev"]:
        if hasattr(scheduler, key):
            tensor = getattr(scheduler, key)
            if isinstance(tensor, torch.Tensor):
                setattr(scheduler, key, tensor.to(device))
    return scheduler


# ============================================================
# RoPE
# ============================================================

def build_rope_cache(T: int, D: int, device):
    assert D % 2 == 0, "RoPE requires even D"
    half = D // 2
    freq = torch.exp(-math.log(10000) * torch.arange(half, device=device) / half)
    t = torch.arange(T, device=device).float()
    angles = t[:, None] * freq[None, :]
    return angles.cos()[None, :, None, :], angles.sin()[None, :, None, :]


def apply_rope_time(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)