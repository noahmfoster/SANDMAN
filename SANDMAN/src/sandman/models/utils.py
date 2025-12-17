import torch
import torch.nn as nn
import math

from typing import Protocol, Callable, Tuple, Optional, Dict, Any, Iterable, Union


# ============================================================
# Types
# ============================================================

# MaskingPolicy may return:
#   (input_mask, target_mask)
# or
#   (input_mask, target_mask, masked_region_ids)
#
# input_mask, target_mask: Optional[BoolTensor[B, T, N]]
# masked_region_ids: Optional[Iterable[int] | LongTensor]
MaskingPolicy = Callable[
    [Dict[str, Any]],
    Union[
        Tuple[Optional[torch.Tensor], Optional[torch.Tensor]],
        Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[Union[Iterable[int], torch.Tensor]]],
    ],
]


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
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss over masked entries"""


# ============================================================
# Targets (Poisson, neuron-level)
# ============================================================

class _PoissonTargetBase:
    def __init__(self, name: str):
        self.name = name
        self.softplus = nn.Softplus()
        self.loss_fn = nn.PoissonNLLLoss(
            log_input=False,
            full=True,
            reduction="none"
        )

    def loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        pred:   [B, T, N] unconstrained
        target: [B, T, N] non-negative
        mask:   [B, T, N] bool
        """
        rate = self.softplus(pred)
        nll = self.loss_fn(rate, target)
        mask_f = mask.to(nll.dtype)

        return (nll * mask_f).sum() / mask_f.sum().clamp(min=1.0)
    
class _MSETargetBase:
    def __init__(self, name: str):
        self.name = name
        self.loss_fn = nn.MSELoss(
            reduction="none"
        )

    def loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        pred:   [B, T, N] unconstrained
        target: [B, T, N] unconstrained
        mask:   [B, T, N] bool
        """
        mse = self.loss_fn(pred, target)
        mask_f = mask.to(mse.dtype)

        return (mse * mask_f).sum() / mask_f.sum().clamp(min=1.0)
    
class SpikeCountTarget(_PoissonTargetBase):
    """
    Target = spike counts per bin.
    """
    def __init__(self, key: str = "spikes_data"):
        super().__init__(name="spike_counts_poisson")
        self.key = key

    def get_target(self, batch: dict) -> torch.Tensor:
        return batch[self.key]  # [B, T, N]


class FiringRateTarget(_PoissonTargetBase):
    """
    Target = per-bin firing rates (λ in Poisson).
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
        fr = batch[self.key]  # [B, T, N]
        if self.bin_width_seconds is not None:
            fr = fr * self.bin_width_seconds
        return fr
    
class SpikeCountMSETarget(_MSETargetBase):
    """
    Target = spike counts per bin.
    """
    def __init__(self, key: str = "spikes_data"):
        super().__init__(name="spike_counts_poisson")
        self.key = key

    def get_target(self, batch: dict) -> torch.Tensor:
        return batch[self.key]  # [B, T, N]


# ============================================================
# Mask utilities
# ============================================================

def _expand_neuron_mask(
    neuron_mask: torch.Tensor,
    T: int,
) -> torch.Tensor:
    """
    neuron_mask: [B, N]
    returns:     [B, T, N]
    """
    return neuron_mask[:, None, :].expand(-1, T, -1)


def _normalize_masking_policy_output(
    out,
    B: int,
    T: int,
    N: int,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[Iterable[int]]]:

    if not isinstance(out, tuple):
        raise ValueError("masking_policy must return a tuple")

    if len(out) == 2:
        input_mask, target_mask = out
        masked_region_ids = None
    elif len(out) == 3:
        input_mask, target_mask, masked_region_ids = out
    else:
        raise ValueError(
            "masking_policy must return (input_mask, target_mask) or "
            "(input_mask, target_mask, masked_region_ids)"
        )

    if input_mask is None and target_mask is None:
        raise ValueError("At least one of input_mask or target_mask must be provided")

    if input_mask is None:
        input_mask = target_mask
    if target_mask is None:
        target_mask = input_mask

    if input_mask.shape != (B, T, N):
        raise ValueError(f"input_mask must be [B,T,N]=({B},{T},{N}), got {input_mask.shape}")
    if target_mask.shape != (B, T, N):
        raise ValueError(f"target_mask must be [B,T,N]=({B},{T},{N}), got {target_mask.shape}")

    return input_mask, target_mask, masked_region_ids

# ============================================================
# Masking Policies
# ============================================================

def region_inpainting_policy(batch: dict):
    """
    Mask ALL neurons from one randomly selected region
    across ALL timesteps.

    Returns:
      input_mask  [B, T, N]
      target_mask [B, T, N]
      masked_region_ids [list[int]]
    """
    spikes = batch["spikes_data"]
    regions_full = batch["neuron_regions_full"]

    B, T, N = spikes.shape
    device = spikes.device

    regions = regions_full[:, :N]
    unique_regions = torch.unique(regions)

    rid = unique_regions[
        torch.randint(len(unique_regions), (1,), device=device)
    ].item()

    neuron_mask = (regions == rid)                 # [B, N]
    mask = neuron_mask[:, None, :].expand(B, T, N)

    return mask, mask, [int(rid)]


def random_neuron_denoising_policy(
    batch: dict,
    mask_prob: float = 0.15,
):
    """
    Randomly mask individual neurons (not regions).
    No region tokens are masked.

    Returns:
      input_mask  [B, T, N]
      target_mask [B, T, N]
      masked_region_ids None
    """
    spikes = batch["spikes_data"]
    B, T, N = spikes.shape
    device = spikes.device

    # Sample neurons once per batch item
    neuron_mask = (
        torch.rand((B, N), device=device) < mask_prob
    )

    mask = neuron_mask[:, None, :].expand(B, T, N)

    return mask, mask, None


def next_step_prediction_policy(batch: dict):
    """
    Predict x[t+1] from x[:t].
    No corruption of inputs.

    Returns:
      input_mask  [B, T, N] = all False
      target_mask [B, T, N] = True for t >= 1
      masked_region_ids None
    """
    spikes = batch["spikes_data"]
    B, T, N = spikes.shape
    device = spikes.device

    input_mask = torch.zeros((B, T, N), dtype=torch.bool, device=device)

    target_mask = torch.zeros((B, T, N), dtype=torch.bool, device=device)
    target_mask[:, 1:, :] = True   # predict future

    return input_mask, target_mask, None

def no_mask_policy(batch):
    spikes = batch["spikes_data"]
    B, T, N = spikes.shape
    mask = torch.ones((B, T, N), dtype=torch.bool, device=spikes.device)
    return mask, mask



# ============================================================
# Example masking policy: region inpainting
# ============================================================

def mask_one_region_policy(batch: dict):
    """
    Randomly selects ONE region and:
      - corrupts all its neurons at all timesteps
      - computes loss on those neurons
      - injects a region-level mask token

    Returns:
      input_mask  [B, T, N]
      target_mask [B, T, N]
      masked_region_ids [list[int]]
    """
    spikes = batch["spikes_data"]              # [B, T, N]
    regions_full = batch["neuron_regions_full"]

    B, T, N = spikes.shape
    device = spikes.device

    regions = regions_full[:, :N]
    unique_regions = torch.unique(regions)

    rid = unique_regions[torch.randint(len(unique_regions), (1,), device=device)].item()

    neuron_mask = (regions == rid)             # [B, N]
    mask = _expand_neuron_mask(neuron_mask, T)

    return mask, mask, [int(rid)]


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
    assert D % 2 == 0
    half = D // 2

    freq = torch.exp(
        -math.log(10000) * torch.arange(0, half, device=device) / half
    )  # [half]

    t = torch.arange(T, device=device).float()  # [T]
    angles = t[:, None] * freq[None, :]         # [T, half]

    cos = angles.cos()[None, :, None, :]        # [1, T, 1, half]
    sin = angles.sin()[None, :, None, :]        # [1, T, 1, half]
    return cos, sin


def apply_rope_time(x, cos, sin):
    """
    x: [B, T, R, D]
    """
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.cat(
        [x1 * cos - x2 * sin,
         x1 * sin + x2 * cos],
        dim=-1
    )
