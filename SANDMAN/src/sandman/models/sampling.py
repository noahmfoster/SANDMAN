import torch
import torch.nn.functional as F
from accelerate import Accelerator
from typing import Dict, List, Tuple, Optional, Any

from sandman.models.diffusion import DiffusionWrapper 
from sandman.models.utils import MaskingPolicy
from sandman.models.training import _move_batch_to_device
from sandman.models.utils import normalize_neuron_mask_output

@torch.no_grad()
def sample_with_diffusion(
    diffusion: DiffusionWrapper,
    batch: Dict[str, Any],
    accelerator: Accelerator,
    *,
    masking_policy: Optional[MaskingPolicy] = None,
    num_steps: Optional[int] = None,
):
    """
    Full diffusion sampling loop (x_T -> x_0) with manual DDPM stepping.
    """
    diffusion.eval()
    batch = _move_batch_to_device(batch, accelerator.device)
    
    spikes = batch["spikes_data"]
    neuron_regions = batch["neuron_regions"]
    eids = batch["eid"].view(-1)
    B, T, N = spikes.shape
    device = spikes.device
    
    # Masking
    if masking_policy is None:
        corrupt_nm = torch.zeros(B, T, N, dtype=torch.bool, device=device)
        hidden_nm = torch.zeros(B, T, N, dtype=torch.bool, device=device)
    else:
        corrupt_nm, hidden_nm, _ = normalize_neuron_mask_output(
            masking_policy(batch),
            B=B, T=T, N=N,
        )
    
    spikes_corrupt = torch.where(corrupt_nm, torch.zeros_like(spikes), spikes)
    
    # Encode to region latents
    x0, region_ids, region_indices, _ = diffusion.model.spikes_to_region_latents(
        spikes_corrupt,
        neuron_regions,
        eids,
        hidden_nm=hidden_nm,
    )  # [B,T,R,K,D]
    
    # Get scheduler parameters
    if num_steps is not None:
        diffusion.scheduler.set_timesteps(num_steps)
    
    timesteps = diffusion.scheduler.timesteps
    alphas_cumprod = diffusion.scheduler.alphas_cumprod.to(device)
    
    # Initialize x_T (noise)
    x = torch.randn_like(x0)
    
    # Reverse diffusion loop with manual DDPM step
    for i, t in enumerate(timesteps):
        t_int = int(t.item()) if torch.is_tensor(t) else int(t)
        
        # Create timestep tensor for the model
        t_tensor = torch.full((B,), t_int, device=device, dtype=torch.long)
        
        # Predict epsilon
        eps_pred = diffusion.model.denoise_region_latents(
            x,
            t_tensor,
            region_ids=region_ids,
            eids=eids,
        )  # [B, T, R, K, D]
        
        # Manual DDPM step (DDIM with eta=1.0 is equivalent to DDPM)
        alpha_prod_t = alphas_cumprod[t_int]
        
        if i < len(timesteps) - 1:
            t_prev = timesteps[i + 1]
            t_prev_int = int(t_prev.item()) if torch.is_tensor(t_prev) else int(t_prev)
            alpha_prod_t_prev = alphas_cumprod[t_prev_int]
        else:
            alpha_prod_t_prev = torch.tensor(1.0, device=device)
        
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        # Predict x0 from epsilon
        sqrt_alpha_prod_t = alpha_prod_t ** 0.5
        sqrt_one_minus_alpha_prod_t = beta_prod_t ** 0.5
        
        x0_pred = (x - sqrt_one_minus_alpha_prod_t * eps_pred) / sqrt_alpha_prod_t
        
        # Compute x_{t-1} using formula from DDPM paper
        sqrt_alpha_prod_t_prev = alpha_prod_t_prev ** 0.5
        sqrt_one_minus_alpha_prod_t_prev = beta_prod_t_prev ** 0.5
        
        # Mean of x_{t-1}
        x_prev_mean = sqrt_alpha_prod_t_prev * x0_pred + sqrt_one_minus_alpha_prod_t_prev * eps_pred
        
        # Add noise (except for the last step)
        if i < len(timesteps) - 1:
            noise = torch.randn_like(x)
            # Variance
            beta_t = diffusion.scheduler.betas[t_int]
            variance = (beta_prod_t_prev / beta_prod_t) * beta_t
            x = x_prev_mean + (variance ** 0.5) * noise
        else:
            x = x_prev_mean
    
    # Decode x_0 to neuron space
    pred_spikes = diffusion.model.region_latents_to_spikes(
        x,
        region_ids,
        region_indices,
        N,
        neuron_regions,
        eids,
    )
    
    return pred_spikes


@torch.no_grad()
def infer_batch(
    diffusion: DiffusionWrapper,
    batch: Dict[str, Any],
    accelerator: Accelerator,
    *,
    masking_policy: Optional[MaskingPolicy] = None,
    use_diffusion: bool = True,
    timestep: Optional[int] = None,
):
    """
    Run inference on a single batch.

    Args:
        diffusion: trained DiffusionWrapper
        batch: data batch dict
        accelerator: Accelerator instance
        masking_policy: optional masking policy (e.g. mask_one_region)
        use_diffusion: if False, bypass diffusion noise (deterministic)
        timestep: optional fixed diffusion timestep (int). If None, t=0.

    Returns:
        pred_spikes: [B, T, N]
    """
    diffusion.eval()

    batch = _move_batch_to_device(batch, accelerator.device)

    spikes = batch["spikes_data"]              # [B,T,N]
    neuron_regions = batch["neuron_regions"]   # [B,N]
    eids = batch["eid"].view(-1)

    B, T, N = spikes.shape
    device = spikes.device

    # ------------------------------------------------------------
    # Masking
    # ------------------------------------------------------------
    if masking_policy is None:
        corrupt_nm = torch.zeros(B, T, N, dtype=torch.bool, device=device)
        hidden_nm = torch.zeros(B, T, N, dtype=torch.bool, device=device)
    else:
        corrupt_nm, hidden_nm, _ = normalize_neuron_mask_output(
            masking_policy(batch),
            B=B, T=T, N=N,
        )

    # ------------------------------------------------------------
    # Corrupt inputs (same semantics as training)
    # ------------------------------------------------------------
    spikes_corrupt = torch.where(corrupt_nm, torch.zeros_like(spikes), spikes)

    # ------------------------------------------------------------
    # Encode → region latents
    # ------------------------------------------------------------
    x0, region_ids, region_indices, _ = diffusion.model.spikes_to_region_latents(
        spikes_corrupt,
        neuron_regions,
        eids,
        hidden_nm=hidden_nm,
    )  # [B,T,R,K,D]

    # ------------------------------------------------------------
    # Diffusion step
    # ------------------------------------------------------------
    if not use_diffusion or diffusion.disable_diffusion:
        x_t = x0
        t = torch.zeros(B, device=device, dtype=torch.long)
    else:
        if timestep is None:
            t = torch.zeros(B, device=device, dtype=torch.long)
        else:
            t = torch.full((B,), int(timestep), device=device, dtype=torch.long)

        eps = torch.randn_like(x0)
        x_t = diffusion.scheduler.add_noise(x0, eps, t)

    # ------------------------------------------------------------
    # Denoise
    # ------------------------------------------------------------
    eps_pred = diffusion.model.denoise_region_latents(
        x_t,
        t,
        region_ids=region_ids,
        eids=eids,
    )

    # ------------------------------------------------------------
    # Recover x0 prediction
    # ------------------------------------------------------------
    if not use_diffusion or diffusion.disable_diffusion:
        x0_pred = eps_pred
    else:
        a_bar = diffusion._alphas_cumprod(t, device=device, dtype=x0.dtype)
        sqrt_a = torch.sqrt(a_bar)[:, None, None, None, None]
        sqrt_one_minus_a = torch.sqrt(1.0 - a_bar)[:, None, None, None, None]
        x0_pred = (x_t - sqrt_one_minus_a * eps_pred) / (sqrt_a + 1e-8)

    # ------------------------------------------------------------
    # Decode → neuron space
    # ------------------------------------------------------------
    pred_spikes = diffusion.model.region_latents_to_spikes(
        x0_pred,
        region_ids,
        region_indices,
        N,
        neuron_regions,
        eids,
    )  # [B,T,N]

    return pred_spikes
