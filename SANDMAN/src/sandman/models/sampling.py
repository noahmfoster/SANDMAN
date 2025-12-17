import torch
from sandman.models.utils import neuron_mask_to_region_time_mask

@torch.no_grad()
def sample_region_latent_ddpm(
    model,                 # your NeuralTransformer
    scheduler,             # diffusers DDPMScheduler
    batch: dict,           # contains spikes_data, neuron_regions, eid, etc.
    masking_policy,        # returns (corrupt_nm, hidden_nm, target_nm) all [B,T,N] bool
    *,
    num_inference_steps: int = 1000,
    eta: float = 0.0,      # not used by DDPM scheduler; kept for API symmetry
    init_noise: torch.Tensor | None = None,
    device: torch.device | None = None,
):
    """
    Returns:
      pred_full: [B,T,N] decoded prediction from sampled x_0 latents.
      x0_latent: [B,T,R,D] the final latent.
    """
    model.eval()
    if device is None:
        device = batch["spikes_data"].device

    spikes = batch["spikes_data"].to(device)                      # [B,T,N]
    neuron_regions = batch["neuron_regions"].to(device)            # [B,N]
    eids = batch["eid"].view(-1).to(device)                        # [B]
    B, T, N = spikes.shape

    # 1) masks (neuron space)
    corrupt_nm, hidden_nm, target_nm = masking_policy(batch)
    corrupt_nm = corrupt_nm.to(device)
    hidden_nm = hidden_nm.to(device)
    target_nm = target_nm.to(device)

    # 2) build conditioning spikes (e.g., zero corrupt positions; keep observed)
    spikes_cond = torch.where(corrupt_nm, torch.zeros_like(spikes), spikes)

    # 3) encode to region latents WITH region-time masking derived from hidden_nm
    #    This returns the *conditioning* latent layout and includes region mask tokens
    x_cond, keys, region_indices, N_total = model.spikes_to_region_latents(
        spikes_cond, neuron_regions, eids, hidden_nm=hidden_nm
    )  # [B,T,R,D]

    # 4) set diffusion timesteps for inference
    scheduler.set_timesteps(num_inference_steps)

    # 5) initialize x_T
    if init_noise is None:
        x = torch.randn_like(x_cond)
    else:
        x = init_noise.to(device)
        assert x.shape == x_cond.shape

    # Optional: if you want to *only* sample masked region-times and keep others fixed,
    # you can “clamp” unmasked positions to x_cond each step.
    # We need a region-time mask [B,T,R] to do that:
    hidden_rm = neuron_mask_to_region_time_mask(hidden_nm, neuron_regions)  # [B,T,R] bool
    hidden_rm = hidden_rm.to(device)

    # 6) reverse diffusion loop
    for t in scheduler.timesteps:
        t_batch = torch.full(
            (x.shape[0],),
            int(t),
            device=x.device,
            dtype=torch.long,
        )


        # a) optionally enforce conditioning on unmasked region-times
        #    (so the model only generates where hidden_rm=True)
        x = torch.where(hidden_rm[..., None], x, x_cond)

        # b) predict epsilon
        #    NOTE: your denoiser should accept timestep t; you should add timestep embedding inside.
        eps_pred = model.denoise_region_latents(x, t_batch)   # [B,T,R,D]  <-- update signature accordingly

        # c) scheduler step: x_{t-1}
        step_out = scheduler.step(eps_pred, t, x)
        x = step_out.prev_sample

    # 7) final clamp (ensure observed region-times exactly match conditioning)
    x0_latent = torch.where(hidden_rm[..., None], x, x_cond)

    # 8) decode latents -> neuron predictions
    pred_full = model.region_latents_to_spikes(x0_latent, keys, region_indices, N_total)

    return pred_full, x0_latent
