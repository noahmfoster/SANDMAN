import torch
import torch.nn as nn

from sandman.models.utils import (
    apply_rope_time,
    build_rope_cache,
    normalize_neuron_mask_output,
    neuron_mask_to_region_time_mask,
)
from sandman.models.utils import TargetSpec

from typing import Dict


# ============================================================
# Region-level encoder (neurons -> region token)
# ============================================================

class RegionEncoder(nn.Module):
    """
    SIMPLE region encoder: [B, T, N_r] -> [B, T, D]
    """
    def __init__(self, d_model: int, n_neurons: int):
        super().__init__()
        self.d_model = int(d_model)
        self.n_neurons = int(n_neurons)
        self.proj = nn.Linear(self.n_neurons, self.d_model)

    def forward(self, spikes_region: torch.Tensor) -> torch.Tensor:
        B, T, N_r = spikes_region.shape
        if N_r != self.n_neurons:
            raise ValueError(f"RegionEncoder expected N_r={self.n_neurons}, got {N_r}")
        return self.proj(spikes_region)


# ============================================================
# Neuron-specific decoder (region token -> neurons)
# ============================================================

class RegionDecoder(nn.Module):
    def __init__(self, d_model: int, n_neurons: int):
        super().__init__()
        self.readout = nn.Linear(d_model, n_neurons)

    def forward(self, region_token: torch.Tensor) -> torch.Tensor:
        return self.readout(region_token)


# ============================================================
# Region codec dictionary
# ============================================================

class RegionCodecDict(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        self._n_neurons = {}

    def _key(self, eid, region_id):
        return f"{int(eid)}:{int(region_id)}"

    def get_codec(self, eid, region_id, n_neurons, device):
        key = self._key(eid, region_id)
        n_neurons = int(n_neurons)

        if key not in self.encoders:
            self.encoders[key] = RegionEncoder(self.d_model, n_neurons).to(device)
            self.decoders[key] = RegionDecoder(self.d_model, n_neurons).to(device)
            self._n_neurons[key] = n_neurons
        elif self._n_neurons[key] != n_neurons:
            raise ValueError(f"Neuron count changed for {key}")

        return self.encoders[key], self.decoders[key]

    def encode(self, spikes, neuron_regions, eids):
        device = spikes.device
        region_tokens = {}
        region_indices = {}

        eid = int(eids[0].item())
        unique_regions = torch.unique(neuron_regions[0])

        for rid in unique_regions:
            rid = int(rid.item())
            idxs = torch.where(neuron_regions[0] == rid)[0]
            spikes_r = spikes.index_select(2, idxs)
            enc, _ = self.get_codec(eid, rid, idxs.numel(), device)
            region_tokens[(eid, rid)] = enc(spikes_r)
            region_indices[(eid, rid)] = idxs

        return region_tokens, region_indices

    def decode_to_full(self, region_tokens, region_indices, N):
        any_tok = next(iter(region_tokens.values()))
        B, T, _ = any_tok.shape
        out = torch.zeros(B, T, N, device=any_tok.device, dtype=any_tok.dtype)

        for (eid, rid), tok in region_tokens.items():
            idxs = region_indices[(eid, rid)]
            _, dec = self.get_codec(eid, rid, idxs.numel(), tok.device)
            out.index_copy_(2, idxs, dec(tok))

        return out


# ============================================================
# Denoising block
# ============================================================

class DenoisingBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        h = self.ln1(x)
        x = x + self.attn(h, h, h)[0]
        h = self.ln2(x)
        return x + self.ff(h)


class TimestepMLP(nn.Module):
    """
    Map integer timesteps -> learned embedding -> [B, D].
    Simple + effective for DDPM.
    """
    def __init__(self, d_model: int, max_steps: int = 2048):
        super().__init__()
        self.embed = nn.Embedding(max_steps, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] long
        return self.mlp(self.embed(t))


# ============================================================
# Neural Transformer
# ============================================================

class NeuralTransformer(nn.Module):
    def __init__(self, d_model, n_heads, n_layers):
        super().__init__()
        self.d_model = int(d_model)
        self.region_codecs = RegionCodecDict(self.d_model)

        self.region_mask_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.region_id_embed = nn.Embedding(512, self.d_model)

        self.blocks = nn.ModuleList([
            DenoisingBlock(self.d_model, n_heads) for _ in range(n_layers)
        ])
        self.final_ln = nn.LayerNorm(self.d_model)

        self.rope_cache = None
        self.t_embed = TimestepMLP(self.d_model, max_steps=2048)

    def _sorted_keys(self, region_tokens):
        return sorted(region_tokens.keys(), key=lambda x: x[1])

    def spikes_to_region_latents(
        self,
        spikes: torch.Tensor,
        neuron_regions: torch.Tensor,
        eids: torch.Tensor,
        *,
        hidden_nm: torch.Tensor,   # [B,T,N] bool
    ):
        """
        Returns:
          x_latent: [B,T,R,D]
          keys, region_indices, N_total

        Region token is replaced at (B,T,region) if ANY neuron in that region is hidden at (B,T).
        """
        region_tokens, region_indices = self.region_codecs.encode(spikes, neuron_regions, eids)
        keys = self._sorted_keys(region_tokens)

        # Derive region-time mask from hidden neuron mask
        hidden_rm = neuron_mask_to_region_time_mask(hidden_nm, neuron_regions)  # [B,T,R]

        out = {}
        for r_idx, k in enumerate(keys):
            tok = region_tokens[k]  # [B,T,D]
            B, T, D = tok.shape

            mask_bt = hidden_rm[:, :, r_idx]  # [B,T]
            if mask_bt.any():
                rid = int(k[1])
                rid_t = torch.tensor(rid, device=tok.device)
                rid_emb = self.region_id_embed(rid_t)[None, None, :]  # [1,1,D]
                mask_tok = self.region_mask_token.expand(B, T, -1) + rid_emb  # [B,T,D]
                tok = torch.where(mask_bt[..., None], mask_tok, tok)

            out[k] = tok

        x = torch.stack([out[k] for k in keys], dim=2)  # [B,T,R,D]
        return x, keys, region_indices, spikes.size(2)

    def denoise_region_latents(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict epsilon in latent space.

        x: [B,T,R,D] = x_t
        t: [B] long timesteps

        returns: eps_pred [B,T,R,D]
        """
        B, T, R, D = x.shape

        # RoPE over time
        if (
            self.rope_cache is None
            or self.rope_cache[0].shape[1] != T
            or self.rope_cache[0].device != x.device
        ):
            self.rope_cache = build_rope_cache(T, D, x.device)
        x = apply_rope_time(x, *self.rope_cache)

        # Add timestep conditioning (broadcast to all tokens)
        t_emb = self.t_embed(t).to(x.dtype)         # [B,D]
        x = x + t_emb[:, None, None, :]             # [B,T,R,D]

        # Transformer blocks
        x = x.view(B, T * R, D)
        for blk in self.blocks:
            x = blk(x)
        x = self.final_ln(x)
        x = x.view(B, T, R, D)

        return x  # interpret as eps_pred

    def region_latents_to_spikes(self, x, keys, region_indices, N):
        region_tokens = {k: x[:, :, i] for i, k in enumerate(keys)}
        return self.region_codecs.decode_to_full(region_tokens, region_indices, N)

    def forward(
        self,
        batch: dict,
        *,
        masking_policy=None,
    ):
        """
        Convenience forward (NO diffusion, NO loss). Useful for quick inference/debug.

        Returns:
            pred_full : [B, T, N]
        """
        spikes = batch["spikes_data"]           # [B,T,N]
        neuron_regions = batch["neuron_regions"]  # [B,N]
        eids = batch["eid"].view(-1)

        B, T, N = spikes.shape
        device = spikes.device

        hidden_nm = None
        if masking_policy is not None:
            _, hidden_nm, _ = masking_policy(batch)

        # Encode neurons -> region tokens
        region_tokens, region_indices = self.region_codecs.encode(
            spikes,
            neuron_regions,
            eids,
        )
        keys = self._sorted_keys(region_tokens)

        # Apply region-time masking (derived from hidden_nm)
        if hidden_nm is not None:
            hidden_rm = neuron_mask_to_region_time_mask(hidden_nm, neuron_regions)  # [B,T,R]

            out = {}
            for r_idx, k in enumerate(keys):
                tok = region_tokens[k]  # [B,T,D]
                mask_bt = hidden_rm[:, :, r_idx]  # [B,T]

                if mask_bt.any():
                    rid = torch.tensor(k[1], device=device)
                    rid_emb = self.region_id_embed(rid)[None, None, :]  # [1,1,D]
                    mask_tok = self.region_mask_token + rid_emb  # [1,1,D] broadcastable

                    tok = torch.where(mask_bt[..., None], mask_tok, tok)
                out[k] = tok
            region_tokens = out

        # Stack + RoPE + transformer (no timestep conditioning here)
        x = torch.stack([region_tokens[k] for k in keys], dim=2)  # [B,T,R,D]

        if self.rope_cache is None or self.rope_cache[0].shape[1] != T or self.rope_cache[0].device != device:
            self.rope_cache = build_rope_cache(T, self.d_model, device)
        x = apply_rope_time(x, *self.rope_cache)

        x = x.view(B, T * len(keys), self.d_model)
        for blk in self.blocks:
            x = blk(x)
        x = self.final_ln(x)
        x = x.view(B, T, len(keys), self.d_model)

        # Decode region latents -> full neuron predictions
        region_tokens_out = {k: x[:, :, i] for i, k in enumerate(keys)}
        pred_full = self.region_codecs.decode_to_full(region_tokens_out, region_indices, N)
        return pred_full


# ============================================================
# Diffusion Wrapper
# ============================================================

class DiffusionWrapper(nn.Module):
    """
    True diffusion training in region-latent space using epsilon-prediction:
        x_t = add_noise(x0, eps, t)
        eps_pred = model(x_t, t)
        loss = MSE(eps_pred, eps)

    Notes:
      - target_spec is kept in the signature for compatibility with your training setup,
        but epsilon-prediction training does not use it directly.
      - masking_policy still controls:
          corrupt_nm: what you overwrite in neuron space before encoding
          hidden_nm:  which region-times receive mask tokens in latent space
          target_nm:  currently unused for epsilon training (kept for future hybrid losses)
    """
    def __init__(self, model, scheduler, target_spec: TargetSpec, masking_policy, reconstruct_loss_weight=1.0):
        super().__init__()
        self.model = model
        self.scheduler = scheduler
        self.target_spec = target_spec
        self.masking_policy = masking_policy
        self.reconstruct_loss_weight = reconstruct_loss_weight
        self.disable_diffusion = False


    def forward(self, batch: dict):
        spikes = batch["spikes_data"]              # [B,T,N]
        neuron_regions = batch["neuron_regions"]   # [B,N]
        eids = batch["eid"].view(-1)

        B, T, N = spikes.shape

        corrupt_nm, hidden_nm, target_nm = normalize_neuron_mask_output(
            self.masking_policy(batch),
            B=B,
            T=T,
            N=N,
        )  # all [B,T,N] bool

        # 1) Corrupt inputs in neuron space (keeps your masking semantics)
        spikes_corrupt = torch.where(corrupt_nm, torch.zeros_like(spikes), spikes)

        # 2) Encode -> region latents (x0) and inject region-mask tokens based on hidden_nm
        x0, keys, region_indices, N_total = self.model.spikes_to_region_latents(
            spikes_corrupt, neuron_regions, eids, hidden_nm=hidden_nm
        )  # x0: [B,T,R,D]

        # 3) Sample timestep and noise, construct x_t
        t = torch.randint(
            0, self.scheduler.config.num_train_timesteps,
            (B,), device=spikes.device, dtype=torch.long
        )

        eps = torch.randn_like(x0)
        if self.disable_diffusion:
            x_t = x0
        else:
            x_t = self.scheduler.add_noise(x0, eps, t)

        # 4) Predict epsilon (true diffusion objective)
        eps_pred = self.model.denoise_region_latents(x_t, t)   # [B,T,R,D]

        # 5) MSE loss in latent space (DDPM epsilon loss)
        diff_loss = torch.mean((eps_pred - eps) ** 2)

        # 6) Reconstion loss in neuron space
        pred_recon = self.model.region_latents_to_spikes(
            x0, keys, region_indices, N_total
        )
        target = self.target_spec.get_target(batch)
        recon_loss = self.target_spec.loss(
            pred=pred_recon,
            target=target,
            mask=torch.ones_like(target, dtype=torch.bool),
        )

        return diff_loss + self.reconstruct_loss_weight * recon_loss