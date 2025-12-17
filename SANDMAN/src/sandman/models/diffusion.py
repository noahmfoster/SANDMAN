import torch
import torch.nn as nn

from sandman.models.utils import apply_rope_time, build_rope_cache
from sandman.models.utils import TargetSpec

from typing import Optional, Iterable, Tuple, Union


# ============================================================
# Region-level encoder (neurons -> region token)
# ============================================================

class RegionEncoder(nn.Module):
    def __init__(self, d_model, encoder_n_heads: int = 4):
        super().__init__()
        self.neuron_proj = nn.Linear(1, d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=encoder_n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, spikes_region):
        """
        spikes_region: [B, T, N_r]
        returns:       [B, T, D]
        """
        B, T, N_r = spikes_region.shape

        # [B, T, N_r, D]
        x = self.neuron_proj(spikes_region[..., None])

        # Treat neurons as tokens *per time*
        x = x.view(B * T, N_r, -1)

        # Self-attention across neurons
        x, _ = self.attn(x, x, x)

        # Pool AFTER attention
        x = x.mean(dim=1)  # [B*T, D]

        x = x.view(B, T, -1)
        return self.norm(x)


# ============================================================
# Neuron-specific decoder (region token -> neurons)
# ============================================================

class RegionDecoder(nn.Module):
    """
    One Linear per (eid, region): D -> N_region_neurons
    """
    def __init__(self, d_model: int, n_neurons: int):
        super().__init__()
        self.n_neurons = int(n_neurons)
        self.readout = nn.Linear(d_model, self.n_neurons)

    def forward(self, region_token: torch.Tensor) -> torch.Tensor:
        """
        region_token: [B, T, D]
        returns:      [B, T, N_r]
        """
        return self.readout(region_token)


# ============================================================
# Dictionary of encoders / decoders keyed by (eid, region)
# ============================================================

class RegionCodecDict(nn.Module):
    """
    Holds (encoder, decoder) pairs for each (eid, region).
    Tracks neuron indices so we can scatter predictions back into original neuron order.
    """
    def __init__(self, d_model: int, encoder_n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.encoder_n_heads = encoder_n_heads
        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        self._decoder_n_neurons = {}  # key -> int (python cache)

    def _key(self, eid, region_id) -> str:
        return f"{int(eid)}:{int(region_id)}"

    def get_codec(self, eid, region_id, n_neurons: int, device: torch.device):
        key = self._key(eid, region_id)

        if key not in self.encoders:
            self.encoders[key] = RegionEncoder(self.d_model, encoder_n_heads=self.encoder_n_heads).to(device)
            self.decoders[key] = RegionDecoder(self.d_model, int(n_neurons)).to(device)
            self._decoder_n_neurons[key] = int(n_neurons)
        else:
            expected = self._decoder_n_neurons.get(key, None)
            if expected is not None and int(n_neurons) != expected:
                raise ValueError(
                    f"Neuron count changed for codec {key}: expected {expected}, got {int(n_neurons)}.\n"
                    "If this is intended (dynamic neuron set), you need a more flexible decoder design."
                )

        return self.encoders[key], self.decoders[key]

    def encode(self, spikes: torch.Tensor, neuron_regions: torch.Tensor, eids: torch.Tensor):
        """
        spikes:         [B, T, N]
        neuron_regions: [B, N]
        eids:           [B]

        Returns:
          region_tokens:  {(eid, region_id): [B, T, D]}
          region_indices: {(eid, region_id): LongTensor[N_r]}
        """
        device = spikes.device
        region_tokens = {}
        region_indices = {}

        # Assumption: one eid per batch
        eid = int(eids[0].item())

        unique_regions = torch.unique(neuron_regions[0])

        for rid in unique_regions:
            region_id = int(rid.item())

            idxs = torch.nonzero(neuron_regions[0] == region_id, as_tuple=False).squeeze(-1)  # [N_r]
            spikes_region = spikes.index_select(dim=2, index=idxs)  # [B, T, N_r]

            encoder, _ = self.get_codec(eid, region_id, n_neurons=idxs.numel(), device=device)
            token = encoder(spikes_region)  # [B, T, D]

            key = (eid, region_id)
            region_tokens[key] = token
            region_indices[key] = idxs

        return region_tokens, region_indices

    def decode_to_full(
        self,
        region_tokens: dict,
        region_indices: dict,
        N_total: int,
    ) -> torch.Tensor:
        """
        region_tokens:  {(eid, region_id): [B, T, D]}
        region_indices: {(eid, region_id): [N_r]}

        Returns:
          pred_full: [B, T, N_total]
        """
        any_token = next(iter(region_tokens.values()))
        B, T, _ = any_token.shape
        device = any_token.device
        dtype = any_token.dtype

        pred_full = torch.zeros((B, T, N_total), device=device, dtype=dtype)

        for (eid, region_id), token in region_tokens.items():
            idxs = region_indices[(eid, region_id)]
            _, decoder = self.get_codec(eid, region_id, n_neurons=idxs.numel(), device=device)
            pred_region = decoder(token)  # [B, T, N_r]

            pred_full.index_copy_(dim=2, index=idxs, source=pred_region)

        return pred_full


# ============================================================
# Transformer over (time × region) tokens
# ============================================================

class NeuralTransformer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int, encoder_n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.region_codecs = RegionCodecDict(d_model, encoder_n_heads=encoder_n_heads)

        # Learned region-mask token (available but optional)
        self.region_mask_token = nn.Parameter(torch.zeros(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.rope_cache = None

    @staticmethod
    def _normalize_masked_region_ids(
        masked_region_ids: Optional[Union[Iterable[int], torch.Tensor]]
    ) -> Optional[set]:
        if masked_region_ids is None:
            return None
        if torch.is_tensor(masked_region_ids):
            # allow LongTensor of region ids
            if masked_region_ids.numel() == 0:
                return set()
            return set(int(x) for x in torch.unique(masked_region_ids).tolist())
        return set(int(x) for x in masked_region_ids)

    def forward(
        self,
        spikes_noisy: torch.Tensor,   # [B, T, N]
        neuron_regions: torch.Tensor, # [B, N]
        eids: torch.Tensor,           # [B]
        timesteps: torch.Tensor,      # [B] (unused for now)
        *,
        masked_region_ids: Optional[Union[Iterable[int], torch.Tensor]] = None,
    ):
        B, T, N = spikes_noisy.shape
        device = spikes_noisy.device

        masked_set = self._normalize_masked_region_ids(masked_region_ids)

        # Encode neurons -> region tokens + keep neuron indices for each region
        region_tokens, region_indices = self.region_codecs.encode(
            spikes_noisy, neuron_regions, eids
        )

        # OPTIONAL: replace selected region tokens with learned mask token
        # This is the inpainting signal. If masked_set is None/empty, no masking happens.
        if masked_set:
            for (eid_k, region_id) in list(region_tokens.keys()):
                if int(region_id) in masked_set:
                    region_tokens[(eid_k, region_id)] = self.region_mask_token.expand(B, T, -1)

        # Stable region ordering for stacking
        keys = sorted(region_tokens.keys(), key=lambda x: x[1])

        x = torch.stack([region_tokens[k] for k in keys], dim=2)  # [B, T, R, D]
        R = x.size(2)

        # RoPE over time
        if (
            self.rope_cache is None
            or self.rope_cache[0].shape[1] != T
            or self.rope_cache[0].device != device
        ):
            self.rope_cache = build_rope_cache(T, self.d_model, device)
        cos, sin = self.rope_cache
        x = apply_rope_time(x, cos, sin)

        # Transformer over flattened (T * R) tokens
        x = x.view(B, T * R, self.d_model)
        x = self.transformer(x)
        x = x.view(B, T, R, self.d_model)

        # Unstack back into dict of region tokens
        updated_region_tokens = {key: x[:, :, i, :] for i, key in enumerate(keys)}

        # Decode into full neuron order [B, T, N]
        pred_full = self.region_codecs.decode_to_full(
            updated_region_tokens, region_indices, N_total=N
        )

        return pred_full, keys


# ============================================================
# Diffusion wrapper (loss in neuron space)
# ============================================================

class DiffusionWrapper(nn.Module):
    """
    Diffusion wrapper with optional inpainting semantics via masked_region_ids.

    Mask semantics:
      - input_mask  [B, T, N] bool : neurons to corrupt (diffuse)
      - target_mask [B, T, N] bool : neurons to score in the loss

    Inpainting semantics (optional):
      - masked_region_ids: iterable/tensor of region IDs whose region tokens are replaced
        by the learned region_mask_token in the transformer.
      - If None/empty, model behaves like a normal transformer (no mask tokens injected).
    """
    def __init__(self, model, scheduler, target_spec: TargetSpec, masking_policy):
        super().__init__()
        self.model = model
        self.scheduler = scheduler
        self.target_spec = target_spec
        self.masking_policy = masking_policy
        self.disable_diffusion = False

    def _run_masking_policy(self, batch: dict):
        out = self.masking_policy(batch)

        if not isinstance(out, tuple):
            raise ValueError("masking_policy must return a tuple")

        if len(out) == 2:
            input_mask, target_mask = out
            masked_region_ids = None
        elif len(out) == 3:
            input_mask, target_mask, masked_region_ids = out
        else:
            raise ValueError(
                "masking_policy must return either (input_mask, target_mask) or "
                "(input_mask, target_mask, masked_region_ids)"
            )

        return input_mask, target_mask, masked_region_ids

    def forward(self, batch: dict) -> torch.Tensor:
        spikes = batch["spikes_data"]  # [B, T, N]
        neuron_regions_full = batch["neuron_regions_full"]
        eids = batch["eid"]
        if eids.ndim == 0:
            eids = eids[None]

        B, T, N = spikes.shape
        device = spikes.device
        neuron_regions = neuron_regions_full[:, :N]

        # Masks + optional masked_region_ids (policy decides)
        input_mask, target_mask, masked_region_ids = self._run_masking_policy(batch)

        if input_mask is None and target_mask is None:
            raise ValueError("masking_policy must return at least one mask")

        # default behavior: if one is None, use the other
        if input_mask is None:
            input_mask = target_mask
        if target_mask is None:
            target_mask = input_mask

        if input_mask.shape != (B, T, N) or target_mask.shape != (B, T, N):
            raise ValueError(
                f"Expected masks [B,T,N]=({B},{T},{N}), got {input_mask.shape} and {target_mask.shape}"
            )

        # Diffusion timestep
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps,
            (B,), device=device
        )

        # Add noise + corrupt only masked inputs
        noise = torch.randn_like(spikes)
        x_noisy_all = self.scheduler.add_noise(spikes, noise, timesteps)
        spikes_noisy = torch.where(input_mask, x_noisy_all, spikes)
        if self.disable_diffusion:
            spikes_noisy = spikes # for debugging: no noise, 

        # Model predicts in neuron space aligned to original ordering
        pred_full, _ = self.model(
            spikes_noisy,
            neuron_regions,
            eids,
            timesteps,
            masked_region_ids=masked_region_ids,  # <-- optional inpainting signal
        )  # [B, T, N]

        # Target
        target = self.target_spec.get_target(batch)[:, :, :N]  # [B, T, N]

        # Loss only on target_mask (TargetSpec handles Poisson + softplus etc.)
        loss = self.target_spec.loss(pred=pred_full, target=target, mask=target_mask)

        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError("Loss exploded")

        if loss.item() < 1e-6:
            print("WARNING: near-zero loss")
        return loss
