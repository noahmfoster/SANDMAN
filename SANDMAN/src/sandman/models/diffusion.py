import math
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sandman.models.utils import (
    apply_rope_time,
    build_rope_cache,
    normalize_neuron_mask_output,
)
from sandman.models.utils import TargetSpec


# ============================================================
# Helpers
# ============================================================

def _unique_sorted_region_ids(neuron_regions: torch.Tensor) -> List[int]:
    """
    neuron_regions: [B, N] (int)
    Returns sorted unique region ids across the whole batch.
    """
    r = torch.unique(neuron_regions)
    r = r[r >= 0]  # in case you use -1 for padding
    return sorted([int(x.item()) for x in r])


def _gather_by_index(x: torch.Tensor, idxs: torch.Tensor) -> torch.Tensor:
    """
    x: [B, T, N, ...], idxs: [M] indices in neuron dimension
    returns: [B, T, M, ...]
    """
    return x.index_select(2, idxs)


def _region_time_hidden_mask(hidden_nm: torch.Tensor, neuron_regions: torch.Tensor, region_ids: List[int]) -> torch.Tensor:
    """
    hidden_nm: [B, T, N] bool
    neuron_regions: [B, N] int
    region_ids: list length R
    returns: hidden_rm [B, T, R] bool where True if ANY neuron in that region is hidden at (B,T)
    """
    B, T, N = hidden_nm.shape
    device = hidden_nm.device
    R = len(region_ids)
    hidden_rm = torch.zeros(B, T, R, device=device, dtype=torch.bool)
    for r_idx, rid in enumerate(region_ids):
        mask_n = (neuron_regions == rid)  # [B, N] bool
        any_hidden = (hidden_nm & mask_n[:, None, :]).any(dim=2)  # [B,T]
        hidden_rm[:, :, r_idx] = any_hidden
    return hidden_rm


# ============================================================
# Diffusion conditioning (DiT-style)
# ============================================================

class TimestepEmbedder(nn.Module):
    """
    Sinusoidal timestep embedding + MLP.
    """
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = int(frequency_embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """
        t: [B] int/float
        returns: [B, dim]
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class AdaLNDiTBlock(nn.Module):
    """
    Transformer block with AdaLN (shift/scale/gate) from a conditioning vector.
    - Pre-norm
    - MHA + MLP
    - Zero-init conditioner last layer -> starts near identity (stable diffusion training)
    """
    def __init__(self, d_model: int, n_heads: int, cond_dim: int, mlp_ratio: int = 4, attn_dropout: float = 0.0):
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.cond_dim = int(cond_dim)

        self.ln1 = nn.LayerNorm(self.d_model, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(self.d_model, self.n_heads, dropout=attn_dropout, batch_first=True)

        self.ln2 = nn.LayerNorm(self.d_model, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, mlp_ratio * self.d_model),
            nn.GELU(),
            nn.Linear(mlp_ratio * self.d_model, self.d_model),
        )

        # Produces: shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp
        self.cond = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.cond_dim, 6 * self.d_model, bias=True),
        )
        nn.init.zeros_(self.cond[-1].weight)
        nn.init.zeros_(self.cond[-1].bias)

    @staticmethod
    def _affine(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D], shift/scale: [B, D]
        return x * (1.0 + scale[:, None, :]) + shift[:, None, :]

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, D]
        cond: [B, cond_dim]
        """
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.cond(cond).chunk(6, dim=-1)

        # Attention
        h = self.ln1(x)
        h = self._affine(h, shift_a, scale_a)
        h = self.attn(h, h, h, need_weights=False)[0]
        x = x + gate_a[:, None, :] * h

        # MLP
        h = self.ln2(x)
        h = self._affine(h, shift_m, scale_m)
        h = self.mlp(h)
        x = x + gate_m[:, None, :] * h

        return x


# ============================================================
# Local pooling (neurons -> small Kenc channels) then collapse to ONE state per region/time
# ============================================================

class RegionAttentionPool(nn.Module):
    """
    Per region/time, pool variable N_r neuron tokens to fixed K queries via cross-attn.

    Inputs:
      x_neur: [B, T, N_r, D]
      key_padding_mask: [B*T, N_r] True where padded (to ignore)
      region_emb: [B, D] used to condition the K queries

    Output:
      z: [B, T, K, D]
    """
    def __init__(self, d_model: int, n_heads: int, K: int):
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.K = int(K)

        self.base_queries = nn.Parameter(torch.randn(self.K, self.d_model) * 0.02)

        self.q_mlp = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.K * self.d_model),
        )
        nn.init.zeros_(self.q_mlp[-1].weight)
        nn.init.zeros_(self.q_mlp[-1].bias)

        self.cross_attn = nn.MultiheadAttention(self.d_model, self.n_heads, batch_first=True)
        self.out_ln = nn.LayerNorm(self.d_model)

    def make_queries(self, region_emb: torch.Tensor) -> torch.Tensor:
        """
        region_emb: [B, D]
        returns queries: [B, K, D]
        """
        B, D = region_emb.shape
        base = self.base_queries[None, :, :].expand(B, -1, -1)  # [B,K,D]
        delta = self.q_mlp(region_emb).view(B, self.K, self.d_model)
        return base + delta

    def forward(self, x_neur: torch.Tensor, key_padding_mask: torch.Tensor, region_emb: torch.Tensor) -> torch.Tensor:
        B, T, N_r, D = x_neur.shape
        assert D == self.d_model

        q = self.make_queries(region_emb)  # [B,K,D]
        q = q[:, None, :, :].expand(B, T, self.K, D).contiguous()  # [B,T,K,D]

        q_bt = q.view(B * T, self.K, D)
        kv_bt = x_neur.view(B * T, N_r, D)

        z_bt = self.cross_attn(
            query=q_bt,
            key=kv_bt,
            value=kv_bt,
            key_padding_mask=key_padding_mask,  # [B*T, N_r]
            need_weights=False,
        )[0]  # [B*T,K,D]

        z = z_bt.view(B, T, self.K, D)
        return self.out_ln(z)


class NeuronTokenEmbed(nn.Module):
    """
    Embed spikes + local neuron-slot identity (session-local) + region ID (+ optional EID).
    This is used ONLY locally (per-region encode/decode); neuron tokens are never global diffusion tokens.
    """
    def __init__(self, d_model: int, max_neurons: int, max_regions: int, max_eids: int, use_eid: bool = True):
        super().__init__()
        self.d_model = int(d_model)
        self.use_eid = bool(use_eid)

        self.spike_proj = nn.Linear(1, self.d_model)

        self.neuron_slot = nn.Embedding(int(max_neurons), self.d_model)
        self.region_emb = nn.Embedding(int(max_regions), self.d_model)
        self.eid_emb = nn.Embedding(int(max_eids), self.d_model) if self.use_eid else None

    def forward(self, spikes: torch.Tensor, neuron_regions: torch.Tensor, eids: torch.Tensor) -> torch.Tensor:
        """
        spikes: [B,T,N]
        neuron_regions: [B,N]
        eids: [B]
        returns: [B,T,N,D]
        """
        B, T, N = spikes.shape
        device = spikes.device

        x = self.spike_proj(spikes[..., None])  # [B,T,N,D]

        idx = torch.arange(N, device=device)
        x = x + self.neuron_slot(idx)[None, None, :, :]

        x = x + self.region_emb(neuron_regions.clamp(min=0, max=self.region_emb.num_embeddings - 1))[:, None, :, :]

        if self.use_eid:
            x = x + self.eid_emb(eids.clamp(min=0, max=self.eid_emb.num_embeddings - 1))[:, None, None, :]

        return x


class RegionIdentity(nn.Module):
    """
    Session-aware region identity embedding: combines (region_id, eid) into one vector.
    """
    def __init__(self, d_model: int, max_regions: int, max_eids: int, use_eid: bool = True):
        super().__init__()
        self.d_model = int(d_model)
        self.use_eid = bool(use_eid)

        self.region_id_embed = nn.Embedding(int(max_regions), self.d_model)
        self.eid_embed = nn.Embedding(int(max_eids), self.d_model) if self.use_eid else None

        self.proj = nn.Sequential(
            nn.LayerNorm(2 * self.d_model),
            nn.Linear(2 * self.d_model, self.d_model),
        )

    def forward(self, region_ids: torch.Tensor, eids: torch.Tensor) -> torch.Tensor:
        """
        region_ids: [B] or [B*R]
        eids: [B] or [B*R]
        returns: [B, D] or [B*R, D]
        """
        r = self.region_id_embed(region_ids)
        if self.use_eid:
            e = self.eid_embed(eids)
        else:
            e = torch.zeros_like(r)
        return self.proj(torch.cat([r, e], dim=-1))


class RegionStateEncoder(nn.Module):
    """
    For one region:
      neuron tokens [B,T,Nr,D_neur] -> pooled [B,T,Kenc,D_neur] -> state [B,T,D_state]
    """
    def __init__(self, d_neur: int, d_state: int, n_heads: int, Kenc: int = 4):
        super().__init__()
        self.d_neur = int(d_neur)
        self.d_state = int(d_state)
        self.Kenc = int(Kenc)

        self.pool = RegionAttentionPool(d_model=self.d_neur, n_heads=n_heads, K=self.Kenc)

        self.to_state = nn.Sequential(
            nn.LayerNorm(self.d_neur),
            nn.Linear(self.d_neur, self.d_state),
        )
        self.collapse = nn.Sequential(
            nn.LayerNorm(self.Kenc * self.d_state),
            nn.Linear(self.Kenc * self.d_state, self.d_state),
        )

    def forward(self, x_neur_pad: torch.Tensor, pad_mask_bt: torch.Tensor, region_emb: torch.Tensor) -> torch.Tensor:
        """
        x_neur_pad: [B,T,Nr,D_neur]
        pad_mask_bt: [B*T, Nr] True where padded
        region_emb: [B, D_neur]
        returns: U: [B,T,D_state]
        """
        z = self.pool(x_neur_pad, key_padding_mask=pad_mask_bt, region_emb=region_emb)  # [B,T,Kenc,D_neur]
        z = self.to_state(z)  # [B,T,Kenc,D_state]
        B, T, K, Ds = z.shape
        u = self.collapse(z.reshape(B, T, K * Ds))  # [B,T,D_state]
        return u


class HyperNeuronDecoder(nn.Module):
    """
    Produces per-neuron readout weights from neuron identity embeddings, then predicts:
      y[b,t,n] = <w[b,n], U[b,t,r(n)]> + b[b,n]

    This keeps diffusion tokens minimal and avoids neuron attention.
    """
    def __init__(self, d_state: int, d_neur_id: int, max_neurons: int, max_regions: int, max_eids: int, use_eid: bool = True):
        super().__init__()
        self.d_state = int(d_state)
        self.d_neur_id = int(d_neur_id)
        self.use_eid = bool(use_eid)

        self.neuron_slot = nn.Embedding(int(max_neurons), self.d_neur_id)
        self.region_emb = nn.Embedding(int(max_regions), self.d_neur_id)
        self.eid_emb = nn.Embedding(int(max_eids), self.d_neur_id) if self.use_eid else None

        self.hyper = nn.Sequential(
            nn.LayerNorm(self.d_neur_id),
            nn.Linear(self.d_neur_id, 2 * self.d_state),
            nn.GELU(),
            nn.Linear(2 * self.d_state, self.d_state + 1),  # w (d_state) + b (1)
        )
        nn.init.zeros_(self.hyper[-1].bias)

    def forward(self, U: torch.Tensor, neuron_regions: torch.Tensor, eids: torch.Tensor, *, r_map: torch.Tensor) -> torch.Tensor:
        """
        U: [B,T,R,D_state] where R corresponds to region_ids order
        neuron_regions: [B,N] region ids (global ids)
        eids: [B]
        r_map: [max_regions] maps global region_id -> local index in [0..R-1], or -1

        returns pred: [B,T,N]
        """
        B, T, R, Ds = U.shape
        _, N = neuron_regions.shape
        device = U.device

        # neuron identity embedding (no spikes)
        idx = torch.arange(N, device=device)
        e = self.neuron_slot(idx)[None, :, :].expand(B, -1, -1)  # [B,N,d_neur_id]
        e = e + self.region_emb(neuron_regions.clamp(min=0, max=self.region_emb.num_embeddings - 1))
        if self.use_eid:
            e = e + self.eid_emb(eids.clamp(min=0, max=self.eid_emb.num_embeddings - 1))[:, None, :]

        wb = self.hyper(e)  # [B,N,Ds+1]
        w = wb[..., :Ds]    # [B,N,Ds]
        b = wb[..., Ds:]    # [B,N,1]

        # map global region ids -> local indices into U's R dimension
        local_r = r_map[neuron_regions.clamp(min=0, max=r_map.numel() - 1)]  # [B,N]
        # any -1 means neuron_region wasn't present in region_ids; clamp to 0 to avoid gather crash
        local_r_safe = local_r.clamp(min=0)

        # gather region state per neuron: [B,T,N,Ds]
        idx_g = local_r_safe[:, None, :, None].expand(B, T, N, Ds)  # [B,T,N,Ds]
        U_neur = U.gather(dim=2, index=idx_g)  # [B,T,N,Ds]

        pred = (U_neur * w[:, None, :, :]).sum(dim=-1) + b[:, None, :, 0]  # [B,T,N]

        # optionally zero out neurons whose region wasn't present
        missing = (local_r < 0)  # [B,N]
        if missing.any():
            pred = torch.where(missing[:, None, :], torch.zeros_like(pred), pred)

        return pred


class NeuralTransformer(nn.Module):
    """
    - Encode neurons -> ONE region-state vector per region/time: U0 [B,T,R,D]
    - Diffusion denoises U_t in that same space
    - Decode to neurons via hypernetwork linear readout (Option C1)

    Methods preserved:
      - spikes_to_region_latents(...)
      - denoise_region_latents(x, t)
      - region_latents_to_spikes(...)
      - forward(batch, masking_policy=None)

    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        *,
        K: int = 8,  # now used as Kenc for local pooling (NOT diffusion channels)
        max_neurons: int = 4096,
        max_regions: int = 512,
        max_eids: int = 4096,
        use_eid: bool = True,
        cond_dim: Optional[int] = None,
        d_neur_id: Optional[int] = None,  # decoder identity width
    ):
        super().__init__()
        self.d_model = int(d_model)     # region-state dimension (D_state)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.Kenc = int(K)
        self.use_eid = bool(use_eid)
        self.max_regions = int(max_regions)

        # Local neuron tokenization for region encoding
        self.tokenizer = NeuronTokenEmbed(
            d_model=self.d_model,
            max_neurons=max_neurons,
            max_regions=max_regions,
            max_eids=max_eids,
            use_eid=use_eid,
        )

        # Session-aware region identity
        self.region_identity = RegionIdentity(
            d_model=self.d_model,
            max_regions=max_regions,
            max_eids=max_eids,
            use_eid=use_eid,
        )

        # Region encoder: local pooling -> ONE state per region/time
        self.region_encoder = RegionStateEncoder(
            d_neur=self.d_model,
            d_state=self.d_model,
            n_heads=self.n_heads,
            Kenc=self.Kenc,
        )

        # Mask token in STATE space (one vector per hidden region/time)
        self.state_mask_token = nn.Parameter(torch.zeros(1, 1, self.d_model))  # [1,1,D]

        # Diffusion conditioning
        self.cond_dim = int(cond_dim) if cond_dim is not None else max(self.d_model // 4, 32)
        self.t_embed = TimestepEmbedder(self.cond_dim)
        self.cond_proj = nn.Linear(self.d_model, self.cond_dim)

        # Diffusion backbone (over tokens T*R)
        self.blocks = nn.ModuleList([
            AdaLNDiTBlock(self.d_model, self.n_heads, cond_dim=self.cond_dim)
            for _ in range(self.n_layers)
        ])
        self.final_ln = nn.LayerNorm(self.d_model)

        # Decoder: hypernetwork linear readout
        d_neur_id = int(d_neur_id) if d_neur_id is not None else max(self.d_model // 2, 64)
        self.decoder = HyperNeuronDecoder(
            d_state=self.d_model,
            d_neur_id=d_neur_id,
            max_neurons=max_neurons,
            max_regions=max_regions,
            max_eids=max_eids,
            use_eid=use_eid,
        )

        self.rope_cache = None

    # --------------------------------------------------
    # Stage module naming (for wrapper freezing)
    # --------------------------------------------------
    def ae_named_modules(self) -> List[str]:
        """
        Autoencoding path:
        spikes -> tokenizer -> region_encoder -> decoder
        plus identity embeddings used by encoder/decoder.
        """
        return [
            "tokenizer",
            "region_identity",
            "region_encoder",
            "state_mask_token",
            "decoder",
        ]

    def diffusion_named_modules(self) -> List[str]:
        """
        Diffusion denoiser path.
        """
        return [
            "blocks",
            "final_ln",
            "t_embed",
            "cond_proj",
            "region_identity",
        ]

    # --------------------------------------------------
    # Conditioning helper
    # --------------------------------------------------
    def _make_cond(self, t: torch.Tensor, *, region_summary: torch.Tensor) -> torch.Tensor:
        """
        region_summary: [B, D] (e.g., mean over regions of region identity embeddings)
        """
        return self.t_embed(t) + self.cond_proj(region_summary)

    # ======================================================
    # Encode: neurons → region STATES (latents)
    # ======================================================
    def spikes_to_region_latents(
        self,
        spikes: torch.Tensor,          # [B,T,N]
        neuron_regions: torch.Tensor,  # [B,N]
        eids: torch.Tensor,            # [B]
        *,
        hidden_nm: torch.Tensor,       # [B,T,N] bool
    ):
        """
        Returns:
          U0: [B,T,R,D] region-state latents
          region_ids: List[int] length R
          region_indices: Dict[(b,rid)->idxs] (kept for drop-in compatibility)
          N: int number of neurons
        """
        B, T, N = spikes.shape
        device = spikes.device

        region_ids = _unique_sorted_region_ids(neuron_regions)
        R = len(region_ids)

        # tokenize neurons (local only)
        x_neur_all = self.tokenizer(spikes, neuron_regions, eids)  # [B,T,N,D]
        hidden_rm = _region_time_hidden_mask(hidden_nm, neuron_regions, region_ids)  # [B,T,R]

        U0 = torch.zeros(B, T, R, self.d_model, device=device, dtype=x_neur_all.dtype)
        region_indices: Dict[Tuple[int, int], torch.Tensor] = {}

        # precompute mapping region id -> local index (for decoder gather)
        # (also returned as region_ids list; wrapper doesn't need r_map)
        for r_idx, rid in enumerate(region_ids):
            idxs_list = []
            max_nr = 0
            for b in range(B):
                idxs = torch.where(neuron_regions[b] == rid)[0]
                idxs_list.append(idxs)
                max_nr = max(max_nr, idxs.numel())
                region_indices[(b, rid)] = idxs

            if max_nr == 0:
                continue

            # pad neurons for this region across batch
            x_pad = torch.zeros(B, T, max_nr, self.d_model, device=device, dtype=x_neur_all.dtype)
            pad_mask = torch.ones(B, T, max_nr, device=device, dtype=torch.bool)  # True = padded

            for b in range(B):
                idxs = idxs_list[b]
                if idxs.numel() == 0:
                    continue
                x_pad[b, :, : idxs.numel()] = _gather_by_index(x_neur_all[b:b+1], idxs).squeeze(0)
                pad_mask[b, :, : idxs.numel()] = False

            # region identity embedding for this region across batch
            rid_batch = torch.full((B,), rid, device=device, dtype=torch.long)
            region_emb = self.region_identity(rid_batch, eids)  # [B,D]

            # encode to ONE state per time
            u = self.region_encoder(
                x_neur_pad=x_pad,
                pad_mask_bt=pad_mask.view(B * T, max_nr),
                region_emb=region_emb,
            )  # [B,T,D]

            # apply state mask where region is hidden at (b,t)
            mask_tok = self.state_mask_token + region_emb[:, None, :]  # [B,1,D]
            u = torch.where(
                hidden_rm[:, :, r_idx][:, :, None],
                mask_tok.expand(B, T, self.d_model),
                u,
            )

            U0[:, :, r_idx, :] = u

        return U0, region_ids, region_indices, N

    # ======================================================
    # Denoise: diffusion backbone over region STATES
    # ======================================================
    def denoise_region_latents(
        self,
        U: torch.Tensor,                 # [B,T,R,D]
        t: torch.Tensor,                 # [B]
        region_ids: Optional[List[int]] = None,
        eids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict epsilon in region-state space. Output shape matches U.
        """
        B, T, R, D = U.shape
        device = U.device

        # Apply RoPE over time (treat R as "channels" dimension)
        # build_rope_cache expects (T, D) and apply_rope_time expects [B,T,*,D]
        U_time = U.view(B, T, R, D)
        if self.rope_cache is None or self.rope_cache[0].shape[1] != T:
            self.rope_cache = build_rope_cache(T, D, device)
        U_time = apply_rope_time(U_time, *self.rope_cache)  # [B,T,R,D]

        # Add region identity as content bias (session-aware)
        if region_ids is None or eids is None:
            # allow calling without ids (e.g., toy usage)
            region_bias = torch.zeros(B, R, D, device=device, dtype=U.dtype)
            region_summary = torch.zeros(B, D, device=device, dtype=U.dtype)
        else:
            rid_t = torch.tensor(region_ids, device=device, dtype=torch.long)  # [R]
            rid_bt = rid_t[None, :].expand(B, R).reshape(B * R)               # [B*R]
            e_bt = eids[:, None].expand(B, R).reshape(B * R)                 # [B*R]
            region_bias = self.region_identity(rid_bt, e_bt).view(B, R, D)   # [B,R,D]
            region_summary = region_bias.mean(dim=1)                         # [B,D]

        U_time = U_time + region_bias[:, None, :, :]  # [B,T,R,D]

        # Flatten tokens: [B, T*R, D]
        seq = U_time.reshape(B, T * R, D)

        # AdaLN cond: timestep + region summary
        cond = self._make_cond(t, region_summary=region_summary)  # [B,cond_dim]

        for blk in self.blocks:
            seq = blk(seq, cond)

        seq = self.final_ln(seq)
        return seq.view(B, T, R, D)

    # ======================================================
    # Decode: region STATES → neuron predictions (hypernetwork)
    # ======================================================
    def region_latents_to_spikes(
        self,
        U: torch.Tensor,                # [B,T,R,D]
        region_ids: List[int],
        region_indices: Dict[Tuple[int, int], torch.Tensor],  # unused but kept for compatibility
        N: int,
        neuron_regions: torch.Tensor,   # [B,N] global region ids
        eids: torch.Tensor,             # [B]
    ) -> torch.Tensor:
        """
        Decode region states back to full neuron predictions [B,T,N].
        """
        B, T, R, D = U.shape
        device = U.device

        # build mapping from global region id -> local index in [0..R-1]
        r_map = torch.full((self.max_regions,), -1, device=device, dtype=torch.long)
        for j, rid in enumerate(region_ids):
            if 0 <= rid < self.max_regions:
                r_map[rid] = j

        pred = self.decoder(U, neuron_regions, eids, r_map=r_map)  # [B,T,N]
        return pred

    def forward(self, batch: dict, *, masking_policy=None) -> torch.Tensor:
        """
        Convenience forward (uses diffusion pass at t=0). Returns [B,T,N] predictions.
        """
        spikes = batch["spikes_data"]             # [B,T,N]
        neuron_regions = batch["neuron_regions"]  # [B,N]
        eids = batch["eid"].view(-1)              # [B]

        B, T, N = spikes.shape
        hidden_nm = torch.zeros(B, T, N, device=spikes.device, dtype=torch.bool)
        if masking_policy is not None:
            _, hidden_nm, _ = normalize_neuron_mask_output(masking_policy(batch), B=B, T=T, N=N)

        U0, region_ids, region_indices, _ = self.spikes_to_region_latents(
            spikes, neuron_regions, eids, hidden_nm=hidden_nm
        )  # [B,T,R,D]

        t0 = torch.zeros(B, device=spikes.device, dtype=torch.long)
        eps_pred = self.denoise_region_latents(U0, t0, region_ids=region_ids, eids=eids)  # [B,T,R,D]

        pred = self.region_latents_to_spikes(
            eps_pred, region_ids, region_indices, N, neuron_regions, eids
        )
        return pred


# ============================================================
# Diffusion Wrapper (updated for state-latent diffusion, drop-in file replacement)
# ============================================================

class DiffusionWrapper(nn.Module):
    """
    Multi-stage training wrapper.

    Stages:
      - "ae":        train encoder/decoder to reconstruct under masking (no diffusion)
      - "diffusion": train latent denoiser (optionally with recon loss)
      - "joint":     train both together

    This wrapper keeps the SAME external training semantics, but now:
      latents are U: [B,T,R,D] (ONE state per region/time).
    """
    def __init__(
        self,
        model: NeuralTransformer,
        scheduler,
        target_spec: TargetSpec,
        masking_policy,
        reconstruct_loss_weight: float = 1.0,
        *,
        stage: str = "joint",
        detach_latents_in_diffusion: bool = True,
        train_region_identity_in_diffusion: bool = True,
    ):
        super().__init__()
        self.model = model
        self.scheduler = scheduler
        self.target_spec = target_spec
        self.masking_policy = masking_policy
        self.reconstruct_loss_weight = float(reconstruct_loss_weight)

        self.stage = stage
        self.detach_latents_in_diffusion = bool(detach_latents_in_diffusion)
        self.train_region_identity_in_diffusion = bool(train_region_identity_in_diffusion)

        self.disable_diffusion = False

        self.set_stage(stage)

    # -----------------------------
    # Stage management
    # -----------------------------
    def set_stage(self, stage: str):
        stage = str(stage).lower()
        if stage not in ("ae", "diffusion", "joint"):
            raise ValueError(f"stage must be one of ['ae','diffusion','joint'], got {stage}")
        self.stage = stage

        for p in self.model.parameters():
            p.requires_grad_(False)

        if stage == "ae":
            self._set_trainable_by_prefix(self.model.ae_named_modules(), True)

        elif stage == "diffusion":
            self._set_trainable_by_prefix(["blocks", "final_ln", "t_embed", "cond_proj"], True)
            if self.train_region_identity_in_diffusion:
                self._set_trainable_by_prefix(["region_identity"], True)

        elif stage == "joint":
            for p in self.model.parameters():
                p.requires_grad_(True)

    def _set_trainable_by_prefix(self, prefixes: List[str], trainable: bool):
        for name, p in self.model.named_parameters():
            if any(name.startswith(pref) for pref in prefixes):
                p.requires_grad_(trainable)

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    # -----------------------------
    # Diffusion math helpers
    # -----------------------------
    def _alphas_cumprod(self, t: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        a = getattr(self.scheduler, "alphas_cumprod", None)
        if a is None:
            raise AttributeError("scheduler must expose alphas_cumprod for x0_pred formula.")
        a = a.to(device=device, dtype=dtype)
        return a.gather(0, t)  # [B]

    # -----------------------------
    # Forward dispatch per stage
    # -----------------------------
    def forward(self, batch: dict) -> torch.Tensor:
        if self.stage == "ae":
            return self.forward_ae(batch)
        elif self.stage == "diffusion":
            return self.forward_diffusion(batch)
        elif self.stage == "joint":
            loss_ae = self.forward_ae(batch)
            loss_diff = self.forward_diffusion(batch)
            return loss_ae + loss_diff
        else:
            raise RuntimeError(f"Unknown stage {self.stage}")

    # -----------------------------
    # Stage A: autoencoding / inpainting
    # -----------------------------
    def forward_ae(self, batch: dict) -> torch.Tensor:
        spikes = batch["spikes_data"]              # [B,T,N]
        neuron_regions = batch["neuron_regions"]   # [B,N]
        eids = batch["eid"].view(-1)               # [B]

        B, T, N = spikes.shape

        corrupt_nm, hidden_nm, target_nm = normalize_neuron_mask_output(
            self.masking_policy(batch),
            B=B, T=T, N=N,
        )

        spikes_corrupt = torch.where(corrupt_nm, torch.zeros_like(spikes), spikes)

        U0, region_ids, region_indices, N_total = self.model.spikes_to_region_latents(
            spikes_corrupt, neuron_regions, eids, hidden_nm=hidden_nm
        )  # [B,T,R,D]

        pred = self.model.region_latents_to_spikes(
            U0, region_ids, region_indices, N_total, neuron_regions, eids
        )  # [B,T,N]

        target = self.target_spec.get_target(batch)
        recon_loss = self.target_spec.loss(pred=pred, target=target, mask=target_nm)
        return recon_loss

    # -----------------------------
    # Stage B: diffusion training
    # -----------------------------
    def forward_diffusion(self, batch: dict) -> torch.Tensor:
        spikes = batch["spikes_data"]              # [B,T,N]
        neuron_regions = batch["neuron_regions"]   # [B,N]
        eids = batch["eid"].view(-1)               # [B]

        B, T, N = spikes.shape
        device = spikes.device

        corrupt_nm, hidden_nm, target_nm = normalize_neuron_mask_output(
            self.masking_policy(batch),
            B=B, T=T, N=N,
        )

        spikes_corrupt = torch.where(corrupt_nm, torch.zeros_like(spikes), spikes)

        U0, region_ids, region_indices, N_total = self.model.spikes_to_region_latents(
            spikes_corrupt, neuron_regions, eids, hidden_nm=hidden_nm
        )  # [B,T,R,D]

        # Optional detach if encoder is frozen
        if self.detach_latents_in_diffusion and (not any(p.requires_grad for p in self.model.region_encoder.parameters())):
            U0 = U0.detach()

        num_steps = int(self.scheduler.config.num_train_timesteps)
        t = torch.randint(0, num_steps, (B,), device=device, dtype=torch.long)

        eps = torch.randn_like(U0)
        U_t = U0 if self.disable_diffusion else self.scheduler.add_noise(U0, eps, t)

        eps_pred = self.model.denoise_region_latents(U_t, t, region_ids=region_ids, eids=eids)

        diff_loss = torch.mean((eps_pred - eps) ** 2)

        if self.reconstruct_loss_weight <= 0:
            return diff_loss

        a_bar = self._alphas_cumprod(t, device=device, dtype=eps_pred.dtype)  # [B]
        sqrt_a = torch.sqrt(a_bar)[:, None, None, None]
        sqrt_one_minus_a = torch.sqrt(1.0 - a_bar)[:, None, None, None]
        U0_pred = (U_t - sqrt_one_minus_a * eps_pred) / (sqrt_a + 1e-8)

        pred_recon = self.model.region_latents_to_spikes(
            U0_pred, region_ids, region_indices, N_total, neuron_regions, eids
        )
        target = self.target_spec.get_target(batch)
        recon_loss = self.target_spec.loss(pred=pred_recon, target=target, mask=target_nm)

        return diff_loss + self.reconstruct_loss_weight * recon_loss
