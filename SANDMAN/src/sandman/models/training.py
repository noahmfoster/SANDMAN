import torch
from typing import Optional, Dict, Any

from accelerate import Accelerator
from diffusers.optimization import get_scheduler
from diffusers.schedulers import DDPMScheduler
from tqdm.notebook import tqdm
import wandb

from sandman.models.utils import move_scheduler_to_device
from sandman.models.diffusion import DiffusionWrapper, NeuralTransformer

from sandman.models.utils import (
    TargetSpec,
    MaskingPolicy,
    SpikeCountMSETarget,
    mask_one_region,
    no_mask_policy,
)


# ============================================================
# Training
# ============================================================

def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """
    Move tensors in batch to device, while avoiding MPS float64 issues
    by downcasting float64 -> float32.
    """
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            # MPS does not support float64
            if v.dtype == torch.float64:
                v = v.float()
            out[k] = v.to(device)
        else:
            out[k] = v
    return out

def _ensure_optimizer_has_all_params(optimizer, model):
    opt_param_ids = set()
    for group in optimizer.param_groups:
        for p in group["params"]:
            opt_param_ids.add(id(p))

    new_params = [p for p in model.parameters() if p.requires_grad and id(p) not in opt_param_ids]
    if new_params:
        optimizer.add_param_group({"params": new_params})
        return len(new_params)
    return 0



def train_epoch(
    loader,
    diffusion: DiffusionWrapper,
    optimizer,
    accelerator: Accelerator,
    epoch: int,
    log_every: int = 50,
    wandb_enabled: bool = True,
):
    diffusion.train()
    total_loss = 0.0

    for step, batch in tqdm(enumerate(loader), total=len(loader)):
        batch = _move_batch_to_device(batch, accelerator.device)

        with accelerator.accumulate(diffusion):
            loss = diffusion(batch)
            _ensure_optimizer_has_all_params(optimizer, diffusion)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

        loss_item = loss.detach().item()
        total_loss += loss_item

        if accelerator.is_main_process and (step % log_every == 0) and wandb_enabled:
            wandb.log({
                "train/loss_step": loss_item,
                # "train/epoch": epoch,
                # "train/step": step,
            })

    avg_loss = total_loss / max(len(loader), 1)

    if accelerator.is_main_process and wandb_enabled:
        wandb.log({
            "train/loss_epoch": avg_loss,
            "epoch": epoch,
        })

    return avg_loss


# ============================================================
# Setup
# ============================================================

def prepare_data_and_model(
    data_loader,
    model_args=None,
    noise_scheduler_args=None,
    *,
    target_spec: Optional[TargetSpec] = None,
    masking_policy: Optional[MaskingPolicy] = None,
    wandb_project: str = "sandman-diffusion",
    wandb_name: str = "region_diffusion_poisson",
    max_epochs: int = 100,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    num_warmup_steps: int = 500,
    wandb_enabled: bool = True,
    reconstruct_loss_weight: float = 0.0,
):
    """
    Returns:
      diffusion, optimizer, lr_scheduler, accelerator, data_loader
    """
    if model_args is None:
        model_args = {"d_model": 12, "n_layers": 6, "n_heads": 2}

    if noise_scheduler_args is None:
        noise_scheduler_args = {
            "num_train_timesteps": 1000,
            "beta_start": 1e-4,
            "beta_end": 0.02,
            "beta_schedule": "linear",
        }

    # Defaults
    if masking_policy is None:
        # Best default for your inpainting goal
        masking_policy = mask_one_region

        # Alternatives you can pass instead:
        # masking_policy = lambda batch: random_neuron_denoising_policy(batch, mask_prob=0.15)
        # masking_policy = next_step_prediction_policy

    if target_spec is None:
        # Works everywhere
        target_spec = SpikeCountMSETarget(key="spikes_data")

    accelerator = Accelerator()
    device = accelerator.device

    # W&B
    if accelerator.is_main_process and wandb_enabled:
        wandb.init(
            project=wandb_project,
            name=wandb_name,
            config={
                "model": "RegionTransformer",
                "target": getattr(target_spec, "name", str(type(target_spec))),
                "diffusion_steps": noise_scheduler_args.get("num_train_timesteps"),
                "max_epochs": max_epochs,
                "lr": lr,
                "weight_decay": weight_decay,
                "num_warmup_steps": num_warmup_steps,
                "masking_policy": getattr(masking_policy, "__name__", "masking_policy"),
                **model_args,
                **noise_scheduler_args,
            }
        )

    # Diffusion scheduler
    noise_scheduler = move_scheduler_to_device(
        DDPMScheduler(**noise_scheduler_args),
        device
    )

    # Model + diffusion wrapper
    model = NeuralTransformer(**model_args)
    diffusion = DiffusionWrapper(
        model=model,
        scheduler=noise_scheduler,
        target_spec=target_spec,
        masking_policy=masking_policy,
        reconstruct_loss_weight=reconstruct_loss_weight,
    )

    if accelerator.is_main_process and wandb_enabled:
        wandb.watch(diffusion.model, log="gradients", log_freq=200)

    optimizer = torch.optim.AdamW(
        diffusion.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=len(data_loader["train"]) * max_epochs
    )

    # Prepare with Accelerate
    diffusion, optimizer, data_loader["train"] = accelerator.prepare(
        diffusion, optimizer, data_loader["train"]
    )

    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in diffusion.model.parameters())
        print(f"Model has {total_params / 1e6:.2f} million parameters.")

    return diffusion, optimizer, lr_scheduler, accelerator, data_loader
