"""
DDPM Training Script for Dual DRR Conditioning
Trains diffusion model in VQ-GAN latent space for X-ray (PA + Lateral) → CTPA generation
Handles DRR rotation and dual-view conditioning
Supports distributed training with PyTorch Lightning DDP
"""

import os
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
import hydra
from omegaconf import DictConfig, OmegaConf

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

from diffusion import GaussianDiffusion
from unet import Unet3D
from dataset.xray_ctpa_dual_drr_dataset import XrayCTPADualDRRDataset
from torch.utils.data import DataLoader


def get_dataloaders(cfg):
    """
    Create dataloaders for X-ray (PA + Lateral) → CTPA paired data.
    DRRs are automatically rotated to correct orientation.
    Each GPU gets 1 sample per batch. Total effective batch = 4 pairs across 4 GPUs.
    """
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    if local_rank == 0:
        print(f"Loading paired X-ray (PA + Lateral) → CTPA dataset...")
        print(f"CTPA dir: {cfg.dataset.ctpa_dir}")
        print(f"PA DRR pattern: {cfg.dataset.get('pa_drr_pattern', '*_pa_drr.png')}")
        print(f"Lateral DRR pattern: {cfg.dataset.get('lateral_drr_pattern', '*_lat_drr.png')}")
        print(f"DRR rotation: {cfg.dataset.get('drr_rotation_angle', 180)} degrees")

    # Create datasets with DRR rotation enabled
    train_dataset = XrayCTPADualDRRDataset(
        ctpa_dir=cfg.dataset.ctpa_dir,
        pa_drr_pattern=cfg.dataset.get('pa_drr_pattern', '*_pa_drr.png'),
        lateral_drr_pattern=cfg.dataset.get('lateral_drr_pattern', '*_lat_drr.png'),
        patch_size=tuple(cfg.dataset.get('patch_size', [128, 128, 128])),
        stride=tuple(cfg.dataset.get('stride', [128, 128, 128])),
        split='train',
        train_split=cfg.dataset.get('train_split', 0.8),
        max_patients=cfg.dataset.get('max_patients', None),
        normalization=cfg.dataset.get('normalization', 'min_max'),
        drr_rotation_angle=cfg.dataset.get('drr_rotation_angle', 180),  # Rotate DRRs
        dual_drr=True
    )

    val_dataset = XrayCTPADualDRRDataset(
        ctpa_dir=cfg.dataset.ctpa_dir,
        pa_drr_pattern=cfg.dataset.get('pa_drr_pattern', '*_pa_drr.png'),
        lateral_drr_pattern=cfg.dataset.get('lateral_drr_pattern', '*_lat_drr.png'),
        patch_size=tuple(cfg.dataset.get('patch_size', [128, 128, 128])),
        stride=tuple(cfg.dataset.get('stride', [128, 128, 128])),
        split='val',
        train_split=cfg.dataset.get('train_split', 0.8),
        max_patients=cfg.dataset.get('max_patients', None),
        normalization=cfg.dataset.get('normalization', 'min_max'),
        drr_rotation_angle=cfg.dataset.get('drr_rotation_angle', 180),  # Rotate DRRs
        dual_drr=True
    )

    # Create dataloaders with optimized worker count
    num_workers_per_gpu = cfg.dataset.get('num_workers', 20)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.model.batch_size,
        shuffle=True,
        num_workers=num_workers_per_gpu,
        pin_memory=True,
        drop_last=True,
        persistent_workers=False,  # Disable persistent workers
        multiprocessing_context='spawn'
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.model.batch_size,
        shuffle=False,
        num_workers=num_workers_per_gpu // 2,
        pin_memory=True,
        persistent_workers=False,  # Disable persistent workers
        multiprocessing_context='spawn'
    )

    if local_rank == 0:
        print(f"✓ Train dataset: {len(train_dataset)} patches")
        print(f"✓ Val dataset: {len(val_dataset)} patches")

    return train_loader, val_loader


def setup_callbacks(cfg):
    """Setup training callbacks for checkpointing and monitoring"""
    return [
        # Best model checkpoint based on validation loss
        ModelCheckpoint(
            dirpath=cfg.model.results_folder,
            filename='best-epoch{epoch:02d}-val_loss{val/loss:.4f}',
            monitor='val/loss',
            mode='min',
            save_top_k=3,
            save_last=False,
            every_n_epochs=1,
            verbose=True
        ),
        # Best model checkpoint based on training loss
        ModelCheckpoint(
            dirpath=cfg.model.results_folder,
            filename='best-train-epoch{epoch:02d}-train_loss{train/loss_epoch:.4f}',
            monitor='train/loss_epoch',
            mode='min',
            save_top_k=2,
            save_last=False,
            every_n_epochs=1,
            verbose=True
        ),
        # Periodic checkpoint every N epochs
        ModelCheckpoint(
            dirpath=cfg.model.results_folder,
            filename='periodic-epoch{epoch:02d}',
            save_top_k=-1,  # Keep all
            every_n_epochs=cfg.model.get('checkpoint_interval', 5),
            save_last=False,
            verbose=True
        ),
        # Always keep latest checkpoint
        ModelCheckpoint(
            dirpath=cfg.model.results_folder,
            filename='last',
            save_top_k=1,
            save_last=True,
            every_n_epochs=1,
            verbose=False
        ),
        # Monitor learning rate
        LearningRateMonitor(logging_interval='step')
    ]


def main():
    """Main training function for 4-GPU DDPM training with dual DRRs."""
    
    # Get local rank from environment (set by torchrun)
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    # Filter out --local-rank from sys.argv before Hydra sees it
    sys.argv = [arg for arg in sys.argv if not arg.startswith('--local-rank')]
    
    # Get absolute path to config directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config")
    
    @hydra.main(config_path=config_path, config_name="base_cfg_ddpm", version_base=None)
    def train(cfg: DictConfig):
        if local_rank == 0:
            print("\n" + "=" * 80)
            print("DDPM TRAINING (4-GPU DDP) - X-ray (PA + Lateral) → CTPA Generation")
            print("=" * 80)
            print(f"\nConfiguration:\n{OmegaConf.to_yaml(cfg)}")
            print("=" * 80 + "\n")

        # Set seed for reproducibility
        pl.seed_everything(cfg.model.seed)

        # Verify VQ-GAN checkpoint exists
        if not os.path.exists(cfg.model.vqgan_ckpt):
            raise FileNotFoundError(
                f"VQ-GAN checkpoint not found: {cfg.model.vqgan_ckpt}\n"
                "Please copy your VQ-GAN checkpoint to /workspace/checkpoints/"
            )

        if local_rank == 0:
            print(f"✓ VQ-GAN checkpoint found: {cfg.model.vqgan_ckpt}")

        # Create dataloaders
        if local_rank == 0:
            print("\nPreparing dataloaders...")
        train_loader, val_loader = get_dataloaders(cfg)

        # Create UNet model for dual DRR conditioning
        if local_rank == 0:
            print(f"Creating UNet3D model...")
            print(f"  - Input size: {cfg.model.diffusion_img_size}")
            print(f"  - Conditioning channels: {cfg.model.cond_dim} (PA + Lateral)")
            print(f"  - Model channels: {cfg.model.diffusion_num_channels}")

        model = Unet3D(
            dim=cfg.model.dim,  # Base channel dimension (64)
            cond_dim=cfg.model.cond_dim,  # 2 for dual DRRs
            dim_mults=cfg.model.dim_mults,
            channels=cfg.model.diffusion_num_channels,
            resnet_groups=8,
            classifier_free_guidance=cfg.model.classifier_free_guidance,
            medclip=cfg.model.medclip
        )

        # Create Gaussian Diffusion wrapper
        if local_rank == 0:
            print(f"Creating Gaussian Diffusion with VQ-GAN integration...")

        diffusion = GaussianDiffusion(
            model,
            vqgan_ckpt=cfg.model.vqgan_ckpt,
            vae_ckpt=cfg.model.get('vae_ckpt', None),
            image_size=cfg.model.diffusion_img_size,
            num_frames=cfg.model.diffusion_depth_size,
            channels=cfg.model.diffusion_num_channels,
            timesteps=cfg.model.timesteps,
            img_cond=True,
            loss_type=cfg.model.loss_type,
            l1_weight=cfg.model.l1_weight,
            perceptual_weight=cfg.model.perceptual_weight,
            discriminator_weight=cfg.model.discriminator_weight,
            classification_weight=cfg.model.classification_weight,
            classifier_free_guidance=cfg.model.classifier_free_guidance,
            medclip=cfg.model.medclip,
            name_dataset=cfg.model.name_dataset,
            dataset_min_value=cfg.model.dataset_min_value,
            dataset_max_value=cfg.model.dataset_max_value,
            learning_rate=cfg.model.learning_rate,
            weight_decay=cfg.model.get('weight_decay', 0.0),
        )

        # Setup callbacks
        callbacks = setup_callbacks(cfg)

        # DDP Strategy with proper configuration
        ddp_strategy = DDPStrategy(
            find_unused_parameters=False,  # No unused parameters, improve performance
            gradient_as_bucket_view=True,
            static_graph=False,
        )

        # Enable gradient checkpointing if specified (reduces memory)
        if cfg.model.get('gradient_checkpointing', True):
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
            if local_rank == 0:
                print("✓ Gradient checkpointing enabled")

        # Create trainer
        if local_rank == 0:
            print(f"\nCreating trainer (4 GPUs, DDP)...")

        trainer = pl.Trainer(
            accelerator='gpu',
            devices=4,
            strategy=ddp_strategy,
            max_epochs=cfg.model.get('max_epochs', 30),
            precision=16 if cfg.model.amp else 32,
            accumulate_grad_batches=cfg.model.gradient_accumulate_every,
            callbacks=callbacks,
            log_every_n_steps=cfg.model.get('log_interval', 50),
            check_val_every_n_epoch=1 if not cfg.model.get('skip_validation', False) else 0,
            enable_progress_bar=True,
            enable_model_summary=local_rank == 0,
            gradient_clip_val=cfg.model.max_grad_norm,
            sync_batchnorm=cfg.model.get('sync_batchnorm', True),
            num_sanity_val_steps=0,  # Skip validation sanity check
            limit_val_batches=0 if cfg.model.get('skip_validation', False) else 0.1,  # Skip if flag set
        )

        # Start training
        resume_ckpt = cfg.model.get('resume_ckpt', None)
        
        if local_rank == 0:
            print("\n" + "=" * 80)
            print("STARTING TRAINING")
            print("=" * 80)
            print(f"Max epochs: {cfg.model.get('max_epochs', 30)}")
            print(f"Batch size per GPU: {cfg.model.batch_size}")
            print(f"Effective batch (with accumulation): {cfg.model.batch_size * 4 * cfg.model.gradient_accumulate_every}")
            print(f"Train samples: {len(train_loader.dataset)}")
            print(f"Val samples: {len(val_loader.dataset)}")
            if resume_ckpt:
                print(f"Resuming from: {resume_ckpt}")
            print("=" * 80 + "\n")

        try:
            trainer.fit(
                diffusion,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                ckpt_path=resume_ckpt
            )
        except KeyboardInterrupt:
            if local_rank == 0:
                print("\n✗ Training interrupted by user")
        finally:
            # Cleanup dataloader workers
            if hasattr(train_loader, '_iterator') and train_loader._iterator is not None:
                train_loader._iterator._shutdown_workers()
            if hasattr(val_loader, '_iterator') and val_loader._iterator is not None:
                val_loader._iterator._shutdown_workers()

        if local_rank == 0:
            print("\n" + "=" * 80)
            print("TRAINING COMPLETE!")
            print("=" * 80)
            print(f"Results saved to: {cfg.model.results_folder}")
            print("=" * 80 + "\n")

    # Run training
    train()


if __name__ == "__main__":
    main()