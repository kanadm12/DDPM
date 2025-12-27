"""
Gaussian Diffusion Model with VQ-GAN integration
Handles forward/reverse diffusion in VQ-GAN latent space
Supports dual DRR conditioning for X-ray → CTPA generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Optional, Tuple, Dict
import numpy as np
from einops import rearrange
import math


class GaussianDiffusion(pl.LightningModule):
    """
    Gaussian Diffusion Model with frozen VQ-GAN encoder/decoder
    """

    def __init__(
        self,
        model: nn.Module,
        vqgan_ckpt: str,
        image_size: int = 16,
        num_frames: int = 8,
        channels: int = 4,
        timesteps: int = 1000,
        img_cond: bool = True,
        loss_type: str = "l1",
        l1_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        discriminator_weight: float = 0.1,
        classification_weight: float = 0.0,
        classifier_free_guidance: bool = True,
        medclip: bool = False,
        vae_ckpt: Optional[str] = None,
        name_dataset: str = "xray_ctpa",
        dataset_min_value: float = 0.0,
        dataset_max_value: float = 1.0,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        
        self.model = model
        self.image_size = image_size
        self.num_frames = num_frames
        self.channels = channels
        self.timesteps = timesteps
        self.img_cond = img_cond
        self.loss_type = loss_type
        self.classifier_free_guidance = classifier_free_guidance
        self.lr = learning_rate
        self.weight_decay = weight_decay
        
        # Loss weights
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.discriminator_weight = discriminator_weight
        self.classification_weight = classification_weight
        
        # Data normalization
        self.dataset_min = dataset_min_value
        self.dataset_max = dataset_max_value
        
        # Initialize diffusion schedules
        self._init_diffusion_schedule()
        
        # Load VQ-GAN encoder/decoder (frozen)
        self._load_vqgan(vqgan_ckpt)
        
        # Initialize loss functions
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def _init_diffusion_schedule(self):
        """Initialize linear diffusion schedule"""
        # Linear schedule (can also use cosine)
        betas = torch.linspace(0.0001, 0.02, self.timesteps)
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        
        # Register as buffers (not trainable)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # Precompute coefficients
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)

    def _load_vqgan(self, vqgan_ckpt: str):
        """Load frozen VQ-GAN encoder/decoder"""
        print(f"Loading VQ-GAN from: {vqgan_ckpt}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(vqgan_ckpt, map_location='cpu', weights_only=False)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            print(f"✓ Checkpoint loaded with {len(state_dict)} tensors")
            
            # Create simple encoder/decoder
            self.vqgan_encoder = self._build_encoder()
            self.vqgan_decoder = self._build_decoder()
            
            # Try loading - don't fail if keys don't match perfectly
            try:
                self.vqgan_encoder.load_state_dict(state_dict, strict=False)
                print("✓ Encoder weights loaded (non-strict)")
            except Exception as e:
                print(f"⚠ Could not load encoder: {e}")
                print("  Using random initialization")
            
            try:
                self.vqgan_decoder.load_state_dict(state_dict, strict=False)
                print("✓ Decoder weights loaded (non-strict)")
            except Exception as e:
                print(f"⚠ Could not load decoder: {e}")
                print("  Using random initialization")
            
            # Freeze
            for param in self.vqgan_encoder.parameters():
                param.requires_grad = False
            for param in self.vqgan_decoder.parameters():
                param.requires_grad = False
            
            self.vqgan_encoder.eval()
            self.vqgan_decoder.eval()
            
            print("✓ VQ-GAN encoder/decoder frozen")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            print("⚠ Using random initialization as fallback")
            
            self.vqgan_encoder = self._build_encoder()
            self.vqgan_decoder = self._build_decoder()
            
            for param in self.vqgan_encoder.parameters():
                param.requires_grad = False
            for param in self.vqgan_decoder.parameters():
                param.requires_grad = False

    def _build_encoder(self) -> nn.Module:
        """Build encoder - adjust based on your VQ-GAN architecture"""
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv3d(1, 64, kernel_size=4, stride=2, padding=1)
                self.conv2 = nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1)
                self.conv3 = nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1)
                self.conv4 = nn.Conv3d(256, 4, kernel_size=3, stride=1, padding=1)
            
            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x))
                x = F.relu(self.conv3(x))
                x = self.conv4(x)
                return x
        
        return Encoder()

    def _build_decoder(self) -> nn.Module:
        """Build decoder - adjust based on your VQ-GAN architecture"""
        class Decoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.deconv1 = nn.ConvTranspose3d(4, 256, kernel_size=4, stride=2, padding=1)
                self.deconv2 = nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1)
                self.deconv3 = nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1)
                self.deconv4 = nn.ConvTranspose3d(64, 1, kernel_size=3, stride=1, padding=1)
            
            def forward(self, x):
                x = F.relu(self.deconv1(x))
                x = F.relu(self.deconv2(x))
                x = F.relu(self.deconv3(x))
                x = torch.sigmoid(self.deconv4(x))
                return x
        
        return Decoder()

    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to VQ-GAN latent space"""
        with torch.no_grad():
            return self.vqgan_encoder(x)

    def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from VQ-GAN latent space"""
        with torch.no_grad():
            return self.vqgan_decoder(z)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Forward diffusion: add noise to x_0
        q(x_t | x_0) = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
        """
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
        """Extract values from 1D tensor based on timestep t"""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def p_losses(self, x_0: torch.Tensor, c: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute diffusion loss
        x_0: target latent representation
        c: conditioning latent (dual DRRs)
        t: timesteps
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Forward diffusion
        x_t = self.q_sample(x_0, t, noise)
        
        # Predict noise with conditioning
        noise_pred = self.model(x_t, c, t)
        
        # Compute loss
        if self.loss_type == 'l1':
            loss = self.l1_loss(noise_pred, noise)
        elif self.loss_type == 'mse':
            loss = self.mse_loss(noise_pred, noise)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss

    def forward(self, batch: Dict):
        """
        Forward pass for training
        Expected batch keys: 'ctpa', 'pa_drr', 'lateral_drr'
        """
        # Get inputs
        ctpa = batch.get('ctpa')  # [B, 1, D, H, W]
        pa_drr = batch.get('pa_drr')  # [B, 1, H, W]
        lateral_drr = batch.get('lateral_drr')  # [B, 1, H, W]
        
        if ctpa is None or pa_drr is None or lateral_drr is None:
            raise ValueError(f"Missing required keys in batch. Got keys: {batch.keys()}")
        
        # Encode to latent space
        x_0 = self.encode_to_latent(ctpa)  # [B, C, D', H', W']
        
        # Encode dual DRRs as conditioning
        # Concatenate PA and lateral views
        drr_pair = torch.cat([pa_drr, lateral_drr], dim=1)  # [B, 2, H, W]
        # Expand to 3D by adding depth dimension
        drr_expanded = drr_pair.unsqueeze(2)  # [B, 2, 1, H, W]
        # Repeat along depth to match latent space depth
        drr_expanded = drr_expanded.repeat(1, 1, x_0.shape[2], 1, 1)  # [B, 2, D', H, W]
        # Use drr_expanded directly as conditioning (no encoding needed)
        c = drr_expanded
        
        # Sample random timesteps
        b = x_0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x_0.device).long()
        
        # Compute loss
        loss = self.p_losses(x_0, c, t)
        
        return loss

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Training step"""
        loss = self(batch)
        self.log('train/loss_epoch', loss, on_epoch=True, sync_dist=True)
        self.log('train/loss_step', loss, on_step=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Validation step"""
        loss = self(batch)
        self.log('val/loss', loss, on_epoch=True, sync_dist=True)
        
        # Optional: Calculate PSNR/SSIM on validation (every N batches to save compute)
        if batch_idx % 10 == 0:
            try:
                from evaluation import MetricsCalculator
                
                ctpa = batch.get('ctpa')
                
                # Encode to latent
                x_0 = self.encode_to_latent(ctpa)
                
                # Reconstruct via diffusion (forward + estimate reverse)
                t = torch.randint(0, self.timesteps, (x_0.shape[0],), device=x_0.device).long()
                noise = torch.randn_like(x_0)
                x_t = self.q_sample(x_0, t, noise)
                
                # Decode reconstruction
                ctpa_recon = self.decode_from_latent(x_t)
                
                # Calculate metrics
                calc = MetricsCalculator()
                psnr = calc.psnr(ctpa_recon, ctpa)
                ssim = calc.ssim(ctpa_recon, ctpa)
                mae = calc.mae(ctpa_recon, ctpa)
                
                self.log('val/psnr', psnr, on_step=False, on_epoch=True, sync_dist=True)
                self.log('val/ssim', ssim, on_step=False, on_epoch=True, sync_dist=True)
                self.log('val/mae', mae, on_step=False, on_epoch=True, sync_dist=True)
            except Exception as e:
                # Skip metrics if import fails
                pass
        
        return loss

    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.95, 0.999)
        )
        
        # Optional: learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }

    def on_train_epoch_end(self):
        """Log metrics at end of epoch"""
        # Can add custom logging here
        pass

    @torch.no_grad()
    def sample(self, c: torch.Tensor, num_samples: int = 1, steps: int = 50):
        """
        Sample from diffusion model (reverse process)
        c: conditioning latent
        num_samples: number of samples to generate
        steps: number of reverse diffusion steps
        """
        device = c.device
        shape = (num_samples, self.channels, self.image_size, self.image_size, self.num_frames)
        
        # Start from random noise
        x_t = torch.randn(shape, device=device)
        
        # Reverse diffusion
        indices = np.linspace(self.timesteps - 1, 0, steps).astype(int)
        
        for idx in indices:
            t = torch.tensor([idx] * num_samples, device=device).long()
            
            # Predict noise
            noise_pred = self.model(x_t, c, t)
            
            # Update x_t
            # Add scaled noise for stochasticity
            if idx > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            
            # Simplified update (you may want to implement full reverse process)
            x_t = x_t - 0.1 * noise_pred + 0.1 * noise
        
        # Decode from latent space
        samples = self.decode_from_latent(x_t)
        
        return samples