"""
Utility functions to load VQ-GAN checkpoints for DDPM conditioning
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class VQGANLoader:
    """Loads and manages VQ-GAN encoder/decoder checkpoints"""
    
    @staticmethod
    def load_vqgan_checkpoint(
        ckpt_path: str,
        device: torch.device = torch.device('cpu')
    ) -> Tuple[nn.Module, nn.Module]:
        """
        Load VQ-GAN encoder and decoder from checkpoint
        
        Args:
            ckpt_path: Path to VQ-GAN checkpoint file
            device: Device to load models on (cpu, cuda:0, etc.)
        
        Returns:
            Tuple of (encoder, decoder) modules
        """
        if not torch.cuda.is_available():
            device = torch.device('cpu')
        
        print(f"Loading VQ-GAN checkpoint from: {ckpt_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(ckpt_path, map_location=device)
            print(f"✓ Checkpoint loaded successfully")
            
            # Extract state dicts based on checkpoint structure
            # Common checkpoint structures:
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Split state dict into encoder and decoder components
            encoder_state = {}
            decoder_state = {}
            
            for key, value in state_dict.items():
                # Handle different naming conventions
                if 'encoder' in key.lower():
                    new_key = key.replace('encoder.', '').replace('_encoder.', '')
                    encoder_state[new_key] = value
                elif 'decoder' in key.lower():
                    new_key = key.replace('decoder.', '').replace('_decoder.', '')
                    decoder_state[new_key] = value
                else:
                    # If no clear prefix, try to infer from architecture
                    encoder_state[key] = value
                    decoder_state[key] = value
            
            print(f"  - Encoder parameters: {len(encoder_state)}")
            print(f"  - Decoder parameters: {len(decoder_state)}")
            
            return encoder_state, decoder_state
            
        except Exception as e:
            print(f"✗ Error loading checkpoint: {e}")
            raise


class FrozenVQGANEncoder(nn.Module):
    """Frozen VQ-GAN encoder for conditioning"""
    
    def __init__(self, encoder_state_dict: dict, device: torch.device = torch.device('cpu')):
        super().__init__()
        self.device = device
        
        # Build encoder architecture (adjust based on your VQ-GAN)
        self.encoder = self._build_encoder()
        
        # Load pretrained weights
        try:
            self.encoder.load_state_dict(encoder_state_dict)
            print("✓ Encoder weights loaded")
        except RuntimeError as e:
            print(f"⚠ Warning: Some weights not loaded - {e}")
        
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.encoder.eval()
        self.encoder.to(device)
    
    def _build_encoder(self) -> nn.Module:
        """
        Build encoder architecture.
        Adjust this based on your actual VQ-GAN encoder architecture
        """
        # Example: Simple 3D encoder
        class SimpleEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv3d(1, 64, kernel_size=3, stride=2, padding=1)
                self.conv2 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
                self.conv3 = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1)
            
            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = torch.relu(self.conv3(x))
                return x
        
        return SimpleEncoder()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space (frozen)"""
        with torch.no_grad():
            return self.encoder(x)


class FrozenVQGANDecoder(nn.Module):
    """Frozen VQ-GAN decoder for reconstruction"""
    
    def __init__(self, decoder_state_dict: dict, device: torch.device = torch.device('cpu')):
        super().__init__()
        self.device = device
        
        # Build decoder architecture
        self.decoder = self._build_decoder()
        
        # Load pretrained weights
        try:
            self.decoder.load_state_dict(decoder_state_dict)
            print("✓ Decoder weights loaded")
        except RuntimeError as e:
            print(f"⚠ Warning: Some weights not loaded - {e}")
        
        # Freeze decoder
        for param in self.decoder.parameters():
            param.requires_grad = False
        
        self.decoder.eval()
        self.decoder.to(device)
    
    def _build_decoder(self) -> nn.Module:
        """
        Build decoder architecture.
        Adjust this based on your actual VQ-GAN decoder architecture
        """
        # Example: Simple 3D decoder
        class SimpleDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.deconv1 = nn.ConvTranspose3d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
                self.deconv2 = nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
                self.deconv3 = nn.ConvTranspose3d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
            
            def forward(self, x):
                x = torch.relu(self.deconv1(x))
                x = torch.relu(self.deconv2(x))
                x = torch.sigmoid(self.deconv3(x))
                return x
        
        return SimpleDecoder()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode latent to output space (frozen)"""
        with torch.no_grad():
            return self.decoder(x)


# Usage in GaussianDiffusion
def integrate_vqgan_in_diffusion(vqgan_ckpt_path: str, device: torch.device):
    """
    Example integration in your GaussianDiffusion class
    
    Usage in diffusion.py:
    ```python
    from ddpm.vqgan_loader import integrate_vqgan_in_diffusion
    
    class GaussianDiffusion(pl.LightningModule):
        def __init__(self, model, vqgan_ckpt, ...):
            super().__init__()
            self.model = model
            
            # Load and freeze VQ-GAN
            encoder_state, decoder_state = integrate_vqgan_in_diffusion(
                vqgan_ckpt, 
                device=self.device
            )
            self.vqgan_encoder = FrozenVQGANEncoder(encoder_state, device)
            self.vqgan_decoder = FrozenVQGANDecoder(decoder_state, device)
        
        def forward(self, x, c):
            # Encode image to latent space
            x_latent = self.vqgan_encoder(x)
            c_latent = self.vqgan_encoder(c)
            
            # Pass through diffusion model
            noise_pred = self.model(x_latent, c_latent)
            
            return noise_pred
    ```
    """
    encoder_state, decoder_state = VQGANLoader.load_vqgan_checkpoint(
        vqgan_ckpt_path,
        device=device
    )
    
    return encoder_state, decoder_state