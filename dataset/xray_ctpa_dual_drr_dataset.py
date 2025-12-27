"""
X-ray CTPA Dual DRR Dataset
Handles loading paired X-ray (PA + Lateral DRRs) and CTPA volumes
All files (CTPA + DRRs) are in same folder with patient ID prefix
Automatically rotates upside-down DRRs to correct orientation
"""

import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Optional, List, Dict
import nibabel as nib  # For NIfTI files (.nii.gz)
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import re


class XrayCTPADualDRRDataset(Dataset):
    """
    Dataset for paired X-ray (PA + Lateral DRRs) → CTPA volumes
    
    Directory structure:
    data/
    ├── 1.3.6.1.4.1.9328.50.4.0001/
    │   ├── 1.3.6.1.4.1.9328.50.4.0001.nii.gz
    │   ├── 1.3.6.1.4.1.9328.50.4.0001_pa_drr.png
    │   └── 1.3.6.1.4.1.9328.50.4.0001_lat_drr.png
    ├── 1.3.6.1.4.1.9328.50.4.0002/
    │   ├── 1.3.6.1.4.1.9328.50.4.0002.nii.gz
    │   ├── 1.3.6.1.4.1.9328.50.4.0002_pa_drr.png
    │   └── 1.3.6.1.4.1.9328.50.4.0002_lat_drr.png
    ...
    
    Supports patch-based loading for large 3D volumes
    Automatically handles DRR rotation for upside-down images
    """
    
    def __init__(
        self,
        ctpa_dir: str,
        pa_drr_pattern: str = "*_pa_drr.png",
        lateral_drr_pattern: str = "*_lat_drr.png",
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        stride: Tuple[int, int, int] = (128, 128, 128),
        split: str = 'train',
        train_split: float = 0.8,
        max_patients: Optional[int] = None,
        normalization: str = 'min_max',
        drr_rotation_angle: int = 180,  # Rotate DRRs by this angle
        dual_drr: bool = True,
    ):
        """
        Args:
            ctpa_dir: Parent directory containing patient subdirectories
            pa_drr_pattern: Glob pattern for PA DRR files (default matches *_pa_drr.png)
            lateral_drr_pattern: Glob pattern for Lateral DRR files (default matches *_lat_drr.png)
            patch_size: Size of patches to extract from 3D volumes
            stride: Stride for patch extraction
            split: 'train' or 'val'
            train_split: Train/val split ratio
            max_patients: Limit number of patients (for testing)
            normalization: 'min_max' or 'z_score'
            drr_rotation_angle: Angle to rotate upside-down DRRs (default 180 degrees)
            dual_drr: Use both PA and Lateral DRRs (True) or single DRR (False)
        """
        self.ctpa_dir = ctpa_dir
        self.pa_drr_pattern = pa_drr_pattern
        self.lateral_drr_pattern = lateral_drr_pattern
        self.patch_size = patch_size
        self.stride = stride
        self.split = split
        self.train_split = train_split
        self.max_patients = max_patients
        self.normalization = normalization
        self.drr_rotation_angle = drr_rotation_angle
        self.dual_drr = dual_drr
        
        # Find all patient data (CTPA + DRR triplets)
        self.patients = self._find_patients()
        
        if len(self.patients) == 0:
            raise ValueError(
                f"No complete patient datasets found in {ctpa_dir}\n"
                f"Expected structure:\n"
                f"  {ctpa_dir}/\n"
                f"  ├── patient_id/\n"
                f"  │   ├── patient_id.nii.gz\n"
                f"  │   ├── patient_id_pa_drr.png\n"
                f"  │   └── patient_id_lat_drr.png\n"
                f"  ├── patient_id/\n"
                f"  ...\n"
            )
        
        # Split into train/val
        self._split_data()
        
        # Generate patch indices
        self.patch_indices = self._generate_patch_indices()
        
        print(f"✓ {self.split.upper()} set: {len(self)} patches from {len(self.patient_list)} patients")

    def _find_patients(self) -> Dict[str, Dict]:
        """
        Find all patient data by scanning subdirectories
        Each subdirectory should contain: patient_id.nii.gz, patient_id_pa_drr.png, patient_id_lat_drr.png
        """
        patients = {}
        
        print(f"Scanning {self.ctpa_dir} for patient data...")
        
        # Get all subdirectories
        subdirs = [d for d in os.listdir(self.ctpa_dir) 
                   if os.path.isdir(os.path.join(self.ctpa_dir, d))]
        
        print(f"Found {len(subdirs)} subdirectories")
        
        for subdir in subdirs:
            subdir_path = os.path.join(self.ctpa_dir, subdir)
            
            # Get patient ID from subdirectory name
            patient_id = subdir
            
            # Find CTPA file (support .nii, .nii.gz, .npy)
            ctpa_files = (
                glob.glob(os.path.join(subdir_path, f"{patient_id}*.nii.gz")) +
                glob.glob(os.path.join(subdir_path, f"{patient_id}*.nii")) +
                glob.glob(os.path.join(subdir_path, f"{patient_id}*.npy"))
            )
            
            # Remove files with _pa_drr or _lat_drr or _lateral_drr in name
            ctpa_files = [f for f in ctpa_files 
                         if not any(x in os.path.basename(f).lower() 
                                   for x in ['_pa_drr', '_lat_drr', '_lateral_drr'])]
            
            # Remove excluded files like swapped_bmd
            ctpa_files = [f for f in ctpa_files if self._is_valid_ctpa(f)]
            
            # Find DRR files
            pa_drr_files = glob.glob(os.path.join(subdir_path, f"{patient_id}*_pa_drr.png"))
            lateral_drr_files = (
                glob.glob(os.path.join(subdir_path, f"{patient_id}*_lat_drr.png")) +
                glob.glob(os.path.join(subdir_path, f"{patient_id}*_lateral_drr.png"))
            )
            
            # Only include patients with complete triplet (CTPA + PA DRR + Lateral DRR)
            if ctpa_files and pa_drr_files and lateral_drr_files:
                patients[patient_id] = {
                    'ctpa': ctpa_files[0],
                    'pa_drr': pa_drr_files[0],
                    'lateral_drr': lateral_drr_files[0],
                    'patient_id': patient_id
                }
                print(f"  ✓ {patient_id}: CTPA + PA + Lateral DRR")
            else:
                missing = []
                if not ctpa_files:
                    missing.append("CTPA")
                if not pa_drr_files:
                    missing.append("PA DRR")
                if not lateral_drr_files:
                    missing.append("Lateral DRR")
                print(f"  ✗ {patient_id}: Missing {', '.join(missing)}")
        
        if self.max_patients:
            # Sort for reproducibility
            sorted_patients = sorted(list(patients.items()), key=lambda x: x[0])
            patients = dict(sorted_patients[:self.max_patients])
        
        print(f"Found {len(patients)} complete patient datasets (CTPA + PA DRR + Lateral DRR)\n")
        
        return patients

    def _split_data(self):
        """Split patients into train/val sets, excluding last 135 files"""
        patient_ids = list(self.patients.keys())
        patient_ids.sort()  # For reproducibility
        
        # Skip last 135 files
        patient_ids = patient_ids[:-135]
        
        print(f"Loaded {len(patient_ids)} patients (skipped last 135)")
        
        split_idx = int(len(patient_ids) * self.train_split)
        
        if self.split == 'train':
            self.patient_list = patient_ids[:split_idx]
        else:  # val
            self.patient_list = patient_ids[split_idx:]
        
        print(f"Split: {len(self.patient_list)} patients for {self.split}")

    def _generate_patch_indices(self) -> List[Tuple[str, Tuple[int, int, int]]]:
        """Generate patch indices for all patients"""
        patch_indices = []
        
        for patient_id in self.patient_list:
            # Load CTPA to get shape
            ctpa_file = self.patients[patient_id]['ctpa']
            ctpa_vol = self._load_ctpa(ctpa_file)
            
            # Skip if file is corrupted
            if ctpa_vol is None:
                continue
            
            ctpa_shape = ctpa_vol.shape
            
            print(f"  {patient_id}: CTPA shape {ctpa_shape}", end="")
            
            # Generate patch indices (smaller patches = faster loading)
            num_patches = 0
            for z in range(0, ctpa_shape[0] - self.patch_size[0] + 1, self.stride[0]):
                for y in range(0, ctpa_shape[1] - self.patch_size[1] + 1, self.stride[1]):
                    for x in range(0, ctpa_shape[2] - self.patch_size[2] + 1, self.stride[2]):
                        patch_indices.append((patient_id, (z, y, x)))
                        num_patches += 1
            
            print(f" → {num_patches} patches")
        
        return patch_indices

    def _load_ctpa(self, ctpa_file: str) -> np.ndarray:
        """Load 3D CTPA volume with error handling for corrupted/empty files"""
        try:
            # Check file size first (skip empty files)
            file_size = os.path.getsize(ctpa_file)
            if file_size < 1000:  # Skip files smaller than 1KB
                print(f"⚠ Warning: File too small (empty): {ctpa_file}")
                return None
            
            if ctpa_file.endswith('.nii.gz') or ctpa_file.endswith('.nii'):
                # Load NIfTI format
                nii = nib.load(ctpa_file)
                ctpa = np.array(nii.dataobj, dtype=np.float32)
                
                # Check if volume is valid
                if ctpa.size == 0 or ctpa.shape[0] < 10:  # Skip if too small
                    print(f"⚠ Warning: Invalid volume shape {ctpa.shape}: {ctpa_file}")
                    return None
                    
            elif ctpa_file.endswith('.npy'):
                ctpa = np.load(ctpa_file).astype(np.float32)
                if ctpa.size == 0 or ctpa.ndim < 3:
                    print(f"⚠ Warning: Invalid NPY file: {ctpa_file}")
                    return None
            else:
                return None
            
            return ctpa
        except (EOFError, FileNotFoundError, IOError, ValueError) as e:
            print(f"⚠ Warning: Could not load CTPA {os.path.basename(ctpa_file)}: {type(e).__name__}")
            return None
        except Exception as e:
            print(f"⚠ Warning: Error loading {os.path.basename(ctpa_file)}: {str(e)[:50]}")
            return None
    
    def _is_valid_ctpa(self, filename: str) -> bool:
        """Check if file should be used as CTPA (exclude specific files)"""
        # Exclude swapped_bmd and similar files
        excluded = ['swapped_bmd', 'bmd', 'metadata', 'info']
        basename = os.path.basename(filename).lower()
        return not any(exc in basename for exc in excluded)

    def _load_drr(self, drr_file: str) -> np.ndarray:
        """
        Load 2D DRR image and rotate if upside-down
        
        Args:
            drr_file: Path to DRR PNG file
        
        Returns:
            Normalized DRR array [H, W]
        """
        # Load image
        img = Image.open(drr_file).convert('L')  # Convert to grayscale
        drr = np.array(img, dtype=np.float32)
        
        # Rotate DRR if specified (default 180 degrees for upside-down images)
        if self.drr_rotation_angle != 0:
            # Rotate using numpy
            # 180 degrees: rotate 90 degrees twice
            k = int(self.drr_rotation_angle / 90) % 4
            if k > 0:
                drr = np.rot90(drr, k=k)
                # After rotation, array axes might be swapped - fix if needed
                if drr.shape != np.array(img).shape:
                    drr = drr.T  # Transpose back if shape changed
        
        return drr

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize image data"""
        if self.normalization == 'min_max':
            x_min = x.min()
            x_max = x.max()
            if x_max > x_min:
                x = (x - x_min) / (x_max - x_min)
            else:
                x = np.zeros_like(x)
        elif self.normalization == 'z_score':
            x_mean = x.mean()
            x_std = x.std()
            if x_std > 0:
                x = (x - x_mean) / x_std
        
        # Clip to [0, 1]
        x = np.clip(x, 0, 1)
        
        return x

    def _extract_patch(self, volume: np.ndarray, patch_start: Tuple[int, int, int]) -> np.ndarray:
        """Extract patch from 3D volume"""
        z, y, x = patch_start
        patch = volume[
            z:z + self.patch_size[0],
            y:y + self.patch_size[1],
            x:x + self.patch_size[2]
        ]
        return patch

    def __len__(self) -> int:
        """Total number of patches"""
        return len(self.patch_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get patch at index
        
        Returns:
            Dict with keys:
                - 'ctpa': 3D CTPA patch [1, D, H, W]
                - 'pa_drr': 2D PA DRR [1, H, W]
                - 'lateral_drr': 2D Lateral DRR [1, H, W]
                - 'patient_id': Patient ID string
                - 'patch_idx': Patch index integer
        """
        patient_id, patch_start = self.patch_indices[idx]
        
        # Get file paths
        ctpa_file = self.patients[patient_id]['ctpa']
        pa_drr_file = self.patients[patient_id]['pa_drr']
        lateral_drr_file = self.patients[patient_id]['lateral_drr']
        
        try:
            # Load CTPA volume
            ctpa_vol = self._load_ctpa(ctpa_file)
            
            # Skip if corrupted or invalid
            if ctpa_vol is None:
                return None
            
            # Load and rotate DRRs
            pa_drr = self._load_drr(pa_drr_file)
            lateral_drr = self._load_drr(lateral_drr_file)
            
            # Skip if any DRR is invalid
            if pa_drr is None or lateral_drr is None:
                return None
            
            # Extract patch from CTPA volume
            ctpa_patch = self._extract_patch(ctpa_vol, patch_start)
            
            # Normalize all
            ctpa_patch = self._normalize(ctpa_patch)
            pa_drr = self._normalize(pa_drr)
            lateral_drr = self._normalize(lateral_drr)
            
            # Convert to tensors and add channel dimension
            ctpa_tensor = torch.from_numpy(ctpa_patch[np.newaxis, ...]).float()  # [1, D, H, W]
            pa_drr_tensor = torch.from_numpy(pa_drr[np.newaxis, ...]).float()  # [1, H, W]
            lateral_drr_tensor = torch.from_numpy(lateral_drr[np.newaxis, ...]).float()  # [1, H, W]
            
            # Resize DRRs to match CTPA spatial dimensions if needed
            target_h, target_w = ctpa_tensor.shape[2:4]
            if pa_drr_tensor.shape[1:] != (target_h, target_w):
                # Resize DRRs using bilinear interpolation
                pa_drr_tensor = torch.nn.functional.interpolate(
                    pa_drr_tensor.unsqueeze(0), 
                    size=(target_h, target_w), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
                lateral_drr_tensor = torch.nn.functional.interpolate(
                    lateral_drr_tensor.unsqueeze(0), 
                    size=(target_h, target_w), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
            
            return {
                'ctpa': ctpa_tensor,
                'pa_drr': pa_drr_tensor,
                'lateral_drr': lateral_drr_tensor,
                'patient_id': patient_id,
                'patch_idx': idx
            }
        
        except Exception as e:
            print(f"Error loading patch {idx} for patient {patient_id}: {e}")
            raise