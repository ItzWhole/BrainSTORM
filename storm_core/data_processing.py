"""
Data Processing Module for STORM Microscopy

Handles TIFF stack loading, peak detection, and subimage extraction.
"""

import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional
from scipy.ndimage import gaussian_filter, median_filter, maximum_filter
from skimage.feature import peak_local_max
import logging

logger = logging.getLogger(__name__)

def find_tiff_files(data_path: Path, pattern: str = "*_MMStack_Default.ome.tif") -> List[Path]:
    """
    Find TIFF files in directory structure
    
    Args:
        data_path: Root directory to search
        pattern: File pattern to match
        
    Returns:
        List of TIFF file paths
    """
    tiff_files = []
    
    for folder in data_path.iterdir():
        if folder.is_dir() and "keras" not in folder.name:
            tif_path = folder / f"{folder.name}_MMStack_Default.ome.tif"
            if tif_path.exists():
                tiff_files.append(tif_path)
    
    # Sort for consistent ordering
    tiff_files.sort()
    logger.info(f"Found {len(tiff_files)} TIFF files")
    
    return tiff_files

def crop_and_sum_stack(stack: np.ndarray, start: int, end: int, csum: int) -> np.ndarray:
    """
    Crop z-stack and sum final slices for peak detection
    
    Args:
        stack: Input stack with shape (Z, H, W)
        start: First slice index (inclusive)
        end: Last slice index (inclusive) 
        csum: Number of final slices to sum
        
    Returns:
        2D summed image (H, W)
    """
    if stack.ndim != 3:
        raise ValueError("Expected stack with shape (Z, H, W)")
    
    # Crop z-range
    cropped = stack[start:end+1]
    
    if csum > cropped.shape[0]:
        raise ValueError(f"csum ({csum}) larger than cropped depth ({cropped.shape[0]})")
    
    # Sum last csum slices
    summed = np.sum(cropped[-csum:], axis=0)
    
    logger.info(f"Cropped stack from {start} to {end}, summed last {csum} slices")
    return summed

def extract_psf_cutouts(stack: np.ndarray, 
                       csum_image: np.ndarray,
                       distance: int,
                       min_distance: int = 5,
                       prominence_sigma: float = 10.0,
                       support_radius: int = 2,
                       start: int = 0,
                       end: Optional[int] = None,
                       plot: bool = True) -> Tuple[List[Tuple], List[int], np.ndarray]:
    """
    Extract PSF cutouts centered on detected peaks
    
    Args:
        stack: 3D array (Z, H, W)
        csum_image: 2D image for peak detection (H, W)
        distance: Cutout size (distance x distance)
        min_distance: Minimum separation between peaks
        prominence_sigma: Peak prominence threshold in noise sigmas
        support_radius: Radius for hot pixel rejection
        start: First z-index to include
        end: Last z-index to include (None = all)
        plot: Whether to visualize detected peaks
        
    Returns:
        cutouts: List of (cutout, z_index) tuples
        group_ids: List of group indices per cutout
        peaks: Array of peak coordinates (row, col)
    """
    Z, H, W = stack.shape
    half = distance // 2
    
    # Handle z-range
    if end is None:
        end = Z
    start = max(0, start)
    end = min(Z, end)
    
    if start >= end:
        raise ValueError("Invalid z-range: start must be < end")
    
    # Smooth image for stable gradients
    smooth = gaussian_filter(csum_image, sigma=1.0)
    
    # Estimate background
    background = gaussian_filter(smooth, sigma=4.0)
    
    # Contrast enhancement (DoG-like)
    contrast = smooth - background
    
    # Robust noise estimation
    bg_mask = contrast < np.percentile(contrast, 50)
    sigma_noise = 1.4826 * np.median(np.abs(contrast[bg_mask] - np.median(contrast[bg_mask])))
    
    if sigma_noise == 0:
        raise ValueError("Estimated noise sigma is zero")
    
    logger.info(f"Estimated noise sigma: {sigma_noise:.3f}")
    
    # Peak detection
    raw_peaks = peak_local_max(
        contrast,
        min_distance=min_distance,
        threshold_abs=prominence_sigma * sigma_noise,
        exclude_border=half
    )
    
    # Hot pixel rejection via spatial support
    med_filtered = median_filter(smooth, size=2 * support_radius + 1)
    peaks = []
    
    for (y, x) in raw_peaks:
        if smooth[y, x] > med_filtered[y, x]:
            peaks.append((y, x))
    
    peaks = np.asarray(peaks)
    logger.info(f"Detected {len(peaks)} peaks after filtering")
    
    # Extract cutouts for specified z-range
    cutouts = []
    group_ids = []
    
    for i, (y, x) in enumerate(peaks):
        for z in range(start, end):
            cutout = stack[z, y-half:y+half+1, x-half:x+half+1]
            if cutout.shape == (distance+1, distance+1):
                cutouts.append((cutout, z))
                group_ids.append(i)
    
    logger.info(f"Extracted {len(cutouts)} cutouts from {len(peaks)} peaks")
    
    # Visualization
    if plot:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(csum_image, cmap="gray")
        ax.set_title(f"Detected Peaks (σ={prominence_sigma}, support={support_radius})")
        
        for (y, x) in peaks:
            ax.plot(x, y, "r.", markersize=6)
        
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_aspect("equal")
        plt.tight_layout()
        plt.show()
    
    return cutouts, group_ids, peaks

def normalize_0_to_1(images: np.ndarray) -> np.ndarray:
    """
    Normalize PSF images to [0, 1] range per image
    
    Args:
        images: Array of images
        
    Returns:
        Normalized images
    """
    images = images.astype(np.float32)
    normalized = []
    
    for img in images:
        img_min = img.min()
        img_max = img.max()
        
        if img_max - img_min > 1e-6:
            norm_img = (img - img_min) / (img_max - img_min)
        else:
            norm_img = img
            
        normalized.append(norm_img)
    
    return np.array(normalized)