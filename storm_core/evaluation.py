"""
Evaluation and Visualization Module for STORM Microscopy

Contains functions for model evaluation, visualization, and analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def rescale_01_to_nm(x: np.ndarray, 
                    start: int, 
                    end: int, 
                    step_nm: float = 25.0, 
                    base_nm: float = -2000.0) -> np.ndarray:
    """
    Rescale [0,1] normalized values back to nanometers
    
    Args:
        x: Values in [0,1] range
        start: Starting stack index
        end: Ending stack index  
        step_nm: Spacing between stacks in nm
        base_nm: Base offset in nm
        
    Returns:
        Values rescaled to nanometers
    """
    nm_min = base_nm + start * step_nm
    nm_max = base_nm + end * step_nm
    return x * (nm_max - nm_min) + nm_min

def plot_true_vs_pred_heatmap(y_true: np.ndarray,
                             y_pred: np.ndarray,
                             bins: int = 100,
                             figsize: Tuple[int, int] = (8, 6),
                             log_scale: bool = True,
                             nm_range: Optional[Tuple[float, float]] = None,
                             ignore_percent: float = 0.0,
                             start: int = 0,
                             end: int = 161) -> None:
    """
    Plot true vs predicted values with global alignment
    
    Args:
        y_true: True values (normalized [0,1])
        y_pred: Predicted values (normalized [0,1])
        bins: Number of histogram bins
        figsize: Figure size
        log_scale: Use log scale for colormap
        nm_range: Range in nanometers (computed if None)
        ignore_percent: Percentage of worst predictions to ignore
        start: Start z-index for nm conversion
        end: End z-index for nm conversion
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    # Compute nm range if not provided
    if nm_range is None:
        nm_min = -2000.0 + start * 25.0
        nm_max = -2000.0 + end * 25.0
        nm_range = (nm_min, nm_max)
    
    nm_min, nm_max = nm_range
    scale = nm_max - nm_min
    
    # Global alignment: find optimal offset
    delta_norm = np.median(y_true - y_pred)
    y_true_aligned = y_true - delta_norm
    
    # Convert to nanometers
    y_true_nm = y_true_aligned * scale + nm_min
    y_pred_nm = y_pred * scale + nm_min
    
    # Compute absolute error and trim worst predictions
    abs_err = np.abs(y_true_nm - y_pred_nm)
    
    if ignore_percent > 0:
        cutoff = np.percentile(abs_err, 100.0 - ignore_percent)
        mask = abs_err <= cutoff
        y_true_nm = y_true_nm[mask]
        y_pred_nm = y_pred_nm[mask]
        abs_err = abs_err[mask]
    
    mae_nm = np.mean(abs_err)
    
    # Create plot
    plt.figure(figsize=figsize)
    plt.hist2d(y_true_nm, y_pred_nm, bins=bins, cmap="viridis",
              norm="log" if log_scale else None)
    plt.colorbar(label="Count")
    plt.xlabel("True Height (nm)")
    plt.ylabel("Predicted Height (nm)")
    plt.title(f"True vs Predicted Heights (MAE = {mae_nm:.1f} nm)")
    plt.grid(True, alpha=0.3)
    
    # Get current axis limits for ideal line
    xlim = plt.xlim()
    ylim = plt.ylim()
    
    # Add ideal line using the data range
    line_min = max(xlim[0], ylim[0])
    line_max = min(xlim[1], ylim[1])
    plt.plot([line_min, line_max], [line_min, line_max], "r--", linewidth=2, label="Ideal")
    
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    logger.info(f"Evaluation: MAE = {mae_nm:.1f} nm, Global offset = {delta_norm * scale:.1f} nm")

def plot_random_psfs(psfs: np.ndarray,
                     heights: np.ndarray,
                     start: int,
                     end: int,
                     n_images: int = 9,
                     step: int = 20,
                     step_nm: float = 25.0,
                     base_nm: float = -2000.0,
                     figsize: Tuple[int, int] = (12, 8),
                     seed: Optional[int] = None) -> None:
    """
    Plot random PSF samples at different heights
    
    Args:
        psfs: Array of PSF images
        heights: Normalized height values [0,1]
        start: Start z-index for nm conversion
        end: End z-index for nm conversion
        n_images: Number of images to display
        step: Height step for sampling
        step_nm: nm per z-step
        base_nm: Base nm offset
        figsize: Figure size
        seed: Random seed
    """
    if seed is not None:
        np.random.seed(seed)
    
    psfs = np.asarray(psfs)
    heights = np.asarray(heights)
    
    # Remove channel dimension if present
    if psfs.ndim == 4 and psfs.shape[-1] == 1:
        psfs = psfs[..., 0]
    
    # Convert heights to discrete indices
    n_heights = end - start
    height_idx = np.round(heights * (n_heights - 1)).astype(int)
    
    # Sample at regular intervals
    targets = np.arange(0, n_heights, step)
    chosen_indices = []
    
    for target in targets:
        candidates = np.where(height_idx == target)[0]
        if len(candidates) > 0:
            chosen_indices.append(np.random.choice(candidates))
    
    n_selected = min(len(chosen_indices), n_images)
    chosen_indices = chosen_indices[:n_selected]
    
    # Create subplot grid
    n_cols = min(4, n_selected)
    n_rows = int(np.ceil(n_selected / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_selected == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Plot selected PSFs
    for i, idx in enumerate(chosen_indices):
        psf = psfs[idx]
        h_norm = heights[idx]
        h_nm = rescale_01_to_nm(h_norm, start, end, step_nm, base_nm)
        
        axes[i].imshow(psf, cmap="gray")
        axes[i].set_title(f"{h_nm:.0f} nm")
        axes[i].axis("off")
    
    # Hide unused subplots
    for i in range(n_selected, len(axes)):
        axes[i].axis("off")
        axes[i].set_visible(False)
    
    plt.suptitle(f"Random PSF Samples at Different Heights", fontsize=14)
    plt.tight_layout()
    plt.show()
    
    logger.info(f"Displayed {n_selected} PSF samples")

def binned_mae_analysis(z_true: np.ndarray,
                       z_pred: np.ndarray,
                       n_bins: int = 100,
                       z_range: Tuple[float, float] = (-2000, 2000)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute MAE binned by true z-values for error analysis
    
    Args:
        z_true: True z-values in nm
        z_pred: Predicted z-values in nm
        n_bins: Number of bins
        z_range: Z-value range for binning
        
    Returns:
        bin_centers: Center of each bin
        mae_per_bin: MAE for each bin
    """
    z_true = np.asarray(z_true)
    z_pred = np.asarray(z_pred)
    
    zmin, zmax = z_range
    bins = np.linspace(zmin, zmax, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    mae_per_bin = np.full(n_bins, np.nan)
    abs_err = np.abs(z_pred - z_true)
    
    for i in range(n_bins):
        mask = (z_true >= bins[i]) & (z_true < bins[i + 1])
        if mask.sum() > 0:
            mae_per_bin[i] = abs_err[mask].mean()
    
    logger.info(f"Computed binned MAE analysis with {n_bins} bins")
    return bin_centers, mae_per_bin