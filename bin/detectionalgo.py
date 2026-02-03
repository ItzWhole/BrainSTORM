#!/usr/bin/env python3
"""
STORM Peak Detection and Localization Algorithms

Comprehensive collection of algorithms for detecting and localizing PSF peaks
in STORM microscopy data with sub-pixel precision.

Author: milab
Optimized for Time Series Analysis Version
"""

import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, maximum_filter
from skimage.feature import peak_local_max

# Default test file path (only used when running as main script)
TIFF_PATH = r"C:\Users\milab\Desktop\data tiff prueba\b3Rub_EM40_100ms_20pM_4mW_ROI2-3.tiff"

def get_frame(frame_idx, path=TIFF_PATH):
    """
    Load a specific frame from a multi-frame TIFF file.

    Parameters
    ----------
    frame_idx : int
        Index of the frame to extract (0-based)
    path : str
        Path to the TIFF file

    Returns
    -------
    numpy.ndarray
        2D array containing the frame data
    """
    with tiff.TiffFile(path) as tif:
        frame = tif.asarray(key=frame_idx)

    plt.figure()
    plt.imshow(frame, cmap="gray")
    plt.colorbar(label="Intensity")
    plt.title(f"Frame {frame_idx}")
    plt.axis("off")
    plt.show()

    return frame


def bandpass_filter(frame, sigma_small=1.2, sigma_large=6.0):
    """
    Apply band-pass filter using Difference of Gaussians (DoG) for PSF enhancement.

    This function enhances PSF features while suppressing background noise by
    subtracting a large-scale smoothed version from a small-scale smoothed version.

    Parameters
    ----------
    frame : numpy.ndarray
        Raw image frame
    sigma_small : float, default=1.2
        Gaussian sigma for PSF-scale smoothing (smaller values preserve finer details)
    sigma_large : float, default=6.0
        Gaussian sigma for background smoothing (larger values remove more background)

    Returns
    -------
    numpy.ndarray
        Band-pass filtered image with enhanced PSF contrast
    """
    frame = frame.astype(np.float32)
    
    # Apply Gaussian filters efficiently
    high_pass = gaussian_filter(frame, sigma=sigma_small)
    low_pass = gaussian_filter(frame, sigma=sigma_large)
    
    # Difference of Gaussians
    return high_pass - low_pass

def find_local_peaks(image, threshold, min_distance=1):
    """
    Find local maxima in an image using efficient neighborhood suppression.
    
    Parameters
    ----------
    image : numpy.ndarray
        2D image array (preferably filtered)
    threshold : float
        Minimum intensity for peak detection
    min_distance : int, default=1
        Minimum distance between peaks (neighborhood size)
        
    Returns
    -------
    numpy.ndarray
        Array of peak coordinates as (row, col) pairs
    """
    # Local maxima detection using maximum filter
    neighborhood = maximum_filter(image, size=min_distance)
    local_max = (image == neighborhood)
    
    # Apply threshold
    detected_peaks = local_max & (image > threshold)
    
    # Extract coordinates
    return np.column_stack(np.nonzero(detected_peaks))


def robust_max(image, k=10):
    """
    Calculate robust maximum using mean of k brightest pixels.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image
    k : int, default=10
        Number of brightest pixels to average
        
    Returns
    -------
    float
        Robust maximum value
    """
    flat = image.ravel()
    brightest = np.partition(flat, -k)[-k:]
    return np.mean(brightest) 

def show_peaks(image, peaks, x_size=4, linewidth=0.6):
    """
    Visualize detected peaks on an image.
    
    Parameters
    ----------
    image : numpy.ndarray
        2D image array
    peaks : numpy.ndarray
        Peak coordinates as (row, col) pairs
    x_size : float, default=4
        Size of the X marker
    linewidth : float, default=0.6
        Thickness of the X lines
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image, cmap='gray')
    
    if len(peaks) > 0:
        rows, cols = peaks[:, 0], peaks[:, 1]
        ax.plot(cols, rows, 'rx', markersize=x_size, mew=linewidth)
        ax.set_title(f'Detected Peaks: {len(peaks)} found')
    else:
        ax.set_title('No peaks detected')

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

def extract_rois(image, peaks, N):
    """
    Extract NxN ROIs centered on given peaks.

    Parameters
    ----------
    image : 2D ndarray
        Input image.
    peaks : array-like, shape (n_peaks, 2)
        Peak coordinates as (row, col).
    N : int
        ROI size (must be odd).

    Returns
    -------
    roi_stack : ndarray, shape (n_rois, N, N)
        Stack of extracted ROIs.
    kept_peaks : ndarray, shape (n_rois, 2)
        Peak coordinates corresponding to returned ROIs.
    """
    if N % 2 == 0:
        raise ValueError("N must be odd so the ROI is centered on the peak.")

    half = N // 2
    rois = []
    kept = []

    H, W = image.shape

    for r, c in peaks:
        r = int(r)
        c = int(c)

        r0 = r - half
        r1 = r + half + 1
        c0 = c - half
        c1 = c + half + 1

        # Discard if ROI does not fit
        if r0 < 0 or c0 < 0 or r1 > H or c1 > W:
            continue

        roi = image[r0:r1, c0:c1]
        rois.append(roi)
        kept.append((r, c))

    if len(rois) == 0:
        return np.empty((0, N, N)), np.empty((0, 2), dtype=int)

    return np.stack(rois), np.array(kept)


def filter_rois_single_central_peak(
    roi_stack,
    threshold_rel=0.2,
    min_distance=2,
    center_size=10,
    strength_factor=1  # New parameter
):
    """
    Filter ROIs to keep those with exactly one peak in the central region.
    If peaks exist outside the center, keep only if the central peak
    is at least strength_factor times as strong as the strongest peak.
    
    Parameters
    ----------
    strength_factor : float, default=0.8
        Minimum relative strength of central peak compared to strongest peak.
        Values closer to 1.0 are more restrictive (closer to original).
        A value of 0.8 means central peak must be at least 0.8 * strongest_peak.

    Returns
    -------
    filtered_rois : ndarray
        Stack of ROIs that passed the filter.
    kept_indices : ndarray
        Indices of ROIs that passed.
    central_peaks : ndarray
        (row, col) of central peak in each kept ROI.
    not_kept_indices : ndarray
        Indices of ROIs that failed the filter.
    """
    n_rois, N, _ = roi_stack.shape
    half_center = center_size // 2
    center = N // 2

    filtered = []
    kept_indices = []
    not_kept_indices = []
    central_peaks = []

    for i, roi in enumerate(roi_stack):
        peaks = peak_local_max(
            roi,
            min_distance=min_distance,
            threshold_rel=threshold_rel
        )

        if len(peaks) == 0:
            not_kept_indices.append(i)
            continue

        # Peaks inside central square
        in_center = []
        for r, c in peaks:
            if (abs(r - center) <= half_center and
                abs(c - center) <= half_center):
                in_center.append((r, c))

        # Must be exactly one peak in center
        if len(in_center) != 1:
            not_kept_indices.append(i)
            continue

        center_peak = in_center[0]
        center_val = roi[center_peak]

        # Find the strongest peak value (could be inside or outside center)
        strongest_val = float('-inf')
        for r, c in peaks:
            peak_val = roi[r, c]
            if peak_val > strongest_val:
                strongest_val = peak_val

        # Check if central peak is strong enough
        # Original: center_val >= strongest_val
        # New: center_val >= strength_factor * strongest_val
        if center_val < strength_factor * strongest_val:
            not_kept_indices.append(i)
            continue

        filtered.append(roi)
        kept_indices.append(i)
        central_peaks.append(center_peak)

    if len(filtered) == 0:
        return (
            np.empty((0, N, N)),
            np.array([], dtype=int),
            np.empty((0, 2), dtype=int),
            np.array(not_kept_indices, dtype=int)
        )

    return (
        np.stack(filtered),
        np.array(kept_indices),
        np.array(central_peaks),
        np.array(not_kept_indices, dtype=int)
    )

def iterative_weighted_centroid(roi, n_iter=5, bg_percentile=10):
    """
    Iterative centroid with background subtraction.
    
    Parameters
    ----------
    roi : 2D array
        ROI containing PSF
    n_iter : int
        Number of iterations
    bg_percentile : float
        Percentile to estimate background
    
    Returns
    -------
    (dy, dx) : sub-pixel offsets relative to ROI center
    """
    # Estimate background from lowest percentile
    bg = np.percentile(roi, bg_percentile)
    
    # Subtract background
    roi_sub = roi - bg
    roi_sub[roi_sub < 0] = 0
    
    y, x = np.indices(roi.shape)
    center = roi.shape[0] // 2
    
    # Initial guess at center
    cy, cx = center, center
    
    for _ in range(n_iter):
        # Create weighting window (circular or Gaussian around current estimate)
        r2 = (y - cy)**2 + (x - cx)**2
        weights = np.exp(-r2 / (2 * 1.5**2))  # Gaussian window
        
        # Weighted centroid
        weighted_intensity = roi_sub * weights
        total_weight = np.sum(weighted_intensity)
        
        if total_weight > 0:
            cy = np.sum(y * weighted_intensity) / total_weight
            cx = np.sum(x * weighted_intensity) / total_weight
    
    # Return offsets relative to center
    dy = cy - center
    dx = cx - center
    
    return dy, dx

def radial_symmetry_center(roi, radius=5):
    """
    Find center using radial symmetry.
    Based on: "Fast, single-molecule localization that achieves 
    theoretically minimum uncertainty" by Parthasarathy (2012).
    
    Returns sub-pixel offsets.
    """
    h, w = roi.shape
    center = h // 2
    
    # Gradient calculation
    gy, gx = np.gradient(roi)
    
    # Compute radial symmetry center
    m00 = np.sum(roi)
    
    if m00 == 0:
        return 0.0, 0.0
    
    # Weighted by gradient magnitude
    grad_mag = np.sqrt(gx**2 + gy**2) + 1e-10  # Avoid division by zero
    wx = gx / grad_mag
    wy = gy / grad_mag
    
    y, x = np.indices(roi.shape)
    
    # Compute weighted averages
    numerator_x = np.sum(wx * (x - center) * roi)
    numerator_y = np.sum(wy * (y - center) * roi)
    
    denominator_x = np.sum(np.abs(wx) * roi)
    denominator_y = np.sum(np.abs(wy) * roi)
    
    if denominator_x == 0 or denominator_y == 0:
        return 0.0, 0.0
    
    dx = numerator_x / denominator_x
    dy = numerator_y / denominator_y
    
    return dy, dx

def moments_localization(roi, threshold_percentile=20):
    """
    Use image moments for localization.
    Works well for any PSF shape.
    """
    # Subtract background
    bg = np.percentile(roi, threshold_percentile)
    roi_sub = roi - bg
    roi_sub[roi_sub < 0] = 0
    
    # Compute moments
    y, x = np.indices(roi.shape)
    m00 = np.sum(roi_sub)
    
    if m00 == 0:
        return 0.0, 0.0
    
    # First moments (centroid)
    m10 = np.sum(x * roi_sub)
    m01 = np.sum(y * roi_sub)
    
    cx = m10 / m00
    cy = m01 / m00
    
    center = roi.shape[0] // 2
    dx = cx - center
    dy = cy - center
    
    return dy, dx

def spline_interpolation_peak(roi, upsample_factor=10):
    """
    Use spline interpolation to find peak with sub-pixel accuracy.
    """
    from scipy.interpolate import RectBivariateSpline
    
    h, w = roi.shape
    center = h // 2
    
    # Find rough peak position (pixel level)
    max_idx = np.argmax(roi)
    peak_y, peak_x = np.unravel_index(max_idx, roi.shape)
    
    # Create spline interpolation
    y_coords = np.arange(h)
    x_coords = np.arange(w)
    
    spline = RectBivariateSpline(y_coords, x_coords, roi, kx=3, ky=3)
    
    # Refine search around peak
    y_fine = np.linspace(max(0, peak_y-2), min(h-1, peak_y+2), 20)
    x_fine = np.linspace(max(0, peak_x-2), min(w-1, peak_x+2), 20)
    
    Y, X = np.meshgrid(y_fine, x_fine, indexing='ij')
    Z = spline(y_fine, x_fine, grid=True)
    
    # Find sub-pixel peak
    max_idx_fine = np.argmax(Z)
    peak_y_fine, peak_x_fine = np.unravel_index(max_idx_fine, Z.shape)
    
    subpixel_y = y_fine[peak_y_fine]
    subpixel_x = x_fine[peak_x_fine]
    
    dy = subpixel_y - center
    dx = subpixel_x - center
    
    return dy, dx

def adaptive_localization(roi, methods=['radial', 'iterative', 'moments']):
    """
    Try multiple methods and use the most consistent result.
    
    Parameters
    ----------
    roi : 2D array
    methods : list of str
        Methods to try: 'radial', 'iterative', 'moments', 'spline'
    
    Returns
    -------
    (dy, dx) : best estimate
    method_used : str
    confidence : float (0-1)
    """
    results = []
    
    if 'radial' in methods:
        dy1, dx1 = radial_symmetry_center(roi)
        results.append(('radial', dy1, dx1))
    
    if 'iterative' in methods:
        dy2, dx2 = iterative_weighted_centroid(roi)
        results.append(('iterative', dy2, dx2))
    
    if 'moments' in methods:
        dy3, dx3 = moments_localization(roi)
        results.append(('moments', dy3, dx3))
    
    if 'spline' in methods:
        dy4, dx4 = spline_interpolation_peak(roi)
        results.append(('spline', dy4, dx4))
    
    # Check consistency
    positions = np.array([(dy, dx) for _, dy, dx in results])
    
    if len(positions) == 0:
        return 0.0, 0.0, 'none', 0.0
    
    # Compute median as robust estimate
    median_dy = np.median(positions[:, 0])
    median_dx = np.median(positions[:, 1])
    
    # Find method closest to median
    distances = []
    for method, dy, dx in results:
        dist = np.sqrt((dy - median_dy)**2 + (dx - median_dx)**2)
        distances.append((dist, method, dy, dx))
    
    # Use the method with smallest deviation from median
    distances.sort()
    best_dist, best_method, best_dy, best_dx = distances[0]
    
    # Confidence based on agreement between methods
    mean_pos = np.mean(positions, axis=0)
    std_pos = np.std(positions, axis=0)
    
    # Confidence decreases with higher spread
    spread = np.sqrt(np.sum(std_pos**2))
    confidence = np.exp(-spread)
    
    return best_dy, best_dx, best_method, confidence

def localize_emitters_xy(frame, peaks, roi_size=25, 
                         method='adaptive', return_confidence=False):
    """
    Main localization function.
    
    Parameters
    ----------
    frame : 2D array
    peaks : (N, 2) array of [y, x] coordinates
    roi_size : int
    method : str or callable
        'radial', 'iterative', 'moments', 'spline', 'adaptive', or custom function
    return_confidence : bool
    
    Returns
    -------
    localizations : list of dicts
    """
    rois, kept_peaks = extract_rois(frame, peaks, roi_size)
    
    localizations = []
    
    for i, (roi, (py, px)) in enumerate(zip(rois, kept_peaks)):
        loc = {
            'pixel_x': px,
            'pixel_y': py,
            'roi_idx': i
        }
        
        # Choose localization method
        if method == 'radial':
            dy, dx = radial_symmetry_center(roi)
            confidence = 1.0
            
        elif method == 'iterative':
            dy, dx = iterative_weighted_centroid(roi)
            confidence = 1.0
            
        elif method == 'moments':
            dy, dx = moments_localization(roi)
            confidence = 1.0
            
        elif method == 'spline':
            dy, dx = spline_interpolation_peak(roi)
            confidence = 1.0
            
        elif method == 'adaptive':
            dy, dx, method_used, confidence = adaptive_localization(roi)
            loc['method_used'] = method_used
            
        elif callable(method):
            # Custom method
            dy, dx = method(roi)
            confidence = 1.0
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        loc.update({
            'subpixel_x': dx,
            'subpixel_y': dy,
            'x': px + dx,
            'y': py + dy,
            'intensity': np.max(roi),
            'snr': np.max(roi) / np.median(roi)
        })
        
        if return_confidence:
            loc['confidence'] = confidence
        
        localizations.append(loc)
    
    return localizations

def compare_localization_methods(roi):
    """
    Compare different localization methods on a single ROI.
    """
    methods = {
        'Radial Symmetry': radial_symmetry_center,
        'Iterative Centroid': iterative_weighted_centroid,
        'Moments': moments_localization,
        'Spline Interpolation': spline_interpolation_peak
    }
    
    results = {}
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()
    
    for idx, (name, func) in enumerate(methods.items()):
        dy, dx = func(roi)
        results[name] = (dx, dy)
        
        # Plot
        ax = axes[idx]
        ax.imshow(roi, cmap='hot', origin='lower')
        center = roi.shape[0] // 2
        ax.plot(center + dx, center + dy, 'bx', markersize=10, mew=2)
        ax.plot(center, center, 'r+', markersize=8, mew=1)
        ax.set_title(f'{name}\ndx={dx:.3f}, dy={dy:.3f}')
        ax.axis('off')
    
    # Summary
    ax = axes[4]
    ax.axis('off')
    summary_text = "Method Comparison:\n\n"
    for name, (dx, dy) in results.items():
        summary_text += f"{name}:\n  dx={dx:.3f}, dy={dy:.3f}\n"
    ax.text(0.1, 0.5, summary_text, fontsize=10, 
            verticalalignment='center', fontfamily='monospace')
    
    axes[5].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return results

#%%

if __name__ == "__main__":
    # Test code - only runs when script is executed directly, not when imported
    a=get_frame(1)
    plt.imshow(a)
    f_a=bandpass_filter(a,1.2,3)

    plt.imshow(f_a)


    filtered_image=f_a
    peaks = find_local_peaks(filtered_image,
                             threshold=0.1 * robust_max(filtered_image, k=10),
                             min_distance=4)



    show_peaks(filtered_image, peaks)

    rois, kept= extract_rois(filtered_image, peaks, 25)

    #%%
    plt.imshow(rois[2]) 

    #%%
    filtered2, kept2, central2,notkept2=filter_rois_single_central_peak(rois)
    #%%
    for i in kept2:
        plt.imshow(rois[i])
        plt.show()

    #%%
    # Test localization methods
    for i in kept2:
        compare_localization_methods(rois[i])

    print('iterative centroid is the best')