#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 11:28:57 2025

@author: lautaro
"""

import os
import numpy as np
import h5py
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
from tensorflow import keras
from collections import Counter
from sklearn.model_selection import GroupShuffleSplit
from tensorflow.keras import layers, models
from matplotlib.colors import LogNorm
import tensorflow as tf
import matplotlib.patches as patches
from scipy.optimize import minimize_scalar
from scipy.ndimage import maximum_filter
from skimage.feature import peak_local_max
from sklearn.metrics import mean_absolute_error
#%%
from pathlib import Path

main_path = "/home/lautaro/Downloads/31 jul 4mw"

p_list = []

for folder in Path(main_path).iterdir():
    if folder.is_dir() and "keras" not in folder.name:
        tif_path = folder / f"{folder.name}_MMStack_Default.ome.tif"
        if tif_path.exists():
            p_list.append(str(tif_path))

# Optional: sort for consistent order (_1, _2, ..., _24)
p_list.sort()


print(f"Found {len(p_list)} files:")
for p in p_list:
    print(p)

# %%
s = [20, 17] #EL SEGUNDO ERA 17!!!!!!!!!!!!!!!!!!! LO CAMBIE A 9

stack_1 = ''
stack_2 = ''

for i in p_list:
    if str(s[0]) in i:
        if 'keras' not in i:
            stack_1 = i
    if str(s[1]) in i:
        if 'keras' not in i:
            stack_2 = i

print(stack_1)
print(stack_2)
#%%
train = tiff.imread(stack_1)
plt.imshow(train[80])
plt.show()

validation = tiff.imread(stack_2)
plt.imshow(validation[80])
plt.show()
#%%
def crop_and_sum_stack(stack, start, end, csum):
    """
    Parameters
    ----------
    stack : np.ndarray
        Input stack with shape (Z, H, W)
    start : int
        First slice index to keep (inclusive)
    end : int
        Last slice index to keep (inclusive)
    csum : int
        Number of final slices to sum after cropping

    Returns
    -------
    summed : np.ndarray
        2D array (H, W) resulting from summing the last `csum` slices
    """
    # --- Crop ---
    cropped = stack[start:end+1]

    if cropped.ndim != 3:
        raise ValueError("Expected stack with shape (Z, H, W)")

    if csum > cropped.shape[0]:
        raise ValueError(
            f"csum ({csum}) is larger than cropped stack depth ({cropped.shape[0]})"
        )

    # --- Sum last csum slices ---
    summed = np.sum(cropped[-csum:], axis=0)

    return summed

#%%
start=0
end=161
csum=30
distance=18

csum_stackim= crop_and_sum_stack(train, start, end, csum)
plt.imshow(csum_stackim)
plt.show()

csum_stackim_val=crop_and_sum_stack(validation, start, end, csum)
plt.imshow(csum_stackim_val)
plt.show()

#%%
from scipy.ndimage import gaussian_filter, median_filter


def extract_psf_cutouts(
    stack,
    csum_stackim,
    distance,
    min_distance=5,
    prominence_sigma=2.0,
    support_radius=1,
    start=0,
    end=None,
    plot=True
):
    """
    Extract PSF cutouts centered on robustly detected peaks.

    Parameters
    ----------
    stack : np.ndarray
        3D array of shape (Z, H, W)
    csum_stackim : np.ndarray
        2D image used for peak detection (H, W)
    distance : int
        Cutout size (distance x distance)
    min_distance : int
        Minimum separation between peaks (pixels)
    prominence_sigma : float
        Required peak prominence in units of background noise sigma
    support_radius : int
        Radius used to reject hot pixels via spatial support
    start : int
        First z-index to include (inclusive)
    end : int or None
        Last z-index to include (exclusive). None → up to Z
    plot : bool
        Whether to plot detected peaks

    Returns
    -------
    cutouts : list of tuples
        (cutout, z_index)
    group_ids : list of int
        Group index per cutout (same id for all z of one peak)
    peaks : np.ndarray
        Array of peak coordinates (row, col)
    """

    Z, H, W = stack.shape
    half = distance // 2

    # ---- Clamp z-range ----
    if end is None:
        end = Z
    start = max(0, start)
    end = min(Z, end)

    if start >= end:
        raise ValueError("Invalid z-range: start must be < end")

    # ---- Smooth to stabilize gradients ----
    smooth = gaussian_filter(csum_stackim, sigma=1.0)

    # ---- Estimate smooth background ----
    background = gaussian_filter(smooth, sigma=4.0)

    # ---- Contrast image (DoG-like) ----
    contrast = smooth - background

    # ---- Robust noise estimate from background only ----
    bg_mask = contrast < np.percentile(contrast, 50)
    sigma_noise = 1.4826 * np.median(
        np.abs(contrast[bg_mask] - np.median(contrast[bg_mask]))
    )

    if sigma_noise == 0:
        raise ValueError("Estimated noise sigma is zero; check input image")

    # ---- Initial peak detection ----
    raw_peaks = peak_local_max(
        contrast,
        min_distance=min_distance,
        threshold_abs=prominence_sigma * sigma_noise,
        exclude_border=half
    )

    # ---- Hot-pixel rejection via spatial support ----
    med = median_filter(smooth, size=2 * support_radius + 1)

    peaks = []
    for (y, x) in raw_peaks:
        if smooth[y, x] > med[y, x]:
            peaks.append((y, x))

    peaks = np.asarray(peaks)

    # ---- Extract cutouts only for z in [start, end) ----
    cutouts = []
    group_ids = []

    for i, (y, x) in enumerate(peaks):
        for z in range(start, end):
            cutout = stack[
                z,
                y - half : y + half + 1,
                x - half : x + half + 1
            ]

            if cutout.shape == (distance + 1, distance + 1):
                cutouts.append((cutout, z))
                group_ids.append(i)

    # ---- Visualization ----
    if plot:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(csum_stackim, cmap="gray")
        ax.set_title(
            f"Detected peaks (σ={prominence_sigma}, support={support_radius})"
        )

        for (y, x) in peaks:
            ax.plot(x, y, "r.", markersize=4)

        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_aspect("equal")
        plt.tight_layout()
        plt.show()

    return cutouts, group_ids, peaks

#%%
def build_astigmatic_psf_network(input_shape):
    inputs = layers.Input(shape=input_shape)

    # Initial processing
    x = layers.Conv2D(64, (5, 5), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)  # Now 15x15

    # Feature extraction blocks
    for filters in [128, 256, 512]:
        # Path 1: Standard 3x3 convolution
        p1 = layers.Conv2D(filters, (3, 3), padding='same',
                           activation='relu')(x)

        # Path 2: Horizontal emphasis (3x5 kernel)
        # Pad height: (top, bottom), width: (left, right)
        p2 = layers.ZeroPadding2D(((1, 1), (2, 2)))(x)  # +2 height, +4 width
        p2 = layers.Conv2D(filters, (3, 5), padding='valid',
                           activation='relu')(p2)

        # Path 3: Vertical emphasis (5x3 kernel)
        p3 = layers.ZeroPadding2D(((2, 2), (1, 1)))(x)  # +4 height, +2 width
        p3 = layers.Conv2D(filters, (5, 3), padding='valid',
                           activation='relu')(p3)

        # Crop all paths to 15x15 (or current spatial dim)
        target_size = x.shape[1:3]  # Get current (H,W)
        p1 = layers.CenterCrop(target_size[0], target_size[1])(p1)
        p2 = layers.CenterCrop(target_size[0], target_size[1])(p2)
        p3 = layers.CenterCrop(target_size[0], target_size[1])(p3)

        # Concatenate along channel dimension
        x = layers.concatenate([p1, p2, p3])
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)

    # Prediction head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(1)(x)

    return models.Model(inputs=inputs, outputs=outputs)

def build_astigmatic_psf_network(input_shape):
    inputs = layers.Input(shape=input_shape)

    # Initial processing
    x = layers.Conv2D(64, (5, 5), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)  # Now 15x15

    # Feature extraction blocks
    for filters in [128, 256, 512]:
        # Path 1: Standard 3x3 convolution
        p1 = layers.Conv2D(filters, (3, 3), padding='same',
                           activation='relu')(x)

        # Path 2: Horizontal emphasis (3x5 kernel)
        # Pad height: (top, bottom), width: (left, right)
        p2 = layers.ZeroPadding2D(((1, 1), (2, 2)))(x)  # +2 height, +4 width
        p2 = layers.Conv2D(filters, (3, 5), padding='valid',
                           activation='relu')(p2)

        # Path 3: Vertical emphasis (5x3 kernel)
        p3 = layers.ZeroPadding2D(((2, 2), (1, 1)))(x)  # +4 height, +2 width
        p3 = layers.Conv2D(filters, (5, 3), padding='valid',
                           activation='relu')(p3)

        # Crop all paths to 15x15 (or current spatial dim)
        target_size = x.shape[1:3]  # Get current (H,W)
        p1 = layers.CenterCrop(target_size[0], target_size[1])(p1)
        p2 = layers.CenterCrop(target_size[0], target_size[1])(p2)
        p3 = layers.CenterCrop(target_size[0], target_size[1])(p3)

        # Concatenate along channel dimension
        x = layers.concatenate([p1, p2, p3])
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)

    # Prediction head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(1)(x)

    return models.Model(inputs=inputs, outputs=outputs)
#%%
def train_val_split_by_group(X, y, group_ids, val_size=0.2, random_state=42):
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=val_size, random_state=random_state)
    train_idx, val_idx = next(splitter.split(X, y, groups=group_ids))
    return (X[train_idx], y[train_idx]), (X[val_idx], y[val_idx])
#%%
cutouts, group_ids, peaks = extract_psf_cutouts(train, csum_stackim,start=start, end=end, distance=distance, min_distance=5,
                            prominence_sigma=10, support_radius=2)
#%%
cutouts_val, group_ids_val, peaks_val = extract_psf_cutouts(validation, csum_stackim_val, start=start, end=end,distance=distance, min_distance=5,
                            prominence_sigma=10, support_radius=2)


#%%
def normalize_0_to_1(images):
    """Quick normalization for PSF images."""
    images = images.astype(np.float32)
    
    # Per-image normalization (most robust)
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

#%%
psfs = np.expand_dims((np.array([cutouts[i][0] for i in range(len(cutouts))])), axis=-1)
heights = np.array([cutouts[i][1] for i in range(len(cutouts))])
group_ids = np.array(group_ids)

print(np.max(heights))

psfs=normalize_0_to_1(psfs)
heights=(heights - np.min(heights)) / (np.max(heights) - np.min(heights))

print(heights)
#%%
psfs_val = np.expand_dims((np.array([cutouts_val[i][0] for i in range(len(cutouts_val))])), axis=-1)
heights_val = np.array([cutouts_val[i][1] for i in range(len(cutouts_val))])
group_ids_val = np.array(group_ids_val)

psfs_val=normalize_0_to_1(psfs_val)
heights_val=heights_val/(np.max(heights_val))

print(heights_val)



#%%
def plot_random_images(images, n_images=9, labels=None, titles=None, 
                       figsize=(12, 8), cmap='gray', random_state=None):
    """
    Plot N random images from an array.
    
    Parameters
    ----------
    images : np.ndarray
        Array of images with shape (N, H, W) or (N, H, W, C)
    n_images : int
        Number of random images to display
    labels : array-like, optional
        Labels for each image (e.g., height values)
    titles : list, optional
        Custom titles for each subplot
    figsize : tuple
        Figure size (width, height)
    cmap : str
        Colormap for grayscale images
    random_state : int, optional
        Random seed for reproducibility
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Input validation
    if images.ndim not in [3, 4]:
        raise ValueError(f"Expected 3D or 4D array, got {images.ndim}D")
    
    # Ensure we don't try to plot more images than available
    n_total = len(images)
    n_images = min(n_images, n_total)
    
    # Randomly select indices
    indices = np.random.choice(n_total, n_images, replace=False)
    
    # Determine grid layout
    n_cols = min(4, n_images)  # Max 4 columns
    n_rows = int(np.ceil(n_images / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Handle single row/column axes
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Plot each image
    for i, (idx, ax) in enumerate(zip(indices, axes)):
        img = images[idx]
        
        # Handle channel dimension
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img.squeeze()  # Remove channel dimension for grayscale
        
        # Display image
        if img.ndim == 2 or (img.ndim == 3 and img.shape[-1] == 1):
            im = ax.imshow(img, cmap=cmap)
        else:
            im = ax.imshow(img)
        
        # Add colorbar for the first image only (to save space)
        if i == 0:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Set title
        title_parts = []
        
        if titles is not None and i < len(titles):
            title_parts.append(str(titles[i]))
        elif labels is not None:
            title_parts.append(f"Label: {labels[idx]:.2f}")
        
        title_parts.append(f"Index: {idx}")
        
        # Add image statistics
        title_parts.append(f"[{img.min():.2f}, {img.max():.2f}]")
        
        ax.set_title(" | ".join(title_parts), fontsize=9)
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
        axes[i].set_visible(False)
    
    plt.suptitle(f"Random Sample of {n_images} Images", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"Displayed {n_images} random images from {n_total} total")
    print(f"Selected indices: {sorted(indices.tolist())}")
    if labels is not None:
        print(f"Selected labels: {labels[indices]}")

#%%
plot_random_images(psfs,n_images=2)
#%%
(X_tr, y_tr), (X_va, y_va) = train_val_split_by_group(psfs, heights, group_ids, val_size=0.2)

#%%
def build_augmenter(distance):
    return tf.keras.Sequential([
        layers.RandomTranslation(height_factor=5/distance, width_factor=3/distance, fill_mode='reflect'),
        layers.RandomRotation(factor=6/180, fill_mode='reflect'),
    ], name="random_jitter")

augmenter = build_augmenter(distance)

#%%
def make_dataset(X, y, batch_size=64, training=False, augmenter=None, shuffle_buf=2048):
    ds = tf.data.Dataset.from_tensor_slices((X.astype('float32'), y.astype('float32')))
    if training:
        ds = ds.shuffle(min(len(X), shuffle_buf),
                        reshuffle_each_iteration=True)
    if training and augmenter is not None:
        ds = ds.map(lambda img, target: (augmenter(img, training=True), target),
                    num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

#%%
train_ds = make_dataset(X_tr, y_tr, batch_size=64, training=True, augmenter=augmenter)
val_ds = make_dataset(X_va, y_va, batch_size=64, training=False)

#%%
print(np.max(y_tr))


#%%
ckpt_path=f'/home/lautaro/Downloads/la_crudisima_realidad/modelo_{distance}.keras'
# Model
model = build_astigmatic_psf_network(input_shape=(distance+1, distance+1, 1))
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.Huber(delta=0.06),
    metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae')]
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        ckpt_path, monitor='val_mae', save_best_only=True, mode='min'),
    # tf.keras.callbacks.EarlyStopping(monitor='val_mae', patience=55, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_mae', factor=0.5, patience=25, min_lr=1e-6),
]

#%%
model.summary()

#%%
# history = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=100,
#     callbacks=callbacks,
#     verbose=1
# )

#%%
# model = tf.keras.models.load_model(ckpt_path)

# model.compile(
#     optimizer=tf.keras.optimizers.Adam(2e-4),  # 🔑 smaller LR
#     loss=tf.keras.losses.Huber(delta=0.02),
#     metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae')]
# )

# history_finetune = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=300,
#     initial_epoch=100,
#     callbacks=[
#         tf.keras.callbacks.ModelCheckpoint(
#             ckpt_path, monitor='val_mae', save_best_only=True, mode='min'
#         ),
#         tf.keras.callbacks.EarlyStopping(
#             monitor='val_mae', patience=40, restore_best_weights=True
#         ),
#         tf.keras.callbacks.ReduceLROnPlateau(
#             monitor='val_mae', factor=0.5, patience=20, min_lr=1e-6
#         ),
#     ],
#     verbose=1
# )

#%%
import numpy as np
import matplotlib.pyplot as plt

def rescale_01_to_nm(x, start, end, step_nm=25.0, base_nm=-2000.0):
    """
    Rescale a [0, 1] variable to nanometers using stack indices.

    Parameters
    ----------
    x : array-like or float
        Values in [0, 1]
    start : int
        Starting stack index
    end : int
        Ending stack index
    step_nm : float, optional
        Spacing between stacks in nm (default: 25)
    base_nm : float, optional
        Base offset in nm (default: -2000)

    Returns
    -------
    x_nm : array-like or float
        Rescaled values in nanometers
    """

    nm_min = base_nm + start * step_nm
    nm_max = base_nm + end * step_nm

    return x * (nm_max - nm_min) + nm_min

def plot_true_vs_pred_heatmap_aligned(
    y_true,
    y_pred,
    bins=100,
    figsize=(6, 6),
    log_scale=True,
    nm_range=(-2000.0+start*25, -2000.0+end*25),
    plot_limits=(-2000.0, 2000.0),
    ignore_percent=15.0,
):
    """
    Drop-in replacement with global alignment.

    Steps:
    1. Compute optimal constant offset Δ that minimizes MAE
       (median(y_true - y_pred))
    2. Shift y_true by -Δ
    3. Discard worst ignore_percent% predictions
    4. Plot heatmap in nm

    Assumes y_true and y_pred are normalif '_20_' in p_valized in [0, 1].
    """

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    nm_min, nm_max = nm_range
    scale = nm_max - nm_min

    # --- Step 1: find optimal alignment offset (normalized units)
    delta_norm = np.median(y_true - y_pred)

    # Apply alignment
    y_true_aligned = y_true - delta_norm

    # --- Convert to nm
    y_true_nm = y_true_aligned * scale + nm_min
    y_pred_nm = y_pred * scale + nm_min

    # --- Step 2: compute absolute error in nm
    abs_err = np.abs(y_true_nm - y_pred_nm)

    # --- Step 3: trim worst errors
    if ignore_percent > 0:
        cutoff = np.percentile(abs_err, 100.0 - ignore_percent)
        mask = abs_err <= cutoff
        y_true_nm = y_true_nm[mask]
        y_pred_nm = y_pred_nm[mask]
        abs_err = abs_err[mask]

    mae_nm = np.mean(abs_err)

    # --- Plot
    plt.figure(figsize=figsize)

    plt.hist2d(
        y_true_nm,
        y_pred_nm,
        bins=bins,
        cmap="viridis",
        norm="log" if log_scale else None,
    )

    plt.colorbar(label="Cantidad")
    plt.xlabel("Valor real (nm)")
    plt.ylabel("Valor predicho (nm)")
    plt.grid()
    # plt.title("Valor real vs predicción (Globally Aligned)")
    plt.title(f"MAE = {mae_nm:.1f} nm")

    # y = x reference
    min_val = min(y_true_nm.min(), y_pred_nm.min())
    max_val = max(y_true_nm.max(), y_pred_nm.max())
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        linewidth=1.5,
        label=(
            "Ideal"
            # f"Desfase global = {delta_norm * scale:.1f} nm"
        ),
    )
    
    plt.xlim(plot_limits)
    plt.ylim(plot_limits)
    
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def plot_true_vs_pred_heatmap_aligned_nm(
    y_true,
    y_pred,
    bins=100,
    figsize=(6, 6),
    log_scale=True,
    ignore_percent=15.0,
    plot_limits=(-2000.0, 2000.0),
):
    """
    Drop-in replacement assuming y_true and y_pred are already in nanometers.

    Steps:
    1. Compute optimal constant offset Δ that minimizes MAE
       (median(y_true - y_pred))
    2. Shift y_true by -Δ
    3. Discard worst ignore_percent% predictions
    4. Plot heatmap in nm with fixed axis limits

    No normalization or denormalization is performed.
    """

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # --- Step 1: optimal global alignment (nm)
    delta_nm = np.median(y_true - y_pred)
    y_true_aligned = y_true - delta_nm

    # --- Step 2: absolute error in nm
    abs_err = np.abs(y_true_aligned - y_pred)

    # --- Step 3: trim worst errors
    if ignore_percent > 0:
        cutoff = np.percentile(abs_err, 100.0 - ignore_percent)
        mask = abs_err <= cutoff
        y_true_aligned = y_true_aligned[mask]
        y_pred = y_pred[mask]
        abs_err = abs_err[mask]

    mae_nm = np.mean(abs_err)

    # --- Plot
    plt.figure(figsize=figsize)

    plt.hist2d(
        y_true_aligned,
        y_pred,
        bins=bins,
        cmap="viridis",
        norm="log" if log_scale else None,
        range=[plot_limits, plot_limits],
    )

    plt.colorbar(label="Cantidad")
    plt.xlabel("Valor real (nm)")
    plt.ylabel("Valor predicho (nm)")
    # plt.title("True vs Predicted Heatmap (Aligned, nm-native)")
    plt.title(f"MAE = {mae_nm:.1f} nm ")

    # y = x reference
    plt.plot(
        plot_limits,
        plot_limits,
        "r--",
        linewidth=1.5,
        label=(
            "Ideal"
            # f"Desfase global= {delta_nm:.1f} nm"
        ),
    )

    plt.xlim(plot_limits)
    plt.ylim(plot_limits)

    plt.legend()
    plt.tight_layout()
    plt.show()
#%%
mp = '/home/lautaro/Downloads/la_crudisima_realidad/'

# ckpt_path = mp + 'modelo_18_nosigmoid100+185epochs_variedroutineESTEESELBUENO.keras'
ckpt_path = mp + 'modelo_24_nosigmoid100+214epochs_variedroutineESTEESELBUENO.keras'
# ckpt_path = mp + 'modelo_30_nosigmoid100+152epochs_variedroutineESTEESELBUENO.keras'

model = tf.keras.models.load_model(ckpt_path)

maes = []
min_maes = []

distance = int(ckpt_path.split('_')[3])
print('ROI:', distance)

#%%
ignore = 0.1
start=0
end=161
csum=30
preds=[]
heights_val=[]

c_list=[p_list[2]] #10 es 0, 8.6 es -3, beads es 2 con prominence 20
print(c_list)
for p in c_list:
    
    p_val = p
    
    if '_20_' not in p_val:
    # if '_20_' in p_val:
    #     raise Exception("Tomar otro stack")
    
        validation = tiff.imread(p_val)
        csum_stackim_val=crop_and_sum_stack(validation, start, end, csum)
        cutouts_val, group_ids_val, peaks_val = extract_psf_cutouts(validation, csum_stackim_val, start=start, end=end,distance=distance, min_distance=5,
                                    prominence_sigma=20, support_radius=2)
        
        psfs_val = np.expand_dims((np.array([cutouts_val[i][0] for i in range(len(cutouts_val))])), axis=-1)
        heights_val = np.array([cutouts_val[i][1] for i in range(len(cutouts_val))])
        group_ids_val = np.array(group_ids_val)
        
        psfs_val=normalize_0_to_1(psfs_val)
        heights_valn=heights_val/(np.max(heights_val))
        
        print(heights_valn)
        
        preds = model.predict(psfs_val, verbose=0).squeeze()
        true_vals = np.array(heights_valn).squeeze()
        print(min(preds),max(preds))
        
        plot_true_vs_pred_heatmap_aligned(true_vals,preds, bins=100,figsize=(6,5), ignore_percent=0)
        print(p_val)
    
#%%

np.savez(
    "beads.npz",
    z_frame31=heights_val,
    z_pred131=preds,
)


#%%
print('Rango:',min(true_vals), max(true_vals))
print('Rango (predicciones):',min(preds), max(preds))

#%%
# model.save(f'/home/lautaro/Downloads/compoundnodisplace.keras')
# model = tf.keras.models.load_model('/home/lautaro/Downloads/la_crudisima_realidad/modelo_18_nosigmoid100+185epochs_variedroutineESTEESELBUENO.keras')

#%%
ignore = 15
maes = []
min_maes = []

#%%
preds = model.predict(psfs_val, verbose=0).squeeze()
true_vals = np.array(heights_val).squeeze()

#%%
print(min(preds),max(preds))
#%%
print(np.max(heights_val))


predsre=rescale_01_to_nm(preds, start, end)
true_valsre=rescale_01_to_nm(true_vals, start, end)

plot_true_vs_pred_heatmap_aligned_nm(true_valsre,predsre, ignore_percent=0, figsize=(6,5))

#%%

import numpy as np
import matplotlib.pyplot as plt

def plot_random_psfs_every_20_heights(
    psfs,
    heights,
    start,
    end,
    step=20,
    step_nm=25.0,
    base_nm=-2000.0,
    seed=None,
):
    """
    Plot a random PSF every `step` heights with denormalized height labels.

    Parameters
    ----------
    psfs : np.ndarray
        Array of PSFs, shape (N, H, W) or (N, H, W, 1)
    heights : np.ndarray
        Normalized heights in [0, 1], shape (N,)
    start, end : int
        Stack index range used for normalization
    step : int, optional
        Height spacing (default: 20)
    step_nm, base_nm : float
        Parameters passed to rescale_01_to_nm
    seed : int or None
        Random seed for reproducibility
    """

    if seed is not None:
        np.random.seed(seed)

    psfs = np.asarray(psfs)
    heights = np.asarray(heights)

    if psfs.ndim == 4 and psfs.shape[-1] == 1:
        psfs = psfs[..., 0]

    # Convert normalized heights to discrete indices
    n_heights = 161
    height_idx = np.round(heights * (n_heights - 1)).astype(int)

    # Target indices every `step`
    targets = np.arange(0, n_heights, step)

    chosen = []
    for t in targets:
        candidates = np.where(height_idx == t)[0]
        if len(candidates) > 0:
            chosen.append(np.random.choice(candidates))

    n = len(chosen)
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 3))

    if n == 1:
        axes = [axes]

    for ax, idx in zip(axes, chosen):
        psf = psfs[idx]
        h_norm = heights[idx]
        h_nm = rescale_01_to_nm(
            h_norm,
            start=start,
            end=end,
            step_nm=step_nm,
            base_nm=base_nm,
        )

        ax.imshow(psf, cmap="gray")
        ax.set_title(f"{h_nm:.0f} nm")
        ax.axis("off")

    plt.suptitle("Random PSFs every 20 heights", y=1.05)
    plt.tight_layout()
    plt.show()
    
plot_random_psfs_every_20_heights(
    psfs,
    heights,
    start=start,
    end=end
)

#%%
z19 = np.load("/home/lautaro/zdatafolder/8.6.npz")
z24 = np.load("/home/lautaro/zdatafolder/z_data24.npz")
z31 = np.load("/home/lautaro/zdatafolder/10.npz")
zpic = np.load("/home/lautaro/zdatafolder/beads.npz")

z_heights_19, z_preds_19 = z19["z_frame31"], z19["z_pred131"]
z_heights_24, z_preds_24 = z24["z_frame24"], z24["z_pred24"]
z_heights_31, z_preds_31 = z31["z_frame31"], z31["z_pred131"]

z_heights_pic, z_preds_pic = zpic["z_frame31"], zpic["z_pred131"]

offset=0

z_preds_19=z_preds_19*4000-2000
z_preds_24=z_preds_24*4000-2000
z_preds_31=z_preds_31*4000-2000

z_preds_pic= z_preds_pic*4000-2000

z_heights_19=(z_heights_19/160*4000-2000)+offset
z_heights_24=(z_heights_24/160*4000-2000)+offset
z_heights_31=(z_heights_31/160*4000-2000)+offset

z_heights_pic= (z_heights_pic/160*4000-2000)+offset

def binned_mae(
    z_true,
    z_pred,
    n_bins=20,
    z_range=(-2000,2000),
    return_counts=False,
):
    """
    Compute MAE(z_pred vs z_true) binned by z_true.

    Parameters
    ----------
    z_true : array-like
        Ground truth heights
    z_pred : array-like
        Predicted heights
    n_bins : int
        Number of height bins
    z_range : tuple or None
        (zmin, zmax). If None, inferred from z_true
    return_counts : bool
        If True, also return number of samples per bin

    Returns
    -------
    bin_centers : ndarray
    mae_per_bin : ndarray
    counts : ndarray (optional)
    """

    z_true = np.asarray(z_true)
    z_pred = np.asarray(z_pred)

    if z_range is None:
        zmin, zmax = z_true.min(), z_true.max()
    else:
        zmin, zmax = z_range

    bins = np.linspace(zmin, zmax, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    mae_per_bin = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)

    abs_err = np.abs(z_pred - z_true)

    for i in range(n_bins):
        mask = (z_true >= bins[i]) & (z_true < bins[i + 1])
        counts[i] = mask.sum()
        if counts[i] > 0:
            mae_per_bin[i] = abs_err[mask].mean()

    if return_counts:
        return bin_centers, mae_per_bin, counts
    return bin_centers, mae_per_bin


# Number of bins
N_BINS = 100

# ---- Compute binned MAE (bin by predicted z) ----
bin_pic, mae_pic = binned_mae(z_preds_pic, z_heights_pic, n_bins=N_BINS)

bin_19, mae_19 = binned_mae(z_preds_19, z_heights_19, n_bins=N_BINS)
bin_24, mae_24 = binned_mae(z_preds_24, z_heights_24, n_bins=N_BINS)
bin_31, mae_31 = binned_mae(z_preds_31, z_heights_31, n_bins=N_BINS)

print(np.mean(mae_24))

# ---- Plot ----
plt.figure(figsize=(7, 5))

plt.plot(bin_pic, mae_pic, color="black", label="Beads")
plt.plot(bin_19, mae_19,  color="tab:blue",  label="Moleculas individuales 8.6 mW")
plt.plot(bin_31, mae_31,  color="tab:green", label="Moleculas individuales 10 mW")
plt.plot(bin_24, mae_24, color="tab:orange",label="Moleculas individuales 14 mW (igual a entrenamiento)")
plt.ylim(0,1500)
plt.xlabel("Altura en Z (nm)")
plt.ylabel("Error medio en módulo (nm)")


plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
#%%



















# --------------------
# Load data
# --------------------
z86      = np.load("/home/lautaro/zdatafolder/z_data31.npz")
z14      = np.load("/home/lautaro/zdatafolder/z_data24.npz")
z10      = np.load("/home/lautaro/zdatafolder/z_data19.npz")
z_beads  = np.load("/home/lautaro/zdatafolder/z_data.npz")

# --------------------
# Extract arrays
# --------------------
z_heights_86, z_preds_86 = z86["z_frame31"], z86["z_pred31"]
z_heights_14, z_preds_14 = z14["z_frame24"], z14["z_pred24"]
z_heights_10, z_preds_10 = z10["z_frame19"], z10["z_pred19"]

z_heights_beads, z_preds_beads = z_beads["z_frame"], z_beads["z_picasso"]

# --------------------
# Rescale to nm
# --------------------
offset = 0

def pred_to_nm(z):
    return z * 4000 - 2000

def height_to_nm(z):
    return (z / 160 * 4000 - 2000) + offset

z_preds_86     = pred_to_nm(z_preds_86)
z_preds_14     = pred_to_nm(z_preds_14)
z_preds_10     = pred_to_nm(z_preds_10)
# z_preds_beads  = pred_to_nm(z_preds_beads)

z_heights_86     = height_to_nm(z_heights_86)
z_heights_14     = height_to_nm(z_heights_14)
z_heights_10     = height_to_nm(z_heights_10)
# z_heights_beads  = height_to_nm(z_heights_beads)

# --------------------
# Binned MAE
# --------------------
N_BINS = 100

bin_beads, mae_beads = binned_mae(z_preds_beads, z_heights_beads, n_bins=N_BINS)
bin_86, mae_86       = binned_mae(z_preds_86, z_heights_86, n_bins=N_BINS)
bin_10, mae_10       = binned_mae(z_preds_10, z_heights_10, n_bins=N_BINS)
bin_14, mae_14       = binned_mae(z_preds_14, z_heights_14, n_bins=N_BINS)

print(np.mean(mae_14))

# --------------------
# Plot
# --------------------
plt.figure(figsize=(7, 5))

plt.plot(bin_beads, mae_beads, color="black",      label="Beads")
plt.plot(bin_86,     mae_86,     color="tab:blue",  label="Moléculas individuales 8.6 mW")
plt.plot(bin_10,     mae_10,     color="tab:green", label="Moléculas individuales 10 mW")
plt.plot(bin_14,     mae_14,     color="tab:orange",
         label="Moléculas individuales 14 mW (entrenamiento)")

plt.ylim(0, 1500)
plt.xlabel("Altura en Z (nm)")
plt.ylabel("Error medio absoluto (nm)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
#%%
def binned_mae_by_true_z(
    z_true,
    z_pred,
    n_bins=100,
    z_range=(-2000, 2000),
    return_counts=False,
):
    """
    Compute MAE(z_pred vs z_true) binned by TRUE z.

    Parameters
    ----------
    z_true : array-like
        True axial positions (nm)
    z_pred : array-like
        Predicted axial positions (nm)
    n_bins : int
        Number of bins
    z_range : tuple
        (zmin, zmax)
    return_counts : bool
        If True, also return counts per bin
    """

    z_true = np.asarray(z_true)
    z_pred = np.asarray(z_pred)

    zmin, zmax = z_range
    bins = np.linspace(zmin, zmax, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    mae = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)

    abs_err = np.abs(z_pred - z_true)

    for i in range(n_bins):
        mask = (z_true >= bins[i]) & (z_true < bins[i + 1])
        counts[i] = mask.sum()
        if counts[i] > 0:
            mae[i] = abs_err[mask].mean()

    if return_counts:
        return bin_centers, mae, counts
    return bin_centers, mae

N_BINS = 100

plt.figure(figsize=(7, 5))

def plot_valid_bins(x, y, **kwargs):
    mask = ~np.isnan(y)
    plt.plot(x[mask], y[mask], **kwargs)

plot_valid_bins(bin_beads, mae_beads, color="black", label="Beads")
plot_valid_bins(bin_86, mae_86, color="tab:blue", label="Moléculas individuales 8.6 mW")
plot_valid_bins(bin_10, mae_10, color="tab:green", label="Moléculas individuales 10 mW")
plot_valid_bins(
    bin_14,
    mae_14,
    color="tab:orange",
    label="Moléculas individuales 14 mW (entrenamiento)",
)

plt.xlabel("Altura real en Z (nm)")
plt.ylabel("Error medio absoluto (nm)")
plt.ylim(0, 1500)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


bin_beads, mae_beads = binned_mae_by_true_z(
    # z_heights_beads,
    z_preds_beads,
    z_heights_beads,
    n_bins=N_BINS
)

bin_86, mae_86 = binned_mae_by_true_z(
    z_heights_86,
    z_preds_86,
    n_bins=N_BINS
)

bin_10, mae_10 = binned_mae_by_true_z(
    z_heights_10,
    z_preds_10,
    n_bins=N_BINS
)

bin_14, mae_14 = binned_mae_by_true_z(
    z_heights_14,
    z_preds_14,
    n_bins=N_BINS
)
#%%
plt.figure(figsize=(7, 5))

def plot_valid_bins(x, y, **kwargs):
    mask = ~np.isnan(y)
    plt.plot(x[mask], y[mask], **kwargs)

def plot_valid_bins_in_range(x, y, zmin, zmax, **kwargs):
    mask = (~np.isnan(y)) & (x >= zmin) & (x <= zmax)
    plt.plot(x[mask], y[mask], **kwargs)

# ---- Plot ----

# Beads: solo entre -780 y 780 nm
plot_valid_bins_in_range(
    bin_beads,
    mae_beads,
    zmin=-780,
    zmax=780,
    color="black",
    label="Picasso",
)

# Moléculas: rango completo
plot_valid_bins(bin_86, mae_86, color="tab:blue",
                label="Modelo 19 px de lado")
plot_valid_bins(bin_10, mae_10, color="tab:green",
                label="Modelo 31 px de lado")
plot_valid_bins(bin_14, mae_14, color="tab:orange",
                label="Modelo 25 px de lado")

# ---- Axes & style ----
plt.xlabel("Altura real en Z (nm)")
plt.ylabel("Error medio absoluto (nm)")
plt.ylim(0, 1000)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
