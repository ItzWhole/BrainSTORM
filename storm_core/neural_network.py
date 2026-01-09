"""
Neural Network Models for STORM Microscopy

Contains the astigmatic PSF regression network and training utilities.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import GroupShuffleSplit
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def build_astigmatic_psf_network(input_shape: Tuple[int, int, int]) -> keras.Model:
    """
    Build neural network for astigmatic PSF height regression
    
    Args:
        input_shape: Input shape (height, width, channels)
        
    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=input_shape)
    
    # Initial processing
    x = layers.Conv2D(64, (5, 5), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    # Multi-path feature extraction blocks
    for filters in [128, 256, 512]:
        # Path 1: Standard 3x3 convolution
        p1 = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
        
        # Path 2: Horizontal emphasis (3x5 kernel)
        p2 = layers.ZeroPadding2D(((1, 1), (2, 2)))(x)
        p2 = layers.Conv2D(filters, (3, 5), padding='valid', activation='relu')(p2)
        
        # Path 3: Vertical emphasis (5x3 kernel)  
        p3 = layers.ZeroPadding2D(((2, 2), (1, 1)))(x)
        p3 = layers.Conv2D(filters, (5, 3), padding='valid', activation='relu')(p3)
        
        # Crop all paths to same size
        target_size = x.shape[1:3]
        p1 = layers.CenterCrop(target_size[0], target_size[1])(p1)
        p2 = layers.CenterCrop(target_size[0], target_size[1])(p2)
        p3 = layers.CenterCrop(target_size[0], target_size[1])(p3)
        
        # Concatenate paths
        x = layers.concatenate([p1, p2, p3])
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    # Prediction head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(1)(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.Huber(delta=0.06),
        metrics=[keras.metrics.MeanAbsoluteError(name='mae')]
    )
    
    logger.info(f"Built astigmatic PSF network with input shape {input_shape}")
    return model

def train_val_split_by_group(X: np.ndarray, 
                           y: np.ndarray, 
                           group_ids: np.ndarray,
                           val_size: float = 0.2,
                           random_state: int = 42) -> Tuple[Tuple[np.ndarray, np.ndarray], 
                                                          Tuple[np.ndarray, np.ndarray]]:
    """
    Split data by groups to prevent data leakage
    
    Args:
        X: Feature array
        y: Target array  
        group_ids: Group identifiers
        val_size: Validation split fraction
        random_state: Random seed
        
    Returns:
        (X_train, y_train), (X_val, y_val)
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    train_idx, val_idx = next(splitter.split(X, y, groups=group_ids))
    
    logger.info(f"Split data: {len(train_idx)} training, {len(val_idx)} validation samples")
    
    return (X[train_idx], y[train_idx]), (X[val_idx], y[val_idx])

def build_augmenter(distance: int) -> keras.Sequential:
    """
    Build data augmentation pipeline
    
    Args:
        distance: Cutout size for scaling augmentation parameters
        
    Returns:
        Keras Sequential augmentation model
    """
    return keras.Sequential([
        layers.RandomTranslation(
            height_factor=5/distance, 
            width_factor=3/distance, 
            fill_mode='reflect'
        ),
        layers.RandomRotation(
            factor=6/180, 
            fill_mode='reflect'
        ),
    ], name="data_augmentation")

def make_dataset(X: np.ndarray,
                y: np.ndarray, 
                batch_size: int = 64,
                training: bool = False,
                augmenter: Optional[keras.Sequential] = None,
                shuffle_buf: int = 2048) -> tf.data.Dataset:
    """
    Create TensorFlow dataset with optional augmentation
    
    Args:
        X: Feature array
        y: Target array
        batch_size: Batch size
        training: Whether this is training data
        augmenter: Data augmentation model
        shuffle_buf: Shuffle buffer size
        
    Returns:
        TensorFlow Dataset
    """
    ds = tf.data.Dataset.from_tensor_slices((
        X.astype('float32'), 
        y.astype('float32')
    ))
    
    if training:
        ds = ds.shuffle(min(len(X), shuffle_buf), reshuffle_each_iteration=True)
        
        if augmenter is not None:
            ds = ds.map(
                lambda img, target: (augmenter(img, training=True), target),
                num_parallel_calls=tf.data.AUTOTUNE
            )
    
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    logger.info(f"Created dataset: batch_size={batch_size}, training={training}")
    return ds

def setup_callbacks(model_path: str, patience: int = 25) -> list:
    """
    Setup training callbacks
    
    Args:
        model_path: Path to save best model
        patience: Patience for learning rate reduction
        
    Returns:
        List of Keras callbacks
    """
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            model_path, 
            monitor='val_mae', 
            save_best_only=True, 
            mode='min'
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_mae', 
            factor=0.5, 
            patience=patience, 
            min_lr=1e-6
        ),
    ]
    
    logger.info(f"Setup callbacks with model path: {model_path}")
    return callbacks