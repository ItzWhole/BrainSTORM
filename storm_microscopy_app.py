#!/usr/bin/env python3
"""
STORM Microscopy Astigmatic Height Regression Application

A clean, modular application for processing TIFF stacks, detecting peaks,
and training neural networks to predict z-heights from PSF appearance.

Author: ItzWhole (restructured from original research code)
"""

import os
import sys
import argparse
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional, Union
import logging

# Add storm_core to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from storm_core.data_processing import (
    find_tiff_files, crop_and_sum_stack, extract_psf_cutouts, normalize_0_to_1
)
from storm_core.neural_network import (
    build_astigmatic_psf_network, train_val_split_by_group, 
    build_augmenter, make_dataset, setup_callbacks
)
from storm_core.evaluation import (
    rescale_01_to_nm, plot_true_vs_pred_heatmap, plot_random_psfs
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class STORMConfig:
    """Configuration class for STORM microscopy analysis"""
    
    def __init__(self):
        # Default parameters
        self.distance = 18  # Cutout size (distance x distance)
        self.start_z = 0
        self.end_z = 161
        self.csum_slices = 30
        self.min_distance = 5
        self.prominence_sigma = 10.0
        self.support_radius = 2
        self.step_nm = 25.0
        self.base_nm = -2000.0
        self.batch_size = 64
        self.epochs = 100
        self.learning_rate = 1e-3
        self.val_split = 0.2
        
        # WSL paths (will be set dynamically)
        self.data_path = None
        self.output_path = None
        self.model_path = None

class STORMTrainer:
    """Main training class for STORM microscopy analysis"""
    
    def __init__(self, config: STORMConfig):
        self.config = config
        
    def load_training_data(self, train_files: List[Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load and process training data from TIFF files"""
        all_cutouts = []
        all_heights = []
        all_group_ids = []
        
        for i, tiff_path in enumerate(train_files):
            logger.info(f"Processing training file {i+1}/{len(train_files)}: {tiff_path.name}")
            
            # Load TIFF stack
            stack = tiff.imread(tiff_path)
            
            # Create summed image for peak detection
            csum_image = crop_and_sum_stack(
                stack, self.config.start_z, self.config.end_z, self.config.csum_slices
            )
            
            # Extract PSF cutouts
            cutouts, group_ids, peaks = extract_psf_cutouts(
                stack, csum_image, self.config.distance,
                min_distance=self.config.min_distance,
                prominence_sigma=self.config.prominence_sigma,
                support_radius=self.config.support_radius,
                start=self.config.start_z,
                end=self.config.end_z,
                plot=True
            )
            
            # Process cutouts
            psfs = np.array([cutout[0] for cutout in cutouts])
            heights = np.array([cutout[1] for cutout in cutouts])
            
            # Adjust group IDs to be globally unique
            adjusted_group_ids = np.array(group_ids) + len(all_cutouts)
            
            all_cutouts.extend(psfs)
            all_heights.extend(heights)
            all_group_ids.extend(adjusted_group_ids)
        
        # Convert to arrays and normalize
        X = np.expand_dims(np.array(all_cutouts), axis=-1)
        y = np.array(all_heights)
        group_ids = np.array(all_group_ids)
        
        # Normalize data
        X = normalize_0_to_1(X)
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
        
        logger.info(f"Loaded {len(X)} training samples from {len(train_files)} files")
        return X, y, group_ids
    
    def train_model(self, X: np.ndarray, y: np.ndarray, group_ids: np.ndarray):
        """Train the neural network model"""
        
        # Split data by groups
        (X_train, y_train), (X_val, y_val) = train_val_split_by_group(
            X, y, group_ids, val_size=self.config.val_split
        )
        
        # Build model
        input_shape = (self.config.distance + 1, self.config.distance + 1, 1)
        model = build_astigmatic_psf_network(input_shape)
        
        # Setup data augmentation
        augmenter = build_augmenter(self.config.distance)
        
        # Create datasets
        train_ds = make_dataset(X_train, y_train, 
                              batch_size=self.config.batch_size, 
                              training=True, augmenter=augmenter)
        val_ds = make_dataset(X_val, y_val, 
                            batch_size=self.config.batch_size, 
                            training=False)
        
        # Setup callbacks
        model_path = self.config.output_path / f"model_distance_{self.config.distance}.keras"
        callbacks = setup_callbacks(str(model_path))
        
        # Train model
        logger.info("Starting model training...")
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.config.epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info(f"Training completed. Model saved to: {model_path}")
        return model, history

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description='STORM Microscopy Height Regression')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing TIFF files')
    parser.add_argument('--output-dir', type=str, default='./output',
                       help='Output directory for results')
    parser.add_argument('--distance', type=int, default=18,
                       help='Cutout size (default: 18)')
    parser.add_argument('--train', action='store_true',
                       help='Train new model')
    parser.add_argument('--predict', type=str,
                       help='Path to TIFF file for prediction')
    parser.add_argument('--model-path', type=str,
                       help='Path to saved model')
    parser.add_argument('--train-indices', nargs='+', type=int, default=[20, 17],
                       help='Indices of files to use for training (default: 20 17)')
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = STORMConfig()
    config.distance = args.distance
    config.data_path = Path(args.data_dir)
    config.output_path = Path(args.output_dir)
    config.output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"STORM Microscopy Application Started")
    logger.info(f"Data directory: {config.data_path}")
    logger.info(f"Output directory: {config.output_path}")
    
    # Find TIFF files
    tiff_files = find_tiff_files(config.data_path)
    
    if args.train:
        logger.info("Training mode selected")
        
        # Select training files based on indices
        train_files = []
        for idx in args.train_indices:
            if idx < len(tiff_files):
                train_files.append(tiff_files[idx])
                logger.info(f"Selected training file {idx}: {tiff_files[idx].name}")
        
        if not train_files:
            logger.error("No valid training files found")
            return
        
        # Initialize trainer and train model
        trainer = STORMTrainer(config)
        X, y, group_ids = trainer.load_training_data(train_files)
        model, history = trainer.train_model(X, y, group_ids)
        
    elif args.predict:
        logger.info(f"Prediction mode: {args.predict}")
        # Prediction pipeline will be implemented
        pass
    else:
        logger.info("Available TIFF files:")
        for i, tiff_file in enumerate(tiff_files):
            logger.info(f"  {i}: {tiff_file}")
        logger.info("Use --train or --predict to run analysis")
        parser.print_help()

if __name__ == "__main__":
    main()