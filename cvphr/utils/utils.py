import os
import re
import cv2
import json
import math
import warnings
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
from pathlib import Path
from decimal import Decimal
from typing import Literal, Tuple
from collections.abc import Mapping, Iterable
from matplotlib.font_manager import FontProperties
import sys
import logging

from config.base_info import (
    proj_dir,
    PATCH_SIZE, 
    UNI_PIXEL, 
    N_BLOCK,
)


# Suppress matplotlib font lookup warnings
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# Create a wrapper class to filter stderr and suppress font warnings
class FilteredStderr:
    """Stderr wrapper for filtering font warnings"""
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        # Copy all attributes
        for attr in ['buffer', 'encoding', 'errors', 'line_buffering', 'closefd']:
            if hasattr(original_stderr, attr):
                try:
                    setattr(self, attr, getattr(original_stderr, attr))
                except:
                    pass
    
    def write(self, text):
        # Filter out font-related warnings
        if isinstance(text, str) and ('findfont' in text.lower() or 'font family' in text.lower()):
            return len(text)  # Return length, pretend to write
        return self.original_stderr.write(text)
    
    def flush(self):
        if hasattr(self.original_stderr, 'flush'):
            return self.original_stderr.flush()
    
    def writelines(self, lines):
        # Filter font warnings
        filtered_lines = [line for line in lines if not (isinstance(line, str) and ('findfont' in line.lower() or 'font family' in line.lower()))]
        if filtered_lines:
            return self.original_stderr.writelines(filtered_lines)
    
    def __getattr__(self, name):
        # Forward all other attributes to original stderr
        return getattr(self.original_stderr, name)

# Install global filter to suppress font warnings
_original_stderr = sys.stderr
if not hasattr(sys.stderr, '_is_filtered'):
    _filtered_stderr = FilteredStderr(_original_stderr)
    _filtered_stderr._is_filtered = True
    sys.stderr = _filtered_stderr


def get_safe_font(font_name='Helvetica', fallback_fonts=['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'], size=9):
    # Use font fallback list, matplotlib will automatically select the first available font
    font_list = [font_name] + fallback_fonts
    return FontProperties(family=font_list, size=size)

def set_safe_font(font_name='Helvetica', fallback_fonts=['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'], size=9):
    # Use font fallback list, matplotlib will automatically select the first available font
    font_list = [font_name] + fallback_fonts
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = font_list
    plt.rcParams['font.size'] = size
    plt.rcParams['axes.unicode_minus'] = False


"""****************************************************************************
*                                                                             *
*                       posreg file and data process                          *
*                                                                             *
****************************************************************************"""

def convert_to_json_serializable(obj):
    """Recursively convert non-serializable Python objects to JSON-serializable format"""
    # Return basic types directly
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # Handle numpy numeric types
    elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Multi-dimensional arrays will be flattened to lists

    # Handle pandas types
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()

    # Handle datetime
    elif isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
        return obj.isoformat()

    # Handle Decimal (commonly used in financial data)
    elif isinstance(obj, Decimal):
        return float(obj)

    # Handle dict/mapping types
    elif isinstance(obj, Mapping):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}

    # Handle iterable objects (exclude strings)
    elif isinstance(obj, Iterable) and not isinstance(obj, str):
        return [convert_to_json_serializable(v) for v in obj]

    # Handle sets
    elif isinstance(obj, set):
        return list(obj)

    # Handle custom objects (try using __dict__)
    elif hasattr(obj, '__dict__'):
        return convert_to_json_serializable(obj.__dict__)

    # Other cases: convert to string or raise exception
    else:
        try:
            return str(obj)
        except Exception as e:
            raise ValueError(f"Object of type {type(obj)} is not JSON serializable: {obj}") from e

def read_json_file(file_path):
    """
    Read JSON file and return parsed Python dict
    """
    # Ensure path is Path object
    json_path = Path(file_path)
    
    # Check if file exists
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    # Check if it's a file
    if not json_path.is_file():
        raise IsADirectoryError(f"Specified path is not a file: {json_path}")
    
    # Read and parse JSON file
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}") from e
    
def record_model_info_2dict(configure, raw_model, criterion, optimizer, scheduler):
    """
    Record basic info of model, loss function, optimizer and training
    """
    # ------------------ 1. Record model architecture hyperparameters ------------------
    # Put some commonly used key attributes in a whitelist first
    basic_attr_keys = [
        "model_name",
        "backbone_name",
        "is_pretrained",
        "freeze_backbone",
        "partial_unfreeze",
        "backbone_out_dim",
        "feature_dim",
        "num_clusters",
        "coord_enc_dims",
        "regressor_dims",
        "reduction_ratio",
        "add_patch_coord",
        "use_psg",
        "use_ca",
        "flag_training",
    ]

    model_arch = {}
    for key in basic_attr_keys:
        if hasattr(raw_model, key):
            model_arch[key] = getattr(raw_model, key)
        else:
            # Explicitly mark as "nokv" when old models don't have this attribute
            model_arch[key] = "nokv"

    # Allow model to declare additional attributes to record:
    # e.g., add self.__model_config_keys__ = ["grid_size", "ca_num_heads"] in PHAR_fineca
    extra_keys = getattr(raw_model, "__model_config_keys__", [])
    for key in extra_keys:
        if hasattr(raw_model, key):
            model_arch[key] = getattr(raw_model, key)

    # Automatically scan __dict__ and record all "simple type" attributes (put in extra_hyperparams)
    simple_types = (int, float, bool, str, list, tuple, dict)
    extra_hparams = {}
    for name, value in raw_model.__dict__.items():
        # Skip private attributes and already recorded keys
        if name.startswith("_"):
            continue
        if name in model_arch:
            continue
        # Skip modules / parameters / Tensors
        if isinstance(value, (nn.Module, nn.Parameter, torch.Tensor)):
            continue
        # Record all other simple types
        if isinstance(value, simple_types):
            extra_hparams[name] = value

    if extra_hparams:
        model_arch["extra_hyperparams"] = extra_hparams

    configure["model_architecture"] = model_arch

    # ------------------ 2. Record network structure (layers) ------------------
    layers_info = {}

    # Record model name
    layers_info["model_name"] = getattr(raw_model, "model_name", type(raw_model).__name__)

    # Record all direct submodules (backbone, feature_extractor, spatial_proj, spatial_pool, token_coord_encoder, regressor, etc.)
    # named_children only traverses the first layer children of current Module, no infinite recursion (more suitable for structure overview)
    for name, module in raw_model.named_children():
        layers_info[name] = str(module)

    # Compatible with previous habit of putting backbone / coord_encoder directly at configure["layers"] top level
    # If some modules are dynamically provided through __getattr__, can also supplement manually:
    for possible_name in ["backbone", "feature_extractor", "coord_encoder", "csmg"]:
        if hasattr(raw_model, possible_name) and possible_name not in layers_info:
            layers_info[possible_name] = str(getattr(raw_model, possible_name))

    configure["layers"] = layers_info

    # ------------------ 3. Record cross attention module config specifically ------------------
    if hasattr(raw_model, "neighbors_cross_attn"):
        attn_module = raw_model.neighbors_cross_attn
        attn_type = type(attn_module).__name__

        attn_cfg = {"type": attn_type}

        # Use hasattr to judge before getattr for safety
        for key in ["feat_dim", "reduced_dim", "num_heads", "head_dim",
                    "num_neighbors", "attn_dropout", "use_neighbor_pos_emb"]:
            if hasattr(attn_module, key):
                attn_cfg[key] = getattr(attn_module, key)

        configure["layers"]["neighbors_cross_attn_config"] = attn_cfg

    # ------------------ 4. Record loss function info ------------------
    configure["criterion"] = {
        "class": type(criterion).__name__,
        "config": getattr(criterion, "__dict__", {}),
    }

    # ------------------ 5. Record optimizer and scheduler config ------------------
    opt_defaults = getattr(optimizer, "defaults", {})
    configure["optimizer_config"] = {
        "class": type(optimizer).__name__,
        "lr": opt_defaults.get("lr", None),
        "betas": opt_defaults.get("betas", None),
        "weight_decay": opt_defaults.get("weight_decay", None),
    }

    sch_name = type(scheduler).__name__
    if "ReduceLROnPlateau" in sch_name:
        configure["scheduler_config"] = {
            "class": sch_name,
            "mode": scheduler.mode,
            "patience": scheduler.patience,
            "factor": scheduler.factor,
            "min_lr": scheduler.min_lrs[0]
            if isinstance(scheduler.min_lrs, list)
            else scheduler.min_lrs,
        }
    elif "CosineAnnealingLR" in sch_name:
        configure["scheduler_config"] = {
            "class": sch_name,
            "T_max": scheduler.T_max,
            "eta_min": scheduler.eta_min,
        }
    else:
        # For other schedulers, simply record class name to avoid KeyError
        configure["scheduler_config"] = {"class": sch_name}

    return configure

def ckpt_load(checkpoint_data, history, model, optimizer, scheduler, device, resume_path, num_epochs, result_dir):
    """
    Load checkpoint and restore training state
    """
    if checkpoint_data and 'history' in checkpoint_data:
        loaded_history = checkpoint_data['history']
        for key in history.keys():
            if key in loaded_history:
                history[key] = loaded_history[key]
        for key, value in loaded_history.items():
            if key not in history:
                history[key] = value

    # Initialize best checkpoint
    best_checkpoint = {
        'epoch': -1,
        'val_loss': float('inf'),
        'model_state_dict': None,
        'optimizer_state_dict': None,
        'scheduler_state_dict': None,
    }

    start_epoch = 0
    if checkpoint_data:
        # Restore model, optimizer, scheduler state
        model_state = checkpoint_data.get('model_state_dict')
        if model_state is None:
            raise ValueError(f"Checkpoint {resume_path} missing model_state_dict, cannot resume training.")
        model.load_state_dict(model_state)

        optimizer_state = checkpoint_data.get('optimizer_state_dict')
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
            # Move optimizer state to target device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        else:
            warnings.warn("Checkpoint missing optimizer_state_dict, will continue training with new optimizer state.")

        scheduler_state = checkpoint_data.get('scheduler_state_dict')
        if scheduler_state is not None:
            try:
                scheduler.load_state_dict(scheduler_state)
            except Exception as exc:
                warnings.warn(f"Failed to restore scheduler state: {exc}")
        start_epoch = checkpoint_data.get('epoch', -1) + 1
        best_checkpoint['epoch'] = checkpoint_data.get('best_epoch', checkpoint_data.get('epoch', best_checkpoint['epoch']))
        best_checkpoint['val_loss'] = checkpoint_data.get('best_val_loss', checkpoint_data.get('val_loss', best_checkpoint['val_loss']))
        if checkpoint_data.get('best_model_state_dict') is not None:
            best_checkpoint['model_state_dict'] = checkpoint_data['best_model_state_dict']
        else:
            best_checkpoint['model_state_dict'] = model_state
        if checkpoint_data.get('best_optimizer_state_dict') is not None:
            best_checkpoint['optimizer_state_dict'] = checkpoint_data['best_optimizer_state_dict']
        elif optimizer_state is not None:
            best_checkpoint['optimizer_state_dict'] = optimizer_state
        if checkpoint_data.get('best_scheduler_state_dict') is not None:
            best_checkpoint['scheduler_state_dict'] = checkpoint_data['best_scheduler_state_dict']
        elif scheduler_state is not None:
            best_checkpoint['scheduler_state_dict'] = scheduler_state
        print(f" - Resuming training from epoch {start_epoch} (target total epochs {num_epochs})")
        if start_epoch >= num_epochs:
            warnings.warn(f"Checkpoint epoch({start_epoch}) has reached or exceeded target epochs({num_epochs}), only evaluation phase will be executed later.")
        best_model_path = Path(result_dir) / 'best_model.pth'
        if not best_model_path.exists() and best_checkpoint['model_state_dict'] is not None:
            # best_model.pth only saves model parameters for subsequent inference/deployment
            torch.save({'model_state_dict': best_checkpoint['model_state_dict']}, best_model_path)
            print(" - Resaved best_model.pth based on checkpoint info (only contains model_state_dict)")
    
    return start_epoch, best_checkpoint, history


"""****************************************************************************
*                                                                             *
*                       posreg model visualization                            *
*                                                                             *
****************************************************************************"""

def training_curve(history, result_dir):
    """Plot simple training and validation loss curves and lr curve"""            
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['lr_history'], label='Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{result_dir}/training_curve.jpg')
    plt.close()

def training_mae_curve_par(history, result_dir, best_checkpoint, loss_type, val_errors_pos, val_errors_dir, val_errors_alt=None):
    """Plot training and validation loss(best pth), lr, mae distribution
    For PAR use, can plot histogram of position and direction error distribution, and loss curves of position and direction"""    
    plt.figure(figsize=(15, 5))
    
    # 1 Loss curves""""""
    plt.subplot(1, 3, 1)
    # 2 loss component curves
    plt.plot(history['train_loss_pos'], label='TLoss Pos', color='royalblue', linewidth=1, linestyle='--', marker='o', markersize=3, alpha=0.6)
    plt.plot(history['val_loss_pos'], label='VLoss Pos', color='skyblue', linewidth=1, linestyle='--', marker='^', markersize=3, alpha=0.6)
    plt.plot(history['train_loss_dir'], label='TLoss Dir', color='red', linewidth=1, linestyle='--', marker='o', markersize=3, alpha=0.6)
    plt.plot(history['val_loss_dir'], label='VLoss Dir', color='coral', linewidth=1, linestyle='--', marker='^', markersize=3, alpha=0.6)
    if val_errors_alt is not None:
        plt.plot(history['train_loss_alt'], label='TLoss Alt', color='olive', linewidth=1, linestyle='--', marker='o', markersize=3, alpha=0.6)
        plt.plot(history['val_loss_alt'], label='VLoss Alt', color='y', linewidth=1, linestyle='--', marker='^', markersize=3, alpha=0.6)
    # Weighted loss curve
    plt.plot(history['train_loss'], label='Train Loss', color='green', linewidth=2)
    plt.plot(history['val_loss'], label='Val Loss', color='orange', linestyle='--',linewidth=2)

    if best_checkpoint['epoch'] != -1:  # Ensure best epoch exists
        best_val_loss = best_checkpoint["val_loss"]
        plt.scatter(best_checkpoint['epoch'], best_checkpoint['val_loss'], 
                   color='red', s=100, label=f'Best Val Loss: {best_val_loss:.4f}')

    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel(loss_type)
    plt.legend()
    plt.grid(True)
    
    # 2 Learning rate changes
    plt.subplot(1, 3, 2)
    plt.plot(history['lr_history'], color='green', linewidth=2)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    
    # 3 Error distribution histogram
    plt.subplot(1, 3, 3)
    plt.hist(val_errors_pos, bins=30, color='purple', alpha=0.5)
    plt.hist(val_errors_dir, bins=30, color='orange', alpha=0.6)
    if val_errors_alt is not None:
        plt.hist(val_errors_alt, bins=30, color='skyblue', alpha=0.7)
    plt.title('Validation Error Distribution')
    plt.xlabel('Absolute Error')
    plt.ylabel('Count')
    if val_errors_alt is not None:
        plt.legend(['Position', 'Direction', 'Altitude'])
    else:
        plt.legend(['Position', 'Direction'])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{result_dir}/training_analysis.jpg', dpi=300, bbox_inches='tight')
    plt.close()
    return best_val_loss



"""****************************************************************************
*                                                                             *
*                              Vis PAR test results                           *
*                                                                             *
****************************************************************************"""

def test_plot_offset_par(result_dir, test_id='', show_title=False, fontsize=9):
    """
    Offset vector plot (pred vs GT) + direction vector plot
    """
    # Set font to Helvetica (with fallback)
    set_safe_font('Helvetica', size=fontsize)
    font_prop = get_safe_font('Helvetica', size=fontsize)

    results_df = pd.read_csv(f'{result_dir}/test_results{test_id}.csv')
    block_centers_x = results_df['block_x'] * PATCH_SIZE + PATCH_SIZE
    block_centers_y = results_df['block_y'] * PATCH_SIZE + PATCH_SIZE
    if max(results_df['x_norm']) > 0.7:
        _SIZE = UNI_PIXEL
    else:
        _SIZE = PATCH_SIZE
    gt_global_x = (block_centers_x + results_df['x_norm'] * _SIZE).astype(int)
    gt_global_y = (block_centers_y + results_df['y_norm'] * _SIZE).astype(int)
    pred_global_x = (block_centers_x + results_df['x_pred'] * _SIZE).astype(int)
    pred_global_y = (block_centers_y + results_df['y_pred'] * _SIZE).astype(int)
    ground_truth = np.stack([gt_global_x, gt_global_y], axis=1)
    predictions = np.stack([pred_global_x, pred_global_y], axis=1)
    # Direction
    gt_x_cosa = results_df['x_cosa'].values * 10
    gt_y_sina = results_df['y_sina'].values * 10
    pred_x_cosa = results_df['x_cosa_pred'].values * 10
    pred_y_sina = results_df['y_sina_pred'].values * 10
    # Altitude
    if 'altitude' in results_df.columns:
        gt_altitude = results_df['altitude'].values
        pred_altitude = results_df['altitude_pred'].values
        alt_error = pred_altitude - gt_altitude
    else:
        alt_error = np.zeros(len(results_df))
    # Plotting
    plt.figure(figsize=(16, 8))
    # plt.figure(figsize=(8, 16))

    # Position offset vectors
    my_grid = [i * PATCH_SIZE for i in range(N_BLOCK+2)]
    plt.subplot(1, 2, 1)
    for x in my_grid:
        plt.axvline(x=x, color='orange', linestyle='--', alpha=0.5)
    for y in my_grid:
        plt.axhline(y=y, color='orange', linestyle='--', alpha=0.5)
    plt.scatter(ground_truth[:, 0], ground_truth[:, 1], c='blue', label='GT. Pos.', s=3, alpha=0.4)
    plt.quiver(
        ground_truth[:, 0], ground_truth[:, 1],
        predictions[:, 0] - ground_truth[:, 0],
        predictions[:, 1] - ground_truth[:, 1],
        angles='xy', scale_units='xy', scale=1, color='green', label='Pos. Err.', 
        alpha=0.7, width=0.002, headwidth=2, headlength=2
    )
    
    if 'altitude' in results_df.columns:
        # Add orange vectors for altitude error (perpendicular to position error vectors)
        for i in range(len(ground_truth)):
            # Calculate position error vector
            pos_error_x = predictions[i, 0] - ground_truth[i, 0]
            pos_error_y = predictions[i, 1] - ground_truth[i, 1]
            
            # Calculate perpendicular direction (rotate 90 degrees counterclockwise)
            perp_x = -pos_error_y
            perp_y = pos_error_x
            
            # Normalize perpendicular direction
            perp_magnitude = np.sqrt(perp_x**2 + perp_y**2)
            if perp_magnitude > 1e-10:  # Avoid division by zero
                perp_x = perp_x / perp_magnitude
                perp_y = perp_y / perp_magnitude
                
                # Magnitude of altitude error (absolute value)
                alt_error_magnitude = abs(alt_error[i])
                
                # Determine direction based on sign of altitude error
                if alt_error[i] < 0:
                    perp_x = -perp_x
                    perp_y = -perp_y
                
                # Plot orange altitude error vectors
                plt.quiver(
                    ground_truth[i, 0], ground_truth[i, 1],
                    perp_x * alt_error_magnitude, perp_y * alt_error_magnitude,
                    angles='xy', scale_units='xy', scale=1, color='orange', 
                    alpha=0.7, width=0.002, headwidth=2, headlength=2,
                    label='Alt Err' if i == 0 else ""
                )
    
    if show_title:
        plt.title('Position Offset Vectors', fontproperties=font_prop, fontsize=fontsize)
    plt.xlabel('X Pixel Coordinates', fontproperties=font_prop, fontsize=fontsize)
    plt.ylabel('Y Pixel Coordinates', fontproperties=font_prop, fontsize=fontsize)
    plt.legend(prop=font_prop, loc='lower left')
    # plt.legend(prop=font_prop, loc='upper right', bbox_to_anchor=(1.0, 1.0))
    plt.grid(True)
    # Set tick label font
    ax1 = plt.gca()
    ax1.tick_params(labelsize=fontsize)
    for label in ax1.get_xticklabels():
        label.set_fontproperties(font_prop)
    for label in ax1.get_yticklabels():
        label.set_fontproperties(font_prop)
        label.set_rotation(90)
        label.set_rotation_mode('anchor')
        label.set_horizontalalignment('right')
        label.set_verticalalignment('center')
    
    # Direction vectors
    plt.subplot(1, 2, 2)
    for x in my_grid:
        plt.axvline(x=x, color='orange', linestyle='--', alpha=0.5)
    for y in my_grid:
        plt.axhline(y=y, color='orange', linestyle='--', alpha=0.5)
        
    linewidth = 0.001
    headw = 1
    headl = 0.5
    plt.quiver(
        ground_truth[:, 0], ground_truth[:, 1],
        predictions[:, 0] - ground_truth[:, 0],
        predictions[:, 1] - ground_truth[:, 1],
        angles='xy', scale_units='xy', scale=1, color='green', label='Pos. Err.', 
        alpha=0.7, width=linewidth, headwidth=headw, headlength=headl, zorder=4
    )
    plt.scatter(ground_truth[:, 0], ground_truth[:, 1], c='blue', s=3, alpha=0.3, zorder=3)
    plt.scatter(predictions[:, 0], predictions[:, 1], c='red', s=3, alpha=0.3, zorder=3)
    
    plt.quiver(
        ground_truth[:, 0], ground_truth[:, 1],
        gt_x_cosa, gt_y_sina,
        angles='xy', scale_units='xy', scale=0.4, color='blue', label='GT. Head.',
        alpha=0.7, width=linewidth, headwidth=headw, headlength=headl, zorder=4
    )
    plt.quiver(
        predictions[:, 0], predictions[:, 1],
        pred_x_cosa, pred_y_sina,
        angles='xy', scale_units='xy', scale=0.4, color='red', label='Pred. Head.',
        alpha=0.8, width=linewidth, headwidth=headw, headlength=headl, zorder=4
    )
    if show_title:
        plt.title('Direction Vectors (cosθ, sinθ)', fontproperties=font_prop, fontsize=fontsize)
    plt.xlabel('X Pixel Coordinates', fontproperties=font_prop, fontsize=fontsize)
    plt.ylabel('Y Pixel Coordinates', fontproperties=font_prop, fontsize=fontsize)
    plt.legend(prop=font_prop, loc='lower left')
    plt.grid(True)
    # Set tick label font
    ax2 = plt.gca()
    ax2.tick_params(labelsize=fontsize)
    for label in ax2.get_xticklabels():
        label.set_fontproperties(font_prop)
    for label in ax2.get_yticklabels():
        label.set_fontproperties(font_prop)
        label.set_rotation(90)
        label.set_rotation_mode('anchor')
        label.set_horizontalalignment('right')
        label.set_verticalalignment('center')
    
    plt.tight_layout()
    # Save as JPG, PDF, SVG formats
    base_path = f'{result_dir}/error_vector_map'
    # plt.savefig(f'{base_path}.jpg', dpi=300)
    plt.savefig(f'{base_path}.pdf', bbox_inches='tight')
    # plt.savefig(f'{base_path}.svg', bbox_inches='tight')
    plt.close()

def test_plot_abserr_par(result_dir, test_id=''):
    """Error vs |x|, |y| relationship plot
    Read result_dir/test_results.csv, analyze relationship between error and absolute coordinate values
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Read data
    results_df = pd.read_csv(f'{result_dir}/test_results{test_id}.csv')

    # Add absolute value columns
    results_df["x_abs"] = results_df["x_norm"].abs()
    results_df["y_abs"] = results_df["y_norm"].abs()

    # Plot relationship between error and |x|, |y|
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.scatterplot(x="x_abs", y="abs_error", data=results_df, ax=axes[0], alpha=0.3, color='darkred')
    sns.regplot(x="x_abs", y="abs_error", data=results_df, ax=axes[0], scatter=False, color='blue')
    axes[0].set_title("Abs Error vs |x|")
    axes[0].set_xlabel("|x|")
    axes[0].set_ylabel("Absolute Error")

    sns.scatterplot(x="y_abs", y="abs_error", data=results_df, ax=axes[1], alpha=0.3, color='darkred')
    sns.regplot(x="y_abs", y="abs_error", data=results_df, ax=axes[1], scatter=False, color='blue')
    axes[1].set_title("Abs Error vs |y|")
    axes[1].set_xlabel("|y|")
    axes[1].set_ylabel("Absolute Error")

    plt.tight_layout()
    plt.savefig(f'{result_dir}/abserr_vs_norm_coord{test_id}.jpg', dpi=300)
    plt.close()

def plot_angle_error_analysis(angle_errors, result_dir, test_id='', show_title=False):
    """
    Plot angle error analysis charts for all test samples
    """
    
    # Set font for academic papers
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.unicode_minus'] = False
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    if show_title:
        fig.suptitle(f'Angle Error Analysis (Test ID: {test_id})', fontsize=14, fontweight='bold')
    
    # 1. Angle error distribution histogram
    axes[0, 0].hist(angle_errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(np.mean(angle_errors), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(angle_errors):.2f}°')
    axes[0, 0].axvline(np.median(angle_errors), color='green', linestyle='--', 
                      label=f'Median: {np.median(angle_errors):.2f}°')
    axes[0, 0].set_xlabel('Angle Error (degrees)', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    if show_title:
        axes[0, 0].set_title('Angle Error Distribution Histogram', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Angle error box plot
    axes[0, 1].boxplot(angle_errors, patch_artist=True, 
                      boxprops=dict(facecolor='lightblue', alpha=0.7))
    axes[0, 1].set_ylabel('Angle Error (degrees)', fontsize=11)
    if show_title:
        axes[0, 1].set_title('Angle Error Box Plot', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Angle error scatter plot (by sample index)
    sample_indices = np.arange(len(angle_errors))
    axes[1, 0].scatter(sample_indices, angle_errors, alpha=0.6, s=10, color='purple')
    axes[1, 0].axhline(np.mean(angle_errors), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(angle_errors):.2f}°')
    axes[1, 0].set_xlabel('Sample Index', fontsize=11)
    axes[1, 0].set_ylabel('Angle Error (degrees)', fontsize=11)
    if show_title:
        axes[1, 0].set_title('Angle Error Scatter Plot', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Cumulative distribution function
    sorted_errors = np.sort(angle_errors)
    cumulative_prob = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    axes[1, 1].plot(sorted_errors, cumulative_prob, linewidth=2, color='orange')
    axes[1, 1].axhline(0.9, color='red', linestyle='--', alpha=0.7, label='90th Percentile')
    axes[1, 1].axhline(0.95, color='green', linestyle='--', alpha=0.7, label='95th Percentile')
    axes[1, 1].set_xlabel('Angle Error (degrees)', fontsize=11)
    axes[1, 1].set_ylabel('Cumulative Probability', fontsize=11)
    if show_title:
        axes[1, 1].set_title('Angle Error Cumulative Distribution Function', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    save_path = f'{result_dir}/angle_error_analysis{test_id}.jpg'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Angle error analysis plot saved to: {save_path}")
    
    # Print statistics
    print(f"\nAngle Error Statistics:")
    print(f"  Mean: {np.mean(angle_errors):.2f}°")
    print(f"  Standard Deviation: {np.std(angle_errors):.2f}°")
    print(f"  Median: {np.median(angle_errors):.2f}°")
    print(f"  Minimum: {np.min(angle_errors):.2f}°")
    print(f"  Maximum: {np.max(angle_errors):.2f}°")
    print(f"  90th Percentile: {np.percentile(angle_errors, 90):.2f}°")
    print(f"  95th Percentile: {np.percentile(angle_errors, 95):.2f}°")
    
def plot_distance_error_analysis(distance_errors, result_dir, test_id='', show_title=False):
    """
    Plot distance error analysis charts for all test samples
    """
    # Set font for academic papers (with fallback)
    set_safe_font('Helvetica', size=9)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    if show_title:
        fig.suptitle(f'Distance Error Analysis (Test ID: {test_id})', fontsize=14, fontweight='bold')
    
    # 1. Distance error distribution histogram
    axes[0, 0].hist(distance_errors, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0, 0].axvline(np.mean(distance_errors), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(distance_errors):.2f}m')
    axes[0, 0].axvline(np.median(distance_errors), color='green', linestyle='--', 
                      label=f'Median: {np.median(distance_errors):.2f}m')
    axes[0, 0].set_xlabel('Distance Error (meters)', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    if show_title:
        axes[0, 0].set_title('Distance Error Distribution Histogram', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Distance error box plot
    axes[0, 1].boxplot(distance_errors, patch_artist=True, 
                      boxprops=dict(facecolor='lightcoral', alpha=0.7))
    axes[0, 1].set_ylabel('Distance Error (meters)', fontsize=11)
    if show_title:
        axes[0, 1].set_title('Distance Error Box Plot', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Distance error scatter plot (by sample index)
    sample_indices = np.arange(len(distance_errors))
    axes[1, 0].scatter(sample_indices, distance_errors, alpha=0.6, s=10, color='darkred')
    axes[1, 0].axhline(np.mean(distance_errors), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(distance_errors):.2f}m')
    axes[1, 0].set_xlabel('Sample Index', fontsize=11)
    axes[1, 0].set_ylabel('Distance Error (meters)', fontsize=11)
    if show_title:
        axes[1, 0].set_title('Distance Error Scatter Plot', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Cumulative distribution function
    sorted_errors = np.sort(distance_errors)
    cumulative_prob = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    axes[1, 1].plot(sorted_errors, cumulative_prob, linewidth=2, color='darkorange')
    axes[1, 1].axhline(0.9, color='red', linestyle='--', alpha=0.7, label='90th Percentile')
    axes[1, 1].axhline(0.95, color='green', linestyle='--', alpha=0.7, label='95th Percentile')
    axes[1, 1].set_xlabel('Distance Error (meters)', fontsize=11)
    axes[1, 1].set_ylabel('Cumulative Probability', fontsize=11)
    if show_title:
        axes[1, 1].set_title('Distance Error Cumulative Distribution Function', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    save_path = f'{result_dir}/distance_error_analysis{test_id}.jpg'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Distance error analysis plot saved to: {save_path}")
    
    # Print statistics
    print(f"\nDistance Error Statistics:")
    print(f"  Mean: {np.mean(distance_errors):.2f}m")
    print(f"  Standard Deviation: {np.std(distance_errors):.2f}m")
    print(f"  Median: {np.median(distance_errors):.2f}m")
    print(f"  Minimum: {np.min(distance_errors):.2f}m")
    print(f"  Maximum: {np.max(distance_errors):.2f}m")
    print(f"  90th Percentile: {np.percentile(distance_errors, 90):.2f}m")
    print(f"  95th Percentile: {np.percentile(distance_errors, 95):.2f}m")

def visualize_test_from_csv_par(result_dir, test_id='', show_samples=2000):
    """Generate visualization charts (position + direction) directly from test_results.csv"""
    # Load test results
    results_df = pd.read_csv(f'{result_dir}/test_results{test_id}.csv')

    # Automatically determine the actual display quantity
    total_samples = len(results_df)
    show_samples = min(total_samples, show_samples)
    
    # Extract position data
    gt_x = results_df['x_norm'].values
    gt_y = results_df['y_norm'].values
    pred_x = results_df['x_pred'].values
    pred_y = results_df['y_pred'].values
    # Direction
    gt_x_cosa = results_df['x_cosa'].values
    gt_y_sina = results_df['y_sina'].values
    pred_x_cosa = results_df['x_cosa_pred'].values
    pred_y_sina = results_df['y_sina_pred'].values

    sample_indices = np.arange(show_samples)

    # --- 1. Scatter comparison plot ---
    plt.figure(figsize=(16, 7))
    # Position
    plt.subplot(1, 2, 1)
    plt.scatter(gt_x, gt_y, c='blue', alpha=0.3, label='GT Pos')
    plt.scatter(pred_x, pred_y, c='red', alpha=0.3, label='Pred Pos')
    plt.title('Position Prediction Scatter')
    plt.xlabel('Normalized X')
    plt.ylabel('Normalized Y')
    plt.legend()
    plt.grid(True)
    # Direction
    plt.subplot(1, 2, 2)
    plt.scatter(gt_x_cosa, gt_y_sina, c='blue', alpha=0.3, label='GT Dir')
    plt.scatter(pred_x_cosa, pred_y_sina, c='red', alpha=0.3, label='Pred Dir')
    plt.title('Direction Prediction Scatter')
    plt.xlabel('cosθ')
    plt.ylabel('sinθ')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{result_dir}/prediction_scatter_from_csv.jpg', dpi=300)
    plt.close()

    # --- 2. Coordinate/direction change curve plot ---
    plt.figure(figsize=(18, 10))
    # Position X
    plt.subplot(2, 2, 1)
    plt.plot(sample_indices, gt_x[:show_samples], 'b--', label='GT X', alpha=0.7, linewidth=1.5)
    plt.plot(sample_indices, pred_x[:show_samples], 'r-', label='Pred X', alpha=0.7, linewidth=1)
    plt.ylabel('Normalized X')
    plt.title('X Coordinate Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    # Position Y
    plt.subplot(2, 2, 3)
    plt.plot(sample_indices, gt_y[:show_samples], 'g--', label='GT Y', alpha=0.7, linewidth=1.5)
    plt.plot(sample_indices, pred_y[:show_samples], 'm-', label='Pred Y', alpha=0.7, linewidth=1)
    plt.xlabel('Sample Index')
    plt.ylabel('Normalized Y')
    plt.title('Y Coordinate Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    # Direction cosθ
    plt.subplot(2, 2, 2)
    plt.plot(sample_indices, gt_x_cosa[:show_samples], 'b--', label='GT cosθ', alpha=0.7, linewidth=1.5)
    plt.plot(sample_indices, pred_x_cosa[:show_samples], 'r-', label='Pred cosθ', alpha=0.7, linewidth=1)
    plt.ylabel('cosθ')
    plt.title('cosθ Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    # Direction sinθ
    plt.subplot(2, 2, 4)
    plt.plot(sample_indices, gt_y_sina[:show_samples], 'g--', label='GT sinθ', alpha=0.7, linewidth=1.5)
    plt.plot(sample_indices, pred_y_sina[:show_samples], 'm-', label='Pred sinθ', alpha=0.7, linewidth=1)
    plt.xlabel('Sample Index')
    plt.ylabel('sinθ')
    plt.title('sinθ Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{result_dir}/coordinate_curves_from_csv.jpg', dpi=300)
    plt.close()

    # --- 3. Error curve plot ---
    plt.figure(figsize=(18, 7))
    # Position error
    plt.subplot(1, 2, 1)
    delta_x = pred_x - gt_x
    delta_y = pred_y - gt_y
    plt.plot(sample_indices, delta_x[:show_samples], 'b-', label='ΔX (Pred - GT)', alpha=0.7)
    plt.plot(sample_indices, delta_y[:show_samples], 'r-', label='ΔY (Pred - GT)', alpha=0.7)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
    stats_text = f"""
    X Error Stats:\nMean: {np.mean(delta_x):.4f}  Std: {np.std(delta_x):.4f}\nY Error Stats:\nMean: {np.mean(delta_y):.4f}  Std: {np.std(delta_y):.4f}
    """
    plt.annotate(stats_text, xy=(0.98, 0.15), xycoords='axes fraction', 
                ha='right', va='bottom', bbox=dict(boxstyle='round', alpha=0.1))
    plt.title('Position Prediction Error (First {} Samples)'.format(show_samples))
    plt.xlabel('Sample Index')
    plt.ylabel('Prediction Error')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    # Direction error
    plt.subplot(1, 2, 2)
    delta_cos = pred_x_cosa - gt_x_cosa
    delta_sin = pred_y_sina - gt_y_sina
    plt.plot(sample_indices, delta_cos[:show_samples], 'b-', label='Δcosθ (Pred - GT)', alpha=0.7)
    plt.plot(sample_indices, delta_sin[:show_samples], 'r-', label='Δsinθ (Pred - GT)', alpha=0.7)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
    stats_text2 = f"""
    cosθ Error Stats:\nMean: {np.mean(delta_cos):.4f}  Std: {np.std(delta_cos):.4f}\nsinθ Error Stats:\nMean: {np.mean(delta_sin):.4f}  Std: {np.std(delta_sin):.4f}
    """
    plt.annotate(stats_text2, xy=(0.98, 0.15), xycoords='axes fraction', 
                ha='right', va='bottom', bbox=dict(boxstyle='round', alpha=0.1))
    plt.title(f'Direction Prediction Error (First {show_samples} Samples, test_id {test_id})')
    plt.xlabel('Sample Index')
    plt.ylabel('Prediction Error')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{result_dir}/error_analysis_from_csv.jpg', dpi=300)
    plt.close()

    # --- 4. Error distribution plot ---
    plt.figure(figsize=(16, 6))
    bins = 50
    alpha = 0.5
    # Position error distribution
    plt.subplot(1, 2, 1)
    plt.hist(delta_x[:show_samples], bins=bins, color='blue', alpha=alpha, label='ΔX (Pred - GT)')
    plt.hist(delta_y[:show_samples], bins=bins, color='red', alpha=alpha, label='ΔY (Pred - GT)')
    stats_text = f"""
    X Error Stats:\nMean: {np.mean(delta_x):.4f}  Std: {np.std(delta_x):.4f}\nY Error Stats:\nMean: {np.mean(delta_y):.4f}  Std: {np.std(delta_y):.4f}
    """
    plt.annotate(stats_text, xy=(0.98, 0.95), xycoords='axes fraction', 
                ha='right', va='top', bbox=dict(boxstyle='round', alpha=0.1))
    plt.title('Position Error Distribution (First {} Samples)'.format(show_samples))
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    # Direction error distribution
    plt.subplot(1, 2, 2)
    plt.hist(delta_cos[:show_samples], bins=bins, color='blue', alpha=alpha, label='Δcosθ (Pred - GT)')
    plt.hist(delta_sin[:show_samples], bins=bins, color='red', alpha=alpha, label='Δsinθ (Pred - GT)')
    stats_text2 = f"""
    cosθ Error Stats:\nMean: {np.mean(delta_cos):.4f}  Std: {np.std(delta_cos):.4f}\nsinθ Error Stats:\nMean: {np.mean(delta_sin):.4f}  Std: {np.std(delta_sin):.4f}
    """
    plt.annotate(stats_text2, xy=(0.98, 0.95), xycoords='axes fraction', 
                ha='right', va='top', bbox=dict(boxstyle='round', alpha=0.1))
    plt.title('Direction Error Distribution (First {} Samples)'.format(show_samples))
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{result_dir}/error_distribution_from_csv.jpg', dpi=300)
    plt.close()

def uniform_rsi_image(rs_img):
    '''Due to different sizes of input remote sensing images and some not being integer multiples of 256
    Input: rs_img = cv2.imread(rs_image_path)
    Call method: cropped_img = uniform_rsi_image(img)
    '''
    # Verify original size
    h, w = rs_img.shape[:2]
    if h == 5128 and w == 5128 : # Version v0: rsi is fixed size 5128, need to crop to 5120
        assert (h == 5128) and (w == 5128), f"Image size should be 5128x5128, got {h}x{w}"
        
        # Calculate crop area
        crop_size = 5120  #RSI_SIZE
        start_x = (w - crop_size) // 2  # (5128-5120)//2 = 4
        start_y = (h - crop_size) // 2
        cropped_rsimg = rs_img[start_y:start_y+crop_size, start_x:start_x+crop_size]
        
        # Verify output size
        assert cropped_rsimg.shape == (5120, 5120, 3), f"Crop failed, got {cropped_rsimg.shape}"
    else:  #Version v1: remote sensing map can be any resolution (integer multiple of 256)
        assert (h % 256 == 0) and (w % 256 == 0), f" * Image size should be integer multiple of 256, got {h}x{w}"
        cropped_rsimg = rs_img
    return cropped_rsimg

def vis_mle_mhe_on_rsimage(result_dir, rsi_id, rs_image_path, test_id='', view_range=None, show_title=False, fontsize=9):
    """
    Visualize test results on original remote sensing image, and draw direction vector arrows at each point
    """
    # Set font to Helvetica (with fallback), size as fontsize
    set_safe_font('Helvetica', size=fontsize)
    font_prop = get_safe_font('Helvetica', size=fontsize)
    
    # Load data and crop from 5128 to 5120
    rs_img = cv2.imread(rs_image_path)
    assert rs_img is not None, "Image read failed, check path"

    cropped_rsimg = uniform_rsi_image(rs_img)

    cropped_rsimg = cv2.cvtColor(cropped_rsimg, cv2.COLOR_BGR2RGB)
    results_df = pd.read_csv(f'{result_dir}/test_results{test_id}.csv')
    plt.figure(figsize=(30, 30), dpi=150)
    ax = plt.gca()
    
    #Filter results_df: if 'rsi_id' column exists, filter by rsi_id, else no filter
    if 'rsi_id' in results_df.columns:
        results_df = results_df[results_df['rsi_id'] == rsi_id].reset_index(drop=True)
    
    # Precompute global coordinates (vectorized operation for performance)
    block_centers_x = results_df['block_x'] * PATCH_SIZE + PATCH_SIZE
    block_centers_y = results_df['block_y'] * PATCH_SIZE + PATCH_SIZE
    if max(results_df['x_norm']) > 0.7:
        _SIZE = UNI_PIXEL
    else:
        _SIZE = PATCH_SIZE
    gt_global_x = (block_centers_x + results_df['x_norm'] * _SIZE).astype(int)
    gt_global_y = (block_centers_y + results_df['y_norm'] * _SIZE).astype(int)
    pred_global_x = (block_centers_x + results_df['x_pred'] * _SIZE).astype(int)
    pred_global_y = (block_centers_y + results_df['y_pred'] * _SIZE).astype(int)
    # Direction
    gt_x_cosa = results_df['x_cosa'].values
    gt_y_sina = results_df['y_sina'].values
    pred_x_cosa = results_df['x_cosa_pred'].values
    pred_y_sina = results_df['y_sina_pred'].values
    target_paths = results_df['target_path'] 

    # If alt_pred and altitude_error columns exist, calculate altitude error vector related info
    if 'alt' in results_df.columns and 'alt_pred' in results_df.columns and 'altitude_error' in results_df.columns:
        gt_alt = results_df['alt'].values
        pred_alt = results_df['alt_pred'].values
        altitude_error = results_df['altitude_error'].values
        # Calculate altitude error vector
        point_alt_error_x, point_alt_error_y = calc_alt_err_vec_batch(
            gt_global_x, gt_global_y, 
            pred_global_x, pred_global_y, 
            gt_alt, pred_alt, output="point")

    def extract_target_patch_id(filepath):
        """
        Extract numbers from file path and connect with dots
        """
        # Get filename (without path)
        filename = os.path.basename(filepath)
        
        # Extract all numbers using regular expression
        numbers = re.findall(r'\d+', filename)
        
        # Connect numbers with dots
        result = '.'.join(numbers)
        
        return result

    # 1. Draw image first (background)
    height, width = cropped_rsimg.shape[:2]
    ax.imshow(cropped_rsimg, zorder=1, alpha=0.6)
    
    # 2. Set grid lines and ticks in PATCH_SIZE units (above image, below points)
    x_ticks = list(range(0, width + 1, PATCH_SIZE))
    y_ticks = list(range(0, height + 1, PATCH_SIZE))
    
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.grid(True, alpha=0.8, zorder=2, color='white')  # Grid lines above image, below points, white color
    ax.set_ylim(height, 0)  # Y-axis ticks from height to 0 top to bottom
    
    # 3. Draw all connection lines (translucent)
    for i in range(len(results_df)):
        ax.plot([gt_global_x[i], pred_global_x[i]], 
                [gt_global_y[i], pred_global_y[i]],
                color='green', linewidth=2, alpha=0.7, zorder=3)
        
        # If alt_pred and altitude_error columns exist, draw altitude error vector
        if 'alt' in results_df.columns and 'alt_pred' in results_df.columns and 'altitude_error' in results_df.columns:
            ax.plot([gt_global_x[i], point_alt_error_x[i]], 
                    [gt_global_y[i], point_alt_error_y[i]],
                    color='orange', linewidth=2, alpha=0.7, zorder=3)

            # Format to integer (0 decimal places)
            alt_err = pred_alt[i] - gt_alt[i]
            alterr_text = f"{int(round(alt_err))}"
            # Calculate midpoint of altitude error vector
            mid_alterr_x = (gt_global_x[i] + point_alt_error_x[i]) / 2
            mid_alterr_y = (gt_global_y[i] + point_alt_error_y[i]) / 2
            
            # Draw distance_error (light green), center point at midpoint of connection line
            light_orange = (1.0, 0.6471, 0.0)  # Orange RGB
            ax.text(
                x=mid_alterr_x,     # Text center x coordinate (midpoint)
                y=mid_alterr_y,     # Text center y coordinate (midpoint)
                s=alterr_text,                 # Text to draw
                fontsize=9, # Font size 9pt  fontsize=font_scale*10, # Font size (unchanged)
                fontproperties=font_prop,  # Use Helvetica font
                color=light_orange,       # Text color (orange)
                weight='normal',  # Font weight
                ha='center',              # Horizontal center alignment
                va='center',            # Vertical center alignment
                transform=ax.transData,  # Coordinate system transformation
                zorder=5  # Text on top layer
            )

        if 0:  # # Add text on image, draw id text info
            tpid = extract_target_patch_id(target_paths[i])

            # Define text content and parameters
            text = tpid #str(node['id'])  # Text to display
            font = cv2.FONT_HERSHEY_SIMPLEX  # Font type
            font_scale = 0.7  # Font size
            text_color = (1.0, 1.0, 1.0)  # Text color (BGR format) white
            text_thickness = 1  # Text line width
            text_position = (gt_global_x[i], gt_global_y[i] + 20)  # Calculate text position (top left of rectangle) offset by certain distance
            # cv2.putText(img_anim, text, text_position, font, font_scale, text_color, text_thickness)
            ax.text(
                x=text_position[0],     # Text bottom left x coordinate
                y=text_position[1],     # Text bottom left y coordinate (note y-axis direction opposite to OpenCV)
                s=text,                 # Text to draw
                fontsize=font_scale*10, # Font size (adjust scale factor)
                color=text_color,       # Text color (RGB string or tuple)
                weight='bold' if text_thickness>1 else 'normal',  # Font weight
                ha='left',              # Horizontal alignment
                va='bottom',            # Vertical alignment
                transform=ax.transData,  # Coordinate system transformation
                zorder=4  # Text on top layer
            )    
            
        if 1:  # # Add text on image, draw distance_error and angle_error results
            # Read distance_error and angle_error from CSV
            distance_error = results_df['distance_error'].iloc[i]
            angle_error = results_df['angle_error'].iloc[i]
            
            # Format to integer (0 decimal places)
            dist_text = f"{int(round(distance_error))}"
            angle_text = f"{int(round(angle_error))}"
            
            font_scale = 0.9  # Font size
            
            # Calculate midpoint of line between prediction point and ground truth
            mid_x = (gt_global_x[i] + pred_global_x[i]) / 2
            mid_y = (gt_global_y[i] + pred_global_y[i]) / 2
            
            # Draw distance_error (light green), center point at midpoint of connection line
            light_green = (0.5, 1.0, 0.5)  # Light green RGB
            ax.text(
                x=mid_x,     # Text center x coordinate (midpoint)
                y=mid_y,     # Text center y coordinate (midpoint)
                s=dist_text,                 # Text to draw
                fontsize=font_scale*10, # Font size 9pt  fontsize=font_scale*10, # Font size (unchanged)
                fontproperties=font_prop,  # Use Helvetica font
                color=light_green,       # Text color (light green)
                weight='normal',  # Font weight
                ha='center',              # Horizontal center alignment
                va='center',            # Vertical center alignment
                transform=ax.transData,  # Coordinate system transformation
                zorder=5  # Text on top layer
            )
            
            # Draw angle_error (light purple), at root of red prediction direction point (prediction point position)
            light_purple = (0.8, 0.5, 1.0)  # Light purple RGB
            ax.text(
                x=pred_global_x[i],     # Prediction point x coordinate
                y=pred_global_y[i],     # Prediction point y coordinate
                s=angle_text,                 # Text to draw
                fontsize=font_scale*10, # Font size 9pt  fontsize=font_scale*10, # Font size (unchanged)
                fontproperties=font_prop,  # Use Helvetica font
                color=light_purple,       # Text color (light purple)
                weight='normal',  # Font weight
                ha='left',              # Horizontal alignment (to the right of prediction point)
                va='center',            # Vertical center alignment
                transform=ax.transData,  # Coordinate system transformation
                zorder=5  # Text on top layer
            ) 
    
    # 4. Draw ground truth points (smaller points)
    ax.scatter(gt_global_x, gt_global_y, 
              c='blue', s=6, alpha=1, 
              edgecolors=(0.68, 0.85, 0.9, 0.8),  # Light blue with 70% transparency ,edgecolors='lightgreen'  # Light green
              linewidths=1.5, 
              marker='o',
              zorder=4  # Points above grid lines
              )  # Remove stroke to reduce memory
    
    # 5. Draw prediction points (smaller points)
    ax.scatter(pred_global_x, pred_global_y, 
              c='red', s=6, alpha=1,
              edgecolors=(1.0, 0.3, 0.3, 0.5),  # Dark red with 50% transparency (R=1, G/B=0.3, alpha 0.5)
              linewidths=1.5, 
              marker='o',
              zorder=4  # Points above grid lines
              )
    # 6. Draw direction vector arrows
    arrow_scale = 40  # Keep original arrow length
    # Ground truth direction (blue arrow)
    ax.quiver(gt_global_x, gt_global_y, gt_x_cosa, gt_y_sina, angles='xy', scale_units='xy', 
              scale=1/arrow_scale, color='blue', alpha=0.5, width=0.002, headwidth=1, headlength=2, zorder=3)
    # Prediction direction (red arrow)
    ax.quiver(pred_global_x, pred_global_y, pred_x_cosa, pred_y_sina, angles='xy', scale_units='xy', 
              scale=1/arrow_scale, color='red', alpha=0.5, width=0.002, headwidth=1, headlength=2, zorder=4)
    
    # 7. Set axis labels and title
    if show_title:
        ax.set_title(f"{rsi_id} Test Results Visualization ({len(results_df)} Samples)", 
                    fontsize=9, fontproperties=font_prop, fontweight='bold')
    ax.set_xlabel('X Pixel Coordinates', fontsize=9, fontproperties=font_prop)
    ax.set_ylabel('Y Pixel Coordinates', fontsize=9, fontproperties=font_prop)
    
    # Set tick label font
    ax.tick_params(labelsize=9)
    for label in ax.get_xticklabels():
        label.set_fontproperties(font_prop)
    for label in ax.get_yticklabels():
        label.set_fontproperties(font_prop)
    
    # 8. Set display range
    if view_range is not None:
        # If specified range provided, use it
        if len(view_range) == 4:
            x_min, x_max, y_min, y_max = view_range
            # Limit to image range
            x_min = max(0, x_min)
            x_max = min(width, x_max)
            y_min = max(0, y_min)
            y_max = min(height, y_max)
            ax.set_xlim(x_min, x_max)
            # Note coordinate axis is reversed, so ymax first
            ax.set_ylim(y_max, y_min)
        else:
            print(f"Warning: view_range parameter format error, should be [x_min, x_max, y_min, y_max], will use auto crop")
            view_range = None  # Set to None, use auto crop
    
    if view_range is None:
        # Auto crop to data area (refer to _plot_ccs_subplot_for_blocks implementation)
        # Collect all point coordinates for auto scaling
        all_xs = list(gt_global_x) + list(pred_global_x)
        all_ys = list(gt_global_y) + list(pred_global_y)
        
        if all_xs and all_ys:
            margin = 40  # Adjustable: leave some blank space, less crowded
            x_min, x_max = min(all_xs), max(all_xs)
            y_min, y_max = min(all_ys), max(all_ys)
            
            # Limit to image range
            x_min = max(0, x_min - margin)
            x_max = min(width, x_max + margin)
            y_min = max(0, y_min - margin)
            y_max = min(height, y_max + margin)
            
            ax.set_xlim(x_min, x_max)
            # Note coordinate axis is reversed, so ymax first
            ax.set_ylim(y_max, y_min)
    
    # Save ultra-high resolution image (may require large memory) - save as JPG, PDF, SVG formats
    base_path = f'{result_dir}/rsi_{rsi_id}_mle_mhe_vis'
    plt.savefig(f'{base_path}.jpg', bbox_inches='tight', dpi=150)
    # plt.savefig(f'{base_path}.pdf', bbox_inches='tight')
    # plt.savefig(f'{base_path}.svg', bbox_inches='tight')
    plt.close()
    
    print(f"Full visualization results saved to: {base_path}.jpg, {base_path}.pdf, {base_path}.svg")
    print(f"Total {len(results_df)} sample points drawn")



"""****************************************************************************
*                                                                             *
*                       gpt-c Training Monitor Tools                          *
*                                                                             *
****************************************************************************"""

def init_gcheck_dir(result_dir):
    """
    Initialize monitoring directory, create result_dir/gcheck subdirectory
    """
    gcheck_dir = os.path.join(result_dir, 'gcheck')  # ♥ gpt-c generate path 'results/xxx/gcheck'
    os.makedirs(gcheck_dir, exist_ok=True)           # ♥ gpt-c create directory if not exists
    return gcheck_dir

def record_prediction_and_save_plots(pos_pred, dir_pred, coords, agl_coords, epoch, history, gcheck_dir,
                                     max_points: int = 2000,
                                     always_plot_first_n: int = 3,
                                     plot_interval: int = 10):
    """
    Record prediction statistics and save scatter plots, including mean/std of position and direction predictions, and save images
    """
    with torch.no_grad():
        # Record mean/std
        pos_mean = pos_pred.mean().item()
        pos_std = pos_pred.std().item()
        dir_mean = dir_pred.mean().item()
        dir_std = dir_pred.std().item()
        history.setdefault('pos_pred_stat', []).append({
            'epoch': epoch + 1,
            'mean': pos_mean,
            'std': pos_std
        })
        history.setdefault('dir_pred_stat', []).append({
            'epoch': epoch + 1,
            'mean': dir_mean,
            'std': dir_std
        })

        # Plot and save (reduce frequency + subsample points)
        ep = epoch + 1
        # Plot only for first N epochs or at specified intervals to avoid time cost per epoch
        should_plot = (ep <= always_plot_first_n) or (ep % plot_interval == 0)
        if should_plot:
            # Subsample: use max_points samples for scatter plot at most
            num_samples = pos_pred.shape[0]
            if num_samples > max_points:
                idx = torch.randperm(num_samples, device=pos_pred.device)[:max_points]
                pos_vis = pos_pred[idx]
                dir_vis = dir_pred[idx]
                coords_vis = coords[idx]
                agl_coords_vis = agl_coords[idx]
            else:
                pos_vis = pos_pred
                dir_vis = dir_pred
                coords_vis = coords
                agl_coords_vis = agl_coords

            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            # Position scatter plot (after subsampling)
            axes[0].scatter(pos_vis[:, 0].cpu(), pos_vis[:, 1].cpu(), alpha=0.5, label="Pred", color='blue')
            axes[0].scatter(coords_vis[:, 0].cpu(), coords_vis[:, 1].cpu(), alpha=0.5, label="True", color='red')
            axes[0].set_xlim(-1.1, 1.1)
            axes[0].set_ylim(-1.1, 1.1)
            axes[0].set_title(f'Position Prediction vs GT (Epoch {epoch+1})')
            axes[0].legend()
            axes[0].grid(True)

            # Direction scatter plot (after subsampling)
            axes[1].scatter(dir_vis[:, 0].cpu(), dir_vis[:, 1].cpu(), alpha=0.5, label="Pred", color='green')
            axes[1].scatter(agl_coords_vis[:, 0].cpu(), agl_coords_vis[:, 1].cpu(), alpha=0.5, label="True", color='orange')
            axes[1].set_xlim(-1.1, 1.1)
            axes[1].set_ylim(-1.1, 1.1)
            axes[1].set_title(f'Direction Prediction vs GT (Epoch {epoch+1})')
            axes[1].legend()
            axes[1].grid(True)

            # Save image
            filename = os.path.join(gcheck_dir, f'pos_dir_vs_gt_epoch{epoch+1:03d}.jpg')
            plt.tight_layout()
            plt.savefig(filename)
            plt.close()
    return history

# Calculate mean of grad_norm_stat for current epoch
def compute_grad_norm_stat_mean(history, epoch):
    """
    Calculate mean of grad_norm_stat only for input epoch, return history.
    """
    stats = history.get('grad_norm_stat', [])
    # Get only current epoch's stats
    epoch_stats = [s for s in stats if s.get('epoch', -1) == epoch+1]
    
    if not epoch_stats:
        return history
    keys = [k for k in epoch_stats[0] if k != 'epoch']
    mean_stat = {'epoch': epoch+1}
    for k in keys:
        mean_stat[k] = float(np.mean([s[k] for s in epoch_stats if k in s]))
    history.setdefault('grad_norm_stat_epoch', []).append(mean_stat)
    return history

def record_gradient_norms(model, history, epoch):
    """
    Record gradient norms of regressor module parameters
    """
    grad_stats = {'epoch': epoch + 1}
    for name, param in model.named_parameters():
        if 'regressor' in name and param.grad is not None:
            grad_stats[name] = param.grad.norm().item()
    history.setdefault('grad_norm_stat', []).append(grad_stats)
    return history

def save_monitor_history(history, gcheck_dir):
    """
    Save training monitoring info as JSON file
    """
    serializable_history = convert_to_json_serializable(history)
    with open(os.path.join(gcheck_dir, 'training_monitor.json'), 'w') as f:
        json.dump(serializable_history, f, indent=2)


"""****************************************************************************
*                                                                             *
*                           posreg diagnos tools                              *
*                                                                             *
****************************************************************************"""

def diagnose_val_batch(result_dir, batch_idx, epoch, patches, coords, agl_coords, pos_pred, dir_pred, loss_pos, loss_dir, loss):
    """Validation phase Loss diagnoser module
    """
    print(f"\n[Debug][{result_dir}] --------")
    print(f"\n[Debug][Epoch {epoch+1} | Batch {batch_idx}] --------")
    
    def print_validity(tensor, name):
        """ Check if tensor contains NaN (Not a Number) or Inf (Infinity) values, True means NaN at that position """
        print(f"{name}: shape={tuple(tensor.shape)}, NaN={torch.isnan(tensor).any().item()}, Inf={torch.isinf(tensor).any().item()}")

    print_validity(patches, "patches")
    print_validity(coords, "coords")
    print_validity(agl_coords, "agl_coords")
    print_validity(pos_pred, "pos_pred")
    print_validity(dir_pred, "dir_pred")
    
    if loss is not None:
        print(f"loss: {loss.item():.6f} | pos_loss: {loss_pos.item():.6f}, dir_loss: {loss_dir.item():.6f}")
        if loss.item() == 0.0:
            print("⚠️ Warning: loss is 0.0!")
    else:
        print("❌ loss is None")

    # Save image for manual analysis
    if torch.isnan(pos_pred).any() or torch.isnan(dir_pred).any():
        save_dir = f"{result_dir}/diagnostics/bad_epoch{epoch+1}_batch{batch_idx}"
        os.makedirs(save_dir, exist_ok=True)
        torchvision.utils.save_image(patches[0].cpu(), os.path.join(save_dir, "bad_patch.jpg"))
        print(f"Saved bad patch image to {save_dir}/bad_patch.jpg")

class MultiTaskLoss(nn.Module):
    def __init__(self, pos_weight=2.0, dir_weight=0.5):
        super().__init__()
        self.pos_criterion = nn.SmoothL1Loss()
        # self.pos_criterion = nn.MSELoss()
        self.dir_criterion = nn.CosineEmbeddingLoss()
        self.pos_weight = pos_weight
        self.dir_weight = dir_weight
        
    def forward(self, pos_pred, pos_true, dir_pred, dir_true):
        # Position loss: SmoothL1
        pos_loss = self.pos_criterion(pos_pred, pos_true)
        
        # Direction loss: cosine similarity (ensure direction consistency)
        # Create target vector (all samples similar)
        target = torch.ones(dir_true.size(0)).to(dir_true.device)
        dir_loss = self.dir_criterion(dir_pred, dir_true, target)
        
        # Dynamically adjust weights
        total_loss = self.pos_weight * pos_loss + self.dir_weight * dir_loss
        return total_loss, pos_loss, dir_loss


"""****************************************************************************
*                                                                             *
*                       uav navigation use function                           *
*                                                                             *
****************************************************************************"""

def vis_waypoints_uavtrajs_on_fig_v4(
    waypoints,
    uav_step,
    csv_path_uav_p2p_nav_records,
    uav_2d3d="2D",
    lang="en",                       # "en" or "cn"
    show_dense_annotations=None,      # None=default by language(en=True, cn=False)
    figsize=(12, 10),
):
    """
    Unified version of vis_waypoints_uavtrajs_on_fig_v4 + vis_waypoints_uavtrajs_on_fig_v4_cn

    Args:
        waypoints: [(lon, lat), ...]
        uav_step: step size in meters
        csv_path_uav_p2p_nav_records: csv path
        uav_2d3d: "2D"/"3D"
        lang: "en" or "cn"
        cn_font_path: SimHei path for Chinese (only used when lang="cn")
        show_dense_annotations: English version previously had many texts; Chinese simplified version omitted them.
            - None: default is (lang=="en")
            - True/False: force enable/disable
        figsize: matplotlib figure size
    """

    if show_dense_annotations is None:
        show_dense_annotations = (lang == "en")

    # ======== language config ========

    def _label(text_en, text_cn):
        return text_cn if lang == "cn" else text_en

    labels = {
        "waypoints": _label("Waypoints", "Hangdian"),
        "wp_line": _label("Waypoints Line", "Hangdian Lianxian"),
        "blk_center": _label("Block Center", "Huanjing Kuaiqu Zhongxin"),
        "traj_real": _label("Real Trajectory", "Shiji Feixing Guiji"),
        "cur_real": _label("Cur Points Real", "Zhenshi Weizhi"),
        "cur_name": _label("Cur Points Name", "Mingyi Weizhi"),
        "cur_pred": _label("Cur Points Pred", "Yuce Weizhi Dian"),
        "step_vec": _label("Step Vector", "Bujin Shiliang"),
        "pred_dir": _label("Pred Direction", "Yuce Hangxiang Jiao"),
        "xlabel": _label("Longitude", "Jingdu"),
        "ylabel": _label("Latitude", "Weidu"),
    }

    # ======== data ========
    wp_lons = [p[0] for p in waypoints]
    wp_lats = [p[1] for p in waypoints]
    df = pd.read_csv(csv_path_uav_p2p_nav_records)

    # ======== plot setup ========
    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")

    # +----+
    # | 01 | Waypoints
    # +----+
    ax.scatter(wp_lons, wp_lats, color="purple", marker="*", s=30 if lang == "cn" else 20,
               label=labels["waypoints"], alpha=0.5)
    ax.plot(wp_lons, wp_lats, color="m", linestyle="--", linewidth=1,
            label=labels["wp_line"], alpha=0.5)

    # dense waypoint annotations (EN original had it; CN simplified omitted)
    if show_dense_annotations:
        for idx, (lon, lat) in enumerate(waypoints):
            ax.text(lon - 0.00005, lat + 0.00005, str(idx),
                    color="m", fontsize=6, alpha=0.6, weight="bold")
            ax.text(lon - 0.0002, lat + 0.0001, f"({lon:.6f},\n{lat:.6f}))",
                    color="violet", fontsize=4, rotation=-45, alpha=0.6)

    # +----+
    # | 02 | Blocks
    # +----+
    earth_radius = 6371000
    block_size_m = 72

    def meters_to_degrees(lat, distance_m):
        lat_deg = distance_m / (earth_radius * 2 * math.pi) * 360
        lon_deg = lat_deg / math.cos(math.radians(lat))
        return lon_deg, lat_deg

    for _, (lon, lat, row, col) in df[["block_center_lon", "block_center_lat", "block_row", "block_col"]].iterrows():
        lon_span, lat_span = meters_to_degrees(lat, block_size_m)
        points = [
            (lon - lon_span / 2, lat - lat_span / 2),
            (lon + lon_span / 2, lat - lat_span / 2),
            (lon + lon_span / 2, lat + lat_span / 2),
            (lon - lon_span / 2, lat + lat_span / 2),
            (lon - lon_span / 2, lat - lat_span / 2),
        ]
        ax.plot(*zip(*points), color="yellowgreen", linestyle=":", linewidth=0.8, alpha=0.5)

        # EN original had block coord annotation; CN simplified omitted
        if show_dense_annotations:
            ax.text(
                lon - lon_span / 2,
                lat + lat_span / 2,
                f"({lon:.6f},\n{lat:.6f}),\n{int(col)}-{int(row)})",
                color="grey",
                fontsize=4,
                verticalalignment="top",
                horizontalalignment="right",
            )

    ax.plot(
        df["block_center_lon"], df["block_center_lat"],
        color="orange", linestyle="-.", marker="s", markersize=4, linewidth=1,
        label=labels["blk_center"], alpha=0.5
    )

    # +----+
    # | 03 | Trajectory / Points
    # +----+
    ax.plot(df["cur_lon_real"], df["cur_lat_real"], color="red", linestyle="-", linewidth=1,
            label=labels["traj_real"], alpha=0.5)

    ax.scatter(df["cur_lon_real"], df["cur_lat_real"], color="red",
               facecolors="none", edgecolors="red", s=6, marker="o",
               label=labels["cur_real"], alpha=0.8)
    ax.scatter(df["cur_lon_name"], df["cur_lat_name"], color="blue",
               s=6, marker="x", label=labels["cur_name"], alpha=0.8)
    ax.scatter(df["cur_lon_pred"], df["cur_lat_pred"], color="forestgreen",
               facecolors="none", edgecolors="forestgreen", s=6, marker="^",
               label=labels["cur_pred"], alpha=0.8)

    # relationship lines
    for i in range(len(df)):
        real_x, real_y = df["cur_lon_real"].iloc[i], df["cur_lat_real"].iloc[i]
        pred_x, pred_y = df["cur_lon_pred"].iloc[i], df["cur_lat_pred"].iloc[i]
        ax.plot([real_x, pred_x], [real_y, pred_y], color="green", linestyle=":", linewidth=0.5, alpha=0.5)

        blk_cx, blk_cy = df["block_center_lon"].iloc[i], df["block_center_lat"].iloc[i]
        ax.plot([real_x, blk_cx], [real_y, blk_cy], color="orange", linestyle="-.", linewidth=0.5, alpha=0.6)

        if i >= 1:
            next_lon_name, next_lat_name = df["next_lon_name"].iloc[i - 1], df["next_lat_name"].iloc[i - 1]
            ax.plot([real_x, next_lon_name], [real_y, next_lat_name],
                    color="blue", linestyle="-.", linewidth=0.5, alpha=0.4)

    # EN original had real-point index annotation; CN simplified omitted
    if show_dense_annotations:
        for idx, (lon, lat) in df[["cur_lon_real", "cur_lat_real"]].iterrows():
            ax.text(lon + 0.00003, lat - 0.00005, str(idx), color="salmon", fontsize=4)

    # cyan helper line (pred -> next nominal)
    for i in range(len(df)):
        ideal_x, ideal_y = df["cur_lon_pred"].iloc[i], df["cur_lat_pred"].iloc[i]
        nx, ny = df["next_lon_name"].iloc[i], df["next_lat_name"].iloc[i]
        ax.plot([ideal_x, nx], [ideal_y, ny], color="cyan", linestyle="--", linewidth=0.5, alpha=0.2)

    # +----+
    # | 04 | Directions (quiver) + arrow legend
    # +----+
    arrow_scale = 0.0001
    arrow_width = 0.001
    arrow_headwidth = 2
    arrow_headlength = 2

    # quiver arrows (do NOT add to legend; we add custom arrow legend later)
    if "ref_direct_llcs_x" in df.columns and "ref_direct_llcs_y" in df.columns:
        for i in range(len(df)):
            ref_x = df["ref_direct_llcs_x"].iloc[i]
            ref_y = df["ref_direct_llcs_y"].iloc[i]
            px = df["cur_lon_pred"].iloc[i]
            py = df["cur_lat_pred"].iloc[i]
            ax.quiver(px, py, ref_x, ref_y,
                      angles="xy", scale_units="xy", scale=1 / arrow_scale,
                      color="skyblue", alpha=0.8, width=arrow_width,
                      headwidth=arrow_headwidth, headlength=arrow_headlength,
                      label="_nolegend_")

    if "direct_pred_llcs_x" in df.columns and "direct_pred_llcs_y" in df.columns:
        for i in range(len(df)):
            dx = 0.6 * df["direct_pred_llcs_x"].iloc[i]
            dy = 0.6 * df["direct_pred_llcs_y"].iloc[i]
            rx = df["cur_lon_real"].iloc[i]
            ry = df["cur_lat_real"].iloc[i]
            # CN version used slightly different scale; keep it
            scale_val = (0.6 / arrow_scale) if lang == "cn" else (1 / arrow_scale)
            ax.quiver(rx, ry, dx, dy,
                      angles="xy", scale_units="xy", scale=scale_val,
                      color="forestgreen", alpha=0.8, width=arrow_width,
                      headwidth=arrow_headwidth, headlength=arrow_headlength,
                      label="_nolegend_")

    # ======== labels/title ========
    plt.xlabel(labels["xlabel"], fontsize=12)
    plt.ylabel(labels["ylabel"], fontsize=12)
    filename_ = csv_path_uav_p2p_nav_records.split("/")[-1]
    city_rsi_traj = filename_.split("_uav_")[0]
    plt.title(f"UAV trajectory of {city_rsi_traj} at step {uav_step} meter {uav_2d3d}.",
                fontsize=14, pad=20)

    # ======== legend: keep all existing items + add arrow-shaped handles ========
    from matplotlib.legend_handler import HandlerPatch
    import matplotlib.patches as mpatches

    class HandlerArrow(HandlerPatch):
        def create_artists(self, legend, orig_handle,
                           xdescent, ydescent, width, height, fontsize, trans):
            p = mpatches.FancyArrow(
                0, 0.3 * height, width, 0,
                length_includes_head=True,
                head_width=0.3 * height,
                head_length=0.2 * width,
                fc=orig_handle.get_facecolor(),
                ec=orig_handle.get_edgecolor(),
                alpha=orig_handle.get_alpha()
            )
            self.update_prop(p, orig_handle, legend)
            p.set_transform(trans)
            return [p]

    # collect existing legend items (from scatter/plot etc.)
    handles, legend_labels = ax.get_legend_handles_labels()

    # append arrow legend items
    ref_arrow = mpatches.FancyArrow(0, 0, 1, 0, fc="skyblue", ec="skyblue", alpha=0.8, linewidth=2)
    pred_arrow = mpatches.FancyArrow(0, 0, 1, 0, fc="forestgreen", ec="forestgreen", alpha=0.8, linewidth=2)
    handles.extend([ref_arrow, pred_arrow])
    legend_labels.extend([labels["step_vec"], labels["pred_dir"]])

    # final legend
    ax.legend(handles, legend_labels,
                handler_map={mpatches.FancyArrow: HandlerArrow()},
                loc="best", fontsize=14)

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    language = '' if lang=='en' else '_cn'
    out_path = os.path.splitext(csv_path_uav_p2p_nav_records)[0] + language + ".jpg"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Flight process visualization image saved to: {out_path}")
    plt.close()

def convert_ccs_to_llcs_vector(
    dir_pred_ccs: tuple[float, float], 
    current_lat: float
) -> tuple[float, float]:
    """
    Convert direction vector from Cartesian coordinate system to longitude-latitude coordinate system, considering latitude's impact on vertical axis.
    """
    # Convert latitude from degrees to radians
    lat_rad = math.radians(current_lat)
    
    # Calculate scale factor for latitude direction
    scale_factor = math.cos(lat_rad)
    
    # Apply scale factor
    dx_llcs = dir_pred_ccs[0]  # Longitude direction (east-west) remains unchanged
    dy_llcs = dir_pred_ccs[1] * scale_factor  # Latitude direction (north-south) scaled by cos(latitude)

    
    return normalize_vector([dx_llcs, dy_llcs])

def convert_llcs_to_ccs_vector(
    dir_pred_llcs: tuple[float, float], 
    current_lat: float
) -> tuple[float, float]:
    """
    Convert direction vector from longitude-latitude coordinate system to Cartesian coordinate system, considering latitude's impact on longitude direction.
    """
    # Convert latitude from degrees to radians
    lat_rad = math.radians(current_lat)
    
    # Calculate scale factor for latitude direction
    scale_factor = math.cos(lat_rad)
    
    # Apply scale factor
    dx_ccs = dir_pred_llcs[0]  # Longitude direction (east-west) remains unchanged
    dy_ccs = dir_pred_llcs[1] / scale_factor  # Latitude direction (north-south) scaled by 1/cos(latitude)
    
    return normalize_vector([dx_ccs, dy_ccs])

def calculate_lon_lat_step_underllcs(
    cur_point_name: tuple[float, float],  # Current longitude-latitude (lon, lat)
    ref_direct_llcs: tuple[float, float],  # Meter-based direction vector [dlon, dlat], latitude scaling considered
    uav_step: float                        # Step length forward (meters)
) -> tuple[float, float]:
    """
    Directly calculate step angle values under llcs longitude-latitude coordinate system.
    """
    lon0, lat0 = cur_point_name
    dlon_m, dlat_m = ref_direct_llcs  # Note: not "meter-based unit vector"

    # Current latitude (radians)
    lat_rad = math.radians(lat0)

    # 1 degree latitude ≈ 111320 meters
    meters_per_deg_lat = 111320

    # 1 degree longitude ≈ 111320 × cos(latitude) meters
    # meters_per_deg_lon = 111320 * math.cos(lat_rad)
    meters_per_deg_lon = 111320   #ref_direct_llcs is unit vector, no need to consider latitude

    # Step components in direction (meters)
    move_lon_m = uav_step * dlon_m
    move_lat_m = uav_step * dlat_m

    # Convert to angle units respectively
    lon_step = move_lon_m / meters_per_deg_lon
    lat_step = move_lat_m / meters_per_deg_lat

    return lon_step, lat_step

def vector2angle(dx, dy):    
    # Convert direction vector to angle (0-360 degrees), note order of dx and dy
    angle_rad = math.atan2(dy, dx)
    angle = math.degrees(angle_rad)
    angle = angle + 360 if angle < 0 else angle
    return angle

def angle2vector(theta):
    theta_rad = theta * np.pi / 180  # Manual conversion to radians
    x_cosa = np.cos(theta_rad)
    y_sina = np.sin(theta_rad)
    return (x_cosa, y_sina)

def normalize_vector(vector):
    """
    Input a 2D vector [a, b], return corresponding unit vector [u, v]
    """
    a, b = vector
    magnitude = math.sqrt(a**2 + b**2)
    
    if magnitude == 0:
        raise ValueError("Cannot normalize zero vector")
    
    u = a / magnitude
    v = b / magnitude
    
    return [u, v]


"""****************************************************************************
*                                                                             *
*                       uav navigation use function                           *
*                                                                             *
****************************************************************************"""
def calc_alt_err_vec_batch(
    gt_x, gt_y,
    pred_x, pred_y,
    gt_alt, pred_alt,
    *,
    output: Literal["vec", "point"] = "vec",
    eps: float = 1e-12
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assume point A is (gt_global_x[i],gt_global_y[i]),
    point B is (pred_global_x[i],pred_global_y[i]),
    vector vec = (pred_alt-gt_alt);
    calc_alt_err_vec calculates end point C starting from A, perpendicular to AB direction 
    based on input variables, where absolute value of vec is length of AC segment. 
    When vec is positive, perpendicular direction is AB rotated 90 degrees counterclockwise;
    when vec is negative, perpendicular direction is AB rotated 90 degrees clockwise.
    Function returns vector calculation form, x-axis positive to right, y-axis positive downward.
    Image coordinate system: x right positive, y down positive.
    """
    # Unified conversion to numpy array and specify float type
    gt_x = np.asarray(gt_x, dtype=float)
    gt_y = np.asarray(gt_y, dtype=float)
    pred_x = np.asarray(pred_x, dtype=float)
    pred_y = np.asarray(pred_y, dtype=float)
    gt_alt = np.asarray(gt_alt, dtype=float)
    pred_alt = np.asarray(pred_alt, dtype=float)

    # Verify input shape consistency
    if not (gt_x.shape == gt_y.shape == pred_x.shape == pred_y.shape == gt_alt.shape == pred_alt.shape):
        raise ValueError(
            "All inputs must have the same shape, got: "
            f"{gt_x.shape}, {gt_y.shape}, {pred_x.shape}, {pred_y.shape}, {gt_alt.shape}, {pred_alt.shape}"
        )

    # 1. Calculate basic parameters
    alt_err = pred_alt - gt_alt  # Altitude error (scalar)
    mag = 4 * np.abs(alt_err)    # Length of AC segment, multiplied by 4 for display effect on map
    dx = pred_x - gt_x           # AB vector x component
    dy = pred_y - gt_y           # AB vector y component
    norm = np.hypot(dx, dy)      # Magnitude of AB vector
    valid = norm > eps           # Valid AB vector flag (avoid division by 0)

    # 2. Calculate unit vector u of AB direction
    ux = np.zeros_like(dx)
    uy = np.zeros_like(dy)
    ux[valid] = dx[valid] / norm[valid]
    uy[valid] = dy[valid] / norm[valid]

    # 3. Core optimization: calculate rotated unit vector first, then scale
    # 3.1 Calculate unit vector of AB rotated 90 degrees
    # In screen coordinate system: 90 degrees counterclockwise rotation (ux, uy) → (uy, -ux)
    rot_ccw_ux, rot_ccw_uy = uy, -ux  # Unit vector of 90 degrees counterclockwise rotation
    # In screen coordinate system: 90 degrees clockwise rotation (ux, uy) → (-uy, ux)
    rot_cw_ux, rot_cw_uy = -uy, ux    # Unit vector of 90 degrees clockwise rotation

    # 3.2 Select rotated unit vector based on alt_err sign
    pos_mask = alt_err > 0  # Mask for positive alt_err
    vx_unit = np.where(pos_mask, rot_ccw_ux, rot_cw_ux)  # Rotated x-direction unit vector
    vy_unit = np.where(pos_mask, rot_ccw_uy, rot_cw_uy)  # Rotated y-direction unit vector

    # 3.3 Scale unit vector by mag to get AC vector
    vx = vx_unit * mag
    vy = vy_unit * mag

    # 4. Handle degenerate case (A≈B, AB direction undefined)
    # Reference AB direction as horizontal right: 90 degrees counterclockwise is up (y negative), 90 degrees clockwise is down (y positive)
    vx[~valid] = 0.0
    vy[~valid] = -alt_err[~valid]  # Equivalent to: alt_err>0 → -mag (up), alt_err<0 → +mag (down)

    # 5. Return result based on output type
    if output == "vec":
        return vx, vy
    elif output == "point":
        cx = gt_x + vx  # C point x coordinate = A point x + AC vector x component
        cy = gt_y + vy  # C point y coordinate = A point y + AC vector y component
        return cx, cy
    else:
        raise ValueError(f"Invalid output type: {output}, must be 'vec' or 'point'")


if __name__ == "__main__":
    pass    