# Bearing-UAV test module: cvphr_test.py
import os
import re
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

# static vars
from config.base_info import (
    UNI_PIXEL,
    reminder_proper_rsi_type,
    DATASET_SPLIT_RATIO,
    PH_LOSS_WEIGHT,
    rsi_type, 
    d_merge_rsis,
    get_rsi_name,
    get_rsidir_dsetdir_cityid,
    flexible_type)
# import tools
from cvphr.utils.utils import (
    convert_to_json_serializable,
    visualize_test_from_csv_par,
    vis_mle_mhe_on_rsimage,
    test_plot_offset_par,
    test_plot_abserr_par,
    plot_angle_error_analysis,
    plot_distance_error_analysis,
    read_json_file)
# import model
from cvphr.models.posaglreg.models import (
    model_kwargs_par_ca_sgm_v5a,
    par_dataloader,
    DATASET_CLASS_DICT)
###############################################
from cvphr.models.posaglreg.models import MODEL_CLASS_DICT


def extract_rsi_id_from_path(rsi_type, target_path):
    """
    Extract rsi_id (2-4 chars between underscores) after rsi_type from path
    """
    pattern = f"{re.escape(rsi_type)}_([^_]{{2,4}})_"
    match = re.search(pattern, target_path)
    return match.group(1) if match else None

def compute_vector_angle_error(dir_pred, agl_coords, return_degrees=True):
    """
    Compute minimum angle between two tensor vectors
    """
    # Ensure inputs are tensors
    if not isinstance(dir_pred, torch.Tensor):
        dir_pred = torch.tensor(dir_pred)
    if not isinstance(agl_coords, torch.Tensor):
        agl_coords = torch.tensor(agl_coords)

    # Compute dot product
    dot_product = torch.sum(dir_pred * agl_coords, dim=-1)

    # Compute vector norms
    norm_pred = torch.norm(dir_pred, p=2, dim=-1)
    norm_gt = torch.norm(agl_coords, p=2, dim=-1)

    # Compute cosine of angle, clamp to avoid numerical errors
    cos_angle = dot_product / (norm_pred * norm_gt + 1e-8)  # Add epsilon to avoid div by zero
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)  # Ensure range [-1, 1]

    # Compute angle in radians
    angle_rad = torch.acos(cos_angle)

    # Convert to degrees if needed
    if return_degrees:
        angle_error = angle_rad * 180.0 / torch.pi
    else:
        angle_error = angle_rad

    return angle_error

def MLE(pos_pred, coords, rs_image_path):
    """
    Compute metric and lat/lon errors (Mean Location Error)
    """
    # Read pixel conversion params from JSON
    rs_json_path = rs_image_path.replace(".jpg", ".json")
    try:
        rs_jdata = read_json_file(rs_json_path)
        lat_per_pixel = rs_jdata['lat_per_pixel']
        lng_per_pixel = rs_jdata['lng_per_pixel']
        latm_per_pixel = rs_jdata['latm_per_pixel']
        lngm_per_pixel = rs_jdata['lngm_per_pixel']
    except Exception as e:
        raise ValueError(f"Failed to read JSON file: {e}. Ensure file exists with required pixel conversion params.")

    # Convert pos_pred to torch.tensor
    if not isinstance(pos_pred, torch.Tensor):
        pos_pred = torch.tensor(pos_pred, dtype=torch.float32)
    # Convert coords to torch.tensor
    if not isinstance(coords, torch.Tensor):
        coords = torch.tensor(coords, dtype=torch.float32)

    # Ensure pos_pred and coords on same device
    if pos_pred.device != coords.device:
        coords = coords.to(pos_pred.device)

    pos_err = pos_pred - coords
    pos_meter_err = UNI_PIXEL * pos_err * torch.tensor([lngm_per_pixel, latm_per_pixel], device=pos_err.device)
    pos_lonlat_err = UNI_PIXEL * pos_err * torch.tensor([lng_per_pixel, lat_per_pixel], device=pos_err.device)

    distance_errors = torch.norm(pos_meter_err, p=2, dim=-1)  # Euclidean distance
    mean_distance_error = torch.mean(distance_errors).item()  # MLE_dis

    # pos_lonlat_err keeps [L, 2] format, L samples of [lon, lat]
    mean_lonlat_error = torch.mean(pos_lonlat_err, dim=0)  # Compute [lon_mean, lat_mean], shape [2]
    return distance_errors, mean_distance_error, mean_lonlat_error

def MHE(dir_pred, agl_coords):
    """
    Compute angle error (Mean Heading Error in degrees)
    """
    # Convert dir_pred to torch.tensor
    if not isinstance(dir_pred, torch.Tensor):
        dir_pred = torch.tensor(dir_pred, dtype=torch.float32)
    # Convert agl_coords to torch.tensor
    if not isinstance(agl_coords, torch.Tensor):
        agl_coords = torch.tensor(agl_coords, dtype=torch.float32)

    # Ensure dir_pred and agl_coords on same device
    if dir_pred.device != agl_coords.device:
        agl_coords = agl_coords.to(dir_pred.device)

    # Compute angle error for each sample
    angle_errors = compute_vector_angle_error(dir_pred, agl_coords, return_degrees=True)

    # Compute mean angle error
    mean_angle_error = torch.mean(angle_errors).item()

    return angle_errors, mean_angle_error

def RECALL_AT_K_PHR(pred_list, gt_list):
    """
    Compute PHR (Position/Heading Recall) via sign consistency between pred and gt
    """
    # Convert to numpy for unified processing
    if not isinstance(pred_list, np.ndarray):
        pred_list = np.array(pred_list)
    if not isinstance(gt_list, np.ndarray):
        gt_list = np.array(gt_list)

    # Ensure consistent shapes
    assert pred_list.shape == gt_list.shape, f"Shape mismatch: {pred_list.shape} vs {gt_list.shape}"
    assert len(pred_list.shape) == 2 and pred_list.shape[1] == 2, f"Expected [L, 2] format, got: {pred_list.shape}"

    num_samples = len(pred_list)
    true_count = 0

    # Iterate each sample
    for i in range(num_samples):
        pred_sub = pred_list[i]  # [2] format with two elements
        gt_sub = gt_list[i]      # [2] format with two elements

        # Compare sign consistency of two elements
        # np.sign returns: 1 for positive, -1 for negative, 0 for zero
        pred_signs = np.sign(pred_sub)  # [2] format, signs
        gt_signs = np.sign(gt_sub)      # [2] format, signs

        # Check if signs of both elements match
        sign_match = (pred_signs == gt_signs).all()

        if sign_match:
            true_count += 1

    # Compute recall
    recall_at_k = 100 * true_count / num_samples if num_samples > 0 else 0.0

    return recall_at_k, num_samples

def LSR_AT_R(distance_errors, TH_LSR=15.0):
    """
    Compute LSR@r (Localization Success Rate at radius r)
    Success = correctly recalled AND distance error < TH_LSR
    """
    # Convert to numpy for unified processing
    if not isinstance(distance_errors, np.ndarray):
        distance_errors = np.array(distance_errors)

    # Ensure 1D array
    if len(distance_errors.shape) > 1:
        distance_errors = distance_errors.flatten()

    num_samples = len(distance_errors)
    cnt = 0

    # If no pred_list and gt_list provided, only check distance error (backward compatible)
    for error in distance_errors:
        if error <= TH_LSR:
            cnt += 1

    # Compute localization success rate
    lsr_at_r = 100 * cnt / num_samples if num_samples > 0 else 0.0

    return lsr_at_r, num_samples


def test_par(dataset_dir,
             test_result_dir,
             train_result_dir,
             test_id='',
             d_rs_image_path='',
             device_id=0,
             factor_bslr=1,
             loss_type='smoothl1',
             pa_loss_weight=PH_LOSS_WEIGHT,
             model_class='',
             dataset_class='',
             model_kwargs=None,
             dataset_kwargs=None):
    """Test position regression model"""
    print("\n ......")
    print("\nTesting Best Model of sgm ... \n")
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = int(32 * factor_bslr)

    # Init model
    model_kwargs = model_kwargs or {}
    model = model_class(**model_kwargs).to(device)
    criterion = nn.SmoothL1Loss()

    # Load model weights
    # Use map_location to remap checkpoint device to current available device
    checkpoint = torch.load(f'{train_result_dir}/best_model.pth', \
                            map_location=lambda storage, loc: storage.cuda(0))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Build metadata_csv path using os.path.join
    metadata_csv = os.path.join(dataset_dir, "metadata", "metadata.csv")
    # Check file exists
    if not os.path.exists(metadata_csv):
        raise FileNotFoundError(f"Cannot find metadata file: {metadata_csv}\n"
                              f"Check dataset_dir: {dataset_dir}")
    print(f" * metadata_csv path: {metadata_csv}")

    _, _, test_loader, test_dataset = par_dataloader(metadata_csv, dataset_class, dataset_kwargs, BATCH_SIZE)

    # Test inference phase
    # Get rsi_type, use default or infer from dataset_kwargs
    if dataset_kwargs is None:
        dataset_kwargs = {}
    rsi_type = dataset_kwargs.get('rsi_type', '254k')  # Default: 254k
    if 'rsi_type' not in dataset_kwargs:
        print(f"Warning: 'rsi_type' not in dataset_kwargs, using default: {rsi_type}")
    rsi_ids = []
    test_loss = 0.0
    pred_positions = []         # Predicted relative positions (pixels) for all test samples
    pred_directions = []        # Predicted vector directions (degrees) for all test samples
    gt_positions = []           # Ground truth relative positions (pixels) for all test samples
    gt_directions = []          # Ground truth vector directions (degrees) for all test samples


    with torch.no_grad():
        flag = 0
        for batch in tqdm(test_loader, desc="Testing"):
            flag += 1
            patches = batch['patches'].to(device)
            coords = batch['coords'].to(device)
            agl_coords = batch['agl_coords'].to(device)
            target_paths = batch['target_path']      # Note: list/tuple of length B
            B = patches.size(0)

            # Predict relative position and vector direction
            pos_pred, dir_pred = model(patches)

            # Handle multi-task, single-task, and joint training modes - compute test loss
            if loss_type == 'multitask':  # Multi-task: different loss functions
                loss, loss_pos, loss_dir = criterion(
                    pos_pred, coords,
                    dir_pred, agl_coords)
            elif loss_type == 'smoothl1':  # Joint training: same loss function
                loss_pos = criterion(pos_pred, coords)
                loss_dir = criterion(dir_pred, agl_coords)
                pos_weight, dir_weight = pa_loss_weight
                loss = pos_weight * loss_pos + dir_weight * loss_dir
            elif loss_type == 'pos_smoothl1':  # Single-task: position
                loss = criterion(pos_pred, coords)
                loss_pos = loss
                loss_dir = loss
            elif loss_type == 'dir_smoothl1':  # Single-task: direction
                loss = criterion(dir_pred, agl_coords)
                loss_pos = loss
                loss_dir = loss
            else:
                raise ValueError(f"Unsupported loss type: {loss_type}")

            test_loss += loss.item() * patches.size(0)

            # Save results for visualization
            pred_positions.append(pos_pred.cpu().numpy())
            pred_directions.append(dir_pred.cpu().numpy())
            gt_positions.append(coords.cpu().numpy())
            gt_directions.append(agl_coords.cpu().numpy())

            # ====== Extract rsi_id for each sample in batch ======
            batch_rsi_ids = []
            for p in target_paths:
                rsi_id = extract_rsi_id_from_path(rsi_type, p)
                if rsi_id is None:
                    raise ValueError(f"Cannot extract rsi_id from path: {p}")
                batch_rsi_ids.append(rsi_id)

            # batch_rsi_ids length = B, aligned with batch sample count
            rsi_ids.extend(batch_rsi_ids)


    # Save test results
    pred_positions  = np.concatenate(pred_positions)
    pred_directions = np.concatenate(pred_directions)
    gt_positions    = np.concatenate(gt_positions)
    gt_directions   = np.concatenate(gt_directions)
    rsi_ids         = np.array(rsi_ids)  # Length == pred_positions.shape[0]

    test_loss = test_loss / len(test_loader.dataset)
    print(f"rsi_ids:", rsi_ids)
    print(f"Test Loss: {test_loss:.4f}")
    print(f'In test_par, rs_image_path={d_rs_image_path}')

    # Compute metrics ==========================================

    # MLE & MHE
    # For single-map datasets, can compute directly; for multi-map, need separate MLE per map
    rsi_ids_set = list(set(rsi_ids))
    # Convert rsi_ids to numpy for boolean indexing (convert once)
    rsi_ids_array = np.array(rsi_ids)
    lst_distance_errors = []
    MLE_each_rsi = []  # For cross-city datasets, stats per map (latitude affects accuracy)
    MLE_each_rsi_lonlat = []  # For cross-city datasets, stats per map
    MLE_dis_mean = 0.0
    MLE_lonlat_mean = 0.0
    for rsi_id in rsi_ids_set:
        rs_image_path = d_rs_image_path[rsi_id]
        # Extract elements from pred_positions and gt_positions where rsi_ids element == rsi_id
        # Use NumPy boolean indexing (efficient)
        mask = rsi_ids_array == rsi_id

        pred_positions_rsi = pred_positions[mask]
        gt_positions_rsi = gt_positions[mask]

        # Compute MLE only for current rsi_id samples
        lst_distance_errors_, MLE_dis_mean_, MLE_lonlat_mean_ = MLE(pred_positions_rsi, gt_positions_rsi, rs_image_path)
        MLE_dis_mean += MLE_dis_mean_

        # Convert MLE_lonlat_mean_ to numpy
        if isinstance(MLE_lonlat_mean_, torch.Tensor):
            MLE_lonlat_mean_np = MLE_lonlat_mean_.cpu().numpy()
        else:
            MLE_lonlat_mean_np = np.array(MLE_lonlat_mean_)

        # Init or accumulate MLE_lonlat_mean
        if isinstance(MLE_lonlat_mean, (int, float)) and MLE_lonlat_mean == 0.0:
            MLE_lonlat_mean = MLE_lonlat_mean_np.copy()
        else:
            MLE_lonlat_mean += MLE_lonlat_mean_np

        # Save per-rsi results (convert to serializable format)
        MLE_each_rsi.append(float(MLE_dis_mean_))
        MLE_each_rsi_lonlat.append(MLE_lonlat_mean_np.tolist())

        # Extend distance errors within loop
        if isinstance(lst_distance_errors_, torch.Tensor):
            lst_distance_errors.extend(lst_distance_errors_.cpu().numpy().tolist())
        else:
            lst_distance_errors.extend(lst_distance_errors_.tolist() if isinstance(lst_distance_errors_, np.ndarray) else lst_distance_errors_)

    # Aggregate results
    MLE_dis_mean = MLE_dis_mean / len(rsi_ids_set)
    if isinstance(MLE_lonlat_mean, np.ndarray):
        MLE_lonlat_mean = MLE_lonlat_mean / len(rsi_ids_set)
    else:
        MLE_lonlat_mean = MLE_lonlat_mean / len(rsi_ids_set) if len(rsi_ids_set) > 0 else np.array([0.0, 0.0])

    lst_angle_errors, MHE_agl_mean = MHE(pred_directions, gt_directions)
    # Recall@k
    recall_at_k, num_samples = RECALL_AT_K_PHR(pred_positions, gt_positions)
    # LSR@r - Localization success rate: correctly recalled AND distance error < threshold
    lsr_at_5, num_samples  = LSR_AT_R(lst_distance_errors, TH_LSR=5)  # Error < 5m
    lsr_at_10, num_samples = LSR_AT_R(lst_distance_errors, TH_LSR=10)  # Error < 10m
    lsr_at_15, num_samples = LSR_AT_R(lst_distance_errors, TH_LSR=15)  # Error < 15m
    lsr_at_20, num_samples = LSR_AT_R(lst_distance_errors, TH_LSR=20)  # Error < 20m
    lsr_at_25, num_samples = LSR_AT_R(lst_distance_errors, TH_LSR=25)  # Error < 25m
    lsr_at_30, num_samples = LSR_AT_R(lst_distance_errors, TH_LSR=30)  # Error < 30m
    # HSR@r - Heading success rate: correctly recalled AND angle error < threshold
    # Note: HSR may need direction vector consistency check, here only check angle error
    hsr_at_5, num_samples  = LSR_AT_R(lst_angle_errors, TH_LSR=5)  # Error < 5 deg
    hsr_at_10, num_samples = LSR_AT_R(lst_angle_errors, TH_LSR=10)  # Error < 10 deg
    hsr_at_15, num_samples = LSR_AT_R(lst_angle_errors, TH_LSR=15)  # Error < 15 deg
    hsr_at_20, num_samples = LSR_AT_R(lst_angle_errors, TH_LSR=20)  # Error < 20 deg
    hsr_at_25, num_samples = LSR_AT_R(lst_angle_errors, TH_LSR=25)  # Error < 25 deg
    hsr_at_30, num_samples = LSR_AT_R(lst_angle_errors, TH_LSR=30)  # Error < 30 deg
    print(f"Recall@k: {recall_at_k:.4f}, num_samples: {num_samples}")
    print(f"LSR@5: {lsr_at_5:.4f}, num_samples: {num_samples}")
    print(f"LSR@10: {lsr_at_10:.4f}, num_samples: {num_samples}")
    print(f"LSR@15: {lsr_at_15:.4f}, num_samples: {num_samples}")
    print(f"HSR@5: {hsr_at_5:.4f}, num_samples: {num_samples}")
    print(f"HSR@10: {hsr_at_10:.4f}, num_samples: {num_samples}")
    print(f"HSR@15: {hsr_at_15:.4f}, num_samples: {num_samples}")

    # Compute 2D mean: mean of x and y
    pos_err = pred_positions - gt_positions
    mae_pos = np.mean(np.abs(pos_err), axis=0)  # Compute [mean_x, mean_y], shape [2]

    # Compute direction error mean (2D)
    dir_err = pred_directions - gt_directions
    mae_dir = np.mean(np.abs(dir_err), axis=0)  # Compute [mean_x, mean_y], shape [2]

    # lonlat error mean (convert to numpy)
    if isinstance(MLE_lonlat_mean, torch.Tensor):
        mae_lonlat = np.abs(MLE_lonlat_mean.cpu().numpy())  # Compute abs, shape [2]
    else:
        mae_lonlat = np.abs(MLE_lonlat_mean)  # If already numpy

    print(f"MLE_lonlat: {mae_lonlat}")
    print(f"MLE_pos: {mae_pos}")
    print(f"MLE_dir: {mae_dir}")
    # Core metrics
    print("\nFinal Results:")
    print(f"MLE_dis_mean: {MLE_dis_mean:.4f}")
    print(f"MHE_agl_mean: {MHE_agl_mean:.4f}")

    # Create metrics dict
    test_mae = {
        'model_name': model.model_name,
        'model_backbone': model.backbone_name,
        'loss_type': loss_type,
        'multiloss': pa_loss_weight,
        'batch_size': BATCH_SIZE,
        'model_class': model_class.__name__,
        'dataset_class': dataset_class.__name__,
        'model_kwargs': model_kwargs,
        'dataset_kwargs': dataset_kwargs,
        'dataset': dataset_dir,
        'metadata': metadata_csv,
        'split_ratio': DATASET_SPLIT_RATIO,  #[0.85, 0.05, 0.1]
        'test_loss': round(test_loss, 4),
        'mae_pos': mae_pos.tolist() if isinstance(mae_pos, np.ndarray) else mae_pos,
        'mae_lonlat': mae_lonlat.tolist() if isinstance(mae_lonlat, np.ndarray) else mae_lonlat,
        'mae_dir': mae_dir.tolist() if isinstance(mae_dir, np.ndarray) else mae_dir,
        'MLE_each_rsi': MLE_each_rsi,
        'MLE_each_rsi_lonlat': MLE_each_rsi_lonlat,
        'mae_dis': round(MLE_dis_mean, 2),
        'mae_agl': round(MHE_agl_mean, 2),
        'recall_at_1': round(recall_at_k, 2),
        'lsr_at_5': round(lsr_at_5, 2),
        'lsr_at_10': round(lsr_at_10, 2),
        'lsr_at_15': round(lsr_at_15, 2),
        'lsr_at_20': round(lsr_at_20, 2),
        'lsr_at_25': round(lsr_at_25, 2),
        'lsr_at_30': round(lsr_at_30, 2),
        'hsr_at_5': round(hsr_at_5, 2),
        'hsr_at_10': round(hsr_at_10, 2),
        'hsr_at_15': round(hsr_at_15, 2),
        'hsr_at_20': round(hsr_at_20, 2),
        'hsr_at_25': round(hsr_at_25, 2),
        'hsr_at_30': round(hsr_at_30, 2)
    }

    # Process all test samples distance and angle error data
    # OBSERVE Dis-Agl ERROR of TEST SAMPLES ==========================================
    lst_angle_errors = np.array(lst_angle_errors)
    lst_distance_errors = np.array(lst_distance_errors)
    # Save angle error stats
    angle_stats = {
        'mean_angle_error': float(np.mean(lst_angle_errors)),
        'std_angle_error': float(np.std(lst_angle_errors)),
        'median_angle_error': float(np.median(lst_angle_errors)),
        'min_angle_error': float(np.min(lst_angle_errors)),
        'max_angle_error': float(np.max(lst_angle_errors)),
        'angle_error_90th_percentile': float(np.percentile(lst_angle_errors, 90)),
        'angle_error_95th_percentile': float(np.percentile(lst_angle_errors, 95))
    }
    distance_stats = {
        'mean_distance_error': float(np.mean(lst_distance_errors)),
        'std_distance_error': float(np.std(lst_distance_errors)),
        'median_distance_error': float(np.median(lst_distance_errors)),
        'min_distance_error': float(np.min(lst_distance_errors)),
        'max_distance_error': float(np.max(lst_distance_errors)),
    }
    # Add angle stats to test_mae dict
    test_mae.update(distance_stats)
    test_mae.update(angle_stats)

    os.makedirs(test_result_dir, exist_ok=True)
    try:  # Write JSON file
        serializable_test_mae = convert_to_json_serializable(test_mae)
        with open(f'{test_result_dir}/test_mae.json', 'w') as f:
            json.dump(serializable_test_mae, f, indent=2)
    except Exception as e:
        print(f"Error saving metrics: {e}")


    # Load original metadata, save computed results
    metadata_df = pd.read_csv(metadata_csv)

    # Ensure only process test set samples
    test_indices = list(test_dataset.indices) if 'test_dataset' in locals() else range(len(pred_positions))

    # Create results DataFrame
    results_df = metadata_df.iloc[test_indices].copy()

    # Add prediction columns
    results_df['rsi_id'] = rsi_ids
    results_df['x_pred'] = pred_positions[:, 0]
    results_df['y_pred'] = pred_positions[:, 1]
    results_df['x_cosa_pred'] = pred_directions[:, 0]
    results_df['y_sina_pred'] = pred_directions[:, 1]
    results_df['x_error'] = pred_positions[:, 0] - gt_positions[:, 0]
    results_df['y_error'] = pred_positions[:, 1] - gt_positions[:, 1]
    results_df['abs_error'] = np.sqrt(results_df['x_error']**2 + results_df['y_error']**2)
    results_df['cos_error'] = pred_directions[:, 0] - gt_directions[:, 0]
    results_df['sin_error'] = pred_directions[:, 1] - gt_directions[:, 1]
    results_df['dir_error'] = np.sqrt(results_df['cos_error']**2 + results_df['sin_error']**2)
    # OBSERVE Dis-Agl ERROR of TEST SAMPLES ==========================================
    # Add angle error to results DataFrame
    results_df['angle_error'] = lst_angle_errors
    results_df['distance_error'] = lst_distance_errors

    # Save full results
    dset_name = dataset_kwargs['dset_name']
    test_result_csv = f'{test_result_dir}/test_results{test_id}.csv'
    results_df.to_csv(test_result_csv, index=False)
    print(f"Test results saved to {test_result_csv}")

    # Save slim results (key columns only)
    slim_df = results_df[[
        'block_x', 'block_y',
        'x_norm', 'y_norm', 'x_pred', 'y_pred',
        'x_error', 'y_error', 'abs_error',
        'theta',
        'x_cosa', 'y_sina', 'x_cosa_pred', 'y_sina_pred',
        'cos_error', 'sin_error', 'dir_error'
    ]]
    slim_df.to_csv(f'{test_result_dir}/test_results{test_id}_slim.csv', index=False)

    # Visualize stats
    visualize_test_from_csv_par(test_result_dir, test_id=test_id)
    # Position regression offset vs direction vector comparison plot
    test_plot_offset_par(test_result_dir, test_id=test_id)

    # 1. Plot error heatmap (Hexbin), 2. Plot error vs |x|, |y|
    test_plot_abserr_par(test_result_dir, test_id=test_id)

    # OBSERVE Dis-Agl ERROR of TEST SAMPLES ==========================================
    # Save distance error visualization
    plot_distance_error_analysis(lst_distance_errors, test_result_dir, test_id=test_id)
    # Save angle error visualization
    plot_angle_error_analysis(lst_angle_errors, test_result_dir, test_id=test_id)

    # Visualize on remote sensing map
    if type(d_rs_image_path) is dict:
        for rsi_id, rs_image_path in d_rs_image_path.items():
            vis_mle_mhe_on_rsimage(test_result_dir, rsi_id, rs_image_path, test_id=test_id)
    else:
        print(f" ? d_rs_image_path is not a dictionary, check it.")

    print(f"\nResults of current rsi saved in: {test_result_dir}")
    return lst_distance_errors, lst_angle_errors


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--rsi_id', type=flexible_type, default=96, help="Remote sensing image ID")
    parser.add_argument('--n_block', type=int, default=15, help="Number of blocks in RS image")
    parser.add_argument('--n_sample', type=int, default=1, help="Samples per block")
    parser.add_argument('--is_3d', type=int, default=1, help="Dataset is 3D view")
    parser.add_argument('--num_epochs', type=int, default=1, help="Training epochs")
    parser.add_argument("--model_class", type=str, default="PARCASGM_v5a", help="Model class")
    parser.add_argument("--dataset_class", type=str, default="RSBlockDatasetPA_v3q", help="Dataset class")        
    parser.add_argument('--device_id', type=int, default=0, help="GPU device ID")
    parser.add_argument('--factor_bslr', type=float, default=0.5, help="Scale factor, base bs=32, lr=1e-4")
    parser.add_argument('--bestpth_dir', type=str, default='', help="Best model path")

    # Parse args
    args = parser.parse_args()
    print(args)
    # dataset params
    rsi_id = args.rsi_id  # Usually unchanged. Large RS image ID, 96=34bc,36bc,37bc,38bc
    n_sample = args.n_sample  # 1/100
    view2d3d = '3d' if args.is_3d else '2d'  # 2d/3d version
    # training params
    device_id = args.device_id
    bestpth_dir = args.bestpth_dir
    model_class = MODEL_CLASS_DICT[args.model_class]
    dataset_class = DATASET_CLASS_DICT[args.dataset_class]


    if 1:
        # Paper best model path might be outside rootdir.
        train_result_dir = f'{bestpth_dir}'

        # Select mode:
        mode = 'test'  # Ensure pth file is ready before testing
        reminder_proper_rsi_type('254k', rsi_type)  # Auto-check if rsi_type is correct

        # Model params
        model_kwargs = model_kwargs_par_ca_sgm_v5a

        test_id = rsi_id
        if rsi_id == 71:
            test_id = '71bc'
        elif rsi_id == 72:
            test_id = '72bc'
        elif rsi_id == 81:
            test_id = '81bc'
        elif rsi_id == 82:
            test_id = '82bc'
        elif rsi_id == 96:
            test_id = '96bc'

        rsi_city_dir, dset_root, dset_name, city_id = get_rsidir_dsetdir_cityid(
            rsi_id, rsi_type, n_sample, view2d3d
        )
        print(f"rsi_city_dir: {rsi_city_dir}")
        print(f"dset_root: {dset_root}")
        print(f"dset_name: {dset_name}")
        print(f"city_id: {city_id}")

        # Remote sensing image name and path
        # Auto-generate identifier from user info
        rsi_name = get_rsi_name(rsi_id)
        print(" * Target rsi:\n", rsi_name)

        # The following 10 lines auto-generate an index dict for dataset RS image paths, like:
        list_of_merge_dset = list(d_merge_rsis.keys())  #['merge_c4_1m4k', 'merge_c4_254k']
        print(f"list_of_merge_dset: {list_of_merge_dset}")

        if rsi_name in list_of_merge_dset:  # Multi large RS images (fused dataset)
            d_rs_image_path = d_merge_rsis[rsi_name]  # Dict at this point
            d_rs_image_path = {k:f"{rsi_city_dir}/{v}" for k,v in d_rs_image_path.items()}  # Dict at this point
        else:  # Single large RS image (dataset)
            d_rs_image_path = {rsi_id:f"{rsi_city_dir}/{rsi_name}"}
        print(" * RS image: ", d_rs_image_path)

        city_rsi_id_str = str(city_id).zfill(3) + str(rsi_id).zfill(3)  # no use

        # Dataset path
        phr_dset_dir = f'{dset_root}/{dset_name}'
        print(" * Dataset: ", phr_dset_dir)

        rsi_id_str = str(rsi_id).zfill(2)
        dset_id_str = str(rsi_id).zfill(2) + str(n_sample).zfill(2) + f'_{view2d3d}'

        # Dataset params dict
        dataset_kwargs = {
            'city_id': city_id,
            'rsi_id': rsi_id,
            'rsi_type': rsi_type,
            'n_sample': n_sample,
            'city_rsi_id_str': city_rsi_id_str,
            'dset_id_str': dset_id_str,
            'rsi_id_str': rsi_id_str,
            'rsi_name': rsi_name,
            'rs_image_path': d_rs_image_path,
            'dset_name': dset_name,
            'dset_dir': phr_dset_dir,
        }
        print(f"dataset_kwargs: {dataset_kwargs}")

        if mode == 'test':  # test    
            #############################################################
            #              PARCASGM_v5 Model Test Only
            #############################################################
            # Default params, no change needed
            factor_bslr = 1/32
            loss_type = 'smoothl1'
            pa_loss_weight = PH_LOSS_WEIGHT
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            test_result_dir = f'{train_result_dir}/test_results_{dset_name}_{rsi_id}_{timestamp}'
            lst_distance_errors, lst_angle_errors = test_par(
                phr_dset_dir,
                test_result_dir,
                train_result_dir,
                test_id=test_id,
                d_rs_image_path=d_rs_image_path,
                device_id=device_id,
                factor_bslr=factor_bslr,
                loss_type=loss_type,
                pa_loss_weight=pa_loss_weight,
                model_class=model_class,
                model_kwargs=model_kwargs,
                dataset_class=dataset_class,
                dataset_kwargs=dataset_kwargs)