# Bearing-UAV train module: cvphr_train.py.
import os
import time
import json
import warnings
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# static vars
from config.base_info import (
    proj_dir, 
    dset_dir,
    rsi_type, 
    d_merge_rsis,
    get_rsi_name,
    get_rsidir_dsetdir_cityid,
    flexible_type, parse_gcth,
    DATASET_SPLIT_RATIO,
    PH_LOSS_WEIGHT)

# import tools
from cvphr.utils.utils import (
    record_model_info_2dict,
    convert_to_json_serializable,
    ckpt_load,
    training_curve,
    training_mae_curve_par,
    init_gcheck_dir,
    record_prediction_and_save_plots,
    record_gradient_norms,
    compute_grad_norm_stat_mean,
    save_monitor_history,
    diagnose_val_batch,
    MultiTaskLoss)
# export model
from cvphr.models.posaglreg.models import (
    # PARCASGM_v5,  # f/N in nonlocalnet.
    PARCASGM_v5a,  # softmax in nonlocalnet.
    RSBlockDatasetPA_v3q,  # pp1 weakened + uav-pp2
    RSBlockDatasetPA_v3q_weather,
    par_dataloader,
    model_kwargs_par_ca_sgm_v5a)

from cvphr.models import MODEL_CLASS_DICT
from cvphr.models import MODEL_KEYWARDS_DICT
from cvphr.models import DATASET_CLASS_DICT  # import dataset class dict
from cvphr.test.cvphr_test import test_par


def train_par(
    dataset_dir,
    device_id=0,
    num_epochs=2,
    factor_bslr=1,
    d_rs_image_path=None,
    loss_type="smoothl1",
    pa_loss_weight=None,
    scheduler_class="ReduceLROnPlateau",
    model_class=None,
    dataset_class=None,
    model_kwargs=None,
    dataset_kwargs=None,
    is_record_gradient_norms=False,
    max_grad_norm=1.5,  # New: gradient clipping threshold
    checkpoint_interval=20,
    resume_checkpoint=None,
    flag_test=False,
    flag_ckpt=False,
):
    """
    Train position regression model (Optimizer + Warmup Cosine Scheduler + Loss enhancement)
    Args:
        dataset_dir: Dataset directory
        num_epochs: Training epochs
        factor_bslr: Scaling factor for batch size and learning rate
        device_id: GPU ID
        d_rs_image_path: Remote sensing image path for visualization (dict for multiple maps)
        model_class: Model class, default PositionRegressionSGM
        dataset_class: Dataset class, default RSBlockDatasetSGM
        model_kwargs: Model init params, dict format
        dataset_kwargs: Dataset init params, dict format
        max_grad_norm: Gradient clipping threshold, default 1.5. None for no clipping
        checkpoint_interval: Checkpoint save interval (unit: epoch), default 20
        resume_checkpoint: Checkpoint path for resume training
        ckpt: Save checkpoints and weights, True-save, False-not save
    """
    # Device config
    if pa_loss_weight is None:
        pa_loss_weight = PH_LOSS_WEIGHT
    if d_rs_image_path is None:
        d_rs_image_path = {}
    if model_class is None or dataset_class is None:
        raise ValueError("model_class and dataset_class cannot be empty")

    device = torch.device(
        f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Key hyperparameters
    inilr = factor_bslr * 1e-4
    BATCH_SIZE = int(32 * factor_bslr)

    # Init model
    model_kwargs = model_kwargs or {}
    model = model_class(**model_kwargs).to(device)

    # Create output dir / Load resume dir
    os.makedirs('results', exist_ok=True)
    resume_path = Path(resume_checkpoint).expanduser() if resume_checkpoint else None
    checkpoint_data = None

    if resume_path: # Resume training
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
        checkpoint_data = torch.load(resume_path, map_location='cpu')
        result_dir = checkpoint_data.get('result_dir')
        if result_dir is None:
            if resume_path.parent.name == 'checkpoints':
                result_dir = str(resume_path.parent.parent)
            else:
                result_dir = str(resume_path.parent)
        print(f" - Resume training detected, loading checkpoint: {resume_path}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Dataset augmentation flag
        if dataset_class == RSBlockDatasetPA_v3q_weather:
            augtype = 'waug'
        else:
            augtype = ''

        # dataset name 2d/3d sign judge
        if 'dset_id_str' in dataset_kwargs:
            dsetidx = dataset_kwargs["dset_id_str"]
        else:
            dsetidx = ''
        
        result_dir = f'results/c4ma/{model.model_name}_d{dsetidx}{augtype}_b{BATCH_SIZE}_l{factor_bslr}_e{num_epochs}_g{max_grad_norm}_{timestamp}'
        print(f" - Training results saved to: {result_dir}")

    os.makedirs(result_dir, exist_ok=True)

    # Data augmentation and loading
    metadata_csv = f"{dataset_dir}/metadata/metadata.csv"
    train_loader, val_loader, _, _ = par_dataloader(metadata_csv, dataset_class, dataset_kwargs, BATCH_SIZE)

    # Loss enhancement options
    # Support SmoothL1Loss and HuberLoss, priority read loss_type from model_kwargs
    pos_weight, dir_weight = pa_loss_weight
    if loss_type == 'huber':
        criterion = nn.HuberLoss(delta=1.0)  # HuberLoss(delta=2.0~5.0)
        print('Using loss function: HuberLoss(delta=1.0)')
    elif loss_type == 'smoothl1' or loss_type == 'pos_smoothl1' or loss_type == 'dir_smoothl1':
        criterion = nn.SmoothL1Loss()
        print(f'Using loss function: SmoothL1Loss(pos_weight={pos_weight}, dir_weight={dir_weight})')
    elif loss_type == 'multitask':
        criterion = MultiTaskLoss(pos_weight=pos_weight, dir_weight=dir_weight)  # Init weights
        print(f'Using loss function: MultiTaskLoss(pos_weight={pos_weight}, dir_weight={dir_weight})')
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    # --------------------------------------

    # scheduler_class = "ReduceLROnPlateau" #scheduler_class
    if scheduler_class == "CosineAnnealingLR":
        # Optimizer: AdamW with weight decay
        optimizer = torch.optim.AdamW(model.parameters(), lr=inilr, weight_decay=1e-4)
        # Warmup + Cosine Scheduler config
        total_steps = num_epochs * len(train_loader)  # Total training steps
        warmup_steps = int(0.1 * total_steps)        # Warmup for first 10% steps
        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=1e-6)  # Add eta_min to avoid too low lr
        # Warmup scheduler (linear increase for first warmup_steps)

        def lr_lambda(step):
            return min((step + 1) / warmup_steps, 1.0)

        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif scheduler_class == "ReduceLROnPlateau":
        optimizer = torch.optim.Adam(model.parameters(), lr=inilr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_class}")

    # Training config
    resume_path_str = str(resume_path) if resume_path else None
    configure = {
        'file_name': Path(__file__).name,  # Record current file name (algorithm version)
        'model_name': model.model_name,
        'model_backbone': model.backbone_name,
        'loss_type': loss_type,
        'pa_loss_weight': pa_loss_weight,
        "optimizer_class": type(optimizer).__name__,
        "criterion_class": type(criterion).__name__,
        "scheduler_class": type(scheduler).__name__,
        'batch_size': BATCH_SIZE,
        'initial_lr': inilr,
        "epochs": num_epochs,
        'model_class': model_class.__name__,
        'dataset_class': dataset_class.__name__,
        'dataset': dataset_dir,
        'metadata': metadata_csv,
        'split_ratio': DATASET_SPLIT_RATIO,  #[0.85, 0.05, 0.1],
        'model_kwargs': model_kwargs,
        'dataset_kwargs': dataset_kwargs,
        'max_grad_norm': max_grad_norm,  # Record gradient clipping config
        'checkpoint_interval': checkpoint_interval,
        'resume_from': resume_path_str,
        'result_dir': result_dir
    }
    # Define model info to save
    # Get raw model (compatible with DataParallel)
    # backbone_name = 'vgg16'  # model_kwargs={'backbone_name': 'vgg16'}
    raw_model = model.module if hasattr(model, 'module') else model
    configure = record_model_info_2dict(configure, raw_model, criterion, optimizer, scheduler)

    serializable_configure = convert_to_json_serializable(configure)
    with open(f'{result_dir}/training_configure.json', 'w') as f:
        json.dump(serializable_configure, f, indent=2)

    # Init training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_loss_pos': [],
        'train_loss_dir': [],
        'val_loss_pos': [],
        'val_loss_dir': [],
        'train_loss_sum_pos_dir': [],
        'val_loss_sum_pos_dir': [],
        'lr_history': [],
        'test_metrics': None,
        'grad_norm_stat_epoch': [],
        'grad_clip_count': 0,  # Record gradient clip count
        'nan_loss_count': 0,   # Record NaN loss count
        'log_no_val_pred': []  # Log error: len(all_val_pos_pred) == 0
    }

    # Mixed precision gradient scaler (only enable on CUDA)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Load checkpoint and resume training state
    start_epoch, best_checkpoint, history = ckpt_load(
        checkpoint_data, history, model, optimizer, scheduler, device, resume_path, num_epochs, result_dir
    )

    checkpoints_dir = Path(result_dir) / 'checkpoints'
    if flag_ckpt:
        checkpoints_dir.mkdir(exist_ok=True)

    # Create dir for monitoring images and stats
    gcheck_dir = init_gcheck_dir(result_dir)

    # Create tensorboard logs
    writer = SummaryWriter(log_dir=f'{result_dir}/tensorboard_logs')

    # ***********************************************************************#
    #                                                                        #
    #                              Training Loop                             #
    #                                                                        #
    # ***********************************************************************
    global_step = (
        start_epoch * len(train_loader) if start_epoch > 0 else 0
    )  # Global step counter
    stop_training = False
    d_tensor_validity = {}

    def _train_one_epoch(epoch):
        """Single epoch training logic."""
        nonlocal global_step
        nonlocal history

        model.train()
        epoch_train_loss = 0.0
        epoch_train_loss_pos = 0.0
        epoch_train_loss_dir = 0.0

        for batch_idx, batch in enumerate(train_loader):
            patches = batch["patches"].to(device, non_blocking=True)
            coords = batch["coords"].to(device, non_blocking=True)
            agl_coords = batch["agl_coords"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            # Forward + Loss: in autocast for TensorCore
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                pos_pred, dir_pred = model(patches)

                loss_pos = criterion(pos_pred, coords)
                loss_dir = criterion(dir_pred, agl_coords)
                pos_weight, dir_weight = pa_loss_weight
                loss = pos_weight * loss_pos + dir_weight * loss_dir

            # Backward (use GradScaler for mixed precision stability)
            scaler.scale(loss).backward()

            # Unscale gradients before clipping to ensure correct norm calculation
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm
                )
                if grad_norm > max_grad_norm:
                    history["grad_clip_count"] += 1
                    print(
                        "Warning: Gradient clipped at batch "
                        f"{batch_idx}, norm: {grad_norm:.4f}"
                    )

            if is_record_gradient_norms:
                # Record gradient norms for each layer of regressor
                history = record_gradient_norms(model, history, epoch)

            # Optimizer step with GradScaler and update scale factor
            scaler.step(optimizer)
            scaler.update()

            batch_size = patches.size(0)
            epoch_train_loss += loss.item() * batch_size
            epoch_train_loss_pos += loss_pos.item() * batch_size
            epoch_train_loss_dir += loss_dir.item() * batch_size

            # Warmup + Cosine Scheduler step
            if scheduler_class == "CosineAnnealingLR":
                # Use warmup_scheduler for first warmup_steps, then cosine scheduler
                if global_step < warmup_steps:
                    warmup_scheduler.step()
                else:
                    scheduler.step()
                global_step += 1

        return epoch_train_loss, epoch_train_loss_pos, epoch_train_loss_dir

    def _validate_one_epoch(epoch):
        """Single epoch validation logic."""
        nonlocal stop_training

        model.eval()
        epoch_val_loss = 0.0
        epoch_val_loss_pos = 0.0
        epoch_val_loss_dir = 0.0
        all_val_pos_pred = []
        all_val_dir_pred = []
        all_val_coords = []
        all_val_agl_coords = []

        with torch.no_grad():
            print(f"Epoch {epoch + 1}/{num_epochs} [Val]")
            for batch_idx, batch in enumerate(val_loader):
                patches = batch["patches"].to(device, non_blocking=True)
                coords = batch["coords"].to(device, non_blocking=True)
                agl_coords = batch["agl_coords"].to(device, non_blocking=True)

                # Enable autocast in validation for speed and consistent numeric distribution
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    pos_pred, dir_pred = model(patches)

                # 'smoothl1', 'huber': joint training, calculate loss directly in validation
                loss_pos = criterion(pos_pred, coords)
                loss_dir = criterion(dir_pred, agl_coords)
                pos_weight, dir_weight = pa_loss_weight
                loss = pos_weight * loss_pos + dir_weight * loss_dir

                # Diagnose if loss is 0
                if loss.item() == 0.0:
                    diagnose_val_batch(
                        result_dir=result_dir,
                        batch_idx=batch_idx,
                        epoch=epoch,
                        patches=patches,
                        coords=coords,
                        agl_coords=agl_coords,
                        pos_pred=pos_pred,
                        dir_pred=dir_pred,
                        loss_pos=loss_pos,
                        loss_dir=loss_dir,
                        loss=loss,
                    )
                    d_tensor_validity["val_loss_0"] = [
                        "val",
                        epoch,
                        batch_idx,
                        loss.item(),
                    ]
                    print(
                        "⚠️⚠️Warning: Validation batch "
                        f"{batch_idx} loss is 0, training will stop!"
                    )
                    stop_training = True

                batch_size = patches.size(0)
                epoch_val_loss += loss.item() * batch_size
                epoch_val_loss_pos += loss_pos.item() * batch_size
                epoch_val_loss_dir += loss_dir.item() * batch_size

                all_val_pos_pred.append(pos_pred.detach().cpu())
                all_val_dir_pred.append(dir_pred.detach().cpu())
                all_val_coords.append(coords.detach().cpu())
                all_val_agl_coords.append(agl_coords.detach().cpu())

        return (
            epoch_val_loss,
            epoch_val_loss_pos,
            epoch_val_loss_dir,
            all_val_pos_pred,
            all_val_dir_pred,
            all_val_coords,
            all_val_agl_coords,
        )

    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs} [Train]")

        # Training phase
        start_time = time.time()
        train_loss, train_loss_pos, train_loss_dir = _train_one_epoch(epoch)
        elapsed_time = time.time() - start_time
        print(f"Training time: {elapsed_time:.2f}s")

        if stop_training:
            print("⚠️⚠️⚠️Warning: Training stopped in training phase!")
            break

        # Validation phase
        start_time = time.time()
        (
            val_loss,
            val_loss_pos,
            val_loss_dir,
            all_val_pos_pred,
            all_val_dir_pred,
            all_val_coords,
            all_val_agl_coords,
        ) = _validate_one_epoch(epoch)
        elapsed_time = time.time() - start_time
        print(f"Validation time: {elapsed_time:.2f}s")

        if stop_training:
            print("⚠️⚠️Warning: Training stopped in validation phase!")
            break

        # Avoid silent issues by checking empty prediction lists
        if len(all_val_pos_pred) == 0:
            str_err = f"[Debug] Epoch {epoch+1}: all_val_pos_pred is empty!"
            print(f"[Debug] Epoch {epoch+1}: all_val_pos_pred is empty! Likely all predictions contained NaN.")
            history['log_no_val_pred'].append(str_err)
        if len(all_val_dir_pred) == 0:
            str_err = f"[Debug] Epoch {epoch+1}: all_val_dir_pred is empty!"
            print(f"[Debug] Epoch {epoch+1}: all_val_dir_pred is empty! Likely all predictions contained NaN.")
            history['log_no_val_pred'].append(str_err)
        if len(all_val_coords) == 0 or len(all_val_agl_coords) == 0:
            warnings.warn("Validation coords/agl_coords list is empty; plots may be blank.")

        # Calculate average loss
        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr_history'].append(optimizer.param_groups[0]['lr'])

        # Record pos and dir loss separately
        train_loss_pos = train_loss_pos / len(train_loader.dataset)
        train_loss_dir = train_loss_dir / len(train_loader.dataset)
        val_loss_pos = val_loss_pos / len(val_loader.dataset)
        val_loss_dir = val_loss_dir / len(val_loader.dataset)
        history['train_loss_pos'].append(train_loss_pos)
        history['train_loss_dir'].append(train_loss_dir)
        history['val_loss_pos'].append(val_loss_pos)
        history['val_loss_dir'].append(val_loss_dir)
        history['train_loss_sum_pos_dir'].append([train_loss, train_loss_pos, train_loss_dir])
        history['val_loss_sum_pos_dir'].append([val_loss, val_loss_pos, val_loss_dir])

        # Record to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Loss/train_pos', train_loss_pos, epoch)
        writer.add_scalar('Loss/train_dir', train_loss_dir, epoch)
        writer.add_scalar('Loss/val_pos', val_loss_pos, epoch)
        writer.add_scalar('Loss/val_dir', val_loss_dir, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

        if scheduler_class == "ReduceLROnPlateau":
            # Adjust learning rate
            scheduler.step(val_loss)

        # Calculate mean grad_norm_stat for current epoch
        if is_record_gradient_norms:
            history = compute_grad_norm_stat_mean(history, epoch)

        # Save best model and update best checkpoint (best_model.pth only saves model_state_dict for deployment)
        if val_loss < best_checkpoint['val_loss']:
            best_checkpoint = {
                'epoch': epoch,
                'val_loss': val_loss,
                'model_state_dict': model.state_dict(),
            }
            # best_model.pth only saves model params (weights and biases), other states in latest_model.pth
            torch.save({'model_state_dict': model.state_dict()}, f'{result_dir}/best_model.pth')
            print(f"New best model saved with val_loss: {val_loss:.4f}")
        
        # Save latest checkpoint (full history for resume; adjust frequency as needed)
        if flag_ckpt:
            checkpoint_payload = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'best_val_loss': best_checkpoint['val_loss'],
                'best_epoch': best_checkpoint['epoch'],
                'result_dir': result_dir,
            }
            latest_path = Path(result_dir) / 'ckpt_latest_model.pth'
            torch.save(checkpoint_payload, latest_path)

            # Save periodic checkpoints at interval, ensure last epoch is saved
            if checkpoint_interval and checkpoint_interval > 0:
                should_save_periodic = ((epoch + 1) % checkpoint_interval == 0) or (epoch + 1 == num_epochs)
                if should_save_periodic:
                    periodic_path = checkpoints_dir / f'epoch_{epoch + 1:04d}.pth'
                    torch.save(checkpoint_payload, periodic_path)

        # Check list before torch.cat to avoid crash even if all batches are skipped
        def _concat_or_empty(tensors, name):
            if isinstance(tensors, list) and len(tensors) > 0:
                return torch.cat(tensors, dim=0)
            warnings.warn(f"Warning: {name} is empty, filling with empty tensor.")
            return torch.empty((0, 2))

        all_val_pos_pred = _concat_or_empty(all_val_pos_pred, "all_val_pos_pred")
        all_val_dir_pred = _concat_or_empty(all_val_dir_pred, "all_val_dir_pred")
        all_val_coords = _concat_or_empty(all_val_coords, "all_val_coords")
        all_val_agl_coords = _concat_or_empty(all_val_agl_coords, "all_val_agl_coords")

        # Plot gcheck images once after final epoch based on validation results
        if epoch + 1 == num_epochs:
            history = record_prediction_and_save_plots(
                all_val_pos_pred,
                all_val_dir_pred,
                all_val_coords,
                all_val_agl_coords,
                epoch,
                history,
                gcheck_dir,
            )

        # Systematized training monitor summary: write to lightweight text log
        log_path = Path(gcheck_dir) / "training_epoch_log.txt"
        with open(log_path, "a") as f_log:
            f_log.write(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
                f"train_pos={train_loss_pos:.6f}, train_dir={train_loss_dir:.6f}, "
                f"val_pos={val_loss_pos:.6f}, val_dir={val_loss_dir:.6f}, "
                f"lr={optimizer.param_groups[0]['lr']:.6e}\n"
            )

        # End of current epoch

    # Save all history to JSON for post-analysis
    save_monitor_history(history, gcheck_dir)

    # Print training statistics
    print("\nTraining Statistics:")
    print(f"Total gradient clips: {history['grad_clip_count']}")
    print(f"Total NaN losses: {history['nan_loss_count']}")

    # Plot training and validation curves
    print("\nPlotting training curves ...")
    training_curve(history, result_dir)

    writer.close()

    model.eval()
    # Joint training mode: position and direction
    val_errors_pos = []
    val_errors_dir = []
    with torch.no_grad():
        for batch in val_loader:
            patches = batch['patches'].to(device)
            pos_pred, dir_pred = model(patches)
            coords = batch['coords'].to(device)
            errors_pos = torch.abs(pos_pred - coords).mean(dim=1).cpu().numpy()
            val_errors_pos.extend(errors_pos)
            agl_coords = batch['agl_coords'].to(device)
            errors_dir = torch.abs(dir_pred - agl_coords).mean(dim=1).cpu().numpy()
            val_errors_dir.extend(errors_dir)
    training_mae_curve_par(history, result_dir, best_checkpoint, loss_type, val_errors_pos, val_errors_dir)

    # Test phase
    if flag_test:
        print("\nTesting Best Model ...")
        # Use custom model and dataset class
        dset_name = dataset_kwargs['dset_name']
        dset_dir = dataset_dir
        train_result_dir = result_dir

        # for test_id, rs_image_path in d_rs_image_path.items():/
        test_id = dataset_kwargs['rsi_id']
        test_result_dir = f'{train_result_dir}/test_results_{dset_name}_{test_id}'
        test_par(
            dset_dir,
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
            dataset_kwargs=dataset_kwargs
        )
    


if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_3d', type=int, default=1, help="Dataset is 3D view")
    parser.add_argument('--rsi_id', type=flexible_type, default=96, help="Remote sensing image ID")
    parser.add_argument('--n_sample', type=int, default=100, help="Sample count per block")
    parser.add_argument('--num_epochs', type=int, default=1, help="Training epochs")
    parser.add_argument('--mode', type=int, default=0, help="Train/test mode, 0-train, 1-test")
    parser.add_argument('--n_block', type=int, default=15, help="Remote sensing image block count")
    parser.add_argument('--device_id', type=int, default=0, help="GPU ID")
    parser.add_argument('--factor_bslr', type=float, default=0.5, help="Scaling factor, base bs=32, lr=1e-4")
    parser.add_argument('--merge', type=int, default=0, help="Downsampling/fusion dataset")
    parser.add_argument('--flag_test', type=int, default=1, help="Run test,1-yes,0-no")
    parser.add_argument('--gcth', type=parse_gcth, default=None, help="Gradient clip threshold max_grad_norm (float supported, 'none' for disable, default: None)")
    parser.add_argument("--model_class", type=str, default="PARCASGM_v5a", help="Model class")
    parser.add_argument("--dataset_class", type=str, default="RSBlockDatasetPA_v3q", help="Dataset class")        
    parser.add_argument('--resume', type=str, default='', help="Resume training from checkpoint")
    parser.add_argument('--checkpoint_interval', type=int, default=20, help="Checkpoint save interval (epoch)")
    parser.add_argument('--flag_ckpt', type=int, default=0, help="Save checkpoints and weights,1-yes,0-no")
    parser.add_argument('--fast_train', type=int, default=0, help="Fast training,1-yes,0-no")
    parser.add_argument('--local_dataset', type=int, default=0, help="Use local dataset,1-yes,0-no")

    args = parser.parse_args()
    print(args)

    # Select mode:
    mode = 'train' if args.mode == 0 else 'test'  # Prepare dataset before training
    flag_test = True if args.flag_test == 1 else False # Run test or not
    # dataset paras
    rsi_id = args.rsi_id  # Remote sensing image ID
    n_sample = args.n_sample  # Suggest mini dataset with sample=1 for debugging
    view2d3d = '3d' if args.is_3d else '2d'  # Dataset 2d/3d version
    # training paras
    device_id = args.device_id  # GPU ID
    factor_bslr = args.factor_bslr  # factor_bslr = 0.5 #0.5-16, 1-b32, 2-b64, 4-b128
    num_epochs = args.num_epochs
    max_grad_norm = args.gcth
    model_class = MODEL_CLASS_DICT[args.model_class]
    model_kwargs = MODEL_KEYWARDS_DICT[args.model_class]  #model_kwargs should match
    dataset_class = DATASET_CLASS_DICT[args.dataset_class]
    resume_path = args.resume if args.resume else None  # Checkpoint path for resume
    ckpt_interval = args.checkpoint_interval
    flag_ckpt = bool(args.flag_ckpt)
    fast_train = bool(args.fast_train)
    local_dataset = bool(args.local_dataset)
    
    print("mode:", mode)
    print("flag_test:", flag_test)
    print("rsi_id:", rsi_id)
    print("n_sample:", n_sample)
    print("view2d3d:", view2d3d)
    print("device_id:", device_id)
    print("factor_bslr:", factor_bslr)
    print("num_epochs:", num_epochs)
    print("max_grad_norm:", max_grad_norm)
    print("model_class:", model_class)
    print("model_kwargs:", model_kwargs)
    print("dataset_class:", dataset_class)
    print("resume_path:", resume_path)
    print("ckpt_interval:", ckpt_interval)
    print("flag_ckpt:", flag_ckpt)
    print("fast_train:", fast_train)
    print("local_dataset:", local_dataset)

    # Remote sensing image name and path
    rsi_city_dir, dset_root, dset_name, city_id = get_rsidir_dsetdir_cityid(
        rsi_id, 
        rsi_type, 
        dset_dir, 
        n_sample, 
        view2d3d, 
        flag_pyr='',  #'' for fixed height
        local_dataset=local_dataset,  # False for mnt dataset, True for project dataset
    )
    # Auto generate ID from user info
    rsi_name = get_rsi_name(rsi_id)
    print(" * Target rsi:\n", rsi_name)

    list_of_merge_dset = list(d_merge_rsis.keys())  #['merge_c4_1m4k', 'merge_c4_254k']
    if rsi_name in list_of_merge_dset:  # Fusion dataset from multiple remote sensing images
        d_rs_image_path = d_merge_rsis[rsi_name]  #dict
        d_rs_image_path = {k:f"{rsi_city_dir}/{v}" for k,v in d_rs_image_path.items()} #dict
        
    else:  # Dataset from single remote sensing image
        # rsi_name = '52.468789_-1.92075_720_720_4326_city.jpg'
        d_rs_image_path = {rsi_id:f"{rsi_city_dir}/{rsi_name}"}
    print(" * Remote sensing images：", d_rs_image_path)

    city_rsi_id_str = str(city_id).zfill(3) + str(rsi_id).zfill(3)  # no use

    # Dataset path
    dset_dir = f'{dset_root}/{dset_name}'
    print(" * Dataset：", dset_dir)

    rsi_id_str = str(rsi_id).zfill(2)
    dset_id_str = str(rsi_id).zfill(2) + str(n_sample).zfill(2) + f'_{view2d3d}'

    # Dataset kwargs
    dataset_kwargs = {
        'city_id': city_id,
        'rsi_id': rsi_id,
        'rsi_type': rsi_type,  # Add rsi_type for rsi_id extraction in test_par
        'n_sample': n_sample,
        'city_rsi_id_str': city_rsi_id_str,
        'dset_id_str': dset_id_str,
        'rsi_id_str': rsi_id_str,
        'rsi_name': rsi_name,
        'rs_image_path': d_rs_image_path,
        'dset_name': dset_name,
        'dset_dir': dset_dir,
        "augname": "Noop",  #Weather aug: Mixed, Rain,Snow,Cloud,Brightness,Noop
    }
    if 'weather' in args.dataset_class:
        dataset_kwargs["augname"] = "Mixed"  #Weather aug: Mixed, Rain,Snow,Cloud,Brightness,Noop, see transform_pipeline_weather

    if mode == 'train':  # Train par (par_dca_sgm5) model
        #############################################################
        #           pr_ca/pr_sca,v5 model training only
        #############################################################

        # No modification needed below
        loss_type = 'smoothl1'  # 'smoothl1', 'huber','multitask','pos_smoothl1','dir_smoothl1'
        scheduler_class = 'ReduceLROnPlateau'
        print(f"Model Backbone: {model_kwargs['backbone_name']}")

        # Use custom model and dataset class
        train_par(
            dataset_dir=dset_dir,
            device_id=device_id,
            num_epochs=num_epochs,
            factor_bslr=factor_bslr,  # 1-b32, 2-b64, 4-b128
            loss_type=loss_type,  # 'smoothl1', 'huber','multitask','pos_smoothl1','dir_smoothl1'
            pa_loss_weight=PH_LOSS_WEIGHT,
            scheduler_class=scheduler_class,
            d_rs_image_path=d_rs_image_path,
            model_class=model_class,  # par_dca_sgm
            model_kwargs=model_kwargs,  # par_dca_sgm
            dataset_class=dataset_class,
            dataset_kwargs=dataset_kwargs,
            # Numerical stability config - prevent NaN loss
            max_grad_norm=max_grad_norm,  # Gradient clip threshold 1 better than 0.5 (stricter)
            checkpoint_interval=ckpt_interval,
            resume_checkpoint=resume_path,
            flag_test=flag_test,
            flag_ckpt=flag_ckpt
        )
                
