# Bearing-Naver procedure for satellite-view navigation test.
# Precondition: Waypoint path files exist in loc2traj/traj_wps_gcs folder
 
import os
import cv2
import math
import json
import torch
import argparse
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List

from config.base_info import (
    id_image_map,  # Predefined image ID mapping (per number in comments)
    PATCH_SIZE, 
    BLOCK_SIZE, 
    MAX_DISTANCE,
    rsijson2info,
    generate_grid_blocks,
    proj_dir,
    rsi_dir_berm_city,  
    rsi_dir_city8_25pp_4096bc)

from naver.runners.visnav import visualize_patches_relation
from naver.runners.visnav import crop_target_patch

from cvphr.utils.utils import (
    vis_waypoints_uavtrajs_on_fig_v4,
    convert_to_json_serializable,
    convert_ccs_to_llcs_vector,
    convert_llcs_to_ccs_vector,
    calculate_lon_lat_step_underllcs,
    vector2angle,
    angle2vector,
    normalize_vector,
    uniform_rsi_image)

from cvphr.models.posaglreg.models import load_config_and_model
from cvphr.models.posaglreg.models import PositionAngleRegressionSGM
from cvphr.models.posaglreg.models import PARCASGM_v5a

from cvphr.utils.utils_transform import transform_pipeline3


class PHR_MODEL_LOADING_BASE:
    """
    PHR model loading base class, provides unified model loading logic
    """
    def __init__(self, 
                 device_id=0,
                 posreg_model_dir='',
                 model_class=PARCASGM_v5a,
                 model_kwargs=None,
                 dataset_kwargs=None):
        """
        Initialize PHR model, provide unified model loading steps
        """
        self.device_id = device_id
        self.posreg_model_dir = posreg_model_dir
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}
        self.dataset_kwargs = dataset_kwargs or {}
        
        # Unified model loading steps
        self._load_model()
        
        # Initialize image preprocessing transform
        self._init_transform()
    
    def _load_model(self):
        """
        Unified model loading logic
        """
        print("\n + Initial Best Model...")
        self.device = torch.device(f"cuda:{self.device_id}" if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model with incoming model_class and model_kwargs
        self.model = self.model_class(**self.model_kwargs).to(self.device)
        
        # Load checkpoint (if path provided)
        if self.posreg_model_dir and os.path.exists(self.posreg_model_dir):
            checkpoint = torch.load(self.posreg_model_dir, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Model weights loaded: {self.posreg_model_dir}")
        else:
            print("⚠ Model path not provided or file does not exist, using randomly initialized model")
        
        self.model.eval()
        print("\n + Best Model Loaded.")
    
    def _init_transform(self):
        """
        Initialize image preprocessing transform
        """
        self.flag_transform_type = 'norm'
        if self.flag_transform_type == 'norm':
            self.nav_transform = transform_pipeline3()
                
        if self.flag_transform_type == 'aug':  # Need to enhance UAV-patch later
            # Add the same image transforms as dataset, future experiments can add simulated weather and noise
            self.nav_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomAffine(degrees=5, translate=(0.05,0.05))
            ])
            
    
    def get_model(self):
        return self.model
    
    def get_device(self):
        return self.device
    
    def get_nav_transform(self):
        return self.nav_transform
        
        
class UAVNavigation(PHR_MODEL_LOADING_BASE):
    """ 
    Bearing-Naver navigation class
    Preconditions:
    loc2nav outputs waypoint sequence list.json file
    generate_grid_blocks outputs block_cnt_point_dict    
    """
    def __init__(self, 
                 start_point: Tuple[float, float],
                 end_point: Tuple[float, float],
                 waypoints: List[Tuple[float, float]],
                 block_cnt_point_dict: Dict[Tuple[float, float], Tuple[int, int]],
                 uav_2d3d= '2d',
                 uav_step: float = 15.0,
                 max_steps: int = 200,
                 output_dir='',
                 th_arrive=None,
                 rs_traj_id='',
                 rs_image_dir='',
                 rsi_type='',
                 device_id=0,
                 posreg_model_dir='',
                 model_class=PARCASGM_v5a,
                 model_kwargs=None,
                 dataset_kwargs=None):

        # Call parent class initialization (model loading)
        super().__init__(
            device_id=device_id,
            posreg_model_dir=posreg_model_dir,
            model_class=model_class,
            model_kwargs=model_kwargs,
            dataset_kwargs=dataset_kwargs
        )
        
        # UAVNavigation specific initialization
        self.start_point = start_point
        self.end_point = end_point
        self.waypoints = waypoints
        self.block_cnt_point_dict = block_cnt_point_dict
        self.uav_2d3d = uav_2d3d
        self.uav_step = uav_step
        self.max_steps = max_steps
        self.output_dir = output_dir
        self.rs_image_dir = rs_image_dir
        self.rsi_type = rsi_type
                
        rsi_json_path = os.path.splitext(self.rs_image_dir)[0] + '.json'
        self.drsi = rsijson2info(rsi_json_path)
        # Basic variables
        self.width, self.height = self.drsi["width_pixel"], self.drsi["height_pixel"]
        self.rsi_cnt_lng, self.rsi_cnt_lat = self.drsi["lng"], self.drsi["lat"]
        # lng_per_pixel, lat_per_pixel = drsi["lng_per_pixel"], drsi["lat_per_pixel"]
        self.latm_per_pixel = self.drsi["latm_per_pixel"]  # 0.14040561622464898,
        self.lng_per_pixel = self.drsi["lng_per_pixel"]
        self.lat_per_pixel = self.drsi["lat_per_pixel"]
        
                
        # th_arrive represents arrival threshold, unit is meters
        if th_arrive is None:
            self.th_arrive = 20.0
        else:
            self.th_arrive = th_arrive
            
        # Save model and data info during initialization
        nav_base_info = {
            'file_name': Path(__file__).name,  #Record current file name (algorithm version)
            'model_name': model_class.get_model_name(),
            'model_class': model_class.__name__,
            'model_kwargs': model_kwargs,
            'dataset_kwargs': dataset_kwargs,
            'posreg_model_dir': posreg_model_dir,
            'output_dir': output_dir,
            'rs_image_dir': rs_image_dir,
            'rsi_type': self.rsi_type,
            'rs_traj_id': rs_traj_id,
            'uav_step': uav_step,
            'th_arrive': self.th_arrive,
            'uav_2d3d': self.uav_2d3d,
        }
        nav_base_info = convert_to_json_serializable(nav_base_info)
        with open(f'{output_dir}/nav_base_info.json', 'w') as f:
            json.dump(nav_base_info, f, indent=2) 
        
        self.ori_angle = 90  #Initial heading angle 90 degrees (due north)
        self.rs_traj_id = rs_traj_id

        # Initialize DataFrame for recording data
        self.records = pd.DataFrame()
        self.nav_transform = self.get_nav_transform()

    
    def lnglat2xy(self, lnglat):
        """Convert longitude/latitude coordinates of a point in remote sensing map to pixel coordinates"""
        # Longitude to x: longitude minus center lng
        xpix = int((lnglat[0] - self.rsi_cnt_lng) / self.lng_per_pixel + self.width / 2)
        # Latitude to y: note use center lat minus latitude
        ypix = int((self.rsi_cnt_lat - lnglat[1]) / self.lat_per_pixel + self.height / 2)
        
        # Ensure bounding box is within image range
        xpix = max(0, xpix)
        xpix = min(self.width, xpix)
        ypix = max(0, ypix)
        ypix = min(self.height, ypix)
        
        return [xpix, ypix]
    
    def xy2lnglat(self, xy):
        """Convert pixel coordinates to longitude/latitude coordinates of a point in remote sensing map"""
        xpix, ypix = xy
        
        # Ensure pixel coordinates are within image range
        xpix = max(0, xpix)
        xpix = min(self.width, xpix)
        ypix = max(0, ypix)
        ypix = min(self.height, ypix)
        
        # Pixel coordinates to longitude/latitude: reverse derive lnglat2xy formula
        lng = (xpix - self.width / 2) * self.lng_per_pixel + self.rsi_cnt_lng
        lat = self.rsi_cnt_lat - (ypix - self.height / 2) * self.lat_per_pixel
        
        return [lng, lat]
        
    def calculate_heading(self, 
                         start_point: Tuple[float, float], 
                         target_point: Tuple[float, float]) -> Tuple[float, Tuple[float, float]]:
        """
        Calculate heading angle and next position
        """
        # Calculate heading angle (due north is 0, clockwise is positive)
        delta_lon = target_point[0] - start_point[0]
        delta_lat = target_point[1] - start_point[1]
        fly_angle_rad = math.atan2(delta_lon, delta_lat)
        
        # Convert to degrees (0-360 degrees)
        fly_angle = math.degrees(fly_angle_rad)
        fly_angle = fly_angle + 360 if fly_angle < 0 else fly_angle
        
        # Calculate actual distance between two points (meters)
        R = 6371000  # Earth radius (meters)
        lat1, lat2 = math.radians(start_point[1]), math.radians(target_point[1])
        lon1, lon2 = math.radians(start_point[0]), math.radians(target_point[0])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        total_distance = R * c  # Total distance (meters)
        
        # Calculate actual movement ratio
        if total_distance == 0:
            return 0, (0, 0)
            
        step_ratio = self.uav_step / total_distance
        
        # Calculate longitude/latitude after movement
        step_lon = start_point[0] + delta_lon * step_ratio
        step_lat = start_point[1] + delta_lat * step_ratio
        
        # Calculate longitude/latitude increment
        cur_step = (step_lon - start_point[0], step_lat - start_point[1])
        
        return fly_angle, cur_step
    
    def get_nearest_block(self, 
                         point: Tuple[float, float]) -> Tuple[Tuple[float, float], Tuple[int, int], List[np.ndarray]]:
        """
        Get nearest remote sensing map block under local uniformity assumption
        """
        # Find nearest block center point
        min_dist = float('inf')
        nearest_block_center = None
        nearest_block_indices = None
        
        for key_, val_ in self.block_cnt_point_dict.items():
            block_indices = val_['block_id']  #ast.literal_eval(key_)  # (row, col)
            block_center = val_['lnglat']
            dist, _ = self.calculate_distance(point, block_center)
            if dist < min_dist:
                min_dist = dist
                nearest_block_center = block_center
                nearest_block_indices = block_indices
        return nearest_block_center, nearest_block_indices
    
    
    # Assume cropped_img is OpenCV image obtained by get_rs_image() method
    def convert_cv2_to_pil(self, cropped_img):
        # 1. Convert BGR channel order to RGB (OpenCV uses BGR by default, PIL uses RGB)
        rgb_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        # 2. Convert to PIL image using Image.fromarray
        image_crop = Image.fromarray(rgb_img)
        return image_crop
        
    def get_rotated_img(self, x, y, w, h, image, angle):
            """
            Rotate around specified point in image and crop rectangular area
            """
            
            # 1. Perform image rotation (expand=True to ensure all rotated content is displayed)
            rotated_image = image.rotate(angle, expand=True)

            # 2. Convert angle to radians
            angle_rad = math.radians(angle)
            
            # 3. Calculate rotation matrix elements
            cos_a = math.cos(angle_rad)  # cos component of rotation matrix
            sin_a = math.sin(angle_rad)  # sin component of rotation matrix
            
            # 4. Calculate new center coordinates after rotation (core formula)
            new_center_x = rotated_image.size[0]/2 + (x - image.size[0]/2)*cos_a + (y - image.size[1]/2)*sin_a
            
            new_center_y = rotated_image.size[1]/2 - (x - image.size[0]/2)*sin_a + (y - image.size[1]/2)*cos_a
            
            # 5. Perform center cropping
            crop_rotated = rotated_image.crop((
                int(new_center_x - w//2),    # Left boundary
                int(new_center_y - h//2),    # Top boundary
                int(new_center_x + w//2),    # Right boundary
                int(new_center_y + h//2)     # Bottom boundary
            ))
            return crop_rotated
    
    def next_point_in_block(self, next_point, nearest_block_center, block_size=512):
        """
        Convert longitude/latitude coordinates to pixel coordinates and relative coordinates with block center as origin.
        """
        # Calculate longitude/latitude difference (unit: degrees)
        delta_lng = next_point[0] - nearest_block_center[0]
        delta_lat = next_point[1] - nearest_block_center[1]
        
        # Calculate relative coordinates (block center as origin, right is positive X, up is positive Y)
        relative_x = delta_lng / self.lng_per_pixel
        relative_y = delta_lat / self.lat_per_pixel
        
        # Calculate pixel coordinates (block top-left as origin, right is positive X, down is positive Y)
        pixel_center = block_size // 2
        pixel_x = pixel_center + relative_x
        pixel_y = pixel_center - relative_y  # Note: Pixel Y-axis points down, opposite to geographic latitude direction
        
        x_norm = relative_x / pixel_center
        y_norm = - relative_y / pixel_center
        
        return pixel_x, pixel_y, relative_x, relative_y, x_norm, y_norm
    
    def pred_point_from_norm(self, x_norm, y_norm, nearest_block_center, block_size=512):
        """
        Convert normalized relative coordinates back to original longitude/latitude coordinates.
        """
        # Calculate relative coordinates (recover from normalized values)
        pixel_center = block_size // 2
        relative_x = x_norm * pixel_center
        relative_y = -y_norm * pixel_center  # Note: Negative sign was taken during normalization, need to restore here
        
        # Calculate longitude/latitude difference
        delta_lng = relative_x * self.lng_per_pixel
        delta_lat = relative_y * self.lat_per_pixel
        
        # Restore original longitude/latitude
        lng = nearest_block_center[0] + delta_lng
        lat = nearest_block_center[1] + delta_lat
        
        return lng, lat

    def get_patches(self, uav_frame_id, fly_angle_ccs, next_point_real, block_center, block_indices: Tuple[int, int]) -> List[np.ndarray]:
        """
        Get four patches of specified block (calculated from nominal next_point_name)
        """
        rsi_img = cv2.imread(self.rs_image_dir)
        assert rsi_img is not None, "Image reading failed, check path"
        cropped_img = uniform_rsi_image(rsi_img)
        
        # Create directory structure
        patches_dir = Path(self.output_dir) / "traj_patches"
        patches_dir.mkdir(parents=True, exist_ok=True)
        patches_check_dir = Path(self.output_dir) / "traj_patches_check"
        patches_check_dir.mkdir(parents=True, exist_ok=True)
        
        # ----------------------------
        # Step 1: Save four neighboring base patches
        # ----------------------------

        # Cut block according to block_indices to get patches
        block_x, block_y = block_indices  #(c,r)
        # Extract current block area (512x512)
        start_x = block_x * 256
        start_y = block_y * 256
        block_img = cropped_img[start_y:start_y+512, start_x:start_x+512]
        
        patches = []
        patches_fdirs = []
        for dx, dy in [(0,0), (1,0), (0,1), (1,1)]:  # Bottom-left, bottom-right, top-right, top-left
            # Calculate position of each base patch
            patch = block_img[
                dy*256 : (dy+1)*256,
                dx*256 : (dx+1)*256
            ]
            # Save base patch
            base_path = patches_dir/f"{self.rs_traj_id}_block_{block_x}_{block_y}_base_{dx}{dy}.png"
            cv2.imwrite(str(base_path), patch)
            patches_fdirs.append(str(base_path))  
            patches.append(patch)      
                
        # ----------------------------
        # Step 2: Get and save patch after UAV flies to next step
        # ----------------------------
        blk_cnt_x, blk_cnt_y = self.lnglat2xy(block_center)  #Pixel coordinates of nominal block center where next uav frame is located
        
        # Cut out UAV field of view patch crop_rotated
        #Pixel coordinates of next actual sub-waypoint where uav actually flies to, target_patch cropping should be executed based on this instead of nominal waypoint
        nextpt_real_x, nextpt_real_y = self.lnglat2xy(next_point_real)  
                
        # Pixel position of target_patch center corresponding to uav real sub-waypoint in block, rpch means real patch
        rpch_in_blk_x, rpch_in_blk_y = nextpt_real_x - start_x, nextpt_real_y - start_y
   
        # theta represents human perspective image, top is 0 degrees, counterclockwise is positive
        # fly_angle represents UAV perspective image direction angle, range 0->360 counterclockwise from right
        theta = fly_angle_ccs  #theta is heading angle fly_angle!

        # Crop target patch, note cropped_img is rsi image, not block image
        target_patch = crop_target_patch(nextpt_real_x, nextpt_real_y, theta, cropped_img)   
        # Convert PIL Image to numpy array and add to patches list *
        target_patch = np.array(target_patch)
        patches.append(target_patch)
        
        center_x, center_y = int(nextpt_real_x), int(nextpt_real_y)       

        print(f" * get_patches/block_center:", block_center)
        print(f" * get_patches/blk_cnt_x, blk_cnt_y", blk_cnt_x, blk_cnt_y)
        print(f" * get_patches/nextpt_real_x, nextpt_real_y", nextpt_real_x, nextpt_real_y)
        print(f" * get_patches/rpch_in_blk_x, rpch_in_blk_y", rpch_in_blk_x, rpch_in_blk_y)
        print(f" * get_patches/center_x, center_y:", center_x, center_y)
        print(f" * get_patches/region of block:", center_y-128,center_y+128,center_x-128,center_x+128)
        print(f" * get_patches/cropped_img.shape:", cropped_img.shape)

        if center_y-128<0 or center_y+128>cropped_img.shape[0] or center_x-128<0 or center_x+128>cropped_img.shape[1]:
            # If target_patch is out of cropped_img boundary, adjust center_x, center_y
            record = {"flag_out_of_map": True}
        else:
            if 1:  # Visualize patches relation
                # Save target patch
                # Current UAV field of view number: remote sensing image id(1..) + trajectory segment number(1..) + sub-trajectory point number(0..)
                viewspan_meter = round(PATCH_SIZE * self.latm_per_pixel)  #Field of view pixel width
                uav_view_stem = self.rs_traj_id + '_' + str(uav_frame_id).zfill(3) + '_' + \
                                str(next_point_real[0]) + '_' + str(next_point_real[1]) + '_' + \
                                str(fly_angle_ccs) + '_' + str(viewspan_meter)
                target_path = patches_dir/f"{uav_view_stem}.jpg"
                patches_fdirs.append(str(target_path)) 
                cv2.imwrite(str(target_path), target_patch)
                # Save patches relation visualization image
                save_dir = patches_check_dir/f"{self.rs_traj_id}_{uav_frame_id}_patches_relation.jpg"
                self.check_patches_relation_logo(patches, rpch_in_blk_x, rpch_in_blk_y, 
                                                 theta, save_dir=save_dir)            
            # Record metadata (including neighbor paths), mainly calculate x_norm, y_norm, and verify if rpch_in_blk_x, rpch_in_blk_y are correct
            pixel_x, pixel_y, relative_x, relative_y, x_norm, y_norm = self.next_point_in_block(next_point_real, block_center, block_size=BLOCK_SIZE)
            
            print(" * get_patches:")
            print(f" * block_x={block_x}, block_y={block_y}")
            print(f" * start_x={start_x}, start_y={start_y}")
            print(f" * blk_cnt_x={blk_cnt_x}, blk_cnt_y={blk_cnt_y}")
            print(f" * nextpt_real_x={nextpt_real_x}, nextpt_real_y={nextpt_real_y}")
            print(f" * rpch_in_blk_x={rpch_in_blk_x}, rpch_in_blk_y={rpch_in_blk_y}")
            print(f" * pixel_x={pixel_x}, pixel_y={pixel_y}")
            print(f" * relative_x={relative_x}, relative_y={relative_y}")
            print(f" * x_norm={x_norm}, y_norm={y_norm}")
            
            self.nav_step['start_xy'] = [start_x, start_y]
            self.nav_step['block_cnt_xy'] = [blk_cnt_x, blk_cnt_y]
            self.nav_step['nextpt_real_xy'] = [nextpt_real_x, nextpt_real_y]
            self.nav_step['rpch_in_blk_xy'] = [rpch_in_blk_x, rpch_in_blk_y]
            self.nav_step['pixel_xy'] = [pixel_x, pixel_y]
            self.nav_step['relative_xy'] = [relative_x, relative_y]
            self.nav_step['norm_xy'] = [x_norm, y_norm]

            # List append is more efficient
            record = {
                "block_x": block_x,
                "block_y": block_y,
                "target_path": patches_fdirs[4],
                "x_norm": x_norm,
                "y_norm": y_norm,
                "angle": fly_angle_ccs,
                "p1_path": patches_fdirs[0],  # Bottom-left
                "p2_path": patches_fdirs[1],  # Bottom-right
                "p3_path": patches_fdirs[2],  # Top-right
                "p4_path": patches_fdirs[3],  # Top-left
                "flag_out_of_map": False
            }

        return patches, patches_fdirs, record

    def position_regression(self,x_norm, y_norm):
        dx = 0
        dy = 0
        x_pred, y_pred = x_norm + dx, y_norm + dy
        x_pred, y_pred = max(-0.5, min(x_pred, 0.5)), max(-0.5, min(y_pred, 0.5))
        return x_pred, y_pred
    
    def position_angle_regression_(self, patches):
        """
        Position and direction regression function
        """
        # Patches preprocessing - explicit normalization required in flight mode
        tensor_patches = []
        for patch in patches:
            # Ensure patch is numpy array format
            if not isinstance(patch, np.ndarray):
                patch = np.array(patch)
            # Patch normalization pipeline processing (consistent training/testing/flight. If investigating weather impact on uav, modify in PHR_MODEL_LOADING_BASE)
            patch = self.nav_transform(patch)
            tensor_patches.append(patch)

        # Merge all patches and add batch dimension [1, 5, C, H, W]
        patches_tensor = torch.stack(tensor_patches).unsqueeze(0)
        # Move to correct device
        patches_tensor = patches_tensor.to(self.device)
        
        # Perform prediction
        with torch.no_grad():
            # DoubleReg model has two outputs
            pos_pred, dir_pred = self.model(patches_tensor)
            pos_pred = pos_pred.cpu().numpy()[0]
            dir_pred = dir_pred.cpu().numpy()[0]
            # predictions.append(pred_coords)
        return pos_pred, dir_pred

    def calculate_distance(self, 
                         point1: Tuple[float, float], 
                         point2: Tuple[float, float],
                         th_arrive: float = None) -> Tuple[float, bool]:
        """
        Calculate distance between two points and determine if arrived
        """
        # Calculate spherical distance using Haversine formula
        R = 6371000  # Earth radius (meters)
        
        lat1, lat2 = math.radians(point1[1]), math.radians(point2[1])
        lon1, lon2 = math.radians(point1[0]), math.radians(point2[0])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c

        return distance, distance < self.th_arrive

    def check_patches_relation_logo(self, patches, x_block, y_block, theta, save_dir=None):
        """
        Visualize relation diagram of five patches, add uavlogo
        """
        p1, p2, p3, p4, target_patch = patches
        
        # Stitch four images into 512x512 block
        top_row = np.hstack((p1, p2))  # Top-left + Top-right
        bottom_row = np.hstack((p3, p4))  # Bottom-left + Bottom-right
        block = np.vstack((top_row, bottom_row))
        
        # Check if block size is correct
        if block.shape != (512, 512, 3):
            block = cv2.resize(block, (512, 512))

        # Convert normalized coordinates to pixel coordinates (center as origin)
        x_pixel, y_pixel = x_block, y_block

        #Calculate unit direction vector x_cosa,y_sina with heading angle theta, see dsmaker_4_rsi() definition
        theta_rad = theta * np.pi / 180  # Manually convert to radians
        x_cosa = np.cos(theta_rad)
        y_sina = np.sin(theta_rad) 

        # Call vpr function to visualize relation of 5 patches
        final_img = visualize_patches_relation(
            block, target_patch,
            x_pixel, y_pixel, x_cosa, y_sina, theta
        )
        
        # Save result image
        if save_dir:
            cv2.imwrite(save_dir, final_img)
            
        print(f"Patches position check is saved to: {save_dir}")

    def calculate_ref_direct_angle(self, cur_point_name, target_waypoint):
        """ 
        Input: Current nominal longitude/latitude coordinates, next waypoint longitude/latitude coordinates
        Output: Reference vector and heading angle in longitude/latitude coordinate system
        """

        # Calculate reference heading angle (due north is 0, clockwise is positive) unit vector
        delta_lon = target_waypoint[0] - cur_point_name[0]
        delta_lat = target_waypoint[1] - cur_point_name[1]
        ref_direct_llcs = normalize_vector([delta_lon, delta_lat])  # Reference direction unit vector in llcs coordinate system

        # Convert direction vector to angle (0-360 degrees)
        ref_angle_llcs = vector2angle(delta_lon, delta_lat)

        return ref_direct_llcs, ref_angle_llcs

    def phreg(self, uav_frame_id, fly_angle_ccs, cur_point_real, block_center, block_indices):
        """
        Unified encapsulated position and direction regression interface.
        """
        # Get four patches corresponding to nearest block and process, get_patches function will unify information conversion to image ccs coordinate system
        # Note block info is nominal waypoint, while target_patch is actual waypoint
        patches, patches_fdirs, record = self.get_patches(uav_frame_id, fly_angle_ccs, cur_point_real, block_center, block_indices)
        if record["flag_out_of_map"] == True or len(patches)!=5:
            # print(f" * | UAV飞出地图边界，退出飞行！！！总步数{uav_frame_id}, 距离终点{distance_ep}米。")
            flag_out_of_map = True
            pos_pred, dir_pred = None, None
            print(f" * | UAV flies out of map boundary, exit flight!!!")
        else:
            flag_out_of_map = False
        
            # Position and direction regression.
            # UAV current real position is at r_i, algorithm regresses nominal position n_i based on r_i
            # pos_pred is unit relative coordinate predicted by regression model, range [-1, 1], dir_pred is unit direction vector predicted by regression model, range [-1, 1]
            pos_pred, dir_pred = self.position_angle_regression_(patches) 
        return pos_pred, dir_pred, flag_out_of_map, patches_fdirs, record
        


    def fly(self) -> None:
        """
        Execute flight process
        """
        uav_frame_id= 0  #uav step frame number
        waypoint_index = 0
        
        # Initialize data table
        df = pd.DataFrame(columns=[
            "block_x", "block_y", "target_path", 
            "x_norm", "y_norm", "angle",
            "p1_path", "p2_path", "p3_path", "p4_path"  # Add neighbor path fields
        ])

        nav_info = []  # Record all flight info, save as json file, each element is dict containing all flight process info of each step
        ob_records = []
        predictions = []
        flag_out_of_map = False
        while waypoint_index < len(self.waypoints) - 1:
            target_waypoint = self.waypoints[waypoint_index + 1]
            print(" * FLY | ......")
            print(f" __---++++ * wp num = {waypoint_index + 1}") 
            
            # Flight loop of current segment: update--predict--judge trilogy until reach current waypoint
            while True:
                
                self.nav_step = {}  # Record flight process info of each step
                
                self.nav_step['uav_frame_id'] = uav_frame_id
                self.nav_step['waypoint_index'] = waypoint_index + 1
                self.nav_step['target_waypoint'] = target_waypoint
                print( f"\n.....................cur waypoint_index={waypoint_index}, uav_frame_id={uav_frame_id}")

                # +----+
                # | 01 |    Initialization
                # +----+                
                # Calculate heading angle (0->360 counterclockwise from right) and calculated displacement cur_step (how much longitude/latitude of uav field of view center needs to move next step)
                # From second frame, calculate heading angle based on longitude/latitude position predicted by position regression
                if uav_frame_id==0:  # Calculate current waypoint increment and next nominal waypoint longitude/latitude coordinates based on direction and step size predicted by position regression point
                    #Starting point position is known, all actual sub-waypoints coincide with nominal sub-waypoints
                    # r_0 = start_point, n_0=r_0
                    cur_point_real = self.start_point  # Real longitude/latitude position r_i
                    cur_point_name = cur_point_real  # Nominal longitude/latitude position n_i
                    fly_angle_ccs = 90.0  #Assume UAV takes off with due north direction
                    #Note: heading angle calculated here is from nominal waypoint to target waypoint, not actual waypoint
                    # fly_angle, cur_step = self.calculate_heading(cur_point_name, target_waypoint)
                print(f"From  cur_point_real {cur_point_real} to wp {target_waypoint}")

                print(f"|step1| --- 1.Originalization ---")
                print(f" >> cur_point_real={cur_point_real}")
                print(f" >> cur_point_name={cur_point_name}")
                print(f" > fly_angle_ccs={fly_angle_ccs}")
                self.nav_step['step_id1'] = ' --- 1.Originalization ---'
                self.nav_step['cur_point_real'] = cur_point_real
                self.nav_step['cur_point_name'] = cur_point_name
                self.nav_step['fly_angle_ccs'] = fly_angle_ccs

                # +----+
                # | 02 |    Predict: position and direction p_i,a_i
                # +----+ 
                # r_i is used to get target_patch, n_i is used to get 4 base_patches (block center)
                # block_center unit is longitude/latitude, block_indices is block row/column number
                block_center, block_indices = self.get_nearest_block(cur_point_name)

                # Position and direction regression: use phreg unified interface
                # UAV current real position is at r_i, algorithm regresses nominal position n_i based on r_i
                # pos_pred is unit relative coordinate predicted by regression model, range [-1, 1], dir_pred is unit direction vector predicted by regression model, range [-1, 1]
                # In gift mode, patches_fdirs represents target_patch_fdir, making self.phreg universal
                pos_pred, dir_pred, flag_out_of_map, patches_fdirs, record = self.phreg(
                    uav_frame_id, fly_angle_ccs, cur_point_real, block_center, block_indices
                )
                
                # Check for out of bounds or invalid record
                if record == {} or flag_out_of_map == True:
                    print(f" * | UAV flies out of map boundary, exit flight!!! Total steps {uav_frame_id}, distance to end point {distance_ep} meters.")
                    flag_out_of_map = True
                    break 
                
                x_pred_, y_pred_ = pos_pred

                # Predict longitude/latitude of current UAV position point, i.e. current nominal longitude/latitude position
                cur_point_pred = self.pred_point_from_norm(x_pred_, y_pred_, block_center, block_size=512)
                direct_pred = normalize_vector(dir_pred)  # Normalize
                direct_pred_llcs = convert_ccs_to_llcs_vector(direct_pred, cur_point_pred[1])
                angle_pred = vector2angle(direct_pred[0], direct_pred[1])
                
                # Use planned uav flight step as prediction value for ideal path simulation, not prediction value from position regression.
                # These lines are for comparison, not used in actual inference
                x_pred, y_pred = self.position_regression(record["x_norm"], record["y_norm"])
                pred_point = self.pred_point_from_norm(x_pred, y_pred, block_center, block_size=512)
                pred_error = [x_pred_ - x_pred, y_pred_ - y_pred]
                pred_err_meter = [x * 36 for x in pred_error]

                print(f"|step2| --- 2.Regression RA ---")
                print(f" <> Nearest block_indices={block_indices}")
                print(f" <> Nearest block_center={block_center}")
                print(f" >> fly_angle_ccs={fly_angle_ccs}")
                self.nav_step['step_id2'] = ' --- 2.Regression RA ---'
                self.nav_step['block_indices'] = block_indices
                self.nav_step['block_center'] = block_center
                self.nav_step['patches_fdirs'] = patches_fdirs
                self.nav_step['pos_pred'] = pos_pred
                self.nav_step['dir_pred'] = dir_pred
                self.nav_step['cur_point_real'] = cur_point_real
                self.nav_step['cur_point_name'] = cur_point_name
                self.nav_step['cur_point_pred'] = cur_point_pred
                self.nav_step['cur_angle_pred'] = angle_pred
                self.nav_step['cur_direct_pred'] = direct_pred
                self.nav_step['fly_angle_ccs2'] = fly_angle_ccs


                # +----+
                # | 03 |    Update heading angle a_i
                # +----+ 
                # In longitude/latitude coordinate system, calculate reference heading angle using predicted position p_i of current step
                ref_direct_llcs, ref_angle_llcs = self.calculate_ref_direct_angle(cur_point_pred, target_waypoint) 

                # Vector coordinate system conversion to get corresponding heading angle in Cartesian coordinate system and its corresponding unit vector
                ref_direct_ccs = convert_llcs_to_ccs_vector(ref_direct_llcs, cur_point_name[1])
                ref_angle_ccs = vector2angle(ref_direct_ccs[0], ref_direct_ccs[1])
                #Navigation strategy: update heading angle based on predicted nominal position and local target waypoint
                fly_angle_ccs = ref_angle_ccs + 360 if ref_angle_ccs < 0 else ref_angle_ccs
                fly_direct_ccs = angle2vector(fly_angle_ccs)  #Direction vector

                print(f"|step3| ---3.Update fly angle ---")
                print(f" >> ref_direct_llcs={ref_direct_llcs}")
                print(f" >> ref_angle_llcs={ref_angle_llcs}")
                print(f" >>> fly_angle_ccs={fly_angle_ccs}")  #Updated heading angle
                print(f" >> fly_direct_ccs={fly_direct_ccs}")
                self.nav_step['step_id3'] = ' ---3.Update fly angle --- '
                self.nav_step['fly_angle_ccs3'] = fly_angle_ccs  #Updated heading angle
                self.nav_step['ref_angle_ccs'] = ref_angle_ccs
                self.nav_step['ref_angle_llcs'] = ref_angle_llcs
                self.nav_step['fly_direct_ccs'] = fly_direct_ccs
                self.nav_step['ref_direct_ccs'] = ref_direct_ccs
                self.nav_step['ref_direct_llcs'] = ref_direct_llcs
                self.nav_step['ref_direct_llcs_length'] = math.hypot(*ref_direct_llcs)


                # +----+
                # | 04 |    Move s_i: UAV move one step (in llcs coordinate system)
                # +----+ 
                # After UAV heading angle is determined, calculate step increment based on step size, note: step increment here is in lon/lat coordinate system
                # cur_step_llcs = calculate_lon_lat_step(cur_point_name, fly_direct_ccs, self.uav_step)
                cur_step_llcs = calculate_lon_lat_step_underllcs(cur_point_pred, ref_direct_llcs, self.uav_step)

                # Get next nominal and actual positions by adding step increment to current nominal and actual positions (two coordinate systems)
                # Next actual waypoint = current actual waypoint + step increment, unit: degree
                # r_i+1 = r_i + s_i
                next_point_real = (cur_point_real[0] + cur_step_llcs[0], cur_point_real[1] + cur_step_llcs[1])
                # n_i+1 = p_i + s_i , Note: not n_i+1 = n_i + s_i, because p_i is the start point of next nominal waypoint
                # That is, actual position is continuous, while nominal position is jumpy, which is the result of predicted position + step increment
                next_point_name = (cur_point_pred[0] + cur_step_llcs[0], cur_point_pred[1] + cur_step_llcs[1])

                print(f"|step4| --- 4.Moving one step ---")
                print(f" >> cur_step_llcs={cur_step_llcs}")
                print(f" >> cur_point_real={cur_point_real}")  
                print(f" >> next_point_real={next_point_real}")
                print(f" >> cur_point_pred={cur_point_pred}")  
                print(f" >> next_point_name={next_point_name}")
                self.nav_step['step_id4'] = ' --- 4.Moving one step --- '
                self.nav_step['cur_step_llcs'] = cur_step_llcs  
                self.nav_step['cur_point_real'] = cur_point_real
                self.nav_step['next_point_real'] = next_point_real
                self.nav_step['cur_point_pred'] = cur_point_pred
                self.nav_step['next_point_name'] = next_point_name

                # +----+
                # | 05 |    Judgment
                # +----+    
                # Judge if reach wp and ep   
                distance_wp_real, reached_wp_real = self.calculate_distance(cur_point_real, target_waypoint)
                distance_wp_pred, reached_wp_pred = self.calculate_distance(cur_point_pred, target_waypoint)
                distance_wp = min(distance_wp_real, distance_wp_pred)
                reached_wp = reached_wp_real or reached_wp_pred

                # Check if current actual UAV lon/lat sub-waypoint reaches end point
                distance_ep, reached_ep = self.calculate_distance(cur_point_real, self.end_point)

                print(f"|step5| --- 5.Check distance ---")
                print(f" >> distance_wp={distance_wp}")
                print(f" >> reached_wp={reached_wp}")
                print(f" >> distance_ep={distance_ep}")
                print(f" >> reached_ep={reached_ep}")
                self.nav_step['step_id5'] = ' --- 5.Check distance --- '
                self.nav_step['distance_wp'] = distance_wp
                self.nav_step['reached_wp'] = reached_wp
                self.nav_step['distance_ep'] = distance_ep
                self.nav_step['reached_ep'] = reached_ep

                # Record current step info
                record_dict = {
                    'uav_frame_id'    : uav_frame_id,
                    'waypoint_index'  : waypoint_index,
                    'distance_wp'     : distance_wp,
                    'distance_ep'     : distance_ep,
                    'reached_waypoint': reached_wp,
                    'reached_endpoint': reached_ep,
                    'cur_lon_name'    : cur_point_name[0],
                    'cur_lat_name'    : cur_point_name[1],
                    'next_lon_name'   : next_point_name[0],
                    'next_lat_name'   : next_point_name[1],
                    'cur_lon_real'    : cur_point_real[0],
                    'cur_lat_real'    : cur_point_real[1],
                    'next_lon_real'   : next_point_real[0],
                    'next_lat_real'   : next_point_real[1],
                    'cur_lon_pred'    : cur_point_pred[0],
                    'cur_lat_pred'    : cur_point_pred[1],  
                    'norm_lon'        : pred_point[0],
                    'norm_lat'        : pred_point[1],
                    'block_center_lon': block_center[0],
                    'block_center_lat': block_center[1],
                    'block_row'       : block_indices[0],
                    'block_col'       : block_indices[1],
                    'ref_direct_llcs_x':ref_direct_llcs[0],
                    'ref_direct_llcs_y':ref_direct_llcs[1],
                    'direct_pred_llcs_x': direct_pred_llcs[0],
                    'direct_pred_llcs_y': direct_pred_llcs[1],
                }
                
                # Add record to DataFrame
                self.records = pd.concat([self.records, pd.DataFrame([record_dict])], ignore_index=True)
                
                nav_info.append(self.nav_step)
                ob_records.append(record)
                
                # Update current position
                cur_point_real = next_point_real
                cur_point_name = next_point_name
                uav_frame_id += 1

                if reached_wp:  #If reach expected waypoint, enter next loop
                    waypoint_index += 1
                    break
                elif uav_frame_id>=self.max_steps:
                    print(f" * | Flight steps exceed {self.max_steps}, exit flight!!! Total steps {uav_frame_id}, distance to end {distance_ep} meters.")
                    break
                
            if reached_ep:
                print(f" * | Reach end point!!! Total steps {uav_frame_id}, distance to end {distance_ep} meters." )
                break 
            elif uav_frame_id>=self.max_steps:
                print(f" * | Flight steps exceed {self.max_steps}, exit flight!!! Total steps {uav_frame_id}, distance to end {distance_ep} meters.")
                break
            elif flag_out_of_map:
                print(f" * | UAV out of map boundary, exit flight!!! Total steps {uav_frame_id}, distance to end {distance_ep} meters.")
                break
        
        # Save flight records for plotting (display on lon/lat map)
        records_dir = self.output_dir + f"/{self.rs_traj_id}_uav_traj_records.csv"
        self.records.to_csv(records_dir, index=False) 

        # Save metadata for future comparison:block_x,block_y,target_path,x_norm,y_norm,angle,p1_path,p2_path,p3_path,p4_path
        df = pd.DataFrame(ob_records)  # Create once
        observe_dir = self.output_dir + f"/{self.rs_traj_id}_uav_traj_observe.csv"
        df.to_csv(observe_dir, index=False)

        # Save json data by navigation stage for human observation
        nav_info_serializable = convert_to_json_serializable(nav_info)
        nav_info_dir = self.output_dir + f"/{self.rs_traj_id}_uav_traj_nav_info.json"
        with open(nav_info_dir, 'w', encoding='utf-8') as f:
            json.dump(nav_info_serializable, f, ensure_ascii=False, indent=4)

        vis_waypoints_uavtrajs_on_fig_v4(self.waypoints, self.uav_step, records_dir, uav_2d3d=self.uav_2d3d)
        print(f" * | Data saved! {records_dir}")
        print(f" * | Data saved! {observe_dir}")
        print(f" * | Data saved! {nav_info_dir}")



def main_nav_test(
    rsi_id='38bc', 
    rsi_type='254k',
    traj_id=0, 
    uav_step =30, 
    uav_2d3d='2d', 
    th_arrive=None, 
    project_dir='',
    cvphr_3d_best_model_dir='',
    cvphr_2d_best_model_dir='',
    flag_suppl=False):
    """
    flag_suppl: Whether to use supplementary waypoints to test model performance on supplementary waypoints
    User needs to specify:
        1. Device info: device_id
        2. Project path: project_dir
        3. Remote sensing image batch: rsi_group
        4. Remote sensing image ID: rsi_id, id-name dict id_image_map, (auto-generate experiment result prefix)
        5. Navigation path: path ID, waypoint json file path (waypoint coordinate list)
        6. Flight step size: uav_step, and arrival threshold th_arrive
        7. Model: model structure config file and training result"""
    
    # 1.Your device info: (default no modification)
    device_id = 0
    
    # 2.Your project path
    project_dir = project_dir  # Modify this path if necessary
    
    # 4. Remote sensing image: rsi_id, id-name dict, see cvphr_base_info/id_image_map,
    rsi_id = rsi_id  #'38bc'  #Remote sensing image path is input in step 7 according to selected batch
    rsi_type = rsi_type  #Remote sensing image resolution and size ('145k', '254k', '504k', '1m4k')
    uav_2d3d = uav_2d3d  # '2d' or '3d'
    
    # 5.Your navigation path info:
    traj_id = traj_id  # 4  # Trajectory ID of the remote sensing image, start from 0, do not exceed the number of trajectories of the remote sensing image
        
    # 6.Your flight step size:
    # Maximum flight distance is designed as 150 times the field of view width, e.g., 254k is 64*150=9600 meters
    max_steps = int(MAX_DISTANCE/uav_step)  # #Maximum UAV flight steps
    # Maximum flight distance is designed as 150 times FOV_UNIT, e.g., 254k is 64*150=9600 meters
    # Recommended step size 15 meters for batch 1 remote sensing maps, 30 meters for batch 2
    th_arrive = th_arrive  # None  # If user does not specify th_arrive, use default value with built-in judgment in nested function
    
    # 7.Your trained pth model path:
    n_block = 15
    if uav_2d3d == '3d':    # Optim 3D Model
        phr_model_dir = cvphr_3d_best_model_dir
    
    if uav_2d3d == '2d':    # Optim 2D Model
        phr_model_dir = cvphr_2d_best_model_dir
        
    # Your remote sensing map path to test
    rsi_dir = rsi_dir_city8_25pp_4096bc
    

    #  ┌─────────┐
    #  │         │
    #  │         │ # Auto-generated info, generally no modification needed  
    #  │         │
    #  └─────────┘     

    # Load model config
    model_class, model_kwargs = load_config_and_model(phr_model_dir)
    model_name = model_class.get_model_name()
    posreg_model_dir = f"{phr_model_dir}/best_model.pth"
    print(f"Model name: {model_name}")
    print(f"Model path: {posreg_model_dir}")
    
    # Get remote sensing image info
    rsi_name = id_image_map[rsi_id]
    rsi_image_path = f"{rsi_dir}/{rsi_name}"
    # Use os.path.splitext() to get JSON path from image path quickly
    rsi_json_path = os.path.splitext(rsi_image_path)[0] + '.json'
    
    # Build trajectory ID
    str_rsi_id = '_0105' if rsi_id == 5 else rsi_id  # Special case handling
    rs_traj_id = f"{str(str_rsi_id).zfill(4)}_{str(traj_id).zfill(2)}"

    # Load waypoint coordinate data
    
    if flag_suppl:
        wp_json_path = f"{project_dir}/loc2traj/traj_wps_gcs/loc2traj_general_{str_rsi_id}/wps_{str(traj_id).zfill(2)}.json"
    else:    
        wp_json_path = f"{project_dir}/loc2traj/traj_wps_gcs/wps{rs_traj_id}.json"
    
    if not os.path.exists(wp_json_path):
        print(f"Waypoint file does not exist: {wp_json_path}, check/skip this experiment.")
        # continue  
    with open(wp_json_path, 'r', encoding='utf-8') as file:
        way_point_centers = json.load(file)
    
    # Extract waypoint info
    end_wp_id = len(way_point_centers) - 1
    start_point = way_point_centers['0']['lnglat']
    end_point = way_point_centers[str(end_wp_id)]['lnglat']
    waypoints = [point['lnglat'] for point in way_point_centers.values()]

    # Create result directory
    subfile = f"nav_{rsi_id}{str(traj_id).zfill(2)}_s{str(uav_step).zfill(2)}_t{th_arrive}_d{uav_2d3d}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nav_result_dir = f"{project_dir}/loc2traj/{subfile}_{model_name}_{timestamp}"
    os.makedirs(nav_result_dir, exist_ok=True)

    # Get remote sensing map info and generate grid blocks
    drsi = rsijson2info(rsi_json_path)
    # Generate block center coordinate dict of remote sensing map rs_id for subsequent block matching
    block_cnt_point_dict = generate_grid_blocks(
        drsi["width_pixel"], drsi["height_pixel"],
        drsi["lng"], drsi["lat"],
        drsi["lng_per_pixel"], drsi["lat_per_pixel"],
        n_block=n_block
    )
    
    # Save grid block info
    block_info_path = f"{nav_result_dir}/{rsi_id}_block_cnt_point_dicts.json"
    with open(block_info_path, "w", encoding="utf-8") as f:
        json.dump(block_cnt_point_dict, f, ensure_ascii=False, indent=4)

    
    # Execute flight test
    if True:  # Flight test switch
        print("=" * 60)
        print("Flying Test")
        print("=" * 60)
        
        # Create navigation object
        nav = UAVNavigation(
            start_point=start_point,  # Coordinates of a point in Beijing
            end_point=end_point,    # Target point coordinates
            waypoints=waypoints, # Waypoint sequence
            block_cnt_point_dict=block_cnt_point_dict, # Block dict
            rs_image_dir=rsi_image_path,  # Remote sensing map
            rsi_type=rsi_type,  #Remote sensing image resolution and size ('145k', '254k', '504k', '1m4k')
            rs_traj_id=rs_traj_id,  # Trajectory ID
            device_id=device_id,
            uav_2d3d=uav_2d3d,
            uav_step=uav_step,
            max_steps=max_steps,
            th_arrive=th_arrive,
            posreg_model_dir=posreg_model_dir,  #Model pth file path
            output_dir=nav_result_dir,   #Trajectory output path
            # model_class=PositionRegressionModel,
            model_class=model_class,
            model_kwargs=model_kwargs,
            dataset_kwargs={}
        )
        print("✓ UAVNavigation instance created successfully")
        print(f"  Device: {nav.device}")
        print(f"  Model type: {type(nav.model)}")
        print(f"  Transform type: {type(nav.nav_transform)}")

        # Execute navigation
        nav.fly()

                           
def parse_args():
    """
    Command line argument parsing: make all test switches and list parameters under original main configurable.
    """
    parser = argparse.ArgumentParser(
        description="UAV navigation test runner (from uav_navigation_v2.py)"
    )

    # Three test mode switches
    parser.add_argument(
        "--nav_test",
        action="store_true",
        default=False,
        help="Enable nav_test mode (disabled by default)",
    )
    parser.add_argument(
        "--cvphr_test",
        action="store_true",
        default=False,
        help="Enable cvphr_test mode (disabled by default)",
    )
    parser.add_argument(
        "--suppl_test",
        action="store_true",
        default=False,
        help="Enable suppl_test mode (disabled by default)",
    )

    # List parameters, support multiple values, e.g.: --th-arrive 20 50
    parser.add_argument(
        "--th_arrive",
        nargs="+",
        type=int,
        default=[20],
        help="Arrival threshold list, default [20]",
    )
    parser.add_argument(
        "--uav_step",
        nargs="+",
        type=int,
        default=[25],
        help="UAV step size list, default [25]",
    )
    parser.add_argument(
        "--rsi_id",
        nargs="+",
        type=str,
        default=["34bc"],
        help="RSI ID list, default ['34bc','36bc','37bc','38bc']",
    )
    parser.add_argument(
        "--traj_id",
        nargs="+",
        type=int,
        default=[50],
        help="Trajectory ID list; e.g.[50,51]",
    )

    # Other parameters
    parser.add_argument(
        "--rsi_type",
        type=str,
        default="254k",
        help="rsi_type, default '254k'",
    )
    parser.add_argument(
        "--uav_2d3d",
        type=str,
        choices=["2d", "3d"],
        default="2d",
        help="UAV navigation uses 2D or 3D model, default '2d'",
    )
    parser.add_argument(
        "--project_dir",
        type=str,
        default=None,
        help="Project root directory, use cvphr_base_info.proj_dir if not passed",
    )
    parser.add_argument(
        "--cvphr_3d_best_model_dir",
        type=str,
        default="./Bearing_UAV/cross_view",
        help="3D best model directory",
    )
    parser.add_argument(
        "--cvphr_2d_best_model_dir",
        type=str,
        default="./Bearing_UAV/satellite_view",
        help="2D best model directory",
    )

    return parser.parse_args()

def main():
    args = parse_args()

    # Import default project_dir
    project_dir = args.project_dir or proj_dir

    # Determine which set of parameters to use based on three test switches
    # Compatible with old logic: if user explicitly gives list via command line, follow command line;
    # otherwise use default values in each test mode.
    if args.nav_test:
        lst_th_arrive = [20]
        lst_uav_step = [25]
        lst_rsi_id = ["34bc"]
        lst_traj_id = [50]
    elif args.cvphr_test:
        lst_th_arrive = [20, 50]
        lst_uav_step = [20, 25, 30]
        lst_rsi_id = ["34bc", "36bc", "37bc", "38bc"]
        lst_traj_id = [50, 51]
    elif args.suppl_test:
        lst_th_arrive = [20]
        lst_uav_step = [25]
        lst_rsi_id = ["34bc", "36bc", "37bc", "38bc"]
        lst_traj_id = list(range(1, 20))
    else:
        # If none of the three switches are on, all follow command line (or default)
        lst_th_arrive = args.th_arrive
        lst_uav_step = args.uav_step
        lst_rsi_id = args.rsi_id
        lst_traj_id = args.traj_id

    rsi_type = args.rsi_type
    uav_2d3d = args.uav_2d3d
    cvphr_3d_best_model_dir = args.cvphr_3d_best_model_dir
    cvphr_2d_best_model_dir = args.cvphr_2d_best_model_dir

    dict_traj = {rsi_id: lst_traj_id for rsi_id in lst_rsi_id}
    print(dict_traj)

    # Test function loop mode
    for th_arrive in lst_th_arrive:
        for uav_step in lst_uav_step:
            for rsi_id in lst_rsi_id:
                for traj_id in dict_traj[rsi_id]:
                    main_nav_test(
                        rsi_id=rsi_id,
                        rsi_type=rsi_type,
                        traj_id=traj_id,
                        uav_step=uav_step,
                        uav_2d3d=uav_2d3d,
                        th_arrive=th_arrive,
                        project_dir=project_dir,
                        cvphr_3d_best_model_dir=cvphr_3d_best_model_dir,
                        cvphr_2d_best_model_dir=cvphr_2d_best_model_dir,
                        flag_suppl=args.suppl_test,  #if U want suppl test
                    )


if __name__ == "__main__":
    # Directly run with default parameters,
    # or run by "test_nav.sh" with more settings.
    main()