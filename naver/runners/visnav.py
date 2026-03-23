# Tool functions of nav.py
import os
import cv2
import csv
import math
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path

from config.base_info import UNI_PIXEL
from config.base_info import PATCH_SIZE
from config.base_info import BLOCK_SIZE
from source.uav_logo import pipeline_add_drone_logo


def visualize_patches_relation(
    block, target_patch,
    x_pixel, y_pixel, x_cosa, y_sina, theta, alt=None, scale_alt=1    
):
    """
    Visualize spatial relationship between 4 neighboring patches of block and target patch, return final_img.
    Args:
        block: Block composed of 4 neighboring patches (top-left, bottom-left, top-right, bottom-right), BGR format
        target_patch: Target patch, BGR format
        x_pixel, y_pixel: Normalized coordinates of target patch in block
        x_cosa, y_sina: Direction vector
        theta: Heading angle (degrees)
        logo_path: Path of UAV logo image
        UNI_PIXEL, PATCH_SIZE, BLOCK_SIZE: Size parameters
    Returns:
        final_img: Visualization result (numpy array)
    """
    # Auto set logo path
    THIS_FILE = Path(__file__).resolve()
    # Project root: .../cvphr
    PROJECT_ROOT = THIS_FILE.parents[2]   # dataset -> data -> cvphr
    # Logo path
    logo_path = PROJECT_ROOT / "source" / "uav_logo" / "plane.png"
    if not logo_path.exists():
        raise FileNotFoundError(f"Cannot load Logo image: {logo_path}")
    
    # Draw green coordinate system, red UAV center & heading, blue UAV field of view on block
    # Draw green cross guide line on block
    cv2.line(block, (0, PATCH_SIZE), (PATCH_SIZE*2, PATCH_SIZE), (0, 255, 0), 1)
    cv2.line(block, (PATCH_SIZE, 0), (PATCH_SIZE, PATCH_SIZE*2), (0, 255, 0), 1)
    # Draw green unit Cartesian coordinate box on block
    cv2.line(block, (UNI_PIXEL, UNI_PIXEL)  , (UNI_PIXEL*3, UNI_PIXEL)  , (0, 255, 0), 1)
    cv2.line(block, (UNI_PIXEL, UNI_PIXEL*3), (UNI_PIXEL*3, UNI_PIXEL*3), (0, 255, 0), 1)
    cv2.line(block, (UNI_PIXEL, UNI_PIXEL)  , (UNI_PIXEL, UNI_PIXEL*3)  , (0, 255, 0), 1)
    cv2.line(block, (UNI_PIXEL*3, UNI_PIXEL), (UNI_PIXEL*3, UNI_PIXEL*3), (0, 255, 0), 1)
    
    # Draw yellow UAV heading angle in block Cartesian coordinate system
    arrow_length = int(UNI_PIXEL/2)  # Arrow length
    # Calculate arrow end coordinates
    arrow_end_x = int(x_pixel + x_cosa * arrow_length)
    # Note "-" is used here because y-axis is positive downward in pixel coordinate system, opposite to block Cartesian coordinate system
    arrow_end_y = int(y_pixel - y_sina * arrow_length)
    arrow_end = (arrow_end_x, arrow_end_y)
    # Draw main line segment
    cv2.line(block, (x_pixel, y_pixel), arrow_end, (0, 255, 255), 2)  # Yellow line
    # Draw three-layer colored (red-orange-yellow) dots for UAV red centroid and heading
    cv2.circle(block, (x_pixel, y_pixel), radius=6, color = (0, 255, 255), thickness=-1)
    cv2.circle(block, (x_pixel, y_pixel), radius=4, color = (0, 165, 255), thickness=-1)
    cv2.circle(block, (x_pixel, y_pixel), radius=2, color = (0,   0, 255), thickness=-1)            
    cv2.circle(block, arrow_end, radius=6, color = (0, 255, 255), thickness=-1)
    cv2.circle(block, arrow_end, radius=4, color = (0, 165, 255), thickness=-1)
    cv2.circle(block, arrow_end, radius=2, color = (0,   0, 255), thickness=-1)

    
    # Add logo on block
    block = pipeline_add_drone_logo(
        background=block,
        logo_path=logo_path,
        position=(x_pixel, y_pixel),
        angle_deg=theta,
        scale=0.4,
        alpha_factor=0.7
    )
    
    # Draw rotated square with [x_pixel,y_pixel] as center and side length of PATCH_SIZE
    half_size = PATCH_SIZE // 2
    # Calculate four vertices of square (before rotation)
    points = np.array([
        [x_pixel - half_size, y_pixel - half_size],  # Top-left
        [x_pixel + half_size, y_pixel - half_size],  # Top-right
        [x_pixel + half_size, y_pixel + half_size],  # Bottom-right
        [x_pixel - half_size, y_pixel + half_size]   # Bottom-left
    ], dtype=np.int32)
    # Create rotation matrix: Note no negative sign for theta, because it's UAV self-rotation!
    M = cv2.getRotationMatrix2D((x_pixel, y_pixel), theta, 1.0)
    # Rotate all points
    rotated_points = cv2.transform(np.array([points]), M)[0]
    # Draw rotated square
    for k in range(4):
        pt1 = tuple(rotated_points[k])
        pt2 = tuple(rotated_points[(k + 1) % 4])
        cv2.line(block, pt1, pt2, (255, 255, 255), 1)  # White line

    # Draw UAV field of view=======When altitude changes, it does not coincide with the standard patch above
    half_size = scale_alt * PATCH_SIZE // 2
    # Calculate four vertices of square (before rotation)
    points = np.array([
        [x_pixel - half_size, y_pixel - half_size],  # Top-left
        [x_pixel + half_size, y_pixel - half_size],  # Top-right
        [x_pixel + half_size, y_pixel + half_size],  # Bottom-right
        [x_pixel - half_size, y_pixel + half_size]   # Bottom-left
    ], dtype=np.int32)
    # Create rotation matrix: Note no negative sign for theta, because it's UAV self-rotation!
    M = cv2.getRotationMatrix2D((x_pixel, y_pixel), theta, 1.0)
    # Rotate all points
    rotated_points = cv2.transform(np.array([points]), M)[0]
    # Draw rotated square
    for k in range(4):
        pt1 = tuple(rotated_points[k])
        pt2 = tuple(rotated_points[(k + 1) % 4])
        cv2.line(block, pt1, pt2, (255, 0, 0), 2)  # Blue line
    
    # Process UAV patch, draw center point, direction line and blue frame
    # Expand 256 pixels to the right and place target image
    
    # Resize target image to 256x256
    target_patch = cv2.resize(target_patch, (PATCH_SIZE, PATCH_SIZE))  #？
    center_x, center_y = UNI_PIXEL, UNI_PIXEL  # Center point of 256x256 image
    # Draw yellow heading
    cv2.line(target_patch, (center_x, center_y), (center_x + arrow_length, center_y), (0, 255, 255), 2)
    # Draw three-layer colored (red-orange-yellow) dots for UAV red centroid and heading
    cv2.circle(target_patch, (center_x, center_y), radius=6, color = (0, 255, 255), thickness=-1)
    cv2.circle(target_patch, (center_x, center_y), radius=4, color = (0, 165, 255), thickness=-1)
    cv2.circle(target_patch, (center_x, center_y), radius=2, color = (0,   0, 255), thickness=-1)
    cv2.circle(target_patch, (center_x + arrow_length, center_y), radius=6, color = (0, 255, 255), thickness=-1)
    cv2.circle(target_patch, (center_x + arrow_length, center_y), radius=4, color = (0, 165, 255), thickness=-1)
    cv2.circle(target_patch, (center_x + arrow_length, center_y), radius=2, color = (0, 0, 255), thickness=-1)

    # Add logo on target_patch
    target_patch = pipeline_add_drone_logo(
        background=target_patch,
        logo_path=logo_path,
        position=(center_x, center_y),
        angle_deg=0,  # Logo in target_img does not need rotation
        scale=0.4,
        alpha_factor=0.7
    )

    # Draw blue square around target_img
    square_size = PATCH_SIZE // 2  # Square side length is half of image size
    # Calculate four vertices of square
    square_points = np.array([
        [center_x - square_size, center_y - square_size],  # Top-left
        [center_x + square_size, center_y - square_size],  # Top-right
        [center_x + square_size, center_y + square_size],  # Bottom-right
        [center_x - square_size, center_y + square_size]   # Bottom-left
    ], dtype=np.int32)
    # Draw square
    for m in range(4):
        pt1 = tuple(square_points[m])
        pt2 = tuple(square_points[(m + 1) % 4])
        cv2.line(target_patch, pt1, pt2, (255, 0, 0), 2)  # Blue line
    
    
    # Build combined image, display angle and relative coordinates in bottom-right block
    # Display normalized coordinate information near center point
    # Create final image (512x768)
    final_img = np.zeros((PATCH_SIZE*2, PATCH_SIZE*3, 3), dtype=np.uint8)
    final_img[:, :BLOCK_SIZE] = block  # Block on left side
    final_img[:PATCH_SIZE, BLOCK_SIZE:] = target_patch  # Target on right side
    
    # Calculate text position to ensure it's within image range
    text_x = BLOCK_SIZE + 20  # 
    text_y = PATCH_SIZE + 20 # Offset 40 pixels up, but not exceeding image boundary
    line_height = 20  # Line spacing

    # Add three lines of text
    x_norm = (x_pixel - PATCH_SIZE) / UNI_PIXEL
    y_norm = (y_pixel - PATCH_SIZE) / UNI_PIXEL
    cv2.putText(final_img, f"theta: {theta:.1f}", (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(final_img, f"x_norm: {x_norm:.2f}", (text_x, text_y + line_height), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(final_img, f"y_norm: {y_norm:.2f}", (text_x, text_y + 2*line_height), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(final_img, f"x_cosa: {x_cosa:.2f}", (text_x, text_y + 3*line_height), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(final_img, f"y_sina: {y_sina:.2f}", (text_x, text_y + 4*line_height), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    if alt != None:            
        cv2.putText(final_img, f"alt: {alt:.2f}", (text_x, text_y + 5*line_height), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return final_img

def crop_target_patch(x_rsi, y_rsi, theta, cropped_img):
    """Crop target patch after rotating cropped_img according to UAV center pixel coordinates in rsi (non-block)"""
    # In OpenCV's rotation operation combining cv2.getRotationMatrix2D and cv2.warpAffine:
    # Rotation direction is determined by the sign of parameter theta (angle):
    # Positive angle (theta > 0): Image rotates counterclockwise
    # Negative angle (theta < 0): Image rotates clockwise
    # Note the -theta here: cropped_img rotates clockwise by theta degrees,
    # which is equivalent to UAV rotating counterclockwise by theta degrees without moving, consistent with definition
    R = cv2.getRotationMatrix2D((x_rsi, y_rsi), -theta, 1)
    rotated_block = cv2.warpAffine(
        cropped_img, R,  (cropped_img.shape[0], cropped_img.shape[1]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )
    
    # Crop target patch
    center_x, center_y = int(x_rsi), int(y_rsi)
    target_patch = rotated_block[
        center_y-UNI_PIXEL:center_y+UNI_PIXEL,
        center_x-UNI_PIXEL:center_x+UNI_PIXEL
    ]
    return target_patch

def calculate_theta_from_direction_vector(x_cosa, y_sina):
    """
    Calculate angle theta from direction vector [x_cosa, y_sina]
    
    Args:
        x_cosa: x component of direction vector (cos value)
        y_sina: y component of direction vector (sin value)
        
    Returns:
        theta: Angle value (degrees), range [0, 360)
        
    Notes:
        - Angle definition: Positive right direction is 0 degrees, counterclockwise is positive
        - Consistent with Cartesian coordinate system:
          First quadrant (x>0, y>0): 0° < theta < 90°
          Second quadrant (x<0, y>0): 90° < theta < 180°  
          Third quadrant (x<0, y<0): 180° < theta < 270°
          Fourth quadrant (x>0, y<0): 270° < theta < 360°
    """
    # Calculate angle (radians) using atan2
    theta_rad = np.arctan2(y_sina, x_cosa)
    
    # Convert to degrees
    theta = np.degrees(theta_rad)
    
    # Ensure angle is in [0, 360) range
    theta = theta + 360 if theta < 0 else theta
    
    return theta

if __name__ == '__main__':
    pass