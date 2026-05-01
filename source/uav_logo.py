import os
import cv2
import math
import numpy as np
from typing import Tuple


def vector2angle(dx, dy):    
    # Convert direction vector to angle (0-360 degrees), note the order of dx and dy
    angle_rad = math.atan2(dy, dx)
    angle = math.degrees(angle_rad)
    angle = angle + 360 if angle < 0 else angle
    return angle

def angle2vector(theta):
    """Input angle, output corresponding direction vector"""
    theta_rad = theta * np.pi / 180  # Manual conversion to radians
    x_cosa = np.cos(theta_rad)
    y_sina = np.sin(theta_rad)
    return (x_cosa, y_sina)

def pipeline_add_drone_logo(
    background: np.ndarray,
    logo_path: str,
    position: Tuple[int, int],
    angle_deg: float,
    scale: float = 0.4,
    alpha_factor: float = 0.7
) -> np.ndarray:
    """
    Overlay and rotate drone Logo on background image (precise center alignment)
    
    Parameters:
        background: Background image (BGR format)
        logo_path: Logo image path (PNG with alpha channel supported)
        position: (x, y) Logo center position on background
        angle_deg: Rotation angle (degrees), clockwise positive
        scale: Scaling factor (default 0.4)
        alpha_factor: Transparency factor (0.0-1.0, 1.0 for fully opaque)
    
    Returns:
        Image after overlay
    """
    # Read Logo image and retain Alpha channel
    logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
    if logo is None:
        raise ValueError(f"Failed to load Logo image: {logo_path}")
    
    # Scale Logo
    if scale != 1.0:
        logo = cv2.resize(logo, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    # Get Logo size
    h, w = logo.shape[:2]
    center_x, center_y = position
    
    # Rotation matrix (rotate around Logo center)
    rotation_mat = cv2.getRotationMatrix2D((w/2, h/2), angle_deg, 1.0)
    
    # Rotate Logo (handle alpha channel case)
    if logo.shape[2] == 4:
        # Separate color and alpha channels
        logo_bgr = logo[:, :, :3]
        logo_alpha = logo[:, :, 3] / 255.0 * alpha_factor  # Apply transparency factor
        
        # Rotate separately
        rotated_bgr = cv2.warpAffine(logo_bgr, rotation_mat, (w, h), flags=cv2.INTER_LINEAR)
        rotated_alpha = cv2.warpAffine(logo_alpha, rotation_mat, (w, h), flags=cv2.INTER_LINEAR)
        
        # Ensure alpha is in [0,1] range
        rotated_alpha = np.clip(rotated_alpha, 0, 1)
    else:
        # Logo without alpha channel
        rotated_bgr = cv2.warpAffine(logo, rotation_mat, (w, h), flags=cv2.INTER_LINEAR)
        rotated_alpha = np.ones((h, w)) * alpha_factor  # Fully opaque
    
    # Calculate Logo boundaries on background (precise center alignment)
    x1 = center_x - w // 2
    y1 = center_y - h // 2
    x2 = x1 + w
    y2 = y1 + h
    
    # Calculate valid area in background
    bg_x1 = max(x1, 0)
    bg_y1 = max(y1, 0)
    bg_x2 = min(x2, background.shape[1])
    bg_y2 = min(y2, background.shape[0])
    
    # Return directly if completely out of frame
    if bg_x1 >= bg_x2 or bg_y1 >= bg_y2:
        return background
    
    # Calculate corresponding area in Logo
    logo_x1 = bg_x1 - x1
    logo_y1 = bg_y1 - y1
    logo_x2 = logo_x1 + (bg_x2 - bg_x1)
    logo_y2 = logo_y1 + (bg_y2 - bg_y1)
    
    # Extract Logo area to overlay
    logo_region = rotated_bgr[logo_y1:logo_y2, logo_x1:logo_x2]
    alpha_region = rotated_alpha[logo_y1:logo_y2, logo_x1:logo_x2]
    
    # Extract background area
    bg_region = background[bg_y1:bg_y2, bg_x1:bg_x2]
    
    # Blend images (consider alpha channel)
    for c in range(3):
        bg_region[:, :, c] = (bg_region[:, :, c] * (1 - alpha_region) + 
                             logo_region[:, :, c] * alpha_region).astype(np.uint8)
    
    # Put blended area back to background
    background[bg_y1:bg_y2, bg_x1:bg_x2] = bg_region
    
    return background

def overlay_drone_logo(
    background: np.ndarray,
    logo_path: str,
    position: Tuple[int, int],
    angle_deg: float,
    scale: float = 1.0,
    alpha_factor: float = 1.0
) -> np.ndarray:
    """
    Overlay and rotate drone Logo on background image (precise center alignment)
    
    Parameters:
        background: Background image (BGR format)
        logo_path: Logo image path (PNG with alpha channel supported)
        position: (x, y) Logo center position on background
        angle_deg: Rotation angle (degrees), clockwise positive
        scale: Scaling factor (default 1.0)
        alpha_factor: Transparency factor (0.0-1.0, 1.0 for fully opaque)
    
    Returns:
        Image after overlay
    """
    # Read Logo image and retain Alpha channel
    logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
    if logo is None:
        raise ValueError(f"Failed to load Logo image: {logo_path}")
    
    # Scale Logo
    if scale != 1.0:
        logo = cv2.resize(logo, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    # Get Logo size
    h, w = logo.shape[:2]
    center_x, center_y = position
    
    # Rotation matrix (rotate around Logo center)
    rotation_mat = cv2.getRotationMatrix2D((w/2, h/2), angle_deg, 1.0)
    
    # Rotate Logo (handle alpha channel case)
    if logo.shape[2] == 4:
        # Separate color and alpha channels
        logo_bgr = logo[:, :, :3]
        logo_alpha = logo[:, :, 3] / 255.0 * alpha_factor  # Apply transparency factor
        
        # Rotate separately
        rotated_bgr = cv2.warpAffine(logo_bgr, rotation_mat, (w, h), flags=cv2.INTER_LINEAR)
        rotated_alpha = cv2.warpAffine(logo_alpha, rotation_mat, (w, h), flags=cv2.INTER_LINEAR)
        
        # Ensure alpha is in [0,1] range
        rotated_alpha = np.clip(rotated_alpha, 0, 1)
    else:
        # Logo without alpha channel
        rotated_bgr = cv2.warpAffine(logo, rotation_mat, (w, h), flags=cv2.INTER_LINEAR)
        rotated_alpha = np.ones((h, w)) * alpha_factor  # Fully opaque
    
    # Calculate Logo boundaries on background (precise center alignment)
    x1 = center_x - w // 2
    y1 = center_y - h // 2
    x2 = x1 + w
    y2 = y1 + h
    
    # Calculate valid area in background
    bg_x1 = max(x1, 0)
    bg_y1 = max(y1, 0)
    bg_x2 = min(x2, background.shape[1])
    bg_y2 = min(y2, background.shape[0])
    
    # Return directly if completely out of frame
    if bg_x1 >= bg_x2 or bg_y1 >= bg_y2:
        return background
    
    # Calculate corresponding area in Logo
    logo_x1 = bg_x1 - x1
    logo_y1 = bg_y1 - y1
    logo_x2 = logo_x1 + (bg_x2 - bg_x1)
    logo_y2 = logo_y1 + (bg_y2 - bg_y1)
    
    # Extract Logo area to overlay
    logo_region = rotated_bgr[logo_y1:logo_y2, logo_x1:logo_x2]
    alpha_region = rotated_alpha[logo_y1:logo_y2, logo_x1:logo_x2]
    
    # Extract background area
    bg_region = background[bg_y1:bg_y2, bg_x1:bg_x2]
    
    # Blend images (consider alpha channel)
    for c in range(3):
        bg_region[:, :, c] = (bg_region[:, :, c] * (1 - alpha_region) + 
                             logo_region[:, :, c] * alpha_region).astype(np.uint8)
    
    # Put blended area back to background
    background[bg_y1:bg_y2, bg_x1:bg_x2] = bg_region
    
    return background

def draw_transparent_circle(
    target_img: np.ndarray,
    center: Tuple[int, int],
    radius: int = 5,
    edge_color: Tuple[int, int, int] = (0, 255, 255),  # Yellow edge (BGR)
    fill_color: Tuple[int, int, int] = (0, 165, 255),  # Orange center (BGR)
    edge_width: int = 1,
    alpha: float = 0.7  # Overall transparency (0.0-1.0)
) -> np.ndarray:
    """
    Draw transparent hollow circle (different colors for edge and center)
    
    Parameters:
        target_img: Target image (will be modified)
        center: (x, y) Circle center coordinates
        radius: Circle radius
        edge_color: Edge color (BGR)
        fill_color: Center color (BGR)
        edge_width: Edge width (pixels)
        alpha: Overall transparency (0.0 fully transparent, 1.0 opaque)
    """
    # Create temporary transparent layer
    overlay = target_img.copy()
    output = target_img.copy()
    
    center_x, center_y = center
    
    # 1. Draw filled orange center circle (radius minus edge width)
    cv2.circle(
        overlay, (center_x, center_y), 
        max(radius - edge_width, 0),  # Ensure inner radius is not less than 0
        fill_color, 
        thickness=-1  # -1 means filled
    )
    
    # 2. Draw yellow edge circle
    cv2.circle(
        overlay, (center_x, center_y), 
        radius, 
        edge_color, 
        thickness=edge_width
    )
    
    # 3. Apply transparency blending
    cv2.addWeighted(
        overlay, alpha,  # Source image and weight
        output, 1 - alpha,  # Target image and weight
        0,  # Gamma value
        output  # Output image
    )
    
    return output



if __name__ == "__main__":
    uav_logo_root = "./source/uav_logo"

    background_name = "blk_0_0_s_000_check.jpg"
    logo_name = "plane.png"

    background = cv2.imread(f'{uav_logo_root}/{background_name}')  # Load background image
    if background is None:
        raise FileNotFoundError("Failed to load background image")

    logo_path = f'{uav_logo_root}/{logo_name}'  # Drone Logo with transparent channel
    if not os.path.exists(logo_path):
        raise FileNotFoundError(f"Failed to load Logo image: {logo_path}")
    
    # Overlay Logo (center (300,200), rotate 58 degrees, scale 0.4x, transparency 0.7)
    result = overlay_drone_logo(
        background=background.copy(),
        logo_path=logo_path,
        position=(300, 200),
        angle_deg=58,
        scale=0.4,
        alpha_factor=0.7
    )

    # cv2.imshow('Result', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    output_stem = background_name.split(".")[0]
    output_path = os.path.join(uav_logo_root, f"{output_stem}_addlogo.jpg")
    cv2.imwrite(output_path, result)