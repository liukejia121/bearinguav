# Base informations.
import json
import math
import argparse
from pathlib import Path

# Project/data roots
from .paths import (
    # ---------- Project root directory
    proj_dir,  #".../cvphr" # PROJECT_ROOT

    # ---------- Dataset root directory
    dset_dir,  # ".../dataset"

    # ---------- Default path for cvphr dataset
    cvphr_dir,
    # .../dataset/posreg_dataset

    # ---------- Path to save full-size remote sensing images: 
    rsi_dir_city8_25pp_4096bc,
    # .../dataset/posreg_dataset/city8_25pp_4096bc

    # Legacy RSI path for Birmingham city
    rsi_dir_berm_city,  #.../dataset/rsi/block_5x5/Bermingham_block"
)

dct_models = {
    "phr5":"PARCASGM_v5a",
    "phr":"PositionHeadingRegression"
}


# Dataset split ratio for training/validation/testing
DATASET_SPLIT_RATIO = [0.85, 0.05, 0.1]  
# CVPHR model training loss weights for pos and head
PH_LOSS_WEIGHT = [0.8, 0.2] 
# #CVPHR model training loss weights for pos, head and alt
# PA_LOSS_WEIGHT = [0.72, 0.18, 0.1]
alpha = 0.1  # 0.1~0.4 is appropriate
PHA_LOSS_WEIGHT = [0.8*(1-alpha), 0.2*(1-alpha), alpha] # Training loss weights

# Mapping: RSI filename -> block id
image_id_map = {}

# c8_25pp_4096bc  Resolution: 0.25 meters/pixel
image_id_map["35.67091338738739_139.69289911300856_1791.95_1024_1024_4326_city.jpg"] = '34bc'   # Tokyo, Japan -- 0.25m/pixel
image_id_map["25.030947387387386_121.51462868800057_1791.95_1024_1024_4326_city.jpg"] = '36bc'  # Taipei, Taiwan -- 0.25m/pixel
image_id_map["1.2897673873873876_103.84197619336068_1791.95_1024_1024_4326_city.jpg"] = '37bc'  # Singapore, Singapore -- 0.25m/pixel
image_id_map["37.75538738738739_-122.4533351740761_1791.95_1024_1024_4326_city.jpg"] = '38bc'   # San Francisco, USA -- 0.25m/pixel

# Fusion RSI dictionary (7 fusion combos for c8 maps; c2=31+32, c3=31+32+33, etc.)
# This tag appears in trained model directory names.
rsi_type = "254k"  # set according to the RSI category in use
# rsi_type = '1m4k'

d_rsi_paras = {
    "1m4k": {
        "pyr":8,  #Pyramid layer index
        "rsi_size": 4096,
        "n_block": 15,
        "alt_unit": 818.1027,
        "fov_unit": 4096 * 256 / 4096,
    },  # 256 m
    "504k": {
        "pyr":4,
        "rsi_size": 4096,
        "n_block": 15,
        "alt_unit": 409.0513,
        "fov_unit": 2048 * 256 / 4096,
    },  # 128 m
    "254k": {
        "pyr":2,
        "rsi_size": 4096,
        "n_block": 15,
        "alt_unit": 204.5257,
        "fov_unit": 1024 * 256 / 4096,
    },  # 64 m
    "124k": {
        "pyr":1,
        "rsi_size": 4096,
        "n_block": 15,
        "alt_unit": 102.2628,
        "fov_unit": 512 * 256 / 4096,
    },  # 35.94 m
    "145k": {
        "rsi_size": 5120,
        "n_block": 19,
        "alt_unit": 115.0,
        "fov_unit": 720 * 256 / 5128,
    },  # 35.94 m
}
ALT_UNIT = 102.2628
d_rsi_paras_pyr = {
    "1m4k": {
        "pyr":8,  #Pyramid layer index
        "rsi_size": 1024,
        "n_block": 3,
        "alt_ref": ALT_UNIT*8,
        "fov_ref": 256,
    },  # 256 m
    "504k": {
        "pyr":4,
        "rsi_size": 2048,
        "n_block": 7,
        "alt_ref": ALT_UNIT*4,
        "fov_ref": 128,
    },  # 128 m
    "254k": {
        "pyr":2,
        "rsi_size": 4096,
        "n_block": 15,
        "alt_ref": ALT_UNIT*2,
        "fov_ref": 64,  #FOV width in meters
    },  # 64 m
    "124k": {
        "pyr":1,
        "rsi_size": 8192,
        "n_block": 31,
        "alt_ref": ALT_UNIT,
        "fov_ref": 32,
    },  # 35.94 m
}


image_id_map["merge_c2a_254k"] = 71  # Fusion 31, 32, 33, 34
image_id_map["merge_c2b_254k"] = 72  # Fusion 31, 32, 33, 34
image_id_map["merge_c3a_254k"] = 81  # Fusion 31, 32, 33, 34
image_id_map["merge_c3b_254k"] = 82  # Fusion 31, 32, 33, 34
image_id_map["merge_c4_1m4k"] = 94  # Fusion 31, 32, 33, 34
image_id_map["merge_c4_254k"] = 96  # Fusion 34bc, 36bc, 37bc, 38bc
image_id_map[""] = 99  # For testing
id_image_map = {v: k for k, v in image_id_map.items()}

d_merge_rsis = {}

d_merge_rsis["merge_c4_254k"] = {
    '34bc': id_image_map['34bc'],
    '36bc': id_image_map['36bc'],
    '37bc': id_image_map['37bc'],
    '38bc': id_image_map['38bc'],
}
d_merge_rsis["merge_c2b_254k"] = {
    '34bc': id_image_map['34bc'],
    '38bc': id_image_map['38bc'],
}
d_merge_rsis["merge_c2a_254k"] = {
    '36bc': id_image_map['36bc'],
    '37bc': id_image_map['37bc'],
}
d_merge_rsis["merge_c3b_254k"] = {
    '34bc': id_image_map['34bc'],
    '36bc': id_image_map['36bc'],
    '38bc': id_image_map['38bc'],
}
d_merge_rsis["merge_c3a_254k"] = {  
    '36bc': id_image_map['36bc'],
    '37bc': id_image_map['37bc'],
    '38bc': id_image_map['38bc'],
}

# Definition of basic variables
# Unified pixel size
UNI_PIXEL = 128
# Base patch size
PATCH_SIZE = 256
# Base block size
BLOCK_SIZE = 512
# Default number of samples per block: 50
# N_SAMPLE   = 100
# Base remote sensing image size
RSI_SIZE = d_rsi_paras[rsi_type]['rsi_size']  # 4096, 5120
# Base number of blocks
N_BLOCK = d_rsi_paras[rsi_type]['n_block']  # 15
# Unit altitude
PYR = d_rsi_paras[rsi_type]['pyr']  #Pyramid layer index
ALTITUDE_UNIT = d_rsi_paras[rsi_type]['alt_unit']  # 115.0 meters
# Field of view per block
# FOV_UNIT = 36.0  # 720*256/5128 ≈ 35.94 m
FOV_UNIT = d_rsi_paras[rsi_type]["fov_unit"]  # ≈ 35.94 m
MAX_DISTANCE = FOV_UNIT * 150  # max flight distance = 150x FOV (e.g., 64*150=9600 m for 254k)



def rsijson2info(rsi_json_path):
    """Load RSI metadata JSON and return as dict."""
    with open(rsi_json_path, "r", encoding="utf-8") as file:
        rsi_jdata = json.load(file)
    return rsi_jdata

def get_rsi_name(rsi_id):
    rsi_to_find = [rsi_id]
    rsi = [k for k, v in image_id_map.items() if v in rsi_to_find]
    return "merge" if not rsi else rsi[0]

def get_rsidir_dsetdir_cityid(
    rsi_id, 
    rsi_type, 
    dset_dir, 
    n_sample, 
    version='', 
    flag_pyr='', 
    local_dataset=False  #Whether to use local dataset
):
    """
    Get RSI map path and dataset naming info. Adjust paths for your server.
    """
    if local_dataset:  # Use dataset under project path
        rsi_city_dir = f"{proj_dir}/datasets/city8_25pp_4096bc"
        dset_root = f"{proj_dir}/datasets/metadatas"
        city_id = rsi_id
        head_stem = 'c1'
        dset_name = f'{head_stem}_{rsi_type}_{rsi_id}_b{N_BLOCK}_s{n_sample}'
        dset_name = dset_name + '_v3d' if version == '3d' else dset_name
        return rsi_city_dir, dset_root, dset_name, city_id
        
    # Select dataset by rsi_id
    else:
        if rsi_id == 96:
            dset_name = f"c4m_{rsi_type}_{rsi_id}bc_b{N_BLOCK}_s{n_sample}"
        elif rsi_id == 71 or rsi_id == 72:
            dset_name = f"c2m_{rsi_type}_{rsi_id}bc_b{N_BLOCK}_s{n_sample}"
        elif rsi_id == 81 or rsi_id == 82:
            dset_name = f"c3m_{rsi_type}_{rsi_id}bc_b{N_BLOCK}_s{n_sample}"
        elif rsi_id in ['31bc', '32bc', '33bc', '34bc', '35bc', '36bc', '37bc', '38bc', '39bc']:
            head_stem = 'c1'
            dset_name = f'{head_stem}_{rsi_type}_{rsi_id}_b{N_BLOCK}_s{n_sample}'
        
        # e.g., c1_254k_34bc_b15_s100, c1_254k_37bc_b15_s100_v3d
        rsi_city_dir = rsi_dir_city8_25pp_4096bc
        dset_root = f"{cvphr_dir}"
        city_id = rsi_id
        if version == '2d':
            dset_name = dset_name
        if version == '3d':
            dset_name = dset_name + '_v3d'
    return rsi_city_dir, dset_root, dset_name, city_id

def get_rs_image_path(rsi_id, rsi_city_dir, rsi_name):
    list_of_merge_dset = list(d_merge_rsis.keys())  # ['merge_c4_1m4k', 'merge_c4_254k']
    if rsi_name in list_of_merge_dset:  # Fusion dataset
        d_rs_image_path = d_merge_rsis[rsi_name]  # Dict at this time
        d_rs_image_path = {k: f"{rsi_city_dir}/{v}" for k, v in d_rs_image_path.items()}  # Dict at this time
    else:  # Single remote sensing image
        d_rs_image_path = {rsi_id: f"{rsi_city_dir}/{rsi_name}"}
    return d_rs_image_path

def reminder_proper_rsi_type(str_correct_rsi_type, your_rsi_type) :
    """Check and warn if rsi_type is inconsistent with expected value."""
    if your_rsi_type != str_correct_rsi_type:
        print("\n⚠️⚠️⚠️\n")
        raise ValueError(
            f"Expected rsi_type={str_correct_rsi_type}, got: {your_rsi_type}. "
            "Please update rsi_type in cvphr_base_info.py."
        )
        
def parse_gcth(value):
    if value.lower() in ('none', 'null'):
        return None
    try:
        val = float(value)
        if val <= 0:
            raise argparse.ArgumentTypeError(f"Value must be positive! Got: {val}")
        return val
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid value: {value}. Use a positive number or 'none'.")

def flexible_type(value):
    """train and test parameters parser tool"""
    try:
        # Try to convert to integer first
        return int(value)
    except ValueError:
        # Keep as string if not pure number
        return value

def generate_grid_blocks(width, height, rsi_cnt_lng, rsi_cnt_lat, lng_per_pixel, lat_per_pixel, n_block=19):
    """
    Compute block centers (pixel + lon/lat) from RSI center and resolution.
    The map is divided into an (n_block+1) x (n_block+1) grid of tiles, each
    tile being a PATCH_SIZE x PATCH_SIZE square. Returns a dict keyed by
    """
    blocks = {}
    block_size = PATCH_SIZE  # Block side length (pixels)
    
    for block_id_x in range(n_block):  # Row index
        for block_id_y in range(n_block):  # Column index
            # Calculate center pixel coordinates of current block
            xpix = block_id_x * block_size + block_size
            ypix = block_id_y * block_size + block_size
            
            # Convert pixel coordinates to longitude/latitude
            lng = (xpix - width / 2) * lng_per_pixel + rsi_cnt_lng
            lat = rsi_cnt_lat - (ypix - height / 2) * lat_per_pixel
            
            # Store in dictionary
            key_ = str((block_id_x, block_id_y))
            blocks[key_] = {
                "block_id": [block_id_x, block_id_y],  # block index
                "xy": [xpix, ypix],  # block center in pixels
                "lnglat": [lng, lat],  # block center lon/lat
            }
    return blocks

def alt_design():
    RSIpix = 4096*2
    P = 256
    Reslution = 0.125  # Unit pixel distance resolution at altitude ~100m
    for i in list(range(4)):
        n = i
        m = 2 ** n
        Res = Reslution * m
        Pmeter = P * Res
        alt = m*100
        Rpix = RSIpix/m
        n_patch = int( Rpix / P )
        n_block = n_patch - 1
        n_rsb = n_block ** 2

        # alt
        a1 = round(alt * 4 / 3, 2)
        a2 = round(alt * 3 / 3, 2)
        a3 = round(alt * 2 / 3, 2)
        dh = round(a1-a3, 2)

        la = [a3, a2, a1]
        for a in la:
            x_pyr = a/100
            y_pyex = math.log2(x_pyr)
            print(' * ', round(a,2), round(x_pyr, 2), round(y_pyex, 2))


if __name__ == '__main__':
    alt_design()
    # A json file demo of rsi image downloaded from GE.
    """
    {
        "image": "52.4796_-1.903_720_720_4326_city.jpg",
        "lat": 52.4796,
        "lng": -1.903,
        "height_meter": 720,
        "width_meter": 720,
        "height_pixel": 5128,
        "width_pixel": 5128,
        "coorsys": 4326,
        "lat_per_pixel": 1.2649154614833242e-06,
        "lng_per_pixel": 2.0768873002120065e-06,
        "latm_per_pixel": 0.14040561622464898,
        "lngm_per_pixel": 0.14040561622464898,
        "left_mid": [
            52.4796,
            -1.9083251390377436
        ],
        "right_mid": [
            52.4796,
            -1.8976748609622565
        ],
        "top_mid": [
            52.48284324324324,
            -1.903
        ],
        "bottom_mid": [
            52.47635675675676,
            -1.903
        ],
        "alt": 1006.4000608804841,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        "time": 20250330
    }
    """