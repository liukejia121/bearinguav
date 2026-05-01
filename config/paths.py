# Definition of main path
from pathlib import Path
import sys

"""
📌用 __file__ 推导路径：
from pathlib import Path
def project_root() -> Path:
    #Return root directory of bearinguav module.
    return Path(__file__).resolve().parents[2]

当前文件位置：bearingnav/model/bearinguav/config/paths.py
路径层级：
level	path
0	paths.py
1	config
2	bearinguav ✅
3	model
4	bearingnav
"""

def project_root() -> Path:
    """
    Dynamically find 'bearinguav' root directory.
    """
    p = Path(__file__).resolve()
    for parent in p.parents:
        if parent.name == "bearinguav":
            return parent
    raise RuntimeError("bearinguav root not found")

def dataset_path(name: str) -> Path:
    """
    Return root directory of dataset.
    """
    p = Path(__file__).resolve()
    for parent in p.parents:
        if parent.name == "bearinguav":
            return parent / name
    raise RuntimeError("bearinguav root not found")

# ---------- Project root directory
proj_dir = project_root()  #"/your/path/of/proj/bearingnav/model/bearinguav" 
print(" * The project dir:", proj_dir)

# # ---------- Dataset root directory
# dset_dir = proj_dir 
# # Folder name for storing cvphr dataset
# dset_cvphr_name = "posreg_dataset"
# # Folder name for storing full-size remote sensing images
# rsi_docu_name = "city8_25pp_4096bc"  #"city8_1mpp_4096"

# ---------- Dataset root directory
# Folder name for storing cvphr dataset
dset_cvphr_name = "Bearing_UAV_90K"  
# Folder name for storing full-size remote sensing images
rsi_docu_name = "city_rsi"  

# ---------- Default folder name and path for storing cvphr dataset
dset_dir = f"{proj_dir}/{dset_cvphr_name}"  #"/your/path/of/proj/bearinguav/Bearing_UAV_90K"
# .../dataset/posreg_dataset
# posreg_dataset_root --> cvphr_dir

# ---------- Path for saving full-size remote sensing images: city8_25pp_4096bc
rsi_dir_city8_25pp_4096bc = f'{dset_dir}/{rsi_docu_name}'
# .../dataset/posreg_dataset/city8_25pp_4096bc

# Legacy RSI path for Birmingham city
rsi_dir_berm_city = f"{dset_dir}/rsi/block_5x5/Bermingham_block"


def setup_project_path(script_path: str = None) -> Path:
    """
    Add project root directory to sys.path to import project modules.
    This function can be used in tool scripts even if the cvphr module is not imported yet.
    
    Args:
        script_path: Path of the script calling this function (usually using __file__). None for auto-detection.
        
    Returns:
        Path: Project root directory path
        
    Example:
        # Use at the beginning of tool scripts
        from config.paths import setup_project_path
        setup_project_path(__file__)
    """
    if script_path is None:
        # Try to get script path from call stack
        import inspect
        frame = inspect.currentframe()
        try:
            caller_frame = frame.f_back
            script_path = caller_frame.f_globals.get('__file__')
        finally:
            del frame
    
    if script_path:
        # Infer project root from script path
        script_dir = Path(script_path).resolve().parent
        # If script is in tools/ directory, project root is parent directory
        # If script is in project root, project root is current directory
        if script_dir.name == 'tools':
            project_root = script_dir.parent
        else:
            # Try to find directory containing config directory upwards
            current = script_dir
            while current != current.parent:
                if (current / 'config').exists() and (current / 'config' / '__init__.py').exists():
                    project_root = current
                    break
                current = current.parent
            else:
                # If not found, assume project root is parent of script directory
                project_root = script_dir.parent
    else:
        # If script path cannot be determined, try to use cvphr module
        try:
            project_root = project_root()
        except (ImportError, AttributeError):
            raise RuntimeError(
                "Cannot automatically determine project root directory. Please explicitly pass the script_path parameter, "
                "e.g.: setup_project_path(__file__)"
            )
    
    # Add project root to sys.path (if not already added)
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    
    return project_root