# Bearing-UAV model.
import os
import cv2
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from importlib import import_module
from typing import Literal, Tuple, List, Dict


from sceneGraphEncodingNet.nets import CSMG, JointNet
from sceneGraphEncodingNet.nets import CSMG_soft, JointNet_soft  #for phr5

from cvphr.utils.utils_transform import transform_pipeline3
from cvphr.utils.utils_transform import transform_pipeline1_gentle
from cvphr.utils.utils_transform import transform_pipeline_weather


"""****************************************************************************
*                                                                             *
*                                  CVPHR-model                                *
*                                                                             *
****************************************************************************"""
class NeighborsCrossAttention(nn.Module):
    def __init__(self, feat_dim=256, reduction_ratio=1):
        """
        Lightweight cross attention module
        Core idea: Use target feature fi as Query to perform attention-weighted aggregation on neighbor_feats to extract relevant context.
        """
        super().__init__()
        reduced_dim = feat_dim // reduction_ratio
        
        self.q_proj = nn.Linear(feat_dim, reduced_dim)
        self.k_proj = nn.Linear(feat_dim, reduced_dim)
        self.v_proj = nn.Linear(feat_dim, reduced_dim)
        
    def forward(self, query, keys):
        # query: [B, num_clusters * D]
        # keys:  [B, 4, num_clusters * D]
        assert query.shape[-1] == self.q_proj.in_features, \
            f"Expected query with dim {self.q_proj.in_features}, got {query.shape[-1]}"

        # query==center_feat，keys==neighbor_feats
        q = self.q_proj(query).unsqueeze(1)  # [B, 1, num_clusters * D/r]
        k = self.k_proj(keys)                # [B, 4, num_clusters * D/r]
        v = self.v_proj(keys)                # [B, 4, num_clusters * D/r]
        
        attn = torch.bmm(q, k.transpose(1,2)) / np.sqrt(q.size(-1))  # [B, 1, 4]
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v).squeeze(1)  # [B, num_clusters * D/r]
        return out

class SimilarityPositionPrior(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.feat_dim = feat_dim
        self.cos = nn.CosineSimilarity(dim=-1)

        # Fixed neighborhood relative coordinates (same as position encoding)
        self.register_buffer('rel_coords', torch.tensor(
            [[-1, -1], [-1, 1], [1, -1], [1, 1]],
            dtype=torch.float32
        ))

    def forward(self, ft, neighbor_feats):
        B, D = ft.size()
        fi_expand = ft.unsqueeze(1).expand(-1, 4, -1)  # [B, 4, D]
        sim = self.cos(fi_expand, neighbor_feats)  # [B, 4]
        weights = torch.softmax(sim, dim=1)  # [B, 4]

        # Weighted relative position
        pos_prior = weights.unsqueeze(2) * self.rel_coords.unsqueeze(0)  # [B, 4, 2]
        pos_prior = pos_prior.sum(dim=1)  # [B, 2]
        return pos_prior

class PositionAngleRegressionSGM(nn.Module):
    def __init__(self, 
                 feature_dim=256, 
                 coord_enc_dims=[64, 256],  # Multiply by n_cluster of csmg for each layer
                 regressor_dims=[256, 64],  # Regression scale
                 reduction_ratio=1,
                 backbone_name='vgg16',
                 num_clusters=4,
                 freeze_backbone=True,
                 partial_unfreeze=False,
                 add_patch_coord=True
                 ):
        """
        PositionAngleRegressionSGM is based on PositionAngleRegressionModel with csmg replacement.
        'par_ca_sgm':  # Replace feat-extract of par_ca with csmg feature extractor of sgm
        """
        super().__init__()
        
        # Save config params as instance attributes
        self.model_name = 'par_ca_sgm'
        self.backbone_name = backbone_name
        self.feature_dim = feature_dim
        self.num_clusters = num_clusters  # Number of scene graph clusters
        self.coord_enc_dims = [x*num_clusters for x in coord_enc_dims]  # Dimension sequence of coordinate encoding
        self.regressor_dims = regressor_dims
        self.reduction_ratio = reduction_ratio
        self.is_pretrained = True
        self.freeze_backbone = freeze_backbone
        self.partial_unfreeze = partial_unfreeze
        self.add_patch_coord = add_patch_coord  # Whether to add neighborhood coordinates
        
        # Backbone config
        if backbone_name == 'vgg16':
            self.backbone = models.vgg16(pretrained=self.is_pretrained)
            # VGG16 features include 13 conv layers and 5 pooling layers
            # Truncate to first 18 layers (remove last 8 layers), output feature map size 28x28, channels 512
            layers = list(self.backbone.features.children())[:-8]
            self.backbone = nn.Sequential(*layers)
            self.backbone_out_dim = 512  # Output channels of VGG16
        if backbone_name == 'resnet18':
            backbone = models.resnet18(pretrained=self.is_pretrained)
            layers = list(backbone.children())[:-2]
            self.backbone = nn.Sequential(*layers)
            self.backbone_out_dim = 512  # Output channels of ResNet50
        if backbone_name == 'resnet50':
            backbone = models.resnet50(pretrained=self.is_pretrained)
            # ResNet50 features, output feature map size 7x7, channels 2048
            layers = list(backbone.children())[:-2]
            self.backbone = nn.Sequential(*layers)
            self.backbone_out_dim = 2048  # Output channels of ResNet50
        
        # === Freeze or partially unfreeze backbone params ===
        if self.freeze_backbone:
            if self.partial_unfreeze:
                if self.backbone_name == 'vgg16':
                    # Unfreeze last several layers of VGG16 (e.g., 10~17)
                    for name, module in self.backbone.named_children():
                        if int(name) >= 14:
                            for p in module.parameters():
                                p.requires_grad = True
                        else:
                            for p in module.parameters():
                                p.requires_grad = False

                elif self.backbone_name.startswith('resnet'):
                    # ResNet unfreeze layer4 (or layer3 + layer4)
                    for name, module in self.backbone.named_children():
                        if any(k in name for k in ['layer4']):
                            for p in module.parameters():
                                p.requires_grad = True
                        else:
                            for p in module.parameters():
                                p.requires_grad = False
            else:
                # Fully freeze
                for p in self.backbone.parameters():
                    p.requires_grad = False

        # Scene graph encoding module
        # Input dim: backbone output dim, Output dim: num_clusters*feature_dim
        self.csmg = CSMG(input_channel=self.backbone_out_dim, output_channel=self.feature_dim, num_clusters=num_clusters)

        # Coordinate encoder
        coord_enc_layers = []
        coord_in_dim = 2  # Input coordinate dim
        # coord_out_dim = self.feature_dim * num_clusters
        for out_dim in self.coord_enc_dims:
            coord_enc_layers.append(nn.Linear(coord_in_dim, out_dim))
            coord_enc_layers.append(nn.ReLU())
            coord_in_dim = out_dim
        # coord_enc_layers.append(nn.Linear(coord_in_dim, coord_out_dim))
        self.coord_encoder = nn.Sequential(*coord_enc_layers)
        
        # Cross attention module
        self.neighbors_cross_attn = NeighborsCrossAttention(
            feat_dim=self.feature_dim*num_clusters, 
            reduction_ratio=reduction_ratio
        )
        
        # Regression head
        self.reg_in_dim = num_clusters * (self.feature_dim + (self.feature_dim // reduction_ratio))

        # Position branch
        #Scheme 1: Initial scheme
        self.pos_in_dim = self.reg_in_dim
        pos_regressor_layers = []
        for out_dim_ in regressor_dims:
            pos_regressor_layers.append(nn.Linear(self.pos_in_dim, out_dim_))
            pos_regressor_layers.append(nn.ReLU())
            self.pos_in_dim = out_dim_
        pos_regressor_layers.append(nn.Linear(self.pos_in_dim, 2))  # Output x, y
        self.pos_regressor = nn.Sequential(*pos_regressor_layers)
        
        # Direction branch
        self.dir_in_dim = self.reg_in_dim
        dir_regressor_layers = []
        for out_dim_ in regressor_dims:
            dir_regressor_layers.append(nn.Linear(self.dir_in_dim, out_dim_))
            dir_regressor_layers.append(nn.ReLU())
            self.dir_in_dim = out_dim_
        dir_regressor_layers.append(nn.Linear(self.dir_in_dim, 2))  # Output cos, sin
        self.dir_regressor = nn.Sequential(*dir_regressor_layers)
    
    def forward(self, patches):
        # patches: [B, 5, C, H, W]
        B = patches.size(0)
        
        # Extract features (shared weights)
        feats = []
        for i in range(5):
            patch = patches[:, i]
            visual_feat = self.backbone(patch)  # torch.Size([32, 512, 32, 32])
            # Pass through scene graph encoding module
            sim_scores, d, d_flatten, _ = self.csmg(visual_feat)  # d_flatten: [B, num_clusters*feature_dim]
            feats.append(d_flatten)

        f1, f2, f3, f4, fi = feats
        # 4 neighborhood coordinates, origin at block center, u as unit length, according to translation image processing coordinate system
        u = 1.0
        known_coords = torch.tensor([[-u,-u], [-u,u], [u,-u], [u,u]],
                                   dtype=torch.float32, device=patches.device) 

        # [4,D] --> [1,4,D] --> [B,4,D]
        coord_embs = self.coord_encoder(known_coords).unsqueeze(0).repeat(B,1,1)  # [B,4,D]

        # Enhance neighborhood features
        if self.add_patch_coord:
            neighbor_feats = torch.stack([f1, f2, f3, f4], dim=1) + coord_embs  # [B,4,D]
        else:
            neighbor_feats = torch.stack([f1, f2, f3, f4], dim=1)  # [B,4,D]
        
        # Cross attention
        ctx_feat = self.neighbors_cross_attn(fi, neighbor_feats)  # [B, 32]
        
        # Feature fusion
        combined = torch.cat([fi, ctx_feat], dim=1)  # [B, 160]
        
        # Regression
        pos_pred = self.pos_regressor(combined)
        dir_pred = self.dir_regressor(combined)

        return pos_pred, dir_pred
    
    @classmethod
    def get_model_name(cls):
        """
        # Get via class method
        model_name = PositionAngleRegressionSGM.get_model_name()
        print(model_name)  # Output: par_sgmdca
        """
        # Create temporary instance and return attribute value
        return cls().model_name

    #  The following three member functions are added for debugging
    # Add numerical stability check function
    def _check_tensor_validity(self, tensor, name="tensor"):
        """Check if tensor contains NaN or infinity values"""
        if torch.isnan(tensor).any():
            warnings.warn(f"Warning: {name} contains NaN values")
            return False
        if torch.isinf(tensor).any():
            warnings.warn(f"Warning: {name} contains Inf values")
            return False
        return True   
        
    def _save_forword_tensor(self, dct_content, debug_dir):
        """Save current important variables when position and angle are NaN"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        debug_dir = f'{debug_dir}/debug_por_forward'
        os.makedirs(debug_dir, exist_ok=True)
        temp_path = f'{debug_dir}/por_forward_tensors_{timestamp}.pt'
        torch.save(dct_content, temp_path)  # Save multiple Tensors (dictionary form)
        print(f"🎯Warning: save err info to {temp_path}")

class PARCASGM_v5(PositionAngleRegressionSGM):
    def __init__(self, 
                 backbone_name='vgg16',
                 feature_dim=256, 
                 coord_enc_dims=[16, 64, 256],   # Coordinate encoding layer dimension sequence
                 regressor_dims=[1024, 256, 64],  # Modified to 3-layer structure
                 reduction_ratio=1,
                 num_clusters=4,
                 freeze_backbone=True,
                 partial_unfreeze=False,
                 add_patch_coord=True):
        """
        PARSGM_v4a：-->v5.
        a Try to replace original csmg encoding with Joint (real sgm)
        b Add "feature similarity" to guide regression accuracy,
        c Basic network settings are fixed to deepened state
            coord_enc_dims=[16, 64, 256],   # Coordinate encoding layer dimension sequence
            regressor_dims=[1024, 256, 64],  # Modified to 3-layer structure
        Code organized more concisely
        """
        super().__init__(
            feature_dim=feature_dim,
            coord_enc_dims=coord_enc_dims,
            regressor_dims=regressor_dims,
            reduction_ratio=reduction_ratio,
            backbone_name=backbone_name,
            num_clusters=num_clusters,
            freeze_backbone=freeze_backbone,
            partial_unfreeze=partial_unfreeze,
            add_patch_coord=add_patch_coord
        )
        
        # Update model name
        self.model_name = 'par_ca_sgm_v5'
        self.sgm = JointNet(None, self.csmg)
        self.sgm.backbone = self.backbone

        # Add auxiliary position guidance module v4a
        self.sim_pos_prior = SimilarityPositionPrior(feat_dim=self.feature_dim)
        self.pos_in_dim = self.reg_in_dim + 2  #Note: 2 nodes added here
        # Position branch v5
        pos_regressor_layers = []
        for out_dim_ in regressor_dims:
            pos_regressor_layers.append(nn.Linear(self.pos_in_dim, out_dim_))
            pos_regressor_layers.append(nn.ReLU())
            self.pos_in_dim = out_dim_
        pos_regressor_layers.append(nn.Linear(self.pos_in_dim, 2))  # Output x, y
        self.pos_regressor = nn.Sequential(*pos_regressor_layers)

    def _model_device(self):
        """Ensure tensors move to the same device as the model parameters."""
        return next(self.parameters()).device

    def _model_dtype(self):
        """Ensure tensors match the model parameter dtype."""
        return next(self.parameters()).dtype

    # Put here: define alias directly in class body after forward_full
    # ===== A: Single patch feature (available for training; no no_grad) =====
    def encode_patch(self, patch: torch.Tensor) -> torch.Tensor:
        """
        Input: [C,H,W] or [B,C,H,W]
        Output: [B, K*D]
        No patch normalization pipeline here, as it cooperates with data loading class which is responsible for normalization
        """
        # if patch.dim() == 3:
        #     patch = patch.unsqueeze(0)
        # out = self.sgm(patch)
        # Key: move input to same device and dtype as model
        # patch = patch.to(device=self._model_device(), dtype=self._model_dtype(), non_blocking=True)
    
        sgm_output = self.sgm(patch)
        return sgm_output['descriptor_flatten']  # d_flatten,[B, K*D]
    
    def forward(self, patches, debug_dir=''):
        # patches: [B, 5, C, H, W]        
        # Extract features (shared weights)
        B = patches.size(0)
        sgm_outputs = []
        _5patch_features = []
        for i in range(5):
            patch = patches[:, i]  # 3, H, W
            sgm_output = self.sgm(patch)  #return_nl=True indicates debug mode
            d_flatten = sgm_output['descriptor_flatten']  # [B, K*D]
            _5patch_features.append(d_flatten)
            sgm_outputs.append(sgm_output)

        f1, f2, f3, f4, uav_patch_feature = _5patch_features
        neighbor_feats = torch.stack([f1, f2, f3, f4], dim=1)  # [B,K,D]

        # 4 neighborhood coordinates, origin at block center, u as unit length, according to translation image processing coordinate system
        u = 1.0
        known_coords = torch.tensor([[-u,-u], [-u,u], [u,-u], [u,u]],
                                   dtype=torch.float32, device=patches.device) 
        # [4,D] --> [1,4,D] --> [B,4,D]
        coord_embs = self.coord_encoder(known_coords).unsqueeze(0).repeat(B,1,1)  # [B,4,D]

        # Enhance neighborhood features
        
        pos_soft_prior = self.sim_pos_prior(uav_patch_feature, neighbor_feats)  # [B, 2]  v5

        if self.add_patch_coord:
            neighbor_feats = neighbor_feats + coord_embs  # [B,4,D]

        # Cross attention
        ctx_feat = self.neighbors_cross_attn(uav_patch_feature, neighbor_feats)  # [B, 32]

        # Feature fusion + position info par_ca_sgm_v4a
        combined = torch.cat([uav_patch_feature, ctx_feat], dim=1)  # [B, D + D']
        combined_with_prior = torch.cat([combined, pos_soft_prior], dim=1)  # [B, D + D' + 2]  v5

        # Regression
        pos_pred = self.pos_regressor(combined_with_prior)
        dir_pred = self.dir_regressor(combined)

        return pos_pred, dir_pred

class PARCASGM_v5a(PARCASGM_v5):
    """
    PARCASGM_v5 class with softmax normalization
    Inherits from PARCASGM_v5, only difference is using CSMG_soft and JointNet_soft
    """
    def __init__(self, 
                 backbone_name='vgg16',
                 feature_dim=256, 
                 coord_enc_dims=[16, 64, 256],   # Coordinate encoding layer dimension sequence
                 regressor_dims=[1024, 256, 64],  # Modified to 3-layer structure
                 reduction_ratio=1,
                 num_clusters=4,
                 freeze_backbone=True,
                 partial_unfreeze=False,
                 add_patch_coord=True):
        """
        PARCASGM_v5a: Version with softmax normalization
        Only difference from PARCASGM_v5 is using CSMG_soft and JointNet_soft
        """
        # Call parent class initialization
        super().__init__(
            backbone_name=backbone_name,
            feature_dim=feature_dim,
            coord_enc_dims=coord_enc_dims,
            regressor_dims=regressor_dims,
            reduction_ratio=reduction_ratio,
            num_clusters=num_clusters,
            freeze_backbone=freeze_backbone,
            partial_unfreeze=partial_unfreeze,
            add_patch_coord=add_patch_coord
        )
        
        # Update model name
        self.model_name = 'phr5'
        
        # Redefine csmg and sgm with soft version
        self.csmg = CSMG_soft(input_channel=self.backbone_out_dim, output_channel=self.feature_dim, num_clusters=num_clusters)
        self.sgm = JointNet_soft(None, self.csmg)
        self.sgm.backbone = self.backbone


class RSBlockDatasetPA_v3q(Dataset):
    """
    Remote sensing data processing class and its processing pipeline design
    # Prerequisites:
        # - PATCH_SIZE
        # - transform_pipeline1_gentle()
        # - transform_pipeline3()
        TPP1 = (Random color jitter + Random Gaussian noise + Random Gaussian blur + Random cutout)
        TPP1_gentle = Gentle version of TPP1, consistent but with compressed params and reduced probability
        TPP2 = (Random affine transform + Perspective angle)
        TPP3 = (ToTensor+Normalize)
    """
    def __init__(self, metadata_csv: str, is_train: bool = True):
        self.df = pd.read_csv(metadata_csv)
        self.is_train = is_train  #True by default except training, False for other scenarios

        # Separate tile and uav for uav weather augmentation
        self.tile_transform = transforms.Compose([
            # transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
            transform_pipeline1_gentle(),
            transform_pipeline3()
        ])
        # Training/validation: uav also uses gentle (TPP2 abandoned in v3q)
        self.uav_transform = transforms.Compose([
            # transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
            transform_pipeline1_gentle(),
            # transform_pipeline_weather(),  #Note: add weather augmentation pipeline later
            transform_pipeline3()
        ])

        # Non-training phase: Minimal transform (ToTensor+Normalize)
        self.test_transform = transform_pipeline3()

        # Column name config (corresponding to column names in metadata.csv)
        self.tile_patch_cols = ['p1_path', 'p2_path', 'p3_path', 'p4_path']
        self.uav_col = 'target_path'

    def __len__(self):
        return len(self.df)

    @staticmethod
    def load_cvimg_to_rgb_pil(path: str) -> Image.Image:
        """Read image with cv2 and convert to RGB PIL.Image with fault tolerance."""
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Fail to read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)

    def prepare_patch(self, path: str, is_uav: bool) -> torch.Tensor:
        """
        Read -> RGB(PIL) -> Select appropriate pipeline -> Return tensor(C,H,W)
        Training:    base = base_transform；uav = uav_transform
        Non-training:  base/uav = test_transform
        """
        img = self.load_cvimg_to_rgb_pil(path)
        if self.is_train:
            tfm = self.uav_transform if is_uav else self.tile_transform
        else:
            tfm = self.test_transform
        return tfm(img)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 4 base patches
        patches = [self.prepare_patch(row[col], is_uav=False) for col in self.tile_patch_cols]
        # 1 uav/target patch
        uav_patch = self.prepare_patch(row[self.uav_col], is_uav=True)
        patches.append(uav_patch)

        patches_tensor = torch.stack(patches, dim=0)  # [5, C, H, W]

        sample = {
            'patches': patches_tensor,
            'coords': torch.tensor([row['x_norm'], row['y_norm']], dtype=torch.float32),
            'ccs_coords': torch.tensor([row['x_uccs'], row['y_uccs']], dtype=torch.float32),
            'agl_coords': torch.tensor([row['x_cosa'], row['y_sina']], dtype=torch.float32),
            'theta': torch.tensor([row['theta']], dtype=torch.float32),
            'block_xy': torch.tensor([row['block_x'], row['block_y']], dtype=torch.int32),
            'target_path': row[self.uav_col],
            # Additional info (default value if CSV has no corresponding column)
            'pyramid':   row.get('pyr', 1),
            'alt_norm':  row.get('alt_norm', 1.0),
            'alt': row.get('alt', 200.0),
            'fov': row.get('fov', 64.0),
        }
        return sample

    # Convenience: External processes (navigation/feature table creation) quickly get paths of 5 images
    def get_patch_paths(self, idx: int) -> List[str]:
        row = self.df.iloc[idx]
        return [row[c] for c in self.base_patch_cols] + [row[self.uav_col]]

class RSBlockDatasetPA_v3q_weather(RSBlockDatasetPA_v3q):
    """Weather augment dataset."""
    def __init__(
        self,
        augname: str,
        metadata_csv: str,
        is_train: bool = True,
        apply_weather_eval: bool = True,
    ):
        super().__init__(metadata_csv, is_train)
        self.apply_weather_eval = apply_weather_eval

        self.eval_uav_transform = transforms.Compose(
            [
                # transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
                transform_pipeline1_gentle(),
                transform_pipeline_weather(augname),
                transform_pipeline3(),
            ]
        )

        self.uav_transform = transforms.Compose(
            [
                # transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
                transform_pipeline1_gentle(),
                transform_pipeline_weather(augname),
                transform_pipeline3(),
            ]
        )

    def prepare_patch(self, path: str, is_uav: bool) -> torch.Tensor:

        img = self.load_cvimg_to_rgb_pil(path)
        if self.is_train:
            tfm = self.uav_transform if is_uav else self.tile_transform
        else:
            if is_uav and self.apply_weather_eval:
                tfm = self.eval_uav_transform
            else:
                tfm = self.test_transform
        return tfm(img)


def par_dataloader(metadata_csv, dataset_class, dataset_kwargs, BATCH_SIZE):
    """
    PAR dataset preprocessing: Load--Augment--Split--Output train/val/test subsets
    """
    # Load full dataset
    if dataset_class == RSBlockDatasetPA_v3q_weather:
        print(dataset_kwargs["augname"]) #Weather augmentation modes: Mixed, individual Rain,Snow,Cloud,Brightness,Noop, see transform_pipeline_weather
        augm_dataset = dataset_class(
            augname=dataset_kwargs["augname"], metadata_csv=metadata_csv, is_train=True
        )
        norm_dataset = dataset_class(
            augname=dataset_kwargs["augname"], metadata_csv=metadata_csv, is_train=False
        )
    else:
        augm_dataset = dataset_class(
            metadata_csv = metadata_csv,
            is_train=True
        )
        norm_dataset = dataset_class(
            metadata_csv = metadata_csv,
            is_train=False
        )

    # Split dataset
    train_size = int(0.85 * len(augm_dataset))
    val_size = int(0.05 * len(augm_dataset))
    test_size = len(augm_dataset) - train_size - val_size
    train_indices, val_indices, test_indices = random_split(
        range(len(augm_dataset)), [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  #If changed, test samples cannot be guaranteed to be unseen
        # generator=torch.Generator().manual_seed(11)
    )
    
    # Create training set (use full augmentation)
    train_dataset = torch.utils.data.Subset(augm_dataset, train_indices.indices)
    
    # Create validation and test sets (only use general_transform)
    val_dataset = torch.utils.data.Subset(norm_dataset, val_indices.indices)
    test_dataset = torch.utils.data.Subset(norm_dataset, test_indices.indices)
    
    # Create data loaders
    num_workers = 8  # Or 8, depending on machine CPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,   # ★ Reuse workers to avoid restart per epoch
        prefetch_factor=2,         # Default 2 is fine, adjust if needed
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    return train_loader, val_loader, test_loader, test_dataset
    
def load_config_and_model(model_config_dir):
    """Before loading pth for navigation simulation, this program can automatically load
    PAR trained weight files according to the specified path,
    and automatically match the corresponding model keyword dictionary to prevent manual configuration errors
    Because model keywords are saved in training_configure.json during PAR model training
    """
    # Build full config file path
    config_path = os.path.join(model_config_dir, "training_configure.json")
    
    # Read JSON file to config, get model keyword and model_class name from the dictionary
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Extract model_kwargs
    model_kwargs = config.get("model_kwargs", {})
    
    # Dynamically import model_class
    model_class_name = config.get("model_class")
    if not model_class_name:
        raise ValueError("|ERR| 'model_class' not found in config")
    else:
        print(f"|TIPS| Now You are loading model {model_class_name}.")
    
    # Assume model class is in 'models' module (adjust according to actual situation)
    try:
        # Replace 'your_module' with actual module name containing model class, e.g., 'models'
        # model_module = import_module("posaglreg_models")  
        model_module = import_module("cvphr.models.posaglreg.models")  #dynamic import
        # Note: The above line will error if "posaglreg_models.py" module is renamed in the future!!!
        model_class = getattr(model_module, model_class_name)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Failed to import {model_class_name}: {str(e)}")
    
    # Output results
    print("Model Class:", model_class)
    print("Model Kwargs:", model_kwargs)
    
    return model_class, model_kwargs


"""****************************************************************************
*                                                                             *
*                                 CVPHR-net input                               *
*                                                                             *
****************************************************************************"""

model_kwargs_par_ca_sgm_v5a={
    'backbone_name': 'vgg16',
    'feature_dim': 256,                 # d=256
    'coord_enc_dims': [16, 64, 256],    # 2 >> nc*[16, 64, d] 
    'regressor_dims': [1024, 256, 64],  #nc*2*d  >> [1024, 256, 64] >> 2
    'reduction_ratio': 1,
    'num_clusters': 4,
    'freeze_backbone': True,
    'partial_unfreeze':False,
    'add_patch_coord':True, 
}


"""****************************************************************************
*                                                                             *
*                            CVPHR-model-dictionary                           *
*                                                                             *
****************************************************************************"""
# Class dictionary of model
MODEL_CLASS_DICT = {
    "PARCASGM_v5":          PARCASGM_v5,
    "PARCASGM_v5a":         PARCASGM_v5a,
}

MODEL_KEYWARDS_DICT = {
    "PARCASGM_v5":          model_kwargs_par_ca_sgm_v5a,
    "PARCASGM_v5a":         model_kwargs_par_ca_sgm_v5a,
}

DATASET_CLASS_DICT = {
    "RSBlockDatasetPA_v3q":         RSBlockDatasetPA_v3q,
    "RSBlockDatasetPA_v3q_weather": RSBlockDatasetPA_v3q_weather,
}