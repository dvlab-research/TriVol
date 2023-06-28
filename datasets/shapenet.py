import os
import glob
import random
from PIL import Image, ImageOps
import numpy as np
import cv2
import imageio
import open3d as o3d

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from .common import *
import csv
import math
from torch_scatter import scatter_mean
import json


def compute_extrinsic_matrix(
    azimuth: float, elevation: float, distance: float
):  # pragma: no cover
    """
    Copied from meshrcnn codebase:
    https://github.com/facebookresearch/meshrcnn/blob/main/shapenet/utils/coords.py#L96
    Compute 4x4 extrinsic matrix that converts from homogeneous world coordinates
    to homogeneous camera coordinates. We assume that the camera is looking at the
    origin.
    Used in R2N2 Dataset when computing calibration matrices.
    Args:
        azimuth: Rotation about the z-axis, in degrees.
        elevation: Rotation above the xy-plane, in degrees.
        distance: Distance from the origin.
    Returns:
        FloatTensor of shape (4, 4).
    """
    azimuth, elevation, distance = float(azimuth), float(elevation), float(distance)

    az_rad = -math.pi * azimuth / 180.0
    el_rad = -math.pi * elevation / 180.0
    sa = math.sin(az_rad)
    ca = math.cos(az_rad)
    se = math.sin(el_rad)
    ce = math.cos(el_rad)
    R_world2obj = torch.tensor(
        [[ca * ce, sa * ce, -se], [-sa, ca, 0], [ca * se, sa * se, ce]]
    )
    R_obj2cam = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    R_world2cam = R_obj2cam.mm(R_world2obj)
    cam_location = torch.tensor([[distance, 0, 0]]).t()
    T_world2cam = -(R_obj2cam.mm(cam_location))
    RT = torch.cat([R_world2cam, T_world2cam], dim=1)
    RT = torch.cat([RT, torch.tensor([[0.0, 0, 0, 1]])])

    # Georgia: For some reason I cannot fathom, when Blender loads a .obj file it
    # rotates the model 90 degrees about the x axis. To compensate for this quirk we
    # roll that rotation into the extrinsic matrix here
    rot = torch.tensor([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    # pyre-fixme[16]: `Tensor` has no attribute `mm`.
    RT = RT.mm(rot.to(RT))

    return RT

class ShapeNetDataset(Dataset):
    def __init__(self, phase, scene_dir, img_wh=[256, 256]):
        self.CACHE = {}
        self.phase = phase  # do something for a real dataset.
        self.scene_dir = scene_dir


        if self.phase == 'train':
            scene_dirs = glob.glob(os.path.join(scene_dir, '*'))
            scene_dirs = sorted(scene_dirs)
            del scene_dirs[::10]
            self.scene_dirs = scene_dirs
        elif self.phase == 'val':
            scene_dirs = glob.glob(os.path.join(scene_dir, '*'))
            scene_dirs = sorted(scene_dirs)
            self.scene_dirs = scene_dirs[::10]
        else:
            # self.phase not support error and return 
            raise NotImplementedError    

        # elif self.phase == 'finetune':
        #     scene_dirs = glob.glob(os.path.join(scene_dir, '*.json'))
        #     self.scene_dirs = scene_dirs * 200
        # elif self.phase == 'demo':
        #     scene_dirs = glob.glob(os.path.join(scene_dir, '*.json'))
        #     self.scene_dirs = scene_dirs * 200
        
        self.W = img_wh[0]
        self.H = img_wh[1]

        if self.phase == 'demo':
            self.num_frames = 36
            self.azim_demo = np.linspace(0.0, 360.0, self.num_frames) + 90.0
            self.elev_demo = np.full([self.num_frames,], 20, dtype=np.float32)

    def __len__(self):
        if self.phase == 'train':
            return min(len(self.scene_dirs), 3000)
        elif self.phase == 'val':
            return 100
        elif self.phase == 'finetune':
            return 200
        elif self.phase == 'demo':
            return self.num_frames
        else:
            return 200
    
    def sample_ray(self, scene_dir, idx):
        idx_rand = random.randint(0, 23)
        file_path = f"{idx_rand:03d}.png"
    
        # read imgs
        image_path = os.path.join(scene_dir, 'images', file_path)
        
        img_ori = Image.open(image_path)
        img_bg = Image.new("RGB", img_ori.size, (255, 255, 255))
        img_bg.paste(img_ori, mask=img_ori.split()[3]) # 3 is the alpha channel
        
        # rgbs
        rgbs = img_bg.resize((self.W, self.H), Image.LANCZOS)      
        rgbs = torchvision.transforms.ToTensor()(rgbs) # (3, H, W)
        rgbs = rgbs.permute(1, 2, 0) # (Hn, Wn, 3) RGB

        # read c2w
        if self.phase == 'demo':
            azim = self.azim_demo[idx]
            elev = self.elev_demo[idx]
            dist = 1.2
        else:
            azim = np.load(os.path.join(scene_dir, 'rotation.npy'))[idx_rand] + 90.0
            elev = np.load(os.path.join(scene_dir, 'elevation.npy'))[idx_rand]
            dist = 1.2
        Rt = compute_extrinsic_matrix(azim, elev, dist)
        c2w = torch.diag(torch.FloatTensor([1, 1, -1, 1])) @ \
                                         torch.inverse(Rt) @ \
                                         torch.diag(torch.FloatTensor([1, -1, -1, 1]))

        # read K        
        camera_angle_x = 0.8575560450553894
        f = 0.5*self.W / np.tan(0.5*camera_angle_x) # original focal length
        fx = fy = f
        cx = self.W / 2
        cy = self.H / 2
        # get all point xyz
        direction = get_ray_directions_opencv(self.W, self.H, fx, fy, cx, cy) # (H, W, 3)
        rays_o, rays_d = get_rays(direction, c2w) # both (H, W, 3)

        return rays_o, rays_d, rgbs, image_path

    def __getitem__(self, i):
        try:
            scene_dir = random.choice(self.scene_dirs)

            if self.phase == 'train':
                rays_o = []
                rays_d = []
                rgbs = []
                for j in range(32):
                    rays_o_, rays_d_, rgbs_, image_path = self.sample_ray(scene_dir, i)
                    rays_o.append(rays_o_.reshape(-1, 3))
                    rays_d.append(rays_d_.reshape(-1, 3))
                    rgbs.append(rgbs_.reshape(-1, 3))
                
                rays_o = torch.cat(rays_o, dim=0)
                rays_d = torch.cat(rays_d, dim=0)
                rgbs = torch.cat(rgbs, dim=0)
                
                idx = torch.randperm(rays_o.shape[0])[:10000]
                rays_o = rays_o[idx]
                rays_d = rays_d[idx]
                rgbs = rgbs[idx]
            else:
                rays_o, rays_d, rgbs, image_path = self.sample_ray(scene_dir, i)

            scene_name = scene_dir.split('/')[-1]
            pcd_path = os.path.join(scene_dir, scene_name + '.ply')
            pcd_color = o3d.io.read_point_cloud(pcd_path)

            points = torch.FloatTensor(np.array(pcd_color.points, dtype=np.float32)) 
            features = torch.FloatTensor(np.array(pcd_color.colors, dtype=np.float32)) 

            aa = points.min(0)[0][None]
            bb = points.max(0)[0][None]  
            
            # random delta_scale from 0.05 to 0.15
            if self.phase == 'train':
                delta_scale = random.uniform(0.05, 0.15)
            else:
                delta_scale = 0.1
            aa = aa - delta_scale * (bb - aa)
            bb = bb + delta_scale * (bb - aa)
            aabb = torch.cat([aa, bb], dim=0)

            C = 4
            resolution = 256
            points = (points - aa) / (bb - aa)
            index_points = (points * (resolution - 1)).long()
            index_rgba = torch.cat([features, torch.ones_like(features[:, 0:1])], dim=1).transpose(0, 1) # [4, N]

            index = index_points[:, 2] + resolution * (index_points[:, 1] + resolution * index_points[:, 0])
            voxels = torch.zeros(C, resolution**3)
            voxels = scatter_mean(index_rgba, index, out=voxels) # B x C x reso^3
            voxels = voxels.reshape(C, resolution, resolution, resolution) # sparce matrix (B x 512 x reso x reso)
        
        except Exception as e:
            print("error:", e)
            return self.__getitem__()

        return {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "rgbs": rgbs,
            "aabb": aabb,
            "voxels": voxels,
            "paths": scene_dir,
            "filename":image_path
        }
