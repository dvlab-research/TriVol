import os
import glob
import random
from PIL import Image
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
from torch_scatter import scatter_mean


class ScanNetDataset(Dataset):
    def __init__(self, phase, scene_dir, img_wh=[640, 512]):
        self.phase = phase  # do something for a real dataset.
        self.scene_dir = scene_dir

        scene_files = glob.glob(os.path.join(scene_dir, '*/*vh_clean_2.ply'))
        if self.phase == 'train':
            scene_files = sorted(scene_files)
            del scene_files[1200:]
            scene_files = scene_files * 2
            random.shuffle(scene_files)
            self.scene_files = scene_files
        elif self.phase == 'val':
            scene_files = sorted(scene_files)
            self.scene_files = scene_files[1200:]
        else:
            self.scene_files = []
        
        self.W = img_wh[0]
        self.H = img_wh[1]

    def __len__(self):
        if self.phase == 'train':
            return len(self.scene_files)
        elif self.phase == 'val':
            return 100
        else:
            return 100
    
    def sample_ray(self, filename):
        nerf_dir = os.path.dirname(filename)
        image_paths = np.loadtxt(os.path.join(nerf_dir, 'images.txt'), dtype=str).tolist()

        image_path = random.choice(image_paths)
            
        # read imgs
        image_path = os.path.join(nerf_dir, 'color', image_path)
        img_ori = Image.open(image_path).convert('RGB')
        W_ori, H_ori = img_ori.size

        # rgbs
        rgbs = img_ori.resize((self.W, self.H), Image.LANCZOS)      
        rgbs = torchvision.transforms.ToTensor()(rgbs) # (3, H, W)
        rgbs = rgbs.permute(1, 2, 0) # (H, W, 3) RGB

        # read c2w
        pose_path = image_path.replace('color', 'pose').replace('.jpg', '.txt')
        c2w = torch.FloatTensor(np.loadtxt(pose_path))

        # read K
        W_scale = float(self.W) / float(W_ori)
        H_scale = float(self.H) / float(H_ori)
        intrinsic_path = os.path.join(nerf_dir, 'intrinsic', 'intrinsic_color.txt')
        intrinsic = np.loadtxt(intrinsic_path)
        fx = intrinsic[0, 0] * W_scale
        fy = intrinsic[1, 1] * H_scale
        cx = intrinsic[0, 2] * W_scale
        cy = intrinsic[1, 2] * H_scale

        # get all point xyz
        direction = get_ray_directions_opencv(self.W, self.H, fx, fy, cx, cy) # (H, W, 3)
        rays_o, rays_d = get_rays(direction, c2w) # both (H, W, 3)

        return rays_o, rays_d, rgbs, image_path

    def __getitem__(self, i):
        filename = self.scene_files[i]

        if self.phase == 'train':
            rays_o = []
            rays_d = []
            rgbs = []
            for j in range(32):
                rays_o_, rays_d_, rgbs_, image_path = self.sample_ray(filename)
                rays_o_ = rays_o_.reshape(-1, 3)
                rays_d_ = rays_d_.reshape(-1, 3)
                rgbs_ = rgbs_.reshape(-1, 3)

                idx = torch.randperm(rays_o_.shape[0])[:10000]
                rays_o.append(rays_o_[idx])
                rays_d.append(rays_d_[idx])
                rgbs.append(rgbs_[idx])
            
            rays_o = torch.cat(rays_o, dim=0)
            rays_d = torch.cat(rays_d, dim=0)
            rgbs = torch.cat(rgbs, dim=0)
        else:
            rays_o, rays_d, rgbs, image_path = self.sample_ray(filename)

        pcd_color = o3d.io.read_point_cloud(filename)
        points_raw = torch.FloatTensor(np.array(pcd_color.points, dtype=np.float32))        
        features = torch.FloatTensor(np.array(pcd_color.colors, dtype=np.float32))
        num_points = points_raw.shape[0]

        # random delta_scale from 0.05 to 0.15
        if self.phase == 'train':
            delta_scale = random.uniform(0.05, 0.15)
            rand_num = random.randint(num_points//2, num_points)
            rand_idx = torch.randperm(num_points)[:rand_num]
            points_raw = points_raw[rand_idx]
            features = features[rand_idx]
        else:
            delta_scale = 0.1
        
        aa = points_raw.min(0)[0][None]
        bb = points_raw.max(0)[0][None]  
        aa = aa - delta_scale * (bb - aa)
        bb = bb + delta_scale * (bb - aa)
        aabb = torch.cat([aa, bb], dim=0)

        C = 4
        resolution = 256
        points = (points_raw - aa) / (bb - aa + 1e-12)
        index_points = (points * (resolution - 1)).long()
        index_rgba = torch.cat([features, torch.ones_like(features[:, 0:1])], dim=1).transpose(0, 1) # [4, N]

        index = index_points[:, 2] + resolution * (index_points[:, 1] + resolution * index_points[:, 0])
        voxels = torch.zeros(C, resolution**3)
        voxels = scatter_mean(index_rgba, index, out=voxels) # B x C x reso^3
        voxels = voxels.reshape(C, resolution, resolution, resolution) # sparce matrix (B x 512 x reso x reso)

        return {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "rgbs": rgbs,
            "aabb": aabb,
            "voxels": voxels,
            "paths": filename,
            "filename": image_path
        }


