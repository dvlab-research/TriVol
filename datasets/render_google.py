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

class RenderGoogleDataset(Dataset):
    def __init__(self, phase, scene_dir, voxel_size, coarse_scale, img_wh=[1024, 1024]):
        self.CACHE = {}
        self.phase = phase  # do something for a real dataset.
        self.voxel_size = voxel_size  # in meter
        self.scene_dir = scene_dir
        ## self.scene_dirs = ['/research/leojia/home/xgxu/Program/GET3D/render_google/img/Shoe/11pro_SL_TRX_FG']
        ## self.scene_dirs_val = ['/research/leojia/home/xgxu/Program/GET3D/render_google/img/Shoe/11pro_SL_TRX_FG']
        self.scene_dirs = []
        self.scene_dirs_val = []
        random.seed(1234)
        ## scene_dir = '/research/leojia/home/xgxu/Program/GET3D/render_google/img/Shoe'
        boost_list = ['Sperry_TopSider_pSUFPWQXPp3', 'Sperry_TopSider_tNB9t6YBUf3', 
        'UGG_Bailey_Button_Triplet_Womens_Boots_Black_7','UGG_Bailey_Bow_Womens_Clogs_Black_7',
        'Tory_Burch_Kiernan_Riding_Boot', 'UGG_Bailey_Button_Womens_Boots_Black_7', 
        'UGG_Cambridge_Womens_Black_7', 'UGG_Classic_Tall_Womens_Boots_Chestnut_7',
        'UGG_Classic_Tall_Womens_Boots_Grey_7','UGG_Jena_Womens_Java_7', 'W_Lou_z0dkC78niiZ',
        'Womens_Hikerfish_Boot_in_Black_Leopard_bVSNY1Le1sm','Womens_Hikerfish_Boot_in_Black_Leopard_ridcCWsv8rW',
        'Womens_Hikerfish_Boot_in_Linen_Leather_Sparkle_Suede_imlP8VkwqIH',
        'Womens_Hikerfish_Boot_in_Linen_Leather_Sparkle_Suede_QktIyAkonrU',
        'Rayna_BootieWP','ASICS_GELBlur33_20_GS_BlackWhiteSafety_Orange']
        for name in os.listdir(scene_dir):
            if name in boost_list:
                continue
            path_this = os.path.join(scene_dir, name)
            point_cloud_path = os.path.join(scene_dir, name+'.ply')
            if not(os.path.exists(point_cloud_path)):
                continue
            if not(os.path.isdir(path_this)):
                continue
            if random.randint(0, 9) >= 2:
                self.scene_dirs.append(path_this)
            else:
                self.scene_dirs_val.append(path_this)

        '''
        if self.phase == 'train':
            scene_dirs = glob.glob(os.path.join(scene_dir, '*/*.json'))
            del scene_dirs[::9]
            self.scene_dirs = scene_dirs
        elif self.phase == 'val':
            scene_dirs = glob.glob(os.path.join(scene_dir, '*/*.json'))
            self.scene_dirs = scene_dirs[::9]
        elif self.phase == 'finetune':
            scene_dirs = glob.glob(os.path.join(scene_dir, '*.json'))
            self.scene_dirs = scene_dirs * 200
        elif self.phase == 'demo':
            scene_dirs = glob.glob(os.path.join(scene_dir, '*.json'))
            self.scene_dirs = scene_dirs * 200
        '''

        H = img_wh[1]
        W = img_wh[0]
        upscale = 1
        self.H = H // upscale
        self.W = W // upscale

        self.coarse_scale = coarse_scale

        if self.phase == 'demo':
            self.num_frames = 36
            self.azim_demo = np.linspace(0.0, 360.0, self.num_frames) + 90.0
            self.elev_demo = np.full([self.num_frames,], 20, dtype=np.float32)

    def __len__(self):
        if self.phase == 'train':
            length = 0
            for mm in range(len(self.scene_dirs)):
                with open(os.path.join(self.scene_dirs[mm], 'transforms.json'), 'r') as f:
                    meta = json.load(f)
                frames = meta['frames']

                length += 1
            return length
            ## return min(len(self.scene_dirs), 1500)
        elif self.phase == 'val':
            return 100
        elif self.phase == 'finetune':
            return 200
        elif self.phase == 'demo':
            return self.num_frames
        else:
            return 200
    
    def sample_ray(self, scene_dir, idx):
        with open(os.path.join(scene_dir, 'transforms.json'), 'r') as f:
            meta = json.load(f)
        frames = meta['frames']
        idx_rand = random.randint(0, len(frames)-1)
        ## idx_rand = 0
        frame = frames[idx_rand]
    
        # read imgs
        image_path = os.path.join(scene_dir, f"{frame['file_path']}")
        if self.phase == 'val': print("Eval image path: ", image_path)
        
        img_ori = Image.open(image_path)
        img_ori = ImageOps.mirror(img_ori)
        img_bg = Image.new("RGB", img_ori.size, (255, 255, 255))
        img_bg.paste(img_ori, mask=img_ori.split()[3]) # 3 is the alpha channel

        W_ori, H_ori = img_bg.size
        imgs = img_bg.resize((self.W, self.H), Image.BILINEAR)      
        imgs = torchvision.transforms.ToTensor()(imgs) # (3, H, W)
        ### print(self.W, self.H)

        # rgbs
        rgbs = img_bg.resize((self.W, self.H), Image.BILINEAR)      
        rgbs = torchvision.transforms.ToTensor()(rgbs) # (3, H, W)
        rgbs = rgbs.permute(1, 2, 0) # (Hn, Wn, 3) RGB

        # read c2w
        '''
        if self.phase == 'demo':
            azim = self.azim_demo[idx]
            elev = self.elev_demo[idx]
            dist = 1.2
        else:
            cam_dir = os.path.dirname(image_path.replace('/img/', '/camera/'))
            azim = np.load(os.path.join(cam_dir, 'rotation.npy'))[idx_rand] + 90.0
            elev = np.load(os.path.join(cam_dir, 'elevation.npy'))[idx_rand]
            dist = 1.2
        '''
        cam_dir = os.path.dirname(image_path.replace('/img/', '/camera/'))
        azim = np.load(os.path.join(cam_dir, 'rotation.npy'))[idx_rand] + 90.0
        elev = np.load(os.path.join(cam_dir, 'elevation.npy'))[idx_rand]
        dist = 1.2

        Rt = compute_extrinsic_matrix(azim, elev, dist)
        c2w = torch.inverse(Rt) @ torch.diag(torch.FloatTensor([1, -1, -1, 1]))
        c2w = torch.diag(torch.FloatTensor([1, 1, -1, 1])) @ c2w

        # read K        
        f = 0.5*self.W / np.tan(0.5*meta['camera_angle_x']) # original focal length
        fx = fy = f
        cx = self.W / 2
        cy = self.H / 2
        # get all point xyz
        direction = get_ray_directions_opencv(self.W, self.H, fx, fy, cx, cy) # (H, W, 3)
        rays_o, rays_d = get_rays(direction, c2w) # both (H, W, 3)

        return rays_o, rays_d, rgbs, imgs, image_path

    def __getitem__(self, i):
        if self.phase == 'train':
            scene_dir = random.choice(self.scene_dirs)
        else:
            scene_dir = random.choice(self.scene_dirs_val)
        ## scene_dir = self.scene_dir
        rays_o, rays_d, rgbs, imgs, image_path = self.sample_ray(scene_dir, i)

        scene_name = scene_dir.split('/')[-1]
        base_name = os.path.basename(scene_name)
        pcd_color_path = scene_dir + '.ply'
        pcd_color = o3d.io.read_point_cloud(pcd_color_path)
        points = torch.FloatTensor(np.array(pcd_color.points, dtype=np.float32))
        aa, bb = points.min(0, True)[0],  points.max(0, True)[0]
        features = torch.FloatTensor(np.array(pcd_color.colors, dtype=np.float32))       

        quantized_coords, quantized_feats, coarse_coords, coarse_weights, points, features = self.CACHE[scene_dir]

        aa = points.min(0)[0][None] - 0.05
        bb = points.max(0)[0][None] + 0.05       
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

        return {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "rgbs": rgbs,
            "imgs": imgs,
            "aabb": aabb,
            "voxels": voxels,
            "filename": image_path,
            "paths": scene_dir
        }


if __name__ == '__main__':
    dataset = ShapeNetGet3DDataset(phase='train', 
                   scene_dir='/research/leojia/home/xgxu/Program/GET3D/render_shapenet_data/shapenet_render_img/img/02958343', 
                   voxel_size=0.01, 
                   coarse_scale=8)
    for i in range(len(dataset)):
        data = dataset[i]
        
