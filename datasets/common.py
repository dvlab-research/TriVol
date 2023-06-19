import os
import glob
import random
from PIL import Image
import numpy as np
import cv2
import open3d as o3d

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from kornia import create_meshgrid
import torchvision
import imageio
import lpips



def get_ray_directions_opencv(W, H, fx, fy, cx, cy):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, focal: image height, width and focal length

    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    directions = \
        torch.stack([(i-cx)/fx, (j-cy)/fy, torch.ones_like(i)], -1) # (H, W, 3)

    return directions


def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate

    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:3, :3].T # (H, W, 3)
    rays_d = rays_d / (torch.norm(rays_d, dim=-1, keepdim=True) + 1e-8)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:3, 3].expand(rays_d.shape) # (H, W, 3)

    return rays_o, rays_d


def minkowski_collate_fn(list_data):
    r"""
    Collation function for MinkowskiEngine.SparseTensor that creates batched
    cooordinates given a list of dictionaries.
    """
    # coordinates_batch, features_batch = ME.utils.sparse_collate(
    #     [d["coordinates"] for d in list_data],
    #     [d["features"] for d in list_data],
    #     dtype=torch.float32,
    # )
    # coordinates_batch = [d["coordinates"] for d in list_data]
    # features_batch = [d["features"] for d in list_data]

    # coarse_coords_batch, occupancy_batch = ME.utils.sparse_collate(
    #     [d["coarse_coords"] for d in list_data],
    #     [d["coarse_weights"] for d in list_data],
    #     dtype=torch.float32,
    # )

    rays_o_batch = torch.stack([d["rays_o"] for d in list_data])
    rays_d_batch = torch.stack([d["rays_d"] for d in list_data])
    rgb_batch = torch.stack([d["rgbs"] for d in list_data])
    img_batch = torch.stack([d["imgs"] for d in list_data])
    # planes_batch = torch.stack([d["planes"] for d in list_data])
    voxels_batch = torch.stack([d["voxels"] for d in list_data])
    aabb_batch = torch.stack([d["aabb"] for d in list_data])
    paths = [d["paths"] for d in list_data]
    
    filenames = [d["filename"] for d in list_data]
    
    return {
        # "coordinates": coordinates_batch,
        # "features": features_batch,
        # "coarse_coords": coarse_coords_batch,
        # "coarse_weights": occupancy_batch,
        "rays_o": rays_o_batch,
        "rays_d": rays_d_batch,
        "rgbs": rgb_batch,
        "imgs": img_batch,
        "voxels": voxels_batch,
        "aabb": aabb_batch,
        "paths":paths,
        "filename": filenames
    }
