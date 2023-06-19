# Copyright (c) NVIDIA Corporation.
# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import os
import argparse
import numpy as np
import open3d as o3d

from pytorch_lightning.core import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import DataLoader
# from torch_scatter import scatter_mean
# import MinkowskiEngine as ME
# from models.minkunet import *
import glob
from PIL import Image
from kornia import create_meshgrid
import random
import string
from pytorch_fid import fid_score, inception
import torchvision
import imageio
import cv2
from datasets import *
from utils import *

from models.unet3d import *

from radiance_fields.mlp import TriVolNeRFRadianceField
from nerfacc.estimators.occ_grid import OccGridEstimator
from radiance_fields.utils import render_image_with_occgrid, Rays
####################################################################################
from skimage.metrics import structural_similarity as ssim_o
from skimage.metrics import peak_signal_noise_ratio as psnr_o
import lpips as lpips_o

def convert_to(img):
    def to_4d(img):
        assert len(img.shape) == 3
        assert img.dtype == np.uint8
        img_new = np.expand_dims(img, axis=0)
        assert len(img_new.shape) == 4
        return img_new

    def to_CHW(img):
        return np.transpose(img, [2, 0, 1])

    def to_tensor(img):
        return torch.Tensor(img)

    return to_tensor(to_4d(to_CHW(img))) / 127.5 - 1

class Measure():
    def __init__(self, net='alex', use_gpu=False):
        self.device = 'cuda' if use_gpu else 'cpu'
        self.model = lpips_o.LPIPS(net=net)
        self.model.to(self.device)

    def measure(self, imgA, imgB):
        return [float(f(imgA, imgB)) for f in [self.psnr, self.ssim, self.lpips]]

    def lpips(self, imgA, imgB, model=None):
        tA = convert_to(imgA).to(self.device)
        tB = convert_to(imgB).to(self.device)
        dist01 = self.model.forward(tA, tB).item()
        return dist01

    def ssim(self, imgA, imgB, gray_scale=True):
        if gray_scale:
            score, diff = ssim_o(cv2.cvtColor(imgA, cv2.COLOR_RGB2GRAY), cv2.cvtColor(imgB, cv2.COLOR_RGB2GRAY), full=True, multichannel=True)
        # multichannel: If True, treat the last dimension of the array as channels. Similarity calculations are done independently for each channel then averaged.
        else:
            score, diff = ssim_o(imgA, imgB, full=True, multichannel=True)
        return score

    def psnr(self, imgA, imgB):
        ### print(imgA.shape, imgB.shape)
        psnr_val = psnr_o(imgA, imgB)
        return psnr_val
####################################################################################


class TriVolModule(LightningModule):
    r"""
    Segmentation Module for MinkowskiEngine.
    """

    def __init__(
        self,
        scene_dir,
        dataset,
        exp_name,
        train_mode,
        val_mode,
        img_wh,
        lr=1e-3,
        weight_decay=0.05,
        voxel_size=0.01,
        batch_size=4,
        val_batch_size=1,
        train_num_workers=8,
        val_num_workers=2,
        max_epochs=200,
        patch_size=64,
        feat_dim=32,
        coarse_scale=8
    ):
        super().__init__()
        for name, value in vars().items():
            if name != "self":
                setattr(self, name, value)
        
        self.trivol_encoder = TriVol_Encoder(in_channels=4, out_channels=args.feat_dim, num_groups=16, nf=16)
        self.radiance_field = TriVolNeRFRadianceField(feat_dim=args.feat_dim)

        # background color
        self.render_bkgd = nn.Parameter(torch.ones(1, 3, dtype=torch.float32), requires_grad=False)

        # model parameters
        grid_resolution = 128
        grid_nlvl = 1
        # render parameters
        self.render_step_size = 5e-3
        self.test_chunk_size = 4096
        
        aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
        aabb = nn.Parameter(aabb, requires_grad=False)
        self.estimator = OccGridEstimator(
            roi_aabb=aabb, resolution=grid_resolution, levels=grid_nlvl)
        self.estimator.binaries = self.estimator.binaries + True
        self.measure = Measure(use_gpu=False)

    def train_dataloader(self):
        if self.dataset == 'shapenet':
            self.train_dataset =  ShapeNetDataset(self.train_mode, 
                                                    scene_dir=self.scene_dir, 
                                                    voxel_size=self.voxel_size, 
                                                    coarse_scale=self.coarse_scale)
        elif self.dataset == 'shapenet_get3d':
            self.train_dataset =  ShapeNetGet3DDataset(self.train_mode, 
                                                    scene_dir=self.scene_dir, 
                                                    voxel_size=self.voxel_size, 
                                                    coarse_scale=self.coarse_scale)
        elif self.dataset == 'render_google':
            self.train_dataset =  RenderGoogleDataset(self.train_mode, 
                                                    scene_dir=self.scene_dir, 
                                                    voxel_size=self.voxel_size, 
                                                    coarse_scale=self.coarse_scale)
        else:
            self.train_dataset =  ScanNetDataset(self.train_mode, 
                                                    scene_dir=self.scene_dir, 
                                                    voxel_size=self.voxel_size, 
                                                    coarse_scale=self.coarse_scale, 
                                                    patch_size=self.patch_size,
                                                    img_wh=self.img_wh)
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=minkowski_collate_fn,
            num_workers=self.train_num_workers,
            shuffle=True,
            pin_memory=False,
            drop_last=True
        )

    def val_dataloader(self):
        if self.dataset == 'shapenet':
            self.val_dataset =  ShapeNetDataset(self.val_mode, 
                                                    scene_dir=self.scene_dir, 
                                                    voxel_size=self.voxel_size, 
                                                    coarse_scale=self.coarse_scale)
        elif self.dataset == 'shapenet_get3d':
            self.val_dataset =  ShapeNetGet3DDataset(self.val_mode, 
                                                    scene_dir=self.scene_dir, 
                                                    voxel_size=self.voxel_size, 
                                                    coarse_scale=self.coarse_scale)
        elif self.dataset == 'render_google':
            self.val_dataset =  RenderGoogleDataset(self.val_mode, 
                                                    scene_dir=self.scene_dir, 
                                                    voxel_size=self.voxel_size, 
                                                    coarse_scale=self.coarse_scale)
        else:
            self.val_dataset =  ScanNetDataset(self.val_mode, 
                                                    scene_dir=self.scene_dir, 
                                                    voxel_size=self.voxel_size, 
                                                    coarse_scale=self.coarse_scale, 
                                                    patch_size=self.patch_size,
                                                    img_wh=self.img_wh)
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            collate_fn=minkowski_collate_fn,
            num_workers=self.val_num_workers
        )

    def forward(self, batch, is_training, batch_idx):
        B, H, W, _ = batch['rays_o'].shape # (B, N, 3)
        assert B == 1, 'only support batch size 1'
        
        device = batch['rays_o'].device
        rays_o = batch['rays_o'].reshape(-1, 3)  # (B*H*W, 3)
        rays_d = batch['rays_d'].reshape(-1, 3)  # (B*H*W, 3)
        rgbs_gt = batch['rgbs'].reshape(-1, 3) # (B*H*W, 3)
        img_path = batch['filename']

        if is_training:
            ## random choice (patch, patch) matrix
            # P = self.patch_size
            # Gx = H // self.patch_size
            # x = random.randint(0, H - (P - 1) * Gx - 1)
            # i = torch.LongTensor([x + Gx*n for n in range(P)])
            # Gy = W // self.patch_size
            # y = random.randint(0, W - (P - 1) * Gy - 1)
            # j = torch.LongTensor([y + Gy*n for n in range(P)])

            # gi, gj = torch.meshgrid(i, j)
            # gi = gi.reshape(-1)
            # gj = gj.reshape(-1)
            # rays_o = rays_o[:, gi, gj]
            # rays_d = rays_d[:, gi, gj]
            # rgbs_gt = rgbs_gt[:, gi, gj]
            # H = W = self.patch_size

            ## random choice (patch x patch,) vector
            num_rays = self.patch_size ** 2
            idx_rand = torch.randperm(H * W)[:num_rays]
            rays_o = rays_o[idx_rand]
            rays_d = rays_d[idx_rand]
            rgbs_gt = rgbs_gt[idx_rand]

        aabb = batch['aabb']
        voxels = batch['voxels']  # [B, 3, 4*32, S, S]
        
        # dense encoder
        voxels_xyz = self.trivol_encoder(voxels) # (B, feat_dim, P, S, S)
        rays = Rays(origins=rays_o, viewdirs=rays_d)

        self.estimator.aabb = batch['aabb'].reshape(6)
        rgbs_prd, acc, depths_prd, _ = render_image_with_occgrid(
                    self.radiance_field,
                    self.estimator,
                    rays,
                    voxels_xyz,
                    aabb,
                    render_step_size=self.render_step_size,
                    render_bkgd=self.render_bkgd,
                    test_chunk_size=self.test_chunk_size,
                    alpha_thre=1e-4
                    )
        if not is_training:
            # # resize to image
            rgbs_prd = rgbs_prd.reshape(B, H, W, 3).permute(0, 3, 1, 2)  # (B, 3, H, W)
            rgbs_gt = rgbs_gt.reshape(B, H, W, 3).permute(0, 3, 1, 2)  # (B, 3, H, W)
            depths_prd = depths_prd.reshape(B, H, W)

        return rgbs_prd, rgbs_gt, depths_prd, img_path

    def training_step(self, batch, batch_idx):
        rgbs_prd, rgbs_gt, depths_prd, img_path = self(batch, is_training=True, batch_idx=batch_idx)
        loss_l2 = 1.0 * ((rgbs_prd - rgbs_gt)**2).mean()
        # loss_tv = 0.01 * (self.tv_loss(voxels_xyz[0]) + self.tv_loss(voxels_xyz[1]) + self.tv_loss(voxels_xyz[2]))
        loss = loss_l2
        
        if batch_idx % 100 == 0:
            psnr_nerf = psnr(rgbs_prd, rgbs_gt)
            self.log("train/loss_l2", loss_l2, on_step=True, on_epoch=True, logger=True, batch_size=self.batch_size)
            self.log("train/loss", loss, on_step=True, on_epoch=True, logger=True, batch_size=self.batch_size)
            self.log("train/psnr_nerf", psnr_nerf, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
            self.log("train/lr", get_learning_rate(self.optimizer), on_epoch=True, logger=True, batch_size=self.batch_size)
 
        return loss
    
    def validation_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        rgbs_prd, rgbs_gt, depths_prd, img_path = self(batch, is_training=False, batch_idx=batch_idx)

        # save image
        rgb_prd = rgbs_prd[0].cpu()
        rgb_gt = rgbs_gt[0].cpu()
        depths_prd = depths_prd[0].cpu()
        depth = visualize_depth(depths_prd) # (3, H, W)
        
        img_path = img_path[0]
        name = img_path.split('/')[-3] + img_path.split('/')[-2] + '_' + img_path.split('/')[-1]
        
        stack_nerf = torch.cat([rgb_gt, rgb_prd, depth], dim=-1) # (3, H, W)
        img_path = os.path.join('logs', self.exp_name, 'fid', f"{self.current_epoch:04d}", "vis", f"{name}")
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        cv2.imwrite(img_path, (stack_nerf.permute(1,2,0)*255.0).cpu().numpy().astype(np.uint8)[..., [2, 1, 0]], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        
        loss = ((rgbs_prd - rgbs_gt)**2).mean()
        psnr_nerf = psnr(rgbs_prd, rgbs_gt)
        
        rgbs_prd_numpy = (rgbs_prd.permute(0, 2, 3, 1).clone().detach().cpu().numpy()) * 255.0
        rgbs_gt_numpy = (rgbs_gt.permute(0, 2, 3, 1).clone().detach().cpu().numpy()) * 255.0
        batch_size = rgbs_prd_numpy.shape[0]
        ssim = 0
        lpips = 0
        for mm in range(batch_size):
            psnr_o, ssim_o, lpips_o = self.measure.measure(rgbs_prd_numpy[mm].astype(np.uint8), 
                                                           rgbs_gt_numpy[mm].astype(np.uint8))
            ssim+=ssim_o
            lpips+=lpips_o
        ssim = ssim/batch_size
        lpips = lpips/batch_size
        ssim = torch.Tensor([ssim]).float()
        lpips = torch.Tensor([lpips]).float()

        output = {"loss": loss, 
                  "psnr_nerf": psnr_nerf,
                  "ssim_nerf": ssim,
                  "lpips_nerf":lpips
                  }
        return output
    
    def validation_epoch_end(self, val_step_outputs):
        loss_val = torch.stack([out['loss'] for out in val_step_outputs]).mean()
        psnr_nerf = torch.stack([out['psnr_nerf'] for out in val_step_outputs]).mean()
        ssim_nerf = torch.stack([out['ssim_nerf'] for out in val_step_outputs]).mean()
        lpips_nerf = torch.stack([out['lpips_nerf'] for out in val_step_outputs]).mean()

        self.log("test/loss", loss_val, on_epoch=True, logger=True, batch_size=self.val_batch_size)
        self.log("test/psnr_nerf", psnr_nerf, on_epoch=True, prog_bar=True, logger=True, batch_size=self.val_batch_size)
        self.log("test/ssim_nerf", ssim_nerf, on_epoch=True, logger=True, batch_size=self.val_batch_size)
        self.log("test/lpips_nerf", lpips_nerf, on_epoch=True, prog_bar=True, logger=True, batch_size=self.val_batch_size)


        paths = [os.path.join('logs', self.exp_name, 'fid', f"{self.current_epoch:04d}", 'real'),
                 os.path.join('logs', self.exp_name, 'fid', f"{self.current_epoch:04d}", 'fake')]
        if self.val_mode == "val" and os.path.exists(paths[0]):
            fid_value = fid_score.calculate_fid_given_paths(paths,
                                                    batch_size=50,
                                                    device='cuda:0',
                                                    dims=2048,
                                                    num_workers=0)
            self.log("test/fid", fid_value, on_epoch=True, prog_bar=True, logger=True, batch_size=self.val_batch_size)
        if self.val_mode == "test":
            img_paths = glob.glob(os.path.join('logs', self.exp_name, 'vis', '*.jpg'))
            img_paths = sorted(img_paths, key=lambda x: int(os.path.basename(x).split('.')[0]))
            writer = imageio.get_writer(os.path.join('logs', self.exp_name, 'vis', 'demo.mp4'), fps=30)
            for im in img_paths:
                writer.append_data(imageio.imread(im))
            writer.close()
            assert True == False

    def configure_optimizers(self):
        self.optimizer = AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: 0.1**(epoch/float(self.max_epochs)))

        return [self.optimizer], [scheduler]


if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--scene_dir", type=str, default="", help="scene dir")
    pa.add_argument("--resume_path", type=str, help="resume ckpt path")
    pa.add_argument("--max_epochs", type=int, default=200, help="Max epochs")
    pa.add_argument("--lr", type=float, default=5e-3, help="Learning rate")
    pa.add_argument("--batch_size", type=int, default=1, help="batch size per GPU")
    pa.add_argument("--ngpus", type=int, default=1, help="num_gpus")
    pa.add_argument("--exp_name", type=str, default="trivol", help="num_gpus")
    pa.add_argument("--train_mode", type=str, default="train")
    pa.add_argument("--val_mode", type=str, default="val", help="test or val")
    pa.add_argument("--voxel_size", type=float, default=0.01, help="the size of voxel")
    pa.add_argument("--patch_size", type=int, default=64, help="the size of sample patch")
    pa.add_argument("--dataset", type=str, default='shapenet', help="the dataset to train")
    pa.add_argument('--img_wh', nargs="+", type=int, default=[640, 512],
                        help='resolution (img_w, img_h) of the image')    
    pa.add_argument("--finetune", action='store_true', default=False, help="is finetune")
    pa.add_argument("--feat_dim", type=int, default=8, help="the dimension of each feature")
    pa.add_argument("--coarse_scale", type=int, default=8, help="coarse scale or ball scale")

    args = pa.parse_args()
    num_devices = min(args.ngpus, torch.cuda.device_count())
    print(f"Testing {num_devices} GPUs.")

    if args.finetune:
        pl_module = TriVolModule.load_from_checkpoint(args.resume_path, 
                                                scene_dir=args.scene_dir, 
                                                dataset=args.dataset,
                                                batch_size=args.batch_size, 
                                                lr=args.lr, 
                                                voxel_size=args.voxel_size, 
                                                max_epochs=args.max_epochs,
                                                exp_name=args.exp_name,
                                                train_mode=args.train_mode,
                                                val_mode=args.val_mode,
                                                patch_size=args.patch_size,
                                                img_wh=args.img_wh)
    else:
         pl_module = TriVolModule( 
                            scene_dir=args.scene_dir, 
                            dataset=args.dataset,
                            batch_size=args.batch_size, 
                            lr=args.lr,
                            voxel_size=args.voxel_size, 
                            max_epochs=args.max_epochs,
                            exp_name=args.exp_name,
                            train_mode=args.train_mode,
                            val_mode=args.val_mode,
                            patch_size=args.patch_size,
                            img_wh=args.img_wh,
                            feat_dim=args.feat_dim,
                            coarse_scale=args.coarse_scale)

    tb_logger = pl_loggers.TensorBoardLogger("logs/%s" % args.exp_name)
    
    checkpoint_callback = ModelCheckpoint(
    monitor="train/psnr_nerf",
    save_top_k=5,
    save_last=True,
    mode="max"
    )

    trainer = Trainer(max_epochs=args.max_epochs, 
                      resume_from_checkpoint=None if args.finetune else args.resume_path,
                      gpus=num_devices, 
                      strategy="ddp", 
                      logger=tb_logger,
                      callbacks=[checkpoint_callback],
                      num_sanity_val_steps=1,
                      check_val_every_n_epoch=1,
                      log_gpu_memory=True,
                      )
    trainer.fit(pl_module)
