# TriVol

## TriVol: Point Cloud Representation via Triple Volumes

This is the official implementation of [TriVol: Point Cloud Representation via Triple Volumes]()&nbsp; (CVPR 2022), a Point Cloud Renderer.

### Installation
* Pytorch / pytorch-lightning
* MinkowskiEngine
* requirements
    ```bash
    pip install -r requirements.txt
    ```

### Training and Rendering
```bash
python train.py --scene_dir /path/to/scannet \
                --dataset scannet \
                --val_mode val \
                --max_epochs 200 \
                --lr 0.001 \
                --voxel_size 0.005 \
                --batch_size 1 \
                --ngpus 4 \
                --patch_size 128 \
                --num_sample 128 \
                --img_wh 648 484 \
                --exp_name Scannet \
```

### Citation
```
@inproceedings{TriVol,
  title={TriVol: Point Cloud Representation via Triple Volumes},
  author={Tao Hu, Xiaogang Xu, Ruihang Chu, Jiaya Jia},
  booktitle={Computer Vision and Pattern Recognition},
  year={2022}
}
```