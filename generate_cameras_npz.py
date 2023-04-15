import numpy as np
import cv2
import torch
import os
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
import json
import trimesh
import glob
import PIL
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt



# image [720, 1280]
# depth [720, 1280]
image_size = 384
trans_totensor = transforms.Compose([
    transforms.CenterCrop(720),
    transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
])
depth_trans_totensor = transforms.Compose([
    transforms.CenterCrop(720),
    transforms.Resize(image_size, interpolation=PIL.Image.NEAREST),
])


out_path_prefix = 'data/fluiddata/'
data_root = '/home/rayne/datasets/fluid_simulation/data/multi_scene_full/'
scenes = ['scan0']
out_names = ['scan0']

for scene, out_name in zip(scenes, out_names):
    out_path = os.path.join(out_path_prefix, out_name)
    os.makedirs(out_path, exist_ok=True)
    print(out_path)

    folders = ["image", "mask", "depth"]
    for folder in folders:
        out_folder = os.path.join(out_path, folder)
        os.makedirs(out_folder, exist_ok=True)

    # load intrinsic
    intrinsic_path = os.path.join(data_root, scene, 'intrinsic.json')
    camera_intrinsic = np.array(json.load(open(intrinsic_path))["intrinsic_matrix"])
    print(camera_intrinsic)
    
    # load pose
    poses = []
    pose_path = os.path.join(data_root, scene, 'scene', 'trajectory.npy')
    poses = np.load(pose_path)
    print(poses.shape)

    # deal with invalid poses
    valid_poses = np.isfinite(poses).all(axis=2).all(axis=1)
    min_vertices = poses[:, :3, 3][valid_poses].min(axis=0)
    max_vertices = poses[:, :3, 3][valid_poses].max(axis=0)
 
    center = (min_vertices + max_vertices) / 2.
    scale = 2. / (np.max(max_vertices - min_vertices) + 3.)
    print(center, scale)

    # we should normalized to unit cube
    scale_mat = np.eye(4).astype(np.float32)
    scale_mat[:3, 3] = -center
    scale_mat[:3 ] *= scale 
    scale_mat = np.linalg.inv(scale_mat)

    # copy image
    cameras = {}
    pcds = []
    H, W = 720, 1280
    print(camera_intrinsic)
    
    K = np.eye(4)
    K[:3, :3] = camera_intrinsic
    print(K)
    
    out_index = 0
    for idx, (valid, pose) in enumerate(zip(valid_poses, poses)):
        print(idx, valid)

        mask = (np.ones((image_size, image_size, 3)) * 255.).astype(np.uint8)

        target_image = os.path.join(out_path, "mask/%03d.png"%(out_index))
        cv2.imwrite(target_image, mask)

        # save pose
        pcds.append(pose[:3, 3])
        pose = K @ np.linalg.inv(pose)
        
        #cameras["scale_mat_%d"%(out_index)] = np.eye(4).astype(np.float32)
        cameras["scale_mat_%d"%(out_index)] = scale_mat
        cameras["world_mat_%d"%(out_index)] = pose

        out_index += 1

    np.savez(os.path.join(out_path, "cameras.npz"), **cameras)
