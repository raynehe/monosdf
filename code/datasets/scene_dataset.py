import os
import torch
import torch.nn.functional as F
import numpy as np

import utils.general as utils
from utils import rend_util
from glob import glob
import cv2
import random
import pickle as pkl
import json

class SceneDataset(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir,
                 img_res,
                 scan_id=0,
                 num_views=-1,  
                 ):

        self.instance_dir = os.path.join('../data', data_dir, 'scan{0}'.format(scan_id))

        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res

        assert os.path.exists(self.instance_dir), "Data directory is empty"
        
        self.num_views = num_views
        assert num_views in [-1, 3, 6, 9]
        
        self.sampling_idx = None

        image_dir = '{0}/image'.format(self.instance_dir)
        image_paths = sorted(utils.glob_imgs(image_dir))
        self.n_images = len(image_paths)

        self.cam_file = '{0}/cameras.npz'.format(self.instance_dir)
        camera_dict = np.load(self.cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_all = []
        self.pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        self.rgb_images = []
        for path in image_paths:
            rgb = rend_util.load_rgb(path)
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(torch.from_numpy(rgb).float())
            
        # used a fake depth image and normal image
        self.depth_images = []
        self.normal_images = []

        for path in image_paths:
            depth = np.ones_like(rgb[:, :1])
            self.depth_images.append(torch.from_numpy(depth).float())
            normal = np.ones_like(rgb)
            self.normal_images.append(torch.from_numpy(normal).float())
            
    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        if self.num_views >= 0:
            image_ids = [25, 22, 28, 40, 44, 48, 0, 8, 13][:self.num_views]
            idx = image_ids[random.randint(0, self.num_views - 1)]
            
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
            "pose": self.pose_all[idx]
        }

        ground_truth = {
            "rgb": self.rgb_images[idx],
            "depth": self.depth_images[idx],
            "normal": self.normal_images[idx],
        }
        
        if self.sampling_idx is not None:
            ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
            ground_truth["depth"] = self.depth_images[idx][self.sampling_idx, :]
            ground_truth["mask"] = torch.ones_like(self.depth_images[idx][self.sampling_idx, :])
            ground_truth["normal"] = self.normal_images[idx][self.sampling_idx, :]
            
            sample["uv"] = uv[self.sampling_idx, :]

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def get_scale_mat(self):
        return np.load(self.cam_file)['scale_mat_0']


# Dataset with monocular depth and normal
class SceneDatasetDN(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir,
                 img_res,
                 scan_id=0,
                 center_crop_type='xxxx',
                 data_type='splishsplash',
                 use_mask=False,
                 num_views=-1
                 ):

        self.instance_dir = os.path.join('/home/rayne/datasets/monosdf', data_dir, 'scan{0}'.format(scan_id))

        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res
        self.num_views = num_views
        self.data_type = data_type
        assert num_views in [-1, 3, 6, 9]
        
        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.sampling_idx = None
        
        def glob_data(data_dir):
            data_paths = []
            data_paths.extend(glob(data_dir))
            data_paths = sorted(data_paths)
            return data_paths
            
        image_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "*_rgb.png"))
        depth_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "*_depth.npy"))
        normal_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "*_normal.npy"))
        particles_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), 'particles', "*.npz"))
        
        # mask is only used in the replica dataset as some monocular depth predictions have very large error and we ignore it
        if use_mask:
            mask_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "*_mask.npy"))
        else:
            mask_paths = None

        self.n_images = len(image_paths)
        
        self.cam_file = '{0}/cameras.npz'.format(self.instance_dir)
        camera_dict = np.load(self.cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_all = []
        self.pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)

            # because we do resize and center crop 384x384 when using omnidata model, we need to adjust the camera intrinsic accordingly
            if center_crop_type == 'center_crop_for_replica':
                scale = 384 / 680
                offset = (1200 - 680 ) * 0.5
                intrinsics[0, 2] -= offset
                intrinsics[:2, :] *= scale
            elif center_crop_type == 'center_crop_for_tnt':
                scale = 384 / 540
                offset = (960 - 540) * 0.5
                intrinsics[0, 2] -= offset
                intrinsics[:2, :] *= scale
            elif center_crop_type == 'center_crop_for_dtu':
                scale = 384 / 1200
                offset = (1600 - 1200) * 0.5
                intrinsics[0, 2] -= offset
                intrinsics[:2, :] *= scale
            elif center_crop_type == 'padded_for_dtu':
                scale = 384 / 1200
                offset = 0
                intrinsics[0, 2] -= offset
                intrinsics[:2, :] *= scale
            elif center_crop_type == 'center_crop_for_fluid':
                scale = 384 / 400
                offset = 0 # CenterCrop
                intrinsics[0, 2] -= offset
                intrinsics[:2, :] *= scale
            elif center_crop_type == 'no_crop':  # for scannet dataset, we already adjust the camera intrinsic duing preprocessing so nothing to be done here
                pass
            else:
                raise NotImplementedError
            
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        # self.all_rays = []

        # if data_dir == 'fluid':
        #     pose_path = os.path.join(self.instance_dir, 'scene', 'trajectory.npy')
        #     self.pose_all = torch.from_numpy(np.load(pose_path)).float()
        #     intrinsic_path = os.path.join(self.instance_dir, 'intrinsic.npy')
        #     self.intrinsics_all = torch.from_numpy(np.load(intrinsic_path)).float()

        #     for i in range(self.n_images):
        #         # crop intrinsics
        #         scale = 384 / 400
        #         offset = 0
        #         self.intrinsics_all[i][0, 2] -= offset
        #         self.intrinsics_all[i][:2, :] *= scale
        #         # compute rays
        #         pose = self.pose_all[i]
        #         focal = self.intrinsics_all[i][0, 0]
        #         directions = self.get_ray_directions(self.img_res[0], self.img_res[1], focal)
        #         rays_o, rays_d = self.get_rays(directions, pose[:3])
        #         self.all_rays.append(torch.cat([rays_o, rays_d], -1))

        self.rgb_images = []
        for path in image_paths:
            rgb = rend_util.load_rgb(path)
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(torch.from_numpy(rgb).float())
            
        self.particles_poss = []
        for path in particles_paths:
            particles_pos, _ = self._read_particles(path)
            self.particles_poss.append(np.stack(particles_pos, 0))
        self.particles_poss = torch.from_numpy(np.stack(self.particles_poss, 0)).float()

        self.depth_images = []
        self.normal_images = []

        for dpath, npath in zip(depth_paths, normal_paths):
            depth = np.load(dpath)
            self.depth_images.append(torch.from_numpy(depth.reshape(-1, 1)).float())
        
            normal = np.load(npath)
            normal = normal.reshape(3, -1).transpose(1, 0)
            # important as the output of omnidata is normalized
            normal = normal * 2. - 1.
            self.normal_images.append(torch.from_numpy(normal).float())

        # load mask
        self.mask_images = []
        if mask_paths is None:
            for depth in self.depth_images:
                mask = torch.ones_like(depth)
                self.mask_images.append(mask)
        else:
            for path in mask_paths:
                mask = np.load(path)
                self.mask_images.append(torch.from_numpy(mask.reshape(-1, 1)).float())

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        if self.num_views >= 0:
            image_ids = [25, 22, 28, 40, 44, 48, 0, 8, 13][:self.num_views]
            idx = image_ids[random.randint(0, self.num_views - 1)]
        
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            # "rays": self.all_rays[idx],
            "particles_poss": self.particles_poss[idx],
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
            "pose": self.pose_all[idx]
        }
        
        ground_truth = {
            "rgb": self.rgb_images[idx],
            "depth": self.depth_images[idx],
            "mask": self.mask_images[idx],
            "normal": self.normal_images[idx]
        }

        if self.sampling_idx is not None:
            ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
            ground_truth["full_rgb"] = self.rgb_images[idx]
            ground_truth["normal"] = self.normal_images[idx][self.sampling_idx, :]
            ground_truth["depth"] = self.depth_images[idx][self.sampling_idx, :]
            ground_truth["full_depth"] = self.depth_images[idx]
            ground_truth["mask"] = self.mask_images[idx][self.sampling_idx, :]
            ground_truth["full_mask"] = self.mask_images[idx]
         
            sample["uv"] = uv[self.sampling_idx, :]

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def get_scale_mat(self):
        return np.load(self.cam_file)['scale_mat_0']

    def _read_particles(self, particle_path):
        """
        read initial particle information and the bounding box information
        """
        if self.data_type == 'blender':
            # particle_info = np.load(osp.join(self.root_dir, self.split, particle_path))
            # with open(osp.join(self.root_dir, self.split, particle_path), 'rb') as fp:
            with open(particle_path, 'rb') as fp:
                particle_info = pkl.load(fp)
            particle_pos = np.array(particle_info['location']).reshape(-1, 3)
            particle_vel = np.array(particle_info['velocity']).reshape(-1, 3)
        elif self.data_type == 'splishsplash':
            # particle_info = np.load(osp.join(self.root_dir, self.split, particle_path))
            particle_info = np.load(particle_path)
            particle_pos = particle_info['pos']
            particle_vel = particle_info['vel']
        else:
            raise NotImplementedError('please enter correct data type')
        # import ipdb;ipdb.set_trace()
        # particle_pos = torch.from_numpy(particle_pos).float()
        # particle_vel = torch.from_numpy(particle_vel).float()
        return particle_pos, particle_vel
    
    def get_ray_directions(self, H, W, focal):
        """
        Get ray directions for all pixels in camera coordinate.
        Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
                ray-tracing-generating-camera-rays/standard-coordinate-systems

        Inputs:
            H, W, focal: image height, width and focal length

        Outputs:
            directions: (H, W, 3), the direction of the rays in camera coordinate
        """
        from kornia import create_meshgrid
        grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
        i, j = grid.unbind(-1)
        # the direction here is without +0.5 pixel centering as calibration is not so accurate
        # see https://github.com/bmild/nerf/issues/24
        directions = \
            torch.stack([(i-W/2)/focal, -(j-H/2)/focal, -torch.ones_like(i)], -1) # (H, W, 3)

        return directions
    
    def get_rays(self, directions, c2w):
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
        rays_d = directions @ c2w[:, :3].T # (H, W, 3)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        # The origin of all rays is the camera origin in world coordinate
        rays_o = c2w[:, 3].expand(rays_d.shape) # (H, W, 3)

        # rays_d = rays_d.view(-1, 3)
        # rays_o = rays_o.view(-1, 3)

        return rays_o, rays_d