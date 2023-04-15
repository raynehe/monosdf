import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import rend_util
from model.embedder import *
from model.density import LaplaceDensity
from model.ray_sampler import ErrorBoundSampler
import matplotlib.pyplot as plt
import numpy as np
from pytorch3d.ops import ball_query
from model.nerf import Embedding

class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            sdf_bounding_sphere,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0,
            sphere_scale=1.0,
            inside_outside=False,
    ):
        super().__init__()

        self.sdf_bounding_sphere = sdf_bounding_sphere
        self.sphere_scale = sphere_scale
        dims = [d_in] + dims + [d_out + feature_vector_size]

        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            dims[0] = input_ch
        print(multires, dims)
        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)

                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)

    def forward(self, input):
        if self.embed_fn is not None:
            input = self.embed_fn(input)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        return x

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[:,:1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients

    def get_outputs(self, x):
        x.requires_grad_(True)
        output = self.forward(x)
        sdf = output[:,:1]
        ''' Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded '''
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf)
        feature_vectors = output[:, 1:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        return sdf, feature_vectors, gradients

    def get_sdf_vals(self, x):
        sdf = self.forward(x)[:,:1]
        ''' Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded '''
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf)
        return sdf


from hashencoder.hashgrid import _hash_encode, HashEncoder
class ImplicitNetworkGrid(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            sdf_bounding_sphere,
            position_encoding_size,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0,
            sphere_scale=1.0,
            inside_outside=False,
            base_size = 16,
            end_size = 2048,
            logmap = 19,
            num_levels=16,
            level_dim=2,
            divide_factor = 1.5, # used to normalize the points range for multi-res grid
            NN_search = {},
            encoding = {},
            use_grid_feature = True
    ):
        super().__init__()

        self.sdf_bounding_sphere = sdf_bounding_sphere
        self.sphere_scale = sphere_scale
        dims = [d_in + position_encoding_size] + dims + [d_out + feature_vector_size]
        self.embed_fn = None
        self.divide_factor = divide_factor
        self.grid_feature_dim = num_levels * level_dim
        self.use_grid_feature = use_grid_feature
        dims[0] += self.grid_feature_dim
        # dims [230, 256, 256, 257]
        self.radius = NN_search['search_radius_scale'] * NN_search['particle_radius']
        self.fix_radius = NN_search['fix_radius']
        self.num_neighbor = NN_search['N_neighbor']
        self.encoding_cfg = encoding
        self.embedding_xyz = Embedding(3, 10)
        self.embedding_dir = Embedding(3, 4)
        if self.encoding_cfg['density']:
            self.embedding_density = Embedding(1, 4)
        
        print(f"using hash encoder with {num_levels} levels, each level with feature dim {level_dim}")
        print(f"resolution:{base_size} -> {end_size} with hash map size {logmap}")
        self.encoding = HashEncoder(input_dim=3, num_levels=num_levels, level_dim=level_dim, 
                    per_level_scale=2, base_resolution=base_size, 
                    log2_hashmap_size=logmap, desired_resolution=end_size)
        
        '''
        # can also use tcnn for multi-res grid as it now supports eikonal loss
        base_size = 16
        hash = True
        smoothstep = True
        self.encoding = tcnn.Encoding(3, {
                        "otype": "HashGrid" if hash else "DenseGrid",
                        "n_levels": 16,
                        "n_features_per_level": 2,
                        "log2_hashmap_size": 19,
                        "base_resolution": base_size,
                        "per_level_scale": 1.34,
                        "interpolation": "Smoothstep" if smoothstep else "Linear"
                    })
        '''
        
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            dims[0] += input_ch - 3
        print("network architecture")
        print(dims)
        
        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            # print('layer', l, 'in', dims[l], 'out', out_dim)
            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)

                elif l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(dims[l]))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)
        self.cache_sdf = None

    def forward(self, input, physical_particles):
        if self.use_grid_feature:
            # normalize point range as encoding assume points are in [-1, 1]
            # print(input.min().item(),input.max().item())
            feature = self.encoding(input / self.divide_factor)
        else:
            feature = torch.zeros_like(input[:, :, :1].repeat(1, 1, 1, self.grid_feature_dim)).squeeze(0)
        # add position embdding here
        # search
        dists_0, indices_0, neighbors_0, radius_0 = self.search(input, physical_particles, self.fix_radius)
        # embedding attributes
        pos_like_feats_0 = self.embedding_local_geometry(dists_0, indices_0, neighbors_0, radius_0, input)
        pos_input_feats_0 = torch.cat(pos_like_feats_0, dim=1).reshape(feature.shape[0], feature.shape[1], -1)
                    
        if self.embed_fn is not None:
            embed = self.embed_fn(input)
            input = torch.cat((embed, feature), dim=-1)
        else:
            input = torch.cat((pos_input_feats_0, feature), dim=-1)

        input = input.reshape(-1, input.shape[-1]) # [131072, 230]
        x = input
        # print('input', round(input.min().item(),2),round(input.max().item(),2))

        for l in range(0, self.num_layers - 1):
            # print('layer', l)
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            x = lin(x)
            # print('x1', round(x.min().item(),2),round(x.max().item(),2))

            if l < self.num_layers - 2:
                x = self.softplus(x)
            # print('x2', round(x.min().item(),2),round(x.max().item(),2))

        return x

    def gradient(self, x, physical_particles):
        x.requires_grad_(True)
        y = self.forward(x, physical_particles)[:,:1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients

    def get_outputs(self, x, physical_particles):
        x.requires_grad_(True) # (1024,98,3)
        output = self.forward(x, physical_particles)
        sdf = output[:,:1] # (100352, 1)

        feature_vectors = output[:, 1:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        return sdf, feature_vectors, gradients

    def get_sdf_vals(self, x, physical_particles):
        sdf = self.forward(x, physical_particles)[:,:1]
        return sdf

    def mlp_parameters(self):
        parameters = []
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            parameters += list(lin.parameters())
        return parameters

    def grid_parameters(self):
        print("grid parameters", len(list(self.encoding.parameters())))
        for p in self.encoding.parameters():
            print(p.shape)
        return self.encoding.parameters()

    def search(self, ray_particles, particles, fix_radius):
        # particles (6320, 3) 不应该是(1, 6320, 3)
        # ray_particles (1024, 128, 3)
        raw_data = particles.unsqueeze(0).repeat(ray_particles.shape[0], 1, 1)
        if fix_radius:
            radius = self.radius
            dists, indices, neighbors = ball_query(p1=ray_particles, 
                                                   p2=raw_data, 
                                                   radius=radius, K=self.num_neighbor)
        # else:
        #     radius = self.get_search_radius(self.radius, ray_particles[:,:,-1] - ro[-1], focal)
        #     dists, indices, neighbors = self._ball_query(ray_particles, raw_data, radius, self.num_neighbor)
        return dists, indices, neighbors, radius

    def embedding_local_geometry(self, dists, indices, neighbors, radius, ray_particles):
        """
        pos like feats
            1. smoothed positions
            2. ref hit pos, i.e., ray position
            3. density
            3. variance
        dir like feats
            1. hit direction, i.e., ray direction
            2. main direction after PCA
        """
        # calculate mask
        nn_mask = dists.ne(0)
        num_nn = nn_mask.sum(-1, keepdim=True)

        # hit pos and hit direction (basic in NeRF formulation)
        pos_like_feats = []
        hit_pos = ray_particles.reshape(-1,3)
        hit_pos_embedded = self.embedding_xyz(hit_pos)
        pos_like_feats.append(hit_pos_embedded)
        # smoothing 
        smoothed_pos, density = self.smoothing_position(ray_particles, neighbors, radius, num_nn, exclude_ray=self.encoding_cfg['exclude_ray'])
        # density
        if self.encoding_cfg['density']:
            density_embedded = self.embedding_density(density.reshape(-1, 1))
            pos_like_feats.append(density_embedded)
        # smoothed pos
        if self.encoding_cfg['smoothed_pos']:
            smoothed_pos_embedded = self.embedding_xyz(smoothed_pos.reshape(-1, 3))
            pos_like_feats.append(smoothed_pos_embedded)
        # variance
        if self.encoding_cfg['var']:
            vec_pp2rp = torch.zeros(ray_particles.shape[0], ray_particles.shape[1], self.num_neighbor, 3).to(neighbors.device)
            vec_pp2rp[nn_mask] = (neighbors - ray_particles.unsqueeze(-2))[nn_mask]
            vec_pp2rp_mean = vec_pp2rp.sum(-2) / (num_nn+1e-12)
            variance = torch.zeros(ray_particles.shape[0], ray_particles.shape[1], self.num_neighbor, 3).to(neighbors.device)
            variance[nn_mask] = ((vec_pp2rp - vec_pp2rp_mean.unsqueeze(-2))**2)[nn_mask]
            variance = variance.sum(-2) / (num_nn+1e-12)
            variance_embedded = self.embedding_xyz(variance.reshape(-1,3))
            pos_like_feats.append(variance_embedded)
        # smoothed dir
        return pos_like_feats
    
    def smoothing_position(self, ray_pos, nn_poses, raduis, num_nn, exclude_ray=True, larger_alpha=0.9, smaller_alpha=0.1):
        dists = torch.norm(nn_poses - ray_pos.unsqueeze(-2), dim=-1)
        weights = torch.clamp(1 - (dists / raduis) ** 3, min=0)
        weighted_nn = (weights.unsqueeze(-1) * nn_poses).sum(-2) / (weights.sum(-1, keepdim=True)+1e-12)
        if exclude_ray:
            pos = weighted_nn
        else:
            if self.encoding_cfg['same_smooth_factor']:
                alpha = torch.ones(ray_pos.shape[0], ray_pos.shape[1], 1) * larger_alpha
            else:
                alpha = torch.ones(ray_pos.shape[0], ray_pos.shape[1], 1) * larger_alpha
                alpha[num_nn.le(20)] = smaller_alpha
            pos = ray_pos * (1-alpha) + weighted_nn * alpha
        return pos, weights.sum(-1, keepdim=True)
    
    def get_particles_direction(self, particles, ro):
        ros = ro.expand(particles.shape[0], -1)
        dirs = particles - ros
        dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
        return dirs

class RenderingNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            position_encoding_size,
            dir_encoding_size,
            normal_size,
            mode,
            d_in,
            d_out,
            dims,
            weight_norm=True,
            multires_view=0,
            NN_search = {},
            encoding = {},
            bias=1.0,
            geometric_init=True,
            per_image_code = False
    ):
        super().__init__()

        self.mode = mode
        # dims = [d_in + feature_vector_size] + dims + [d_out]
        dims = [d_in + feature_vector_size + position_encoding_size + dir_encoding_size + normal_size] + dims + [d_out]


        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.per_image_code = per_image_code
        if self.per_image_code:
            # nerf in the wild parameter
            # parameters
            # maximum 1024 images
            self.embeddings = nn.Parameter(torch.empty(1024, 32))
            std = 1e-4
            self.embeddings.data.uniform_(-std, std)
            dims[0] += 32

        print("rendering network architecture:")
        print(dims)

        self.num_layers = len(dims)
        self.radius = NN_search['search_radius_scale'] * NN_search['particle_radius']
        self.fix_radius = NN_search['fix_radius']
        self.num_neighbor = NN_search['N_neighbor']
        self.encoding_cfg = encoding
        self.embedding_xyz = Embedding(3, 10)
        self.embedding_dir = Embedding(3, 4)
        if self.encoding_cfg['density']:
            self.embedding_density = Embedding(1, 4)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    # torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.normal_(lin.weight, mean=0.0, std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires_view > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            
            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, points, normals, view_dirs, feature_vectors, indices, physical_particles, rays = [], ro = []):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)
        # add position embdding here
        # search
        dists_0, indices_0, neighbors_0, radius_0 = self.search(points, physical_particles, self.fix_radius)
        # embedding attributes
        pos_like_feats_0, dirs_like_feats_0, num_nn_0 = self.embedding_local_geometry(dists_0, indices_0, neighbors_0, radius_0, points, rays, ro)
        pos_input_feats_0 = torch.cat(pos_like_feats_0, dim=1) # 100352, 198
        dir_input_feats_0 = torch.cat(dirs_like_feats_0, dim=1) # 100352, 54
        # pos_input_feats_0 = pos_input_feats_0.reshape(feature_vectors.shape[0], feature_vectors.shape[1], -1)
        # feature_vectors [100352, 256]
        # normals [1024, 98, 3] -> [100352, 3]

        if self.mode == 'idr':
            rendering_input = torch.cat([pos_input_feats_0, dir_input_feats_0, normals, feature_vectors], dim=-1)
        elif self.mode == 'nerf':
            rendering_input = torch.cat([dir_input_feats_0, feature_vectors], dim=-1)
        else:
            raise NotImplementedError

        if self.per_image_code:
            image_code = self.embeddings[indices].expand(rendering_input.shape[0], -1)
            rendering_input = torch.cat([rendering_input, image_code], dim=-1)
            
        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x) # (100352, 543)

            if l < self.num_layers - 2:
                x = self.relu(x)
        
        x = self.sigmoid(x)
        return x

    def search(self, ray_particles, particles, fix_radius):
        raw_data = particles.unsqueeze(0).repeat(ray_particles.shape[0], 1, 1)
        if fix_radius:
            radius = self.radius
            dists, indices, neighbors = ball_query(p1=ray_particles, 
                                                   p2=raw_data, 
                                                   radius=radius, K=self.num_neighbor)
        # else:
        #     radius = self.get_search_radius(self.radius, ray_particles[:,:,-1] - ro[-1], focal)
        #     dists, indices, neighbors = self._ball_query(ray_particles, raw_data, radius, self.num_neighbor)
        return dists, indices, neighbors, radius
    
    def embedding_local_geometry(self, dists, indices, neighbors, radius, ray_particles, rays = [], ro = [], sigma_only=False):
        """
        pos like feats
            1. smoothed positions
            2. ref hit pos, i.e., ray position
            3. density
            3. variance
        dir like feats
            1. hit direction, i.e., ray direction
            2. main direction after PCA
        """
        # calculate mask
        nn_mask = dists.ne(0)
        num_nn = nn_mask.sum(-1, keepdim=True)

        # hit pos and hit direction (basic in NeRF formulation)
        pos_like_feats = []
        hit_pos = ray_particles.reshape(-1,3)
        hit_pos_embedded = self.embedding_xyz(hit_pos)
        pos_like_feats.append(hit_pos_embedded)
        if not sigma_only:
            hit_dir = rays[:,3:]
            hit_dir_embedded = self.embedding_dir(hit_dir)
            hit_dir_embedded = torch.repeat_interleave(hit_dir_embedded, repeats=ray_particles.shape[1], dim=0)
            dir_like_feats = []
            dir_like_feats.append(hit_dir_embedded)
        # smoothing 
        smoothed_pos, density = self.smoothing_position(ray_particles, neighbors, radius, num_nn, exclude_ray=self.encoding_cfg['exclude_ray'])
        if not sigma_only:
            smoothed_dir = self.get_particles_direction(smoothed_pos.reshape(-1, 3), ro)
        # density
        if self.encoding_cfg['density']:
            density_embedded = self.embedding_density(density.reshape(-1, 1))
            pos_like_feats.append(density_embedded)
        # smoothed pos
        if self.encoding_cfg['smoothed_pos']:
            smoothed_pos_embedded = self.embedding_xyz(smoothed_pos.reshape(-1, 3))
            pos_like_feats.append(smoothed_pos_embedded)
        # variance
        if self.encoding_cfg['var']:
            vec_pp2rp = torch.zeros(ray_particles.shape[0], ray_particles.shape[1], self.num_neighbor, 3).to(neighbors.device)
            vec_pp2rp[nn_mask] = (neighbors - ray_particles.unsqueeze(-2))[nn_mask]
            vec_pp2rp_mean = vec_pp2rp.sum(-2) / (num_nn+1e-12)
            variance = torch.zeros(ray_particles.shape[0], ray_particles.shape[1], self.num_neighbor, 3).to(neighbors.device)
            variance[nn_mask] = ((vec_pp2rp - vec_pp2rp_mean.unsqueeze(-2))**2)[nn_mask]
            variance = variance.sum(-2) / (num_nn+1e-12)
            variance_embedded = self.embedding_xyz(variance.reshape(-1,3))
            pos_like_feats.append(variance_embedded)
        # smoothed dir
        if not sigma_only:
            if self.encoding_cfg['smoothed_dir']:
                smoothed_dir_embedded = self.embedding_dir(smoothed_dir)
                dir_like_feats.append(smoothed_dir_embedded)
        if not sigma_only:
            return pos_like_feats, dir_like_feats, num_nn
        else:
            return pos_like_feats
    
    def smoothing_position(self, ray_pos, nn_poses, raduis, num_nn, exclude_ray=True, larger_alpha=0.9, smaller_alpha=0.1):
        dists = torch.norm(nn_poses - ray_pos.unsqueeze(-2), dim=-1)
        weights = torch.clamp(1 - (dists / raduis) ** 3, min=0)
        weighted_nn = (weights.unsqueeze(-1) * nn_poses).sum(-2) / (weights.sum(-1, keepdim=True)+1e-12)
        if exclude_ray:
            pos = weighted_nn
        else:
            if self.encoding_cfg['same_smooth_factor']:
                alpha = torch.ones(ray_pos.shape[0], ray_pos.shape[1], 1) * larger_alpha
            else:
                alpha = torch.ones(ray_pos.shape[0], ray_pos.shape[1], 1) * larger_alpha
                alpha[num_nn.le(20)] = smaller_alpha
            pos = ray_pos * (1-alpha) + weighted_nn * alpha
        return pos, weights.sum(-1, keepdim=True)
    
    def get_particles_direction(self, particles, ro):
        ros = ro.expand(particles.shape[0], -1)
        dirs = particles - ros
        dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
        return dirs


class MonoSDFNetwork(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.feature_vector_size = conf.get_int('feature_vector_size')
        self.scene_bounding_sphere = conf.get_float('scene_bounding_sphere', default=1.0)
        self.white_bkgd = conf.get_bool('white_bkgd', default=False)
        self.bg_color = torch.tensor(conf.get_list("bg_color", default=[1.0, 1.0, 1.0])).float().cuda()

        Grid_MLP = conf.get_bool('Grid_MLP', default=False)
        self.Grid_MLP = Grid_MLP
        if Grid_MLP:
            self.implicit_network = ImplicitNetworkGrid(self.feature_vector_size, 0.0 if self.white_bkgd else self.scene_bounding_sphere, **conf.get_config('implicit_network'))    
        else:
            self.implicit_network = ImplicitNetwork(self.feature_vector_size, 0.0 if self.white_bkgd else self.scene_bounding_sphere, **conf.get_config('implicit_network'))
        
        self.rendering_network = RenderingNetwork(self.feature_vector_size, **conf.get_config('rendering_network'))
        
        self.density = LaplaceDensity(**conf.get_config('density'))
        sampling_method = conf.get_string('sampling_method', default="errorbounded")
        self.ray_sampler = ErrorBoundSampler(self.scene_bounding_sphere, **conf.get_config('ray_sampler'))
        # for name, parms in self.implicit_network.named_parameters():	
        #     print('-->name:', name)
        #     print('-->para:', parms)
        #     print('-->grad_requirs:',parms.requires_grad)
        #     print('-->grad_value:',parms.grad)
        #     print('-->is_leaf:',parms.is_leaf)
        #     print("===")
        

    def forward(self, input, indices):
        # Parse model input
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]
        physical_particles = input["particles_poss"]
        # rays = input["rays"]

        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)
        ro = cam_loc[0] # [9.6846, 0.0000, 4.6888]
        
        # we should use unnormalized ray direction for depth
        ray_dirs_tmp, _ = rend_util.get_camera_params(uv, torch.eye(4).to(pose.device)[None], intrinsics)
        depth_scale = ray_dirs_tmp[0, :, 2:]
        
        batch_size, num_pixels, _ = ray_dirs.shape

        cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)
        rays = torch.cat([cam_loc, ray_dirs], dim=-1)

        
        z_vals, z_samples_eik = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, self, physical_particles)
        N_samples = z_vals.shape[1]

        # ray_dirs_1 = torch.tensor([ [0, 0, 1] for _ in range (ray_dirs.shape[0]) ]).cuda()
        # ray_dirs_2 = torch.tensor([ [0, 1, 0] for _ in range (ray_dirs.shape[0]) ]).cuda()
        # ray_dirs_3 = torch.tensor([ [1, 0, 0] for _ in range (ray_dirs.shape[0]) ]).cuda()
        # points_1 = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs_1.unsqueeze(1)
        # points_2 = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs_2.unsqueeze(1)
        # points_3 = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs_3.unsqueeze(1)
        # print('points_1', round(points_1.min().item(),2),round(points_1.max().item(),2))
        # print('points_2', round(points_2.min().item(),2),round(points_2.max().item(),2))
        # print('points_3', round(points_3.min().item(),2),round(points_3.max().item(),2))
        # points.min(), points.max() = -13.0, 22.69
        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        points_flat = points.reshape(-1, 3)


        dirs = ray_dirs.unsqueeze(1).repeat(1,N_samples,1)
        dirs_flat = dirs.reshape(-1, 3)

        sdf, feature_vectors, gradients = self.implicit_network.get_outputs(points, physical_particles)
        print('sdf', round(sdf.min().item(),2),round(sdf.max().item(),2))
        gradients_flat = gradients.reshape(-1, 3)
        
        rgb_flat = self.rendering_network(points, gradients_flat, dirs, feature_vectors, indices, physical_particles, rays, ro)
        # rgb_flat = self.rendering_network(pos_input_feats_0, gradients, dir_input_feats_0, feature_vectors, indices)
        rgb = rgb_flat.reshape(-1, N_samples, 3)

        weights = self.volume_rendering(z_vals, sdf)

        rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, 1)
        
        depth_values = torch.sum(weights * z_vals, 1, keepdims=True) / (weights.sum(dim=1, keepdims=True) +1e-8)
        # we should scale rendered distance to depth along z direction
        depth_values = depth_scale * depth_values
        
        # white background assumption
        if self.white_bkgd:
            acc_map = torch.sum(weights, -1)
            rgb_values = rgb_values + (1. - acc_map[..., None]) * self.bg_color.unsqueeze(0)

        output = {
            'rgb':rgb,
            'rgb_values': rgb_values,
            'depth_values': depth_values,
            'z_vals': z_vals,
            'depth_vals': z_vals * depth_scale,
            'sdf': sdf.reshape(z_vals.shape),
            'weights': weights,
        }

        if self.training:
            # Sample points for the eikonal loss
            n_eik_points = batch_size * num_pixels
            
            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-self.scene_bounding_sphere, self.scene_bounding_sphere).cuda()

            # add some of the near surface points
            eik_near_points = (cam_loc.unsqueeze(1) + z_samples_eik.unsqueeze(2) * ray_dirs.unsqueeze(1)).reshape(-1, 3)
            eikonal_points = torch.cat([eikonal_points, eik_near_points], 0)
            # add some neighbour points as unisurf
            neighbour_points = eikonal_points + (torch.rand_like(eikonal_points) - 0.5) * 0.01   
            eikonal_points = torch.cat([eikonal_points, neighbour_points], 0)
            # eikonal_points = eikonal_points.reshape(eik_near_points.shape[0], eikonal_points.shape[0]//eik_near_points.shape[0], eikonal_points.shape[1])
            # dists, indices, neighbors, radius = self.search(eikonal_points, physical_particles, self.fix_radius)
            # eikonal_pos_like_feats, _, _ = self.embedding_local_geometry(dists, indices, neighbors, radius, eikonal_points, rays, ro)
            # eikonal_pos_like_feats = torch.cat(eikonal_pos_like_feats, dim=1)
            # grad_theta = self.implicit_network.gradient(eikonal_pos_like_feats)
                   
            # eikonal_points [4096, 3]
            eikonal_points = eikonal_points.unsqueeze(1)
            grad_theta = self.implicit_network.gradient(eikonal_points, physical_particles)
            
            # split gradient to eikonal points and heighbour ponits
            output['grad_theta'] = grad_theta[:grad_theta.shape[0]//2]
            output['grad_theta_nei'] = grad_theta[grad_theta.shape[0]//2:]
        
        # compute normal map
        normals = gradients / (gradients.norm(2, -1, keepdim=True) + 1e-6)
        normals = normals.reshape(-1, N_samples, 3)
        normal_map = torch.sum(weights.unsqueeze(-1) * normals, 1)
        
        # transform to local coordinate system
        rot = pose[0, :3, :3].permute(1, 0).contiguous()
        normal_map = rot @ normal_map.permute(1, 0)
        normal_map = normal_map.permute(1, 0).contiguous()
        
        output['normal_map'] = normal_map

        return output

    def volume_rendering(self, z_vals, sdf):
        density_flat = self.density(sdf)
        density = density_flat.reshape(-1, z_vals.shape[1])  # (batch_size * num_pixels) x N_samples

        dists = z_vals[:, 1:] - z_vals[:, :-1]
        dists = torch.cat([dists, torch.tensor([1e10]).cuda().unsqueeze(0).repeat(dists.shape[0], 1)], -1)

        # LOG SPACE
        free_energy = dists * density
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), free_energy[:, :-1]], dim=-1)  # shift one step
        alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))  # probability of everything is empty up to now
        weights = alpha * transmittance # probability of the ray hits something here

        return weights
