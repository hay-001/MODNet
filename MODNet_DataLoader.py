from __future__ import print_function

import torch
import torch.utils.data as data
from torch.utils.data.dataloader import default_collate

import os
import numpy as np
import scipy.spatial as sp

from MODNet_Utils import pca_alignment
from MODNet_Utils  import parse_arguments
from plyfile import PlyData, PlyElement


##################################New Dataloader Class###########################

def my_collate(batch):

    batch = list(filter(lambda x : x is not None, batch))

    return default_collate(batch)

class RandomPointcloudPatchSampler(data.sampler.Sampler):

    def __init__(self, data_source, patches_per_shape, seed=None, identical_epochs=False):
        self.data_source = data_source
        self.patches_per_shape = patches_per_shape
        self.seed = seed
        self.identical_epochs = identical_epochs
        self.total_patch_count = None

        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**32-1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        self.total_patch_count = 0
        for shape_ind, _ in enumerate(self.data_source.shape_names):
            self.total_patch_count = self.total_patch_count + min(self.patches_per_shape, self.data_source.shape_patch_count[shape_ind])

    def __iter__(self):

        if self.identical_epochs:
            self.rng.seed(self.seed)

        return iter(self.rng.choice(sum(self.data_source.shape_patch_count), size=self.total_patch_count, replace=False))

    def __len__(self):
        return self.total_patch_count


class PointcloudPatchDataset(data.Dataset):

    def __init__(self, root=None, shapes_list_file=None, patch_radius=0.03, points_per_patch=400,patch_num=3,
                 seed=None, train_state='train', shape_name=None):

        self.root = root
        self.shapes_list_file = shapes_list_file

        self.patch_radius = patch_radius
        self.points_per_patch = points_per_patch
        self.patch_num = patch_num
        self.seed = seed
        self.train_state = train_state

        # initialize rng for picking points in a patch
        if self.seed is None:
            self.seed = np.random.random_integers(0, 2 ** 10 - 1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        self.shape_patch_count = []
        self.patch_radius_absolute = []
        self.gt_shapes = []
        self.noise_shapes = []

        self.shape_names = []
        if self.train_state == 'evaluation' and shape_name is not None:
            noise_pts = np.load(os.path.join(self.root, shape_name + '.npy'))
            noise_kdtree = sp.cKDTree(noise_pts)
            self.noise_shapes.append({'noise_pts': noise_pts, 'noise_kdtree': noise_kdtree})
            self.shape_patch_count.append(noise_pts.shape[0])
            bbdiag = float(np.linalg.norm(noise_pts.max(0) - noise_pts.min(0), 2))
            self.patch_radius_absolute.append(bbdiag * self.patch_radius)






        elif self.train_state == 'train':
            with open(os.path.join(self.root, self.shapes_list_file)) as f:
                self.shape_names = f.readlines()
            self.shape_names = [x.strip() for x in self.shape_names]
            self.shape_names = list(filter(None, self.shape_names))

            for shape_ind, shape_name in enumerate(self.shape_names):
                print('getting information for shape %s' % shape_name)
                if shape_ind % 5 == 0:
                    gt_pts = np.load(os.path.join(self.root, shape_name + '.npy'))
                    gt_normal = np.load(os.path.join(self.root, shape_name + '_normal.npy'))
                    gt_kdtree = sp.cKDTree(gt_pts)
                    self.gt_shapes.append({'gt_pts': gt_pts, 'gt_normal': gt_normal, 'gt_kdtree': gt_kdtree})
                    self.noise_shapes.append({'noise_pts': gt_pts, 'noise_kdtree': gt_kdtree})
                    noise_pts = gt_pts
                else:
                    noise_pts = np.load(os.path.join(self.root, shape_name + '.npy'))
                    noise_kdtree = sp.cKDTree(noise_pts)
                    self.noise_shapes.append({'noise_pts': noise_pts, 'noise_kdtree': noise_kdtree})

                self.shape_patch_count.append(noise_pts.shape[0])
                bbdiag = float(np.linalg.norm(noise_pts.max(0) - noise_pts.min(0), 2))
                self.patch_radius_absolute.append(bbdiag * self.patch_radius)




    def patch_sampling(self, patch_pts):

        if patch_pts.shape[0] > self.points_per_patch:

            sample_index = np.random.choice(range(patch_pts.shape[0]), self.points_per_patch, replace=False)

        else:

            sample_index = np.random.choice(range(patch_pts.shape[0]), self.points_per_patch)

        return sample_index


    def patch_sampling_total( self, patch_pts,gt_points_per_patch):

        if patch_pts.shape[0] > gt_points_per_patch:

            sample_index = np.random.choice(range(patch_pts.shape[0]), gt_points_per_patch, replace=False)

        else:

            sample_index = np.random.choice(range(patch_pts.shape[0]), gt_points_per_patch)

        return sample_index


    def __getitem__(self, index):

        # find shape that contains the point with given global index

        shape_ind, patch_ind = self.shape_index(index)
        noise_shape = self.noise_shapes[shape_ind]
        i=0
        multi_patchs=0
        gt_multi_patchs = 0
        gt_multi_normals = 0
        multi_support_radius=0
        for i in range(self.patch_num):
            patch_radius = self.patch_radius_absolute[shape_ind]/int(self.patch_radius/0.01)*(i+int(self.patch_radius/0.01))
            # For noise_patch

            noise_patch_idx = noise_shape['noise_kdtree'].query_ball_point(noise_shape['noise_pts'][patch_ind], patch_radius)
            if self.train_state == 'train':
                if len(noise_patch_idx) < 3:
                    return None
                noise_patch_pts = noise_shape['noise_pts'][noise_patch_idx] - noise_shape['noise_pts'][patch_ind]
            else:
                if len(noise_patch_idx) < 3:
                    print('expand radius')
                    noise_patch_idx = noise_shape['noise_kdtree'].query_ball_point(noise_shape['noise_pts'][patch_ind],
                                                                                   patch_radius * 1.5)
                    noise_patch_pts = noise_shape['noise_pts'][noise_patch_idx] - noise_shape['noise_pts'][patch_ind]
                else:
                    noise_patch_pts = noise_shape['noise_pts'][noise_patch_idx] - noise_shape['noise_pts'][patch_ind]



            if i==0:
                noise_patch_pts, noise_patch_inv = pca_alignment(noise_patch_pts)
            else:
                noise_patch_pts = np.array(np.linalg.inv(noise_patch_inv) * np.matrix(noise_patch_pts.T)).T

            noise_patch_pts /= ( self.patch_radius_absolute[shape_ind]/int(self.patch_radius/0.01)*5)

            noise_sample_idx = self.patch_sampling(noise_patch_pts)
            noise_patch_pts  = noise_patch_pts[noise_sample_idx]

            support_radius = np.linalg.norm(noise_patch_pts.max(0) - noise_patch_pts.min(0), 2) / noise_patch_pts.shape[0]
            support_radius = np.expand_dims(support_radius, axis=0)

            noise_patch_pts = np.expand_dims(noise_patch_pts, axis=0)
            if i==0:
                multi_patchs =  noise_patch_pts
                multi_support_radius =  support_radius
            else:
                multi_patchs = np.concatenate((multi_patchs, noise_patch_pts), axis=0)
                multi_support_radius = np.concatenate((multi_support_radius, support_radius), axis=0)

            # For multi_patch
            if self.train_state == 'train':
                # For gt_patch
                gt_shape = self.gt_shapes[shape_ind // 5]

                gt_patch_idx = gt_shape['gt_kdtree'].query_ball_point(noise_shape['noise_pts'][patch_ind], patch_radius)

                if len(gt_patch_idx) < 3:
                    return None
                gt_patch_pts = gt_shape['gt_pts'][gt_patch_idx]
                gt_patch_pts -= noise_shape['noise_pts'][patch_ind]
                gt_patch_pts /= ( self.patch_radius_absolute[shape_ind]/int(self.patch_radius/0.01)*5)
                gt_patch_pts = np.array(np.linalg.inv(noise_patch_inv) * np.matrix(gt_patch_pts.T)).T

                gt_patch_normal = gt_shape['gt_normal'][gt_patch_idx]
                gt_patch_normal = np.array(np.linalg.inv(noise_patch_inv) * np.matrix(gt_patch_normal.T)).T
                gt_sample_idx = self.patch_sampling(gt_patch_pts)


                #total
                if i==self.patch_num-1:
                    gt_total_sample_idx = self.patch_sampling_total(gt_patch_pts,500)
                    gt_total_patch_pts = gt_patch_pts[gt_total_sample_idx]
                    gt_total_patch_normal = gt_patch_normal[gt_total_sample_idx]


                gt_patch_pts = gt_patch_pts[gt_sample_idx]
                gt_patch_normal = gt_patch_normal[gt_sample_idx]

                gt_patch_pts = np.expand_dims(gt_patch_pts, axis=0)
                gt_patch_normal = np.expand_dims(gt_patch_normal, axis=0)

                if i==0:
                    gt_multi_patchs =  gt_patch_pts
                    gt_multi_normals =  gt_patch_normal
                else:
                    gt_multi_patchs = np.concatenate((gt_multi_patchs, gt_patch_pts), axis=0)
                    gt_multi_normals = np.concatenate((gt_multi_normals, gt_patch_normal), axis=0)



        if self.train_state == 'evaluation':
            return torch.from_numpy(multi_patchs), torch.from_numpy(noise_patch_inv), \
                   noise_shape['noise_pts'][patch_ind]
        else:


            return torch.from_numpy(multi_patchs),\
                   torch.from_numpy(gt_multi_patchs),torch.from_numpy(gt_multi_normals),\
                   torch.from_numpy(gt_total_patch_pts),torch.from_numpy(gt_total_patch_normal), \
                   torch.from_numpy(multi_support_radius)

    def __len__(self):
        return sum(self.shape_patch_count)

    def shape_index(self, index):
        shape_patch_offset = 0
        shape_ind = None
        for shape_ind, shape_patch_count in enumerate(self.shape_patch_count):
            if (index >= shape_patch_offset) and (index < shape_patch_offset + shape_patch_count):
                shape_patch_ind = index - shape_patch_offset
                break
            shape_patch_offset = shape_patch_offset + shape_patch_count

        return shape_ind, shape_patch_ind


if __name__ == '__main__':
    parameters = parse_arguments()
    parameters.trainset = './Dataset/Train'
    parameters.summary_dir = './Summary/Train/logs'
    parameters.network_model_dir = './Summary/Train'
    parameters.batchSize = 64
    parameters.lr = 1e-4
    parameters.workers = 4
    parameters.nepoch = 50
    print(parameters)

    if not os.path.exists(parameters.summary_dir):
        os.makedirs(parameters.summary_dir)
    if not os.path.exists(parameters.network_model_dir):
        os.makedirs(parameters.network_model_dir)
    print("Random Seed: ", parameters.manualSeed)
    np.random.seed(parameters.manualSeed)
    torch.manual_seed(parameters.manualSeed)

    # optionally resume from a checkpoint
    train_dataset = PointcloudPatchDataset(
        root=parameters.trainset,
        shapes_list_file='train.txt',
        patch_radius=0.05,
        seed=parameters.manualSeed,
        train_state='train',
        patch_num=3,
        points_per_patch = 200
    )
    train_datasampler = RandomPointcloudPatchSampler(
        train_dataset,
        patches_per_shape=8000,
        seed=parameters.manualSeed,
        identical_epochs=False)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_datasampler,
        shuffle=(train_datasampler is None),
        collate_fn=my_collate,
        batch_size=parameters.batchSize,
        num_workers=int(parameters.workers),
        pin_memory=True)
    num_batch = len(train_dataloader)
    for epoch in range(parameters.start_epoch, parameters.nepoch):
        for batch_ind, data_tuple in enumerate(train_dataloader):
            multi_patches, gt_patch, gt_normal, gt_patch_total, gt_normal_total, support_radius = data_tuple

            multi_patches = multi_patches.float().cuda(non_blocking=True)

            # vertex = [tuple(item) for item in multi_patches[0,0,:,:].cpu().numpy()]
            # vertex = np.array(vertex, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
            # PlyData([PlyElement.describe(vertex, 'vertex')], text=True).write("/home/hay/LAB/Pointfilter1.0/Dataset/"+str(batch_ind)+'_1.ply')
            # #
            # vertex = [tuple(item) for item in multi_patches[0,1,:,:].cpu().numpy()]
            # vertex = np.array(vertex, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
            # PlyData([PlyElement.describe(vertex, 'vertex')], text=True).write("/home/hay/LAB/Pointfilter1.0/Dataset/"+str(batch_ind)+'_2.ply')
            # #
            # vertex = [tuple(item) for item in multi_patches[0,2,:,:].cpu().numpy()]
            # vertex = np.array(vertex, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
            # PlyData([PlyElement.describe(vertex, 'vertex')], text=True).write("/home/hay/LAB/Pointfilter1.0/Dataset/"+str(batch_ind)+'_3.ply')



            # vertex = [tuple(item) for item in gt_patch[0,0,:,:].cpu().numpy()]
            # vertex = np.array(vertex, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
            # PlyData([PlyElement.describe(vertex, 'vertex')], text=True).write("/home/hay/LAB/Pointfilter1.0/Dataset/gt"+str(batch_ind)+'_1.ply')
            # #
            # vertex = [tuple(item) for item in gt_patch[0,1,:,:].cpu().numpy()]
            # vertex = np.array(vertex, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
            # PlyData([PlyElement.describe(vertex, 'vertex')], text=True).write("/home/hay/LAB/Pointfilter1.0/Dataset/gt"+str(batch_ind)+'_2.ply')
            # #
            # vertex = [tuple(item) for item in gt_patch[0,2,:,:].cpu().numpy()]
            # vertex = np.array(vertex, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
            # PlyData([PlyElement.describe(vertex, 'vertex')], text=True).write("/home/hay/LAB/Pointfilter1.0/Dataset/gt"+str(batch_ind)+'_3.ply')



            # gt_patch = gt_patch.float().cuda(non_blocking=True)
            # gt_normal = gt_normal.float().cuda(non_blocking=True)
            # support_radius = parameters.support_multiple * support_radius
            # support_radius = support_radius.float().cuda(non_blocking=True)
            # support_angle = (parameters.support_angle / 360) * 2 * np.pi

            #
            # print(gt_patch.shape)
            # print(gt_patch_total.shape)

            # print(gt_patch.shape)
            # print(gt_normal.shape)
            # print(support_radius.shape)

            print('[%d: %d/%d ] \n' % (epoch, batch_ind, num_batch))

