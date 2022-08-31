
# coding=utf-8

from __future__ import print_function
from tensorboardX import SummaryWriter
from MODNet_Network_Architecture import modnet
from MODNet_DataLoader import PointcloudPatchDataset, RandomPointcloudPatchSampler, my_collate
from MODNet_Utils import parse_arguments, adjust_learning_rate, compute_bilateral_loss_with_repulsion

import os
import numpy as np
import torch.utils.data
import torch.optim as optim
import torch.backends.cudnn as cudnn

torch.backends.cudnn.benchmark = True

def train(opt):
    if not os.path.exists(opt.summary_dir):
        os.makedirs(opt.summary_dir)
    if not os.path.exists(opt.network_model_dir):
        os.makedirs(opt.network_model_dir)
    print("Random Seed: ", opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    denoisenet = modnet(patch_point_nums=400).cuda()
    optimizer = optim.SGD(
        denoisenet.parameters(),
        lr=opt.lr,
        momentum=opt.momentum)
    train_writer = SummaryWriter(opt.summary_dir)


    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format("opt.resume"))
            checkpoint = torch.load('./Summary/Train/model_base.pth')
            opt.start_epoch = checkpoint['epoch']
            denoisenet.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(opt.resume, checkpoint['epoch']))

        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))
    train_dataset = PointcloudPatchDataset(
        root=opt.trainset,
        shapes_list_file='train.txt',
        patch_radius=0.03,
        seed=opt.manualSeed,
        train_state='train'
        ,points_per_patch=400)
    train_datasampler = RandomPointcloudPatchSampler(
        train_dataset,
        patches_per_shape=2000,
        seed=opt.manualSeed,
        identical_epochs=False)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_datasampler,
        shuffle=(train_datasampler is None),
        collate_fn=my_collate,
        batch_size=opt.batchSize,
        num_workers=int(opt.workers),
        pin_memory=True)
    num_batch = len(train_dataloader)
    for epoch in range(opt.start_epoch, opt.nepoch):
        adjust_learning_rate(optimizer, epoch, opt)
        print('lr is %.10f' % (optimizer.param_groups[0]['lr']))
        for batch_ind, data_tuple in enumerate(train_dataloader):
            denoisenet.train()
            optimizer.zero_grad()

            multi_patches,gt_patch, gt_normal, gt_patch_total, gt_normal_total,support_radius = data_tuple

            multi_patches = multi_patches.float().cuda(non_blocking=True)


            gt_patch = gt_patch.float().cuda(non_blocking=True)
            gt_normal = gt_normal.float().cuda(non_blocking=True)
            gt_patch_total = gt_patch_total.float().cuda(non_blocking=True)
            gt_normal_total = gt_normal_total.float().cuda(non_blocking=True)
            support_radius = opt.support_multiple * support_radius
            support_radius = support_radius.float().cuda(non_blocking=True)
            support_angle =  (opt.support_angle / 360) * 2 * np.pi
            multi_patches = multi_patches.transpose(3, 2).contiguous()




            pred_pts_min,pred_pts_mid,pred_pts_max,pred_pts_total,loss_weight = denoisenet(multi_patches)


            loss_min = 100 * compute_bilateral_loss_with_repulsion(pred_pts_min, gt_patch[:,0,:,:], gt_normal[:,0,:,:],
                                                               support_radius[:,0].unsqueeze(1), support_angle, opt.repulsion_alpha)

            loss_mid = 100 * compute_bilateral_loss_with_repulsion(pred_pts_mid, gt_patch[:,1,:,:], gt_normal[:,1,:,:],
                                                               support_radius[:,1].unsqueeze(1), support_angle, opt.repulsion_alpha)

            loss_max = 100 * compute_bilateral_loss_with_repulsion(pred_pts_max, gt_patch[:,2,:,:], gt_normal[:,2,:,:],
                                                              support_radius[:,2].unsqueeze(1), support_angle, opt.repulsion_alpha)

            loss_total = 100 * compute_bilateral_loss_with_repulsion(pred_pts_total, gt_patch_total, gt_normal_total,
                                                               support_radius[:,2].unsqueeze(1), support_angle, opt.repulsion_alpha)


            loss=0.2*(loss_min+loss_mid+loss_max)+loss_total


            loss.backward()
            optimizer.step()

            print(
                '[%d: %d/%d] train loss: %f '
                % (epoch, batch_ind, num_batch, loss.item()))

            train_writer.add_scalar('loss', loss.data.item(), epoch * num_batch + batch_ind)
            train_writer.add_scalar('loss_min', loss_min.data.item(), epoch * num_batch + batch_ind)
            train_writer.add_scalar('loss_mid', loss_mid.data.item(), epoch * num_batch + batch_ind)
            train_writer.add_scalar('loss_max', loss_max.data.item(), epoch * num_batch + batch_ind)
            train_writer.add_scalar('loss_total', loss_total.data.item(), epoch * num_batch + batch_ind)


        checpoint_state = {
            'epoch': epoch + 1,
            'state_dict': denoisenet.state_dict(),
            'optimizer': optimizer.state_dict()}

        if epoch == (opt.nepoch - 1):

            torch.save(checpoint_state, '%s/model_full_ae.pth' % opt.network_model_dir)

        torch.save(checpoint_state, '%s/model_full_ae_%d.pth' % (opt.network_model_dir, epoch))

if __name__ == '__main__':
    parameters = parse_arguments()
    parameters.trainset = './Dataset/Train'
    parameters.summary_dir = './Summary/Train/logs'
    parameters.network_model_dir = './Summary/Train'
    parameters.batchSize = 200
    parameters.lr = 1e-4
    parameters.workers = 16
    parameters.nepoch = 25
    print(parameters)
    train(parameters)