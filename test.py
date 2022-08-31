import os
import torch
import numpy as np
from MODNet_Network_Architecture import modnet
from MODNet_DataLoader import PointcloudPatchDataset
from MODNet_Utils import parse_arguments
from plyfile import PlyData, PlyElement
import copy
from time import *

def eval(opt,txt):

    with open(os.path.join(opt.testset, txt), 'r') as f:
        shape_names = f.readlines()
    shape_names = [x.strip() for x in shape_names]
    shape_names = list(filter(None, shape_names))

    if not os.path.exists(parameters.save_dir):
        os.makedirs(parameters.save_dir)


    for shape_id, shape_name in enumerate(shape_names):


        xyz=np.loadtxt(os.path.join(opt.testset, shape_name + '.xyz'))
        np.save(os.path.join(opt.save_dir, shape_name + '.npy'),xyz.astype('float32'))



        for eval_index in range(opt.eval_iter_nums):
            test_dataset = PointcloudPatchDataset(
                root=opt.save_dir,
                shape_name=shape_name,
                patch_radius=0.03,
                train_state='evaluation',
                points_per_patch=400)
            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=opt.batchSize,
                num_workers=int(opt.workers))



            modnet_eval = modnet().cuda()
            model_filename = os.path.join(parameters.eval_dir, 'model_full_ae.pth')
            checkpoint = torch.load(model_filename)
            modnet_eval.load_state_dict(checkpoint['state_dict'])
            modnet_eval.cuda()
            modnet_eval.eval()


            radius_min = copy.deepcopy(test_dataset.patch_radius_absolute)
            radius_max = np.multiply(0.333*5, radius_min)

            pts_min = np.empty((0, 3), dtype='float32')
            pts_mid = np.empty((0, 3), dtype='float32')
            pts_max = np.empty((0, 3), dtype='float32')
            pts_total = np.empty((0, 3), dtype='float32')

            begin_time = time()
            for batch_ind, data_tuple in enumerate(test_dataloader):

                multi_patchs, noise_inv, noise_disp = data_tuple
                multi_patchs = multi_patchs.float().cuda()
                noise_inv = noise_inv.float().cuda()
                multi_patchs = multi_patchs.transpose(3, 2).contiguous()





                pred_pts_min,pred_pts_mid,pred_pts_max,pred_pts_total,loss_weight = modnet_eval(multi_patchs)

                pred_pts_min = pred_pts_min.unsqueeze(2)
                pred_pts_min = torch.bmm(noise_inv, pred_pts_min)
                pts_min = np.append(pts_min,
                                    np.squeeze(pred_pts_min.data.cpu().numpy()) * radius_max + noise_disp.numpy(),
                                    axis=0)

                pred_pts_mid = pred_pts_mid.unsqueeze(2)
                pred_pts_mid = torch.bmm(noise_inv, pred_pts_mid)
                pts_mid = np.append(pts_mid,
                                    np.squeeze(pred_pts_mid.data.cpu().numpy()) * radius_max + noise_disp.numpy(),
                                    axis=0)

                pred_pts_max = pred_pts_max.unsqueeze(2)
                pred_pts_max = torch.bmm(noise_inv, pred_pts_max)
                pts_max = np.append(pts_max,
                                    np.squeeze(pred_pts_max.data.cpu().numpy()) * radius_max + noise_disp.numpy(),
                                    axis=0)


                pred_pts_total = pred_pts_total.unsqueeze(2)
                pred_pts_total = torch.bmm(noise_inv, pred_pts_total)
                pts_total = np.append(pts_total,
                                     np.squeeze(pred_pts_total.data.cpu().numpy()) * radius_max + noise_disp.numpy(),
                                     axis=0)
            end_time=time()
            run_time=end_time-begin_time
            print('%s: time: %f'%(shape_name,run_time))
            np.savetxt(os.path.join(opt.save_dir, shape_name+ '.xyz')
                                    , pts_total, fmt='%.8f')


            np.save(os.path.join(opt.save_dir, shape_name + '_pred_iter_' + str(eval_index + 1)+ '.npy'),
                    pts_total.astype('float32'))

            np.savetxt(os.path.join(opt.save_dir, shape_name + '_pred_iter_' + str(eval_index + 1) +"_total"+ '.xyz')
                                    , pts_total, fmt='%.8f')






if __name__ == '__main__':

    parameters = parse_arguments()
    file=['0.005,10000','0.005,20000','0.005,50000',
          '0.01,10000','0.01,20000','0.01,50000',
          '0.015,10000','0.015,20000','0.015,50000',]

    for i in range(len(file)):
        print('\n'+file[i]+'\n')
        parameters.testset = './Dataset/Test/'+file[i]
        parameters.eval_dir = './Summary/pre_train_model/'
        parameters.batchSize = 60
        parameters.workers = 16
        parameters.save_dir = './Summary/Test/'+file[i]
        parameters.eval_iter_nums = 1
        parameters.patch_radius = 0.03
        eval(parameters,'test_'+file[i]+'.txt')




