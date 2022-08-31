import scipy.spatial as sp
import numpy as np
import torch
import os
from Customer_Module.chamfer_distance.dist_chamfer import chamferDist
from plyfile import PlyData, PlyElement

nnd = chamferDist()

def npy2ply(filename, save_filename):
    pts = np.load(filename)
    vertex = [tuple(item) for item in pts]
    vertex = np.array(vertex, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    PlyData([PlyElement.describe(vertex, 'vertex')], text=True).write(save_filename)



def Eval_With_Charmfer_Distance(file_ads,a,type=None,eval_iter_nums=1):
    cd_list = []
    print('Errors under Chamfer Distance')
    for shape_id, shape_name in enumerate(shape_names):
        xyz=np.loadtxt(os.path.join('./Dataset/Test/gt', shape_name[:-a] + '.xyz'))
        np.save(os.path.join('./Dataset/Test/gt', shape_name[:-a] + '.npy'),xyz.astype('float32'))

        gt_pts = np.load(os.path.join('./Dataset/Test/gt', shape_name[:-a] + '.npy'))
        pred_pts = np.load(os.path.join(file_ads+file[i], shape_name + '_pred_iter_'+str(eval_iter_nums)+type+'.npy'))
        with torch.no_grad():
            gt_pts_cuda = torch.from_numpy(np.expand_dims(gt_pts, axis=0)).cuda().float()
            pred_pts_cuda = torch.from_numpy(np.expand_dims(pred_pts, axis=0)).cuda().float()
            dist1, dist2 = nnd(pred_pts_cuda, gt_pts_cuda)
            chamfer_errors = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)
            print('%12s：  %.3f' % (shape_names[shape_id], round(chamfer_errors.item() * 100000, 3)))
            cd_list.append(round(chamfer_errors.item() * 10000, 3))
    cd_arrey = np.array(cd_list)
    print('%s:  %.3f' % ('Mean CD', cd_arrey.mean()))
    return cd_arrey.mean()


def Eval_With_Mean_Square_Error(file_ads,a,type=None,eval_iter_nums=1):
    mse_list = []
    print('\nErrors under Mean Square Error')
    for shape_id, shape_name in enumerate(shape_names):
        gt_pts = np.load(os.path.join('./Dataset/Test/gt', shape_name[:-a] + '.npy'))
        gt_pts_tree = sp.cKDTree(gt_pts)
        pred_pts = np.load(os.path.join(file_ads+file[i], shape_name + '_pred_iter_'+str(eval_iter_nums)+type+'.npy'))
        pred_dist, _ = gt_pts_tree.query(pred_pts, 10)
        print('%12s：  %.3f' % (shape_names[shape_id], round(pred_dist.mean() * 1000, 3)))
        mse_list.append(round(pred_dist.mean() * 1000, 3))
    mse_arrey = np.array(mse_list)
    print('%s:  %.3f' % ('Mean MSE', mse_arrey.mean()))
    return mse_arrey.mean()


if __name__ == '__main__':
    file=['0.005,10000','0.005,20000','0.005,50000',
          '0.01,10000','0.01,20000','0.01,50000',
          '0.015,10000','0.015,20000','0.015,50000',]
    file_ads = './Summary/Test/'

    cd_mean=[]
    mse_mean=[]

    for i in range(len(file)):


        with open(os.path.join('./Dataset/Test/'+file[i], 'test_'+file[i]+'.txt'), 'r') as f:
            shape_names = f.readlines()
        shape_names = [x.strip() for x in shape_names]
        shape_names = list(filter(None, shape_names))

        print('\n'+file[i]+'\n')
        if i>2 and i<6:
            nu=5
        else:
            nu=6

        cd=Eval_With_Charmfer_Distance(file_ads, nu, '_total')
        mse=Eval_With_Mean_Square_Error(file_ads, nu, '_total')

        cd_mean.append(cd)
        mse_mean.append(mse)


    print("mean cd: %f ,mean mse: %f "
          %(np.array(cd_mean).mean(),np.array(mse_mean).mean()))



