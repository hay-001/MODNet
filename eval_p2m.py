import scipy.spatial as sp
import numpy as np
import torch
import pytorch3d.loss
import pytorch3d.structures

import os
from plyfile import PlyData, PlyElement


from pytorch3d.loss.point_mesh_distance import point_face_distance
import point_cloud_utils as pcu


def npy2ply(filename, save_filename):
    pts = np.load(filename)
    vertex = [tuple(item) for item in pts]
    vertex = np.array(vertex, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    PlyData([PlyElement.describe(vertex, 'vertex')], text=True).write(save_filename)


def load_off(off_dir):

    verts, faces = pcu.load_mesh_vf(off_dir)
    verts = torch.FloatTensor(verts)
    faces = torch.LongTensor(faces)
    all_meshes = {'verts': verts, 'faces': faces}
    return all_meshes



def normalize_sphere(pc, radius=1.0):
    """
    Args:
        pc: A batch of point clouds, (B, N, 3).
    """
    ## Center
    p_max = pc.max(dim=-2, keepdim=True)[0]
    p_min = pc.min(dim=-2, keepdim=True)[0]
    center = (p_max + p_min) / 2    # (B, 1, 3)
    pc = pc - center
    ## Scale
    scale = (pc ** 2).sum(dim=-1, keepdim=True).sqrt().max(dim=-2, keepdim=True)[0] / radius  # (B, N, 1)
    pc = pc / scale
    return pc, center, scale

def normalize_pcl(pc, center, scale):
    return (pc - center) / scale


def Eval_With_P2M(file_ads,a):


    p2m_list = []
    print('Errors under P2F')
    for shape_id, shape_name in enumerate(shape_names):
        xyz=np.loadtxt(os.path.join('./Dataset/Test/gt', shape_name[:-a] + '.xyz'))
        np.save(os.path.join('./Dataset/Test/gt', shape_name[:-a] + '.npy'),xyz.astype('float32'))

        meshes=load_off(os.path.join('./Dataset/Test/gt_mesh', shape_name[:-(a+6)] + '.off'))
        pred_pts = np.load(os.path.join(file_ads + file[i],
                                        shape_name + '_pred_iter_1' + '.npy'))
        pcl = pred_pts
        pcl=torch.from_numpy(pcl).to('cuda:0')
        verts = meshes['verts'].to('cuda:0')
        faces = meshes['faces'].to('cuda:0')
        assert verts.dim() == 2 and faces.dim() == 2, 'Batch is not supported.'

        # Normalize mesh
        verts, center, scale = normalize_sphere(verts.unsqueeze(0))
        verts = verts[0]
        # Normalize pcl
        pcl = normalize_pcl(pcl.unsqueeze(0), center=center, scale=scale)
        pcl = pcl[0]

        # print('%.6f %.6f' % (verts.abs().max().item(), pcl.abs().max().item()))

        # Convert them to pytorch3d structures
        pcls = pytorch3d.structures.Pointclouds([pcl])
        meshes = pytorch3d.structures.Meshes([verts], [faces])
        chamfer_errors=pytorch3d.loss.point_mesh_face_distance(meshes, pcls)
        p2m_list.append(round(chamfer_errors.item() * 10000 , 3))
        #print('%12s：  %.3f' % (shape_names[shape_id], round(chamfer_errors.item() * 10000 , 3)))

    p2m_arrey = np.array(p2m_list)
    print('%s:  %.3f' % ('Mean p2f', p2m_arrey.mean()))
    return p2m_arrey.mean()



if __name__ == '__main__':
    file=['0.005,10000','0.005,20000','0.005,50000',
          '0.01,10000','0.01,20000','0.01,50000',
          '0.015,10000','0.015,20000','0.015,50000',]

    p2m_mean=[]

    for i in range(len(file)):

        file_ads = './Summary/Test/'

        with open(os.path.join('./Dataset/Test/'+file[i], 'test_'+file[i]+'.txt'), 'r') as f:
            shape_names = f.readlines()
        shape_names = [x.strip() for x in shape_names]
        shape_names = list(filter(None, shape_names))

        print('\n'+file[i]+'\n')
        if i>2 and i<6:
            nu=5
        else:
            nu=6

        p2m=Eval_With_P2M(file_ads, nu)
        p2m_mean.append(p2m)

    print("mean p2f: %f"
          %(np.array(p2m_mean).mean()))



