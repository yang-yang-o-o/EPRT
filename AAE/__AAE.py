from math import sqrt
import torch
import cv2
import numpy as np
import os
import sys
import configparser

from .dataset import AAE_dataset
from . import model
from .weight_load import load

class AAE():
        def __init__(self,obj_name,train=True,use_cuda=False) -> None:

                weight_path = {'obj_01':'AAE/weight/T_Less_obj_01.npz',
                                'gear':'AAE/weight/weight.npz',
                                'mesh_01':'AAE/weight/mesh_01.npz',
                                'mesh_02':'AAE/weight/mesh_02.npz',
                                'mesh_03':'AAE/weight/mesh_03.npz',
                                'mesh_04':'AAE/weight/mesh_04.npz',
                                'mesh_05':'AAE/weight/mesh_05.npz'}
                cfg_path    = {'obj_01':'AAE/cfg/T_Less_obj_01.cfg',
                                'gear':'AAE/cfg/gear.cfg',
                                'mesh_01':'AAE/cfg/mesh_01.cfg',
                                # 'mesh_02':'AAE/cfg/mesh_02.cfg',
                                'mesh_02':'AAE/cfg/mesh_02_1_m8_30.cfg',
                                'mesh_03':'AAE/cfg/mesh_03.cfg',
                                'mesh_04':'AAE/cfg/mesh_04.cfg',
                                'mesh_05':'AAE/cfg/mesh_05.cfg'}
                self.data_path   = {'obj_01':'AAE/data/T_Less_obj_01',
                                'gear':'AAE/data/gear',
                                'mesh_01':'AAE/data/mesh_01',
                                # 'mesh_02':'AAE/data/mesh_02',
                                'mesh_02':'AAE/data/mesh_02_1_m8_30',
                                'mesh_03':'AAE/data/mesh_03',
                                # 'mesh_04':'AAE/data/mesh_04',
                                'mesh_04':'AAE/data/mesh_04_4_3.0',
                                'mesh_05':'AAE/data/mesh_05'}

                self.train = train
                self.use_cuda = use_cuda
                self.Model = model.AAE(train=self.use_cuda)
                load(self.Model,weight_path[obj_name],self.use_cuda) 
                if self.use_cuda:
                        self.Model.cuda()
                self.Model.eval()

                args = configparser.ConfigParser()
                args.read(cfg_path[obj_name])
                self.dataset_args = { k:v for k,v in
                                        args.items('Dataset') +
                                        args.items('Paths') +
                                        args.items('Augmentation')+
                                        args.items('Queue') +
                                        args.items('Embedding')+
                                        args.items('Network')+
                                        args.items('Training')}

                self.Dataset = AAE_dataset(self.data_path[obj_name],train=self.train,**self.dataset_args)
                self.Rs = self.Dataset.viewsphere_for_embedding()

        def inference(self,ROI,k_test,predict_bbx):
                """
                input
                -----    
                        ROI : cv mat

                output
                ------     
                        T   : world to camera
                """
                im = cv2.resize(ROI,(128,128))
                # cv2.imshow("AAE_input",im)
                x = im/255.
                x = np.expand_dims(x, 0)
                x = x.transpose((0,3,1,2))
                x = torch.from_numpy(x).float()
                if self.use_cuda:
                        x = x.cuda()
                encode = self.Model.latent_code(x)
                idcs,rendered_bbx = self.Model.similarity(encode)
                R = self.Rs[idcs] # world to camera，将世界坐标系下的点转换到相机坐标系下

                # 
                Rs_est, ts_est = self.Dataset.refine_T(R,predict_bbx,rendered_bbx,k_test)
                T = np.concatenate((np.concatenate((Rs_est,ts_est.reshape((1,3)).T),axis=1),np.array([[0,0,0,1]])),axis=0) # world to camera
                return T

        def reconstruct(self,ROI):
                """
                input
                -----    
                        ROI : cv mat
                        
                output
                ------     
                        reconstruct_img   : cv mat
                """
                im = cv2.resize(ROI,(128,128))
                cv2.imshow('ROI',im)
                x = im/255.
                x = np.expand_dims(x, 0)
                x = x.transpose((0,3,1,2))
                x = torch.from_numpy(x).float()
                if self.train:
                        x = x.cuda()    
                reconstruction = self.Model(x)
                if self.train:
                        reconstruction = reconstruction.cpu()
                reconstruction = reconstruction.detach().permute((0,2,3,1)).numpy()
                cv2.imshow('reconstruct',cv2.resize(reconstruction[0],(128,128)))
                encode = self.Model.latent_code(x)
                idcs,rendered_bb = self.Model.similarity(encode)
                R = self.Dataset.viewsphere_for_embedding()[idcs]

                rendered_view = self.Dataset.render_rot( R,downSample = 1)
                cv2.imshow('rendered_view', cv2.resize(rendered_view,(128,128)))
                key = cv2.waitKey(0)
                if key == 27:
                        exit(-1)
        def get_AABB_bbx(self,pointcloud):
                '''
                input
                -----
                        pointcloud:     .ply file
                output
                ------
                        corners:        4x8
                        diameter:       object diameter
                '''
                import open3d
                mesh = open3d.io.read_point_cloud(pointcloud)
                vertices             = np.c_[np.array(mesh.points), np.ones((len(mesh.points), 1))].transpose() # 4xn
                
                min_x = np.min(vertices[0,:])
                max_x = np.max(vertices[0,:])
                min_y = np.min(vertices[1,:])
                max_y = np.max(vertices[1,:])
                min_z = np.min(vertices[2,:])
                max_z = np.max(vertices[2,:])
                corners = np.array([[min_x, min_y, min_z],
                                        [min_x, min_y, max_z],
                                        [min_x, max_y, min_z],
                                        [min_x, max_y, max_z],
                                        [max_x, min_y, min_z],
                                        [max_x, min_y, max_z],
                                        [max_x, max_y, min_z],
                                        [max_x, max_y, max_z]])

                corners = np.concatenate((np.transpose(corners), np.ones((1,8)) ), axis=0)
                diameter = sqrt(pow((max_x-min_x),2)+pow((max_y-min_y),2)+pow((max_z-min_z),2))
                return corners ,diameter,vertices
        def ProjectCorner3D(self,image,corner3D,T,mtx,show_point=False,show_edge=False,color=(0,255,0)):
                '''
                Input:
                        Point:  points expressed in the world frame , 4xn , mm
                        T    :  extrinsic_calibration_matrix        , 4x4
                        mtx  :  intrinsic_calibration_matrix        , 3x3
                Output:
                        dot:    Pixel coordinates on the image , 2xn , float
                '''
                dot_c = np.dot(T,corner3D)[:3,:]
                dot = np.dot(mtx,dot_c)
                dot[0,:] /=dot[2,:]
                dot[1,:] /=dot[2,:]
                if show_point:
                        for d in dot.T:
                                cv2.circle(image,(int(d[0]),int(d[1])),2,color)
                if show_edge:
                        edges_corners = np.array([[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]])
                        for edge in edges_corners:
                                cv2.line(image,(int(dot[0,edge[0]]),int(dot[1,edge[0]])), (int(dot[0,edge[1]]),int(dot[1,edge[1]])), color,1)
                return dot[:2,:]