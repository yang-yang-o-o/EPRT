import cv2
import numpy as np
import glob
from numpy import random
import configparser

import torch
from torch.utils.data.dataset import Dataset
import imgaug.augmenters as aug

from .pysixd_stuff import view_sampler

class AAE_dataset(Dataset):
    def __init__(self,data_path,train=True,**kw) -> None:
        super().__init__()
        self.train = train
        self.shape = (int(kw['h']), int(kw['w']), int(kw['c']))
        self._kw = kw
        self.dataset_path = data_path
        self.train_x = None
        self.train_y = None
        self.mask_x  = None
        self.bg_imgs = None
        self.train_img_num = None
        self.train_bg_num = None
        self.size = None
        # self.noof_training_imgs = int(kw['noof_training_imgs'])
        # self.dataset_path = data_path

        # self.bg_img_paths = glob.glob(kw['background_images_glob'])
        # self.noof_bg_imgs = min(int(kw['noof_bg_imgs']), len(self.bg_img_paths))


        # self.train_x = np.empty( (self.noof_training_imgs,) + self.shape, dtype=np.uint8 )
        # self.mask_x = np.empty( (self.noof_training_imgs,) + self.shape[:2], dtype= bool)
        # self.noof_obj_pixels = np.empty( (self.noof_training_imgs,), dtype= bool)
        # self.train_y = np.empty( (self.noof_training_imgs,) + self.shape, dtype=np.uint8 )
        # self.bg_imgs = np.empty( (self.noof_bg_imgs,) + self.shape, dtype=np.uint8 )
        # if np.float(eval(self._kw['realistic_occlusion'])):
        #     self.random_syn_masks
        if self.train:
            data = np.load(self.dataset_path + '/316be0760ed7447e347dc6de42feb477.npz')
            bg_imgs   = np.load(self.dataset_path + '/e53920ed550389388b32218664351a92.npy')
            self.train_x = np.array(data['train_x']).astype(np.uint8)
            self.train_y = np.array(data['train_y']).astype(np.uint8)
            self.mask_x  = np.array(data['mask_x'])
            self.bg_imgs = np.array(bg_imgs) # 这里不能直接使用np.load得到的bg_imgs，需要先使用类属性来备份。否则DataLoader会报错 cannot serialize '_io.BufferedReader' object
            
            self.train_img_num = self.train_x.shape[0]
            self.train_bg_num = self.bg_imgs.shape[0]

            self.size = self.train_x.shape[0]
        else:
            pass

        self.augmentation = aug.Sequential([
                            #Sometimes(0.5, PerspectiveTransform(0.05)),
                            #Sometimes(0.5, CropAndPad(percent=(-0.05, 0.1))),
                            aug.Sometimes(0.5, aug.Affine(scale=(1.0, 1.2))),
                            aug.Sometimes(0.5, aug.CoarseDropout( p=0.2, size_percent=0.05) ),
                            aug.Sometimes(0.5, aug.GaussianBlur(1.2*np.random.rand())),
                            aug.Sometimes(0.5, aug.Add((-25, 25), per_channel=0.3)),
                            aug.Sometimes(0.3, aug.Invert(0.2, per_channel=True)),
                            aug.Sometimes(0.5, aug.Multiply((0.6, 1.4), per_channel=0.5)),
                            aug.Sometimes(0.5, aug.Multiply((0.6, 1.4))),
                            aug.Sometimes(0.5, aug.contrast.LinearContrast((0.5, 2.2), per_channel=0.3))
                            ], random_order=False)

    def __len__(self):
        return self.size
    def __getitem__(self, index):
        train_x = self.train_x[index] # 128x128x3
        train_y = self.train_y[index] # 128x128x3
        train_x_mask = self.mask_x[index] # 128x128x3
        train_x_mask = np.expand_dims(train_x_mask,2)
        train_x_mask = np.repeat(train_x_mask,3,2)
        bg_id = random.randint(0,self.train_bg_num)
        bg_img = self.bg_imgs[bg_id] # 128x128x3
        train_x[np.where(train_x_mask)] = bg_img[np.where(train_x_mask)]

        train_x = self.augmentation.augment_image(train_x)
        
        train_x = cv2.cvtColor(train_x,cv2.COLOR_BGR2RGB).transpose((2,0,1))/255.0
        train_y = cv2.cvtColor(train_y,cv2.COLOR_BGR2RGB).transpose((2,0,1))/255.0

        train_x = torch.from_numpy(train_x)
        train_y = torch.from_numpy(train_y)

        return (train_x, train_y) 

    def get(self,index):
        train_x = self.train_x[index] # 128x128x3
        train_y = self.train_y[index] # 128x128x3
        train_x_mask = self.mask_x[index] # 128x128x3
        train_x_mask = np.expand_dims(train_x_mask,2)
        train_x_mask = np.repeat(train_x_mask,3,2)
        bg_id = random.randint(0,self.train_bg_num)
        bg_img = self.bg_imgs[bg_id] # 128x128x3
        train_x[np.where(train_x_mask)] = bg_img[np.where(train_x_mask)]

        train_x = self.augmentation.augment_image(train_x)
        
        return train_x, train_x_mask, train_y, bg_img 
    def viewsphere_for_embedding(self):
        kw = self._kw
        num_cyclo = int(kw['num_cyclo']) # 36
        azimuth_range = (0, 2 * np.pi)  # 方位角
        elev_range = (-0.5 * np.pi, 0.5 * np.pi) # 俯仰角
        views, _ = view_sampler.sample_views(
            int(kw['min_n_views']), # 2562
            float(kw['radius']), # 700
            azimuth_range,
            elev_range
        )
        Rs = np.empty( (len(views)*num_cyclo, 3, 3) )
        i = 0
        for view in views:
            for cyclo in np.linspace(0, 2.*np.pi, num_cyclo):
                rot_z = np.array([[np.cos(-cyclo), -np.sin(-cyclo), 0], [np.sin(-cyclo), np.cos(-cyclo), 0], [0, 0, 1]])
                Rs[i,:,:] = rot_z.dot(view['R'])
                i += 1
        return Rs # world to camera，将世界坐标系下的点转换到相机坐标系下
    def refine_T(self,R,predicted_bb,rendered_bb,K_test):
        '''
        Input
        -----
            R               :   world to camera ，将世界坐标系下的点转换到相机坐标系下 √
            predicted_bb    :   [x_lefttop,y_lefttop,w,h]   √
            rendered_bb     :   [x_lefttop,y_lefttop,w,h]   √
            k_test          :   intrinsic matrix            √

        Output
        ------
            Rs_est          :   world to camera ，将世界坐标系下的点转换到相机坐标系下  √
            ts_est          :   物体坐标系原点在相机坐标系下的表示（表示在相机坐标系下，指向物体坐标系原点的向量）  √
        '''
        Rs_est = R.copy()
        K_train = np.array(eval(self._kw['k'])).reshape(3,3)
        # K_test = np.array([[909.3773280487426,                  0.,              642.9053428238576],
        #                    [               0.,   909.3773280487426,              358.0199195001692],
        #                    [               0.,                  0.,                             1.]])
        K00_ratio = K_test[0,0] / K_train[0,0]  
        K11_ratio = K_test[1,1] / K_train[1,1]

        mean_K_ratio = np.mean([K00_ratio,K11_ratio])

        ts_est = np.empty((1,3))
        render_radius = eval(self._kw['radius'])

        diag_bb_ratio = np.linalg.norm(np.float32(rendered_bb[2:])) / np.linalg.norm(np.float32(predicted_bb[2:]))
        z = diag_bb_ratio * mean_K_ratio * render_radius

        center_obj_x_train = rendered_bb[0] + rendered_bb[2]/2. - K_train[0,2]
        center_obj_y_train = rendered_bb[1] + rendered_bb[3]/2. - K_train[1,2]

        center_obj_x_test = predicted_bb[0] + predicted_bb[2]/2 - K_test[0,2]
        center_obj_y_test = predicted_bb[1] + predicted_bb[3]/2 - K_test[1,2]
        
        center_obj_mm_x = center_obj_x_test * z / K_test[0,0] - center_obj_x_train * render_radius / K_train[0,0]  
        center_obj_mm_y = center_obj_y_test * z / K_test[1,1] - center_obj_y_train * render_radius / K_train[1,1]  

        t_est = np.array([center_obj_mm_x, center_obj_mm_y, z])
        ts_est = t_est

        # correcting the rotation matrix 
        # the codebook consists of centered object views, but the test image crop is not centered
        # we determine the rotation that preserves appearance when translating the object
        d_alpha_x = - np.arctan(t_est[0]/t_est[2]) # 绕y旋转的是负角度，下面R_corr_y中没有隐式乘-1。
        d_alpha_y = - np.arctan(t_est[1]/t_est[2]) # 绕x旋转的是正角度，下面R_corr_x中d_alpha_y又隐式的乘了-1，负负得正。
        R_corr_x = np.array([[1,0,0],              # 绕相机x轴转
                            [0,np.cos(d_alpha_y),-np.sin(d_alpha_y)],
                            [0,np.sin(d_alpha_y),np.cos(d_alpha_y)]]) 
        R_corr_y = np.array([[np.cos(d_alpha_x),0,-np.sin(d_alpha_x)],
                            [0,1,0],                # 绕相机y轴转
                            [np.sin(d_alpha_x),0,np.cos(d_alpha_x)]]) 
        R_corrected = np.dot(R_corr_y,np.dot(R_corr_x,Rs_est))
        # Rs_est 是sphere上采样得到的，相机光心正对物体坐标系原点时的旋转矩阵，是world to camera的，如果测试时物体处于
        # 相机坐标系的x和y均为正的象限，那么要从光心正对物体坐标系原点变换到光心位于物体坐标系原点的左上方，相机坐标系就
        # 需要绕x轴转一个正的角度，绕y轴转一个负角度。
        Rs_est = R_corrected

        return (Rs_est, ts_est) # Rs_est : world to camera 
                                # ts_est : 物体坐标系原点在相机坐标系下的表示（表示在相机坐标系下，指向物体坐标系原点的向量）

    def renderer(self):
        from .meshrenderer import meshrenderer, meshrenderer_phong
        if self._kw['model'] == 'cad':
            renderer = meshrenderer.Renderer(
               [self._kw['model_path']],
               int(self._kw['antialiasing']),
               self.dataset_path,
               float(self._kw['vertex_scale'])
            )
        elif self._kw['model'] == 'reconst':
            renderer = meshrenderer_phong.Renderer(
               [self._kw['model_path']],
               int(self._kw['antialiasing']),
               self.dataset_path,
               float(self._kw['vertex_scale'])
            )
        else:
            'Error: neither cad nor reconst in model path!'
            exit()
        return renderer
    def render_rot(self, R, t=None ,downSample = 1):
        kw = self._kw
        h, w = self.shape[:2]
        radius = float(kw['radius'])
        render_dims = eval(kw['render_dims'])
        K = eval(kw['k'])
        K = np.array(K).reshape(3,3)
        K[:2,:] = K[:2,:] / downSample

        clip_near = float(kw['clip_near'])
        clip_far = float(kw['clip_far'])
        pad_factor = float(kw['pad_factor'])

        t = np.array([0, 0, float(kw['radius'])])

        bgr_y, depth_y = self.renderer().render( # 这里如果是函数调用，而不是@lazy_property，可能会出现多次缓存的问题 
            obj_id=0,
            W=render_dims[0]/downSample,
            H=render_dims[1]/downSample,
            K=K.copy(),
            R=R,
            t=t,
            near=clip_near,
            far=clip_far,
            random_light=False
        )

        ys, xs = np.nonzero(depth_y > 0)
        obj_bb = view_sampler.calc_2d_bbox(xs, ys, render_dims)
        x, y, w, h = np.array(obj_bb).astype(np.int32)

        size = int(np.maximum(h, w) * pad_factor)

        left = int(np.maximum(x+w/2-size/2, 0))
        right = int(np.minimum(x+w/2+size/2, bgr_y.shape[1]))
        top = int(np.maximum(y+h/2-size/2, 0))
        bottom = int(np.minimum(y+h/2+size/2, bgr_y.shape[0]))

        bgr_y = bgr_y[top:bottom, left:right]
        return cv2.resize(bgr_y, self.shape[:2])

if __name__ == '__main__':
    args = configparser.ConfigParser()
    args.read('./cfg/AAE.cfg')
    dataset_args = { k:v for k,v in
        args.items('Dataset') +
        args.items('Paths') +
        args.items('Augmentation')+
        args.items('Queue') +
        args.items('Embedding')+
        args.items('Network')+
        args.items('Training')}

    data_path = './data'
    _Dataset = AAE_dataset(data_path,**dataset_args)
    train_x, train_x_mask, train_y, bg_img = _Dataset.__getitem__(0)
    cv2.imshow('train_x',train_x)
    cv2.imshow('train_x_mask',train_x_mask.astype(np.uint8)*255)
    cv2.imshow('train_y',train_y)
    cv2.imshow('bg_img',bg_img)
    cv2.waitKey(0) 