import torch
import cv2
import numpy as np
import os
import sys
import configparser

from dataset import AAE_dataset
import model
from weight_load import load
import mouse_cut

def test():
    AAE = model.AAE()
    load(AAE,'weight.npz') 
    AAE.cuda()
    AAE.eval()

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

    data_path = 'F:\AAE_D2C_YOLO\AAE\AAE_pytorch\data'
    _Dataset = AAE_dataset(data_path,**dataset_args)
    ############ 整个复现的模型应该是没有问题的

    Cut = mouse_cut.image_cut()


    # im = cv2.imread('F:/AAE_D2C_YOLO/AAE/AAEgear/crop_img\\999.png')
    image = cv2.imread('F:\AAE_D2C_YOLO\AAE\AAEgear/RGB_view_41.bmp')
    im, center = Cut.cut(image)
    # cv2.imwrite('cut_img.png',im)
    # predicted_bb = [Cut.point1[0],Cut.point1[1],Cut.point2[0]-Cut.point1[0],Cut.point2[1]-Cut.point1[1]]

    im = cv2.resize(im,(128,128))
    # im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    x = im/255.
    x = np.expand_dims(x, 0)
    x = x.transpose((0,3,1,2))
    x = torch.from_numpy(x).float().cuda()
    reconstruction = AAE(x).cpu().detach().permute((0,2,3,1)).numpy()
    cv2.imshow('reconstruct',cv2.resize(reconstruction[0],(128,128)))
    encode = AAE.latent_code(x)
    idcs,rendered_bb = AAE.similarity(encode)
    R = _Dataset.viewsphere_for_embedding()[idcs]
    print(R)
    print(center)
    print(cv2.Rodrigues(R)[0].transpose())
    print(Cut.imgpoont2t(center,0.35))
    
    # R_est, t_est = _Dataset.refine_T(R,predicted_bb,rendered_bb)
    # np.savez('T.npz',R_est,t_est)

    # pred_view = _Dataset.render_rot( R_est,downSample = 1)
    pred_view = _Dataset.render_rot( R,downSample = 1)
    cv2.imshow('pred_view', cv2.resize(pred_view,(128,128)))
    cv2.waitKey(0)

if __name__ == "__main__":
    test()