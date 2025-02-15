import torch
import cv2
import numpy as np
import os
import sys
import configparser

from dataset import AAE_dataset
import model
from weight_load import load

def test():
    AAE = model.AAE()
    load(AAE,'AAE/weight.npz') 
    AAE.cuda()
    AAE.eval()

    args = configparser.ConfigParser()
    args.read('AAE/cfg/AAE.cfg')
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
    train_x, train_x_mask, train_y, bg_img = _Dataset.get(1)
    cv2.imshow('train_x',train_x)
    cv2.imshow('train_x_mask',train_x_mask.astype(np.uint8)*255)
    cv2.imshow('train_y',train_y)
    cv2.imshow('bg_img',bg_img)

    # # train_x, train_y = _Dataset.__getitem__(0)
    # im = cv2.imread('F:/AAE_D2C_YOLO/AAE/AAEgear/crop_img\\1.png')
    # im = cv2.resize(im,(128,128))
    # # im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    # x = im/255.
    # x = np.expand_dims(x, 0)
    # x = x.transpose((0,3,1,2))
    # x = torch.from_numpy(x).float().cuda()
    # reconstruction = AAE(x).cpu().detach().permute((0,2,3,1)).numpy()
    # # reconstruction = np.squeeze(reconstruction,0).transpose((1,2,0))*255
    # # reconstruction = reconstruction.astype(np.uint8)
    # # reconstruction = cv2.cvtColor(reconstruction,cv2.COLOR_RGB2BGR)
    # cv2.imshow('reconstruct',cv2.resize(reconstruction[0],(128,128)))
    # encode = AAE.latent_code(x)
    # idcs = AAE.similarity(encode)
    # R = _Dataset.viewsphere_for_embedding()[idcs]
    # print(R)
    
    # pred_view = _Dataset.render_rot( R,downSample = 1)
    # cv2.imshow('pred_view', cv2.resize(pred_view,(128,128)))
    # cv2.waitKey(0)

    train_x, train_y = _Dataset.__getitem__(1)
    train_x = torch.unsqueeze(train_x,0).float().cuda()
    reconstruction = AAE(train_x).cpu().detach().permute((0,2,3,1)).numpy()
    cv2.imshow('reconstruct',cv2.resize(reconstruction[0],(128,128)))
    encode = AAE.latent_code(train_x)
    idcs, rendered_bb = AAE.similarity(encode)
    R = _Dataset.viewsphere_for_embedding()[idcs]
    # import mouse_cut
    # Cut = mouse_cut.image_cut()
    # cut_img, _ = Cut.cut('F:\AAE_D2C_YOLO\AAE\AAEgear/RGB_view_41.bmp')
    # predicted_bb = [Cut.point1[0],Cut.point1[1],Cut.point2[0]-Cut.point1[0],Cut.point2[1]-Cut.point1[1]]
    # R_est, t_est = _Dataset.refine_T(R,predicted_bb,rendered_bb)
    # cv2.imwrite('cut_img.png',cut_img)
    # np.savez('T.npz',R_est,t_est)
    print(R)
    pred_view = _Dataset.render_rot( R,downSample = 1)
    cv2.imshow('pred_view', cv2.resize(pred_view,(128,128)))
    cv2.waitKey(0)


if __name__ == "__main__":
    test()