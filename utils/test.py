import imp
import os
import sys
from time import time

from cv2 import imwrite
import numpy as np
sys.path.append(os.getcwd())
import cv2

import YOLOv4_tiny.__Detect as __Detect
import AAE.__AAE as __AAE
import D2C.__D2C as __D2C

def yolo_test():
    _Detect = __Detect.Detect(obj_name="mesh_03",use_cuda=False)

    # image_dir = 'data/image/mesh_02'
    # image_dir = 'data/2'
    # image_dir = 'D:\Postgraduate_files\Final_paper\code\Experiment-5.4\image3'
    # image_dir = 'F:\AAE_D2C_YOLO\Release\data\image\mesh_05_real'
    image_dir = 'F:\AAE_D2C_YOLO\Release\data\image\mesh_03_real'
    image_paths = [os.path.join(image_dir,i) for i in os.listdir(image_dir) if i[-3:]=='png' or i[-3:]=='bmp' or i[-3:]=='jpg']
    for image_path in image_paths:
        img = cv2.imread(image_path)
        # img = cv2.resize(img,(864,1152))
        boxes = _Detect.detect(img)
        image = _Detect.plot()
        cv2.destroyAllWindows()
        cv2.imshow('detect',image)
        print(image_path)
        crop_img , predict_bbx,_ = _Detect.crop()
        # cv2.imshow('crop',crop_img[0])
        key = cv2.waitKey(0)
        if key == 27:
            break

def yolo_test_usbcamera():
    import time
    _Detect = __Detect.Detect(obj_name="mesh_05",use_cuda=True)
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        # frame_ = argumentation(frame)
        boxes = _Detect.detect(frame,RunningTimePrint=True)
        image = _Detect.plot()
        cv2.imshow('1',image)
        key = cv2.waitKey(1)
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            break
        if key == 32:
            cv2.imwrite('data\image\mesh_02/{}.png'.format(time.time()),frame)

def AAE_test():
    _AAE = __AAE.AAE(obj_name='mesh_04',train=False)
    # im = cv2.imread('data/AAE/gear/crop_img/1.png')
    # im = cv2.imread('data/AAE/T_Less_obj_01/crop_img/1.png')
    # im = cv2.imread('data/AAE/mesh_01/crop_img/1639485343.8137717.png')
    # _AAE.reconstruct(im)

    image_dir = 'data/AAE/mesh_03/crop_img'
    image_paths = [os.path.join(image_dir,i) for i in os.listdir(image_dir) if i[-3:]=='png']
    for image_path in image_paths:
        img = cv2.imread(image_path)
        _AAE.reconstruct(img)

def AAE_runningtime():
    import numpy as np
    import time
    _AAE = __AAE.AAE(obj_name='mesh_05',train=False,use_cuda=True)
    image_dir = cv2.imread('crop_img.png')
    K_test = np.array([(640+480)/2., 0, 640/2, 0, (640+480)/2., 480/2, 0, 0, 1]).reshape(3,3)
    predict_bbx = {0: [170, 262, 71, 52]}
    t0 = time.time()
    for i in range(100):
        T = _AAE.inference(image_dir,K_test,predict_bbx[0])
    print("average running time: {} ms".format((time.time()-t0)/100*1000))

def D2C_test():
    lib_path = "D2C/libtest_localization.so"  
    mesh_path = b"D2C/3D_models/Gear.stl"
    cfg_path = b"D2C/test_images/test1_1_camera_calib.yml"
    _D2C = __D2C.D2C(lib_path,mesh_path,cfg_path)
    frame = cv2.imread("D2C/RGB_view_41.bmp")
    R_vec = b"0.26758555,-0.53488473,0.05784623"
    t_vec = b"0.08163072436711435,-0.02079111854001027,0.35"
    _D2C.inference_image(frame,R_vec,t_vec)

def argumentation(image):
    import imgaug.augmenters as aug
    import numpy as np
    augmentation = aug.Sequential([
                            #Sometimes(0.5, PerspectiveTransform(0.05)),
                            #Sometimes(0.5, CropAndPad(percent=(-0.05, 0.1))),
                            aug.Sometimes(0.5, aug.Affine(scale=(1.0, 1.2))),
                            # aug.Sometimes(0.5, aug.CoarseDropout( p=0.2, size_percent=0.05) ),
                            aug.Sometimes(0.5, aug.GaussianBlur(1.5*np.random.rand())),
                            aug.Sometimes(0.5, aug.Add((-25, 25), per_channel=0.8)),
                            aug.Sometimes(0.3, aug.Invert(0.5, per_channel=True)),
                            aug.Sometimes(0.5, aug.Multiply((0.6, 1.4), per_channel=0.8)),
                            aug.Sometimes(0.5, aug.Multiply((0.6, 1.4))),
                            aug.Sometimes(0.5, aug.contrast.LinearContrast((0.5, 2.2), per_channel=0.8))
                            ], random_order=False)
    image = augmentation.augment_image(image)
    return image

def casual_test():
    # image = cv2.imread('data\AAE\mesh_02_real/1/0_2.png')
    image = cv2.imread('data\image\mesh_02/1.png')
    cv2.imshow('1',image)
    image_ = argumentation(image).copy()
    cv2.imshow('2',image_)
    cv2.waitKey()
    # _AAE = __AAE.AAE(obj_name='mesh_02',train=False)
    # _AAE.reconstruct(image_)
    _Detect = __Detect.Detect(obj_name="mesh_02",use_cuda=False)
    boxes = _Detect.detect(image_)
    image = _Detect.plot()
    cv2.imshow('detect',image)
    key = cv2.waitKey(0)

def diameter():
    path = "AAE\data\mesh_05\diameter.npy"
    d = np.load(path)
    print(d)
if __name__ == '__main__':
    # yolo_test_usbcamera()
    yolo_test()
    # AAE_runningtime()
    # AAE_test()
    # D2C_test()
    # diameter()


