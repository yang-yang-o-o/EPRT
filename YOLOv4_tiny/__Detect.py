from .model import Yolov4
from .do_detect import do_detect
import cv2
import time
from .utils import plot_boxes_cv2 , load_class_names
import torch
import os
import numpy as np
import xml.dom.minidom

class Detect():
    def __init__(self,obj_name,use_cuda=True) -> None:
        
        weight_path = {'obj_01':'YOLOv4_tiny/weight/T_Less_obj_01_epoch100.pth',
                        
                        'gear':'-',

                        'mesh_01':'YOLOv4_tiny/weight/mesh_01/mesh_01_fullgradient_epoch101.pth', # best for mesh_01
                        # 'mesh_01':'YOLOv4_tiny/weight/mesh_01/mesh_01_epoch100.pth',
                        # 'mesh_01_fullgradient':'YOLOv4_tiny/weight/mesh_01/mesh_01_fullgradient_epoch101.pth', # √
                        
                        # 'mesh_02':'YOLOv4_tiny/weight/mesh_02/250d_-45_45/Yolov4_epoch101.pth', #best for mesh_02
                        # 'mesh_02':'YOLOv4_tiny/weight/mesh_02/mesh_02_epoch100.pth', # 非全梯度，没有去除圆形，CAD
                        # 'mesh_02_250d':'YOLOv4_tiny/weight/mesh_02/mesh_02_250d_epoch100.pth', # 非全梯度，没有去除圆形，reconst
                        # 'mesh_02_250d_-45_45':'YOLOv4_tiny/weight/mesh_02/250d_-45_45/Yolov4_epoch51.pth', # 非全梯度，去除圆形，reconst
                        # 'mesh_02_250d_fullgradient':'YOLOv4_tiny/weight/mesh_02/250d/mesh_02_250d_epoch51.pth', # 全梯度，没有去除圆形，reconst ，Early stopping 确定了51
                        # 'mesh_02_250d_-45_45_fullgradient':'YOLOv4_tiny/weight/mesh_02/250d_-45_45_fullgradient/Yolov4_epoch11.pth', # 全梯度，去除圆形，reconst

                        'mesh_02':'YOLOv4_tiny/weight/mesh_02/100d/Yolov4_epoch101.pth',# 61、81、91、101

                        'mesh_03':'YOLOv4_tiny/weight/mesh_03/mesh_03_fullgradient_epoch81.pth', # best for mesh_03
                        # 'mesh_03':'YOLOv4_tiny/weight/mesh_03/mesh_03_epoch100.pth',
                        # 'mesh_03_fullgradient':'YOLOv4_tiny/weight/mesh_03/mesh_03_fullgradient_epoch81.pth'  # √

                        'mesh_04':'YOLOv4_tiny/weight/mesh_04/Yolov4_epoch51.pth',

                        'mesh_05':'YOLOv4_tiny/weight/mesh_05/Yolov4_epoch51.pth',
                        # 'mesh_05':'YOLOv4_tiny/weight/mesh_05/Yolov4_epoch11.pth',
                        }

        self.use_cuda = use_cuda
        self.Model = Yolov4(n_classes=1,inference=True)
        self.Model.load_state_dict(torch.load(weight_path[obj_name], map_location=lambda storage, loc: storage))
        if self.use_cuda:
            self.Model.cuda()
        self.image = None
        self.boxes = None

    def detect(self,image,RunningTimePrint=False):
        """
        imput:      image , cv mat

        output:    [B,bboxes]    bboxes:[[x1,y1,x2,y2,cof,cof,ID],...]  ,两个conf是一样的
        """
        self.image = image.copy()
        resized = cv2.resize(self.image, (416, 416))
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        boxes = do_detect(self.Model, resized, 0.9, 0.5, use_cuda=self.use_cuda,
        # boxes = do_detect(self.Model, resized, 0.95, 0.5, use_cuda=self.use_cuda,
                        inference_time=RunningTimePrint,
                        non_maximum_suppression_time=RunningTimePrint)  #  [B,bboxes]    bboxes:[[x1,y1,x2,y2,cof,cof,ID],...]
        self.boxes = np.array(boxes)
        return self.boxes
    
    def plot(self):
        tmp_boxes = self.boxes[0]

        image = self.image.copy()
        image = plot_boxes_cv2(image, tmp_boxes)
        return image

    def plot_(self,image):
        tmp_boxes = self.boxes[0]
        image = plot_boxes_cv2(image, tmp_boxes)
        return image
    
    def crop(self):
        tmp_boxes = self.boxes[0]
        
        crop_img = {}
        predict_bbx = {}
        predict_bbx_rectify = {}
        width = self.image.shape[1]
        height = self.image.shape[0]
        for i in range(tmp_boxes.shape[0]):
            box = tmp_boxes[i]
            x1 = int(box[0] * width) # box的每个值是特征图中的占比，这里乘以输入图像的宽高也就得到在输入图像中的Bbox坐标
            y1 = int(box[1] * height)
            x2 = int(box[2] * width)
            y2 = int(box[3] * height)
            predict_bbx[i] = [x1,y1,x2-x1,y2-y1]
            if (x2 - x1) > (y2 - y1):
                _y1 = max(0,y1-((x2 - x1) - (y2 - y1))//2) 
                _y2 = min(height,y2+((x2 - x1) - (y2 - y1))//2) 
                y1 = _y1
                y2 = _y2
            else:
                _x1 = max(0,x1-((y2 - y1) - (x2 - x1))//2) 
                _x2 = min(width,x2+((y2 - y1) - (x2 - x1))//2) 
                x1 = _x1
                x2 = _x2
            offset = 20
            left_up_x = max(0,x1-offset)
            left_up_y = max(0,y1-offset)
            right_bottom_x = min(width,x2+offset)
            right_bottom_y = min(height,y2+offset)
            crop_img[i] = self.image[left_up_y:right_bottom_y,left_up_x:right_bottom_x,:].copy()
            predict_bbx_rectify[i] = [x1,y1,x2-x1,y2-y1]
        return crop_img , predict_bbx , predict_bbx_rectify
    
    def load_xml(self,xml_path):
        dom = xml.dom.minidom.parse(xml_path)
        root =dom.documentElement
        bbox = []
        for ob in root.getElementsByTagName("object"):
            bndbox = ob.getElementsByTagName('bndbox')[0]
            xmin = int(bndbox.getElementsByTagName('xmin')[0].childNodes[0].data)/640.0
            ymin = int(bndbox.getElementsByTagName('ymin')[0].childNodes[0].data)/480.0
            xmax = int(bndbox.getElementsByTagName('xmax')[0].childNodes[0].data)/640.0
            ymax = int(bndbox.getElementsByTagName('ymax')[0].childNodes[0].data)/480.0
            bbox.append([xmin,ymin,xmax,ymax,1.0,1.0,0])
        return [bbox]  #[B,bboxes]    bboxes:[[x1,y1,x2,y2,cof,cof,ID],...]  ,两个conf是一样的