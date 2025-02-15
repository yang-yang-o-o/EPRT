from .model import Yolov4
from .do_detect import do_detect
import cv2
import time
from .utils import plot_boxes_cv2 , load_class_names
import torch
import os

def inference():
    model = Yolov4(n_classes=1,inference=True)

    # pretrained_dict = torch.load('1.pth')
    # conv_0 = pretrained_dict['models.0.conv1.weight']
    # model.load_state_dict(pretrained_dict)
    model.load_state_dict(torch.load('YOLOv4_tiny/Yolov4_epoch100.pth', map_location=lambda storage, loc: storage))
    
    use_cuda = True
    if use_cuda:
        model.cuda()
    
    # image_dir = 'E:/T-LESS/01/rgb'
    # image_paths = [os.path.join(image_dir,i) for i in os.listdir(image_dir) if i[-3:]=='png']
    
    # image_dir = 'data/test'
    image_dir = 'data/image'
    image_paths = [os.path.join(image_dir,i) for i in os.listdir(image_dir) if i[-3:]=='png']

    # image_dir = 'data/test3'
    # image_paths = [os.path.join(image_dir,i) for i in os.listdir(image_dir) if i[-3:]=='png']

    for image_path in image_paths:

        img = cv2.imread(image_path)
        resized = cv2.resize(img, (416, 416))
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        boxes = do_detect(model, resized, 0.9, 0.5, use_cuda)  #  [B,bboxes]    bboxes:[[x1,y1,x2,y2,cof,cof,ID],...]

        # image = plot_boxes_cv2(img, boxes[0])
        image = plot_boxes_cv2(img, boxes[0],crop=True)
        print(image_path)
        cv2.imshow('1', image)
        key = cv2.waitKey(0)

        if key == 27:
            break

if __name__ == "__main__":
    inference()
