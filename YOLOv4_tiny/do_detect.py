import numpy as np
import torch
import time
from .utils import nms_cpu

def do_detect(model, img, conf_thresh, nms_thresh, use_cuda=True,inference_time=True,non_maximum_suppression_time=True):
    model.eval()
    t0 = time.time()

    if type(img) == np.ndarray and len(img.shape) == 3:  # cv2 image
        img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    elif type(img) == np.ndarray and len(img.shape) == 4:
        img = torch.from_numpy(img.transpose(0, 3, 1, 2)).float().div(255.0)
    else:
        print("unknow image type")
        exit(-1)

    # img = img.expand(4,-1,-1,-1)

    if use_cuda:
        img = img.cuda()
    img = torch.autograd.Variable(img)
    
    t1 = time.time()

    output = model(img)

    t2 = time.time()

    if inference_time:
        print('-----------------------------------')
        print('           Preprocess : %f' % (t1 - t0))
        print('      Model Inference : %f' % (t2 - t1))
        print('-----------------------------------')


    # output = [torch.load('boxes.pt') , torch.load('confs.pt')]
    return post_processing(img, conf_thresh, nms_thresh, output, non_maximum_suppression_time)

def post_processing(img, conf_thresh, nms_thresh, output,non_maximum_suppression_time):

    # anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
    # num_anchors = 9
    # anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    # strides = [8, 16, 32]
    # anchor_step = len(anchors) // num_anchors

    # [batch, num, 1, 4]
    box_array = output[0]
    # [batch, num, num_classes]
    confs = output[1]

    t1 = time.time()

    if type(box_array).__name__ != 'ndarray':
        box_array = box_array.cpu().detach().numpy()
        confs = confs.cpu().detach().numpy()

    num_classes = confs.shape[2]

    # [batch, num, 4]
    box_array = box_array[:, :, 0]

    # [batch, num, num_classes] --> [batch, num]
    max_conf = np.max(confs, axis=2)  # 每个框找出最大的类别置信度
    max_id = np.argmax(confs, axis=2) # 每个框找出最大的类别置信度对应的类别ID

    t2 = time.time()

    bboxes_batch = []
    for i in range(box_array.shape[0]):# 处理每个B
       
        argwhere = max_conf[i] > conf_thresh    # 大于conf_thresh为True，否则为False
        l_box_array = box_array[i, argwhere, :] # 找出置信度大于conf_thresh的所有Bbox
        l_max_conf = max_conf[i, argwhere]      # 找出置信度大于conf_thresh的所有置信度
        l_max_id = max_id[i, argwhere]          # 找出置信度大于conf_thresh的所有类别ID

        bboxes = []
        # nms for each class
        for j in range(num_classes):# 处理每个类

            cls_argwhere = l_max_id == j
            ll_box_array = l_box_array[cls_argwhere, :]
            ll_max_conf = l_max_conf[cls_argwhere]
            ll_max_id = l_max_id[cls_argwhere]

            keep = nms_cpu(ll_box_array, ll_max_conf, nms_thresh) # keep为ll_max_conf中置信度较大的的下标
            
            if (keep.size > 0):#挑出keep所对应的框，置信度，ID
                ll_box_array = ll_box_array[keep, :]
                ll_max_conf = ll_max_conf[keep]
                ll_max_id = ll_max_id[keep]

                for k in range(ll_box_array.shape[0]): # 遍历每个keep中存的框，将每个框的信息[x1,y1,x2,y2,cof,cof,ID]添加到bboxes
                    bboxes.append([ll_box_array[k, 0], ll_box_array[k, 1], ll_box_array[k, 2], ll_box_array[k, 3], ll_max_conf[k], ll_max_conf[k], ll_max_id[k]])
        
        bboxes_batch.append(bboxes)

    t3 = time.time()

    if non_maximum_suppression_time:
        print('-----------------------------------')
        print('       max and argmax : %f' % (t2 - t1))
        print('                  nms : %f' % (t3 - t2))
        print('Post processing total : %f' % (t3 - t1))
        print('-----------------------------------')
    
    return bboxes_batch     # [B,bboxes]    bboxes:[[x1,y1,x2,y2,cof,cof,ID],...]    x1,y1,x2,y2：每个值都是一个比例，是特征图中的占比，也等于原始图像中的占比
