import math
import numpy as np
import torch

def plot_boxes_cv2(img, boxes, savename=None, class_names=None, color=None,crop=False):
    '''
    input:
            boxes : [[xmin,ymin,xmax,ymax],[...],...] , nx7
    output:
            image with boxes
    '''
    import cv2
    img = np.copy(img)
    colors = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)
    crop_img = {}

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    width = img.shape[1]
    height = img.shape[0]
    for i in range(boxes.shape[0]):
        box = boxes[i]
        x1 = int(box[0] * width) # box的每个值是特征图中的占比，这里乘以输入图像的宽高也就得到在输入图像中的Bbox坐标
        y1 = int(box[1] * height)
        x2 = int(box[2] * width)
        y2 = int(box[3] * height)

        if color:
            rgb = color
        else:
            # rgb = (255, 0, 0)
            rgb = (0, 255, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue = get_color(0, offset, classes)
            if color is None:
                rgb = (red, green, blue)
            img = cv2.putText(img, class_names[cls_id], (x1, y1-5), cv2.FONT_HERSHEY_PLAIN, 1, rgb, 1)
        
        if crop:
            # import time
            # center = [(x1+x2)//2,(y1+y2)//2]
            # h = y2 - y1
            # w = x2 - x1
            # crop_img = img[center[1]-64:center[1]+64,center[0]-64:center[0]+64,:]
            # crop_img = img[center[1]-32:center[1]+32,center[0]-32:center[0]+32,:]
            crop_img[i] = img[y1:y2,x1:x2,:].copy()
            # cv2.imwrite("C:/Users/Yang/Desktop/AAE/crop_img/"+f"{time.time()}".split('.')[1]+".png", crop_img)
        
        img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 1)
        cv2.putText(img,f"{i}",(x1, y1),1,1,(0,255,0),1)
    if savename:
        print("save plot results to %s" % savename)
        cv2.imwrite(savename, img)  
    return img

def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names

def get_region_boxes(boxes_and_confs):

    # print('Getting boxes from boxes and confs ...')

    boxes_list = []
    confs_list = []

    for item in boxes_and_confs:
        boxes_list.append(item[0])
        confs_list.append(item[1])

    # boxes: [batch, num1 + num2 + num3, 1, 4]
    # confs: [batch, num1 + num2 + num3, num_classes]
    boxes = torch.cat(boxes_list, dim=1)
    confs = torch.cat(confs_list, dim=1)
        
    return [boxes, confs]

def nms_cpu(boxes, confs, nms_thresh=0.5, min_mode=False):
    # print(boxes.shape)
    """Pure Python NMS baseline."""
    # x1、y1、x2、y2、以及score赋值
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    #每一个检测框的面积
    areas = (x2 - x1) * (y2 - y1)
    #按照score置信度降序排序
    order = confs.argsort()[::-1]

    keep = []   #保留的结果框集合
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]

        keep.append(idx_self)#保留该类剩余box中得分最高的一个
        #得到相交区域,左上及右下
        xx1 = np.maximum(x1[idx_self], x1[idx_other])
        yy1 = np.maximum(y1[idx_self], y1[idx_other])
        xx2 = np.minimum(x2[idx_self], x2[idx_other])
        yy2 = np.minimum(y2[idx_self], y2[idx_other])

        #计算相交的面积,不重叠时面积为0
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h  # idx_self分别和所有idx_other的相交面积

        #计算IoU：重叠面积 /（面积1+面积2-重叠面积）
        if min_mode:
            over = inter / np.minimum(areas[order[0]], areas[order[1:]])    # order[0]和每个其他的框都计算一个相交面积与两者的最小面积的比值，over的维度为areas[order[1:]]的维度
        else:
            over = inter / (areas[order[0]] + areas[order[1:]] - inter)     # order[0]和每个其他的框都计算一个相交面积与两者的并集面积的比值，也就是IOU，over的维度为areas[order[1:]]的维度

        #保留IoU小于阈值的box
        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]     #因为over数组的长度比order数组少一个,所以这里要将所有下标后移一位

        # 因为order一开始就是按照置信度降序排的，第一次while循环把置信度最高的一个保存到了keep中，然后在
        # 后面找出与已经放入keep中的框的IOU小于阈值的框放入重新作为order，因为order一开始就是降序的，
        # 计算了IOU，又用阈值过滤后找出的框的构成的order仍然是按置信度降序排的，每次while都将order中置信度最高的框加入keep
    
    return np.array(keep)