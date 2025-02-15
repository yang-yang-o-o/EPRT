# -*- coding: utf-8 -*-
'''
@Time          : 2020/05/06 21:09
@Author        : Tianxiaomo
@File          : dataset.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''
import os
import random
import sys

import cv2
import numpy as np

import torch
from torch.utils.data.dataset import Dataset


def rand_uniform_strong(min, max):
    if min > max:
        swap = min
        min = max
        max = swap
    return random.random() * (max - min) + min


def rand_scale(s):
    scale = rand_uniform_strong(1, s)
    if random.randint(0, 1) % 2:
        return scale
    return 1. / scale


def rand_precalc_random(min, max, random_part):
    if max < min:
        swap = min
        min = max
        max = swap
    return (random_part * (max - min)) + min


def fill_truth_detection(bboxes, num_boxes, classes, flip, dx, dy, sx, sy, net_w, net_h):
    '''
    return：
            bboxes   ：   nx5  [[x1,y1,x2,y2,cls_id],...]       根据dx, dy, sx, sy将bboxes每个框的坐标原点改到裁剪后的图像原点，然后去除bboxes中超出裁剪后图像边界的框，然后取前num_boxes
                       个框按net_w, net_h换算到网络输入的图像中并根据flip决定是否水平翻转后返回这前num_boxes个框。
            min_w_h  ：   2,    同时返回这前num_boxes个框换算前的最小宽和最小高
    '''
    if bboxes.shape[0] == 0:
        return bboxes, 10000
    np.random.shuffle(bboxes)
    bboxes[:, 0] -= dx
    bboxes[:, 2] -= dx
    bboxes[:, 1] -= dy
    bboxes[:, 3] -= dy# 将边界框的坐标原点设置为裁剪后的原点

    bboxes[:, 0] = np.clip(bboxes[:, 0], 0, sx)
    bboxes[:, 2] = np.clip(bboxes[:, 2], 0, sx)

    bboxes[:, 1] = np.clip(bboxes[:, 1], 0, sy)
    bboxes[:, 3] = np.clip(bboxes[:, 3], 0, sy) # 将边界框的坐标限定在裁剪后的图像尺寸内

    out_box = list(np.where(((bboxes[:, 1] == sy) & (bboxes[:, 3] == sy)) |
                            ((bboxes[:, 0] == sx) & (bboxes[:, 2] == sx)) |
                            ((bboxes[:, 1] == 0) & (bboxes[:, 3] == 0)) |
                            ((bboxes[:, 0] == 0) & (bboxes[:, 2] == 0)))[0]) # 超界的边界框在bboxes中的行号列表
    list_box = list(range(bboxes.shape[0]))
    for i in out_box:
        list_box.remove(i)
    bboxes = bboxes[list_box]   # 将超界的边界框从bboxes中移除

    if bboxes.shape[0] == 0:
        return bboxes, 10000

    bboxes = bboxes[np.where((bboxes[:, 4] < classes) & (bboxes[:, 4] >= 0))[0]]

    if bboxes.shape[0] > num_boxes:
        bboxes = bboxes[:num_boxes] # 取前num_boxes个bboxes。

    min_w_h = np.array([bboxes[:, 2] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1]]).min()# 找出所有边界框w的h中的最小值

    bboxes[:, 0] *= (net_w / sx)
    bboxes[:, 2] *= (net_w / sx)
    bboxes[:, 1] *= (net_h / sy)
    bboxes[:, 3] *= (net_h / sy)# 边界框按比例换算到网络输入的图像中

    if flip:# 水平翻转边界框
        temp = net_w - bboxes[:, 0]
        bboxes[:, 0] = net_w - bboxes[:, 2]
        bboxes[:, 2] = temp

    return bboxes, min_w_h


def rect_intersection(a, b):
    minx = max(a[0], b[0])
    miny = max(a[1], b[1])

    maxx = min(a[2], b[2])
    maxy = min(a[3], b[3])
    return [minx, miny, maxx, maxy]


def image_data_augmentation(mat, w, h, pleft, ptop, swidth, sheight, flip, dhue, dsat, dexp, gaussian_noise, blur,
                            truth):
    '''
    output:     sized   [sheight, swidth, 3] ->(resize)-> (w, h, 3)    , 和原始图像有交集的区域值为原始图像，不相交的区域值为原始图像的均值
                        然后水平翻转，改变HSV，高斯模糊，高斯噪声，最后输出(w, h, 3)

    '''
    try:
        img = mat
        oh, ow, _ = img.shape
        pleft, ptop, swidth, sheight = int(pleft), int(ptop), int(swidth), int(sheight)     # 定义在原数图像坐标系下
        # crop  ，求裁剪区域和原始图像mat的交集坐标 new_src_rect : [minx, miny, maxx, maxy]，这个坐标定义在原始图像坐标系下
        src_rect = [pleft, ptop, swidth + pleft, sheight + ptop]  # x1,y1,x2,y2
        img_rect = [0, 0, ow, oh]
        new_src_rect = rect_intersection(src_rect, img_rect)  # 交集

        # 将定义在原始图像坐标系下的交集区域坐标转换到裁剪区域坐标系下
        dst_rect = [max(0, -pleft), max(0, -ptop), max(0, -pleft) + new_src_rect[2] - new_src_rect[0],
                    max(0, -ptop) + new_src_rect[3] - new_src_rect[1]]
        # new_src_rect和dst_rect分别为交集区域在原始图像坐标系和裁剪区域坐标系下的坐标

        # cv2.Mat sized

        if (src_rect[0] == 0 and src_rect[1] == 0 and src_rect[2] == img.shape[0] and src_rect[3] == img.shape[1]):
            sized = cv2.resize(img, (w, h), cv2.INTER_LINEAR)#如果裁剪区域和原始图像重合，直接将原始图像缩放到输入网络的宽和高
        else:
            cropped = np.zeros([sheight, swidth, 3])# 创建裁剪图像
            cropped[:, :, ] = np.mean(img, axis=(0, 1)) # 定义与裁剪区域等大的张量，每个值设为原始图像的均值

            cropped[dst_rect[1]:dst_rect[3], dst_rect[0]:dst_rect[2]] = \
                img[new_src_rect[1]:new_src_rect[3], new_src_rect[0]:new_src_rect[2]]# 将交集从原始图像拷贝到张量
            #得到的cropped拥有和原始图像的交集区域，其他的区域为原始图像的均值

            # resize
            sized = cv2.resize(cropped, (w, h), cv2.INTER_LINEAR)# 张量缩放到输入网络的宽和高

        # flip  ，水平翻转
        if flip:
            # cv2.Mat cropped
            sized = cv2.flip(sized, 1)  # 0 - x-axis, 1 - y-axis, -1 - both axes (x & y)

        # HSV augmentation
        # cv2.COLOR_BGR2HSV, cv2.COLOR_RGB2HSV, cv2.COLOR_HSV2BGR, cv2.COLOR_HSV2RGB
        if dsat != 1 or dexp != 1 or dhue != 0:
            if img.shape[2] >= 3:
                hsv_src = cv2.cvtColor(sized.astype(np.float32), cv2.COLOR_RGB2HSV)  # RGB to HSV
                hsv = cv2.split(hsv_src)
                hsv[1] *= dsat
                hsv[2] *= dexp
                hsv[0] += 179 * dhue
                hsv_src = cv2.merge(hsv)
                sized = np.clip(cv2.cvtColor(hsv_src, cv2.COLOR_HSV2RGB), 0, 255)  # HSV to RGB (the same as previous)
            else:
                sized *= dexp

        if blur:
            if blur == 1:
                dst = cv2.GaussianBlur(sized, (17, 17), 0)
                # cv2.bilateralFilter(sized, dst, 17, 75, 75)
            else:
                ksize = (blur / 2) * 2 + 1
                dst = cv2.GaussianBlur(sized, (ksize, ksize), 0)

            if blur == 1: # ? ? ?
                img_rect = [0, 0, sized.cols, sized.rows]
                for b in truth:
                    left = (b.x - b.w / 2.) * sized.shape[1]
                    width = b.w * sized.shape[1]
                    top = (b.y - b.h / 2.) * sized.shape[0]
                    height = b.h * sized.shape[0]
                    roi(left, top, width, height)
                    roi = roi & img_rect
                    dst[roi[0]:roi[0] + roi[2], roi[1]:roi[1] + roi[3]] = sized[roi[0]:roi[0] + roi[2],
                                                                          roi[1]:roi[1] + roi[3]]

            sized = dst

        if gaussian_noise:
            noise = np.array(sized.shape)
            gaussian_noise = min(gaussian_noise, 127)
            gaussian_noise = max(gaussian_noise, 0)
            cv2.randn(noise, 0, gaussian_noise)  # mean and variance
            sized = sized + noise
    except:
        print("OpenCV can't augment image: " + str(w) + " x " + str(h))
        sized = mat

    return sized


def filter_truth(bboxes, dx, dy, sx, sy, xd, yd):
    '''
    Input:
        bboxes      :   nx5,Bbox在裁剪图像中的边界框
        dx          :   裁剪图像复制到混合图像的区域的左上角在裁剪图像坐标系下的x坐标
        dy          :   裁剪图像复制到混合图像的区域的左上角在裁剪图像坐标系下的y坐标
        sx          :   裁剪图像复制到混合图像的区域的宽
        sy          :   裁剪图像复制到混合图像的区域的高
        xd          :   裁剪图像复制到混合图像的区域的左上角在混合图像坐标系下的x坐标
        yd          :   裁剪图像复制到混合图像的区域的左上角在混合图像坐标系下的y坐标
    Output:
        bboxes      :   nx5,Bbox在4张图像混合图像中的边界框,[x1,y1,x2,y2]
    '''
    bboxes[:, 0] -= dx
    bboxes[:, 2] -= dx
    bboxes[:, 1] -= dy
    bboxes[:, 3] -= dy

    bboxes[:, 0] = np.clip(bboxes[:, 0], 0, sx)
    bboxes[:, 2] = np.clip(bboxes[:, 2], 0, sx)

    bboxes[:, 1] = np.clip(bboxes[:, 1], 0, sy)
    bboxes[:, 3] = np.clip(bboxes[:, 3], 0, sy)

    out_box = list(np.where(((bboxes[:, 1] == sy) & (bboxes[:, 3] == sy)) |
                            ((bboxes[:, 0] == sx) & (bboxes[:, 2] == sx)) |
                            ((bboxes[:, 1] == 0) & (bboxes[:, 3] == 0)) |
                            ((bboxes[:, 0] == 0) & (bboxes[:, 2] == 0)))[0])
    list_box = list(range(bboxes.shape[0]))
    for i in out_box:
        list_box.remove(i)
    bboxes = bboxes[list_box]

    bboxes[:, 0] += xd
    bboxes[:, 2] += xd
    bboxes[:, 1] += yd
    bboxes[:, 3] += yd

    return bboxes


def blend_truth_mosaic(out_img, img, bboxes, w, h, cut_x, cut_y, i_mixup,
                       left_shift, right_shift, top_shift, bot_shift):
    '''
    Output:
        out_img         :   [self.cfg.h, self.cfg.w, 3], 混合了4张个区域的图像，4个区域靠cut_x和cut_y来划分
        bboxes          ：  nx5,    [[x1,y1,x2,y2,cls_id],...]      前4个通道为Bbox在4张图像混合图像中的边界框,[x1,y1,x2,y2]
    '''
    # 考虑到会越界而做的约束处理，因为这个处理可能会包含更多的均值区域进交集区域
    left_shift = min(left_shift, w - cut_x) # 当left_shift > w - cut_x，left_shift + cut_x会越界，所以限定left_shift最大为w - cut_x，相当于left_shift间隔往回缩，包含更多的均值区域进交集区域
    top_shift = min(top_shift, h - cut_y)   # 限定top_shift最大为h - cut_y
    right_shift = min(right_shift, cut_x)   # 限定right_shift最大为cut_x
    bot_shift = min(bot_shift, cut_y)       # 限定bot_shift最大为cut_y

    if i_mixup == 0:# 左上角
        bboxes = filter_truth(bboxes, left_shift, top_shift, cut_x, cut_y, 0, 0)
        out_img[:cut_y, :cut_x] = img[top_shift:top_shift + cut_y, left_shift:left_shift + cut_x]# 将可能包含均值区域的交集区域的左上角复制到out_img的左上角
    if i_mixup == 1:# 右上角
        bboxes = filter_truth(bboxes, cut_x - right_shift, top_shift, w - cut_x, cut_y, cut_x, 0)
        out_img[:cut_y, cut_x:] = img[top_shift:top_shift + cut_y, cut_x - right_shift:w - right_shift]# 将可能包含均值区域的交集区域的右上角复制到out_img的右上角
    if i_mixup == 2:# 左下角
        bboxes = filter_truth(bboxes, left_shift, cut_y - bot_shift, cut_x, h - cut_y, 0, cut_y)
        out_img[cut_y:, :cut_x] = img[cut_y - bot_shift:h - bot_shift, left_shift:left_shift + cut_x]# 将可能包含均值区域的交集区域的左下角复制到out_img的左下角
    if i_mixup == 3:# 右下角
        bboxes = filter_truth(bboxes, cut_x - right_shift, cut_y - bot_shift, w - cut_x, h - cut_y, cut_x, cut_y)
        out_img[cut_y:, cut_x:] = img[cut_y - bot_shift:h - bot_shift, cut_x - right_shift:w - right_shift]# 将可能包含均值区域的交集区域的右下角复制到out_img的右下角

    return out_img, bboxes


def draw_box(img, bboxes):
    for b in bboxes:
        img = cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 1)
    return img


class Yolo_dataset(Dataset):
    def __init__(self, lable_path, cfg, train=True):
        super(Yolo_dataset, self).__init__()
        if cfg.mixup == 2:
            print("cutmix=1 - isn't supported for Detector")
            raise
        elif cfg.mixup == 2 and cfg.letter_box:
            print("Combination: letter_box=1 & mosaic=1 - isn't supported, use only 1 of these parameters")
            raise

        self.cfg = cfg
        self.train = train

        truth = {}
        f = open(lable_path, 'r', encoding='utf-8')
        for line in f.readlines():
            data = line.split(" ")
            truth[data[0]] = []
            for i in data[1:]:
                truth[data[0]].append([int(float(j)) for j in i.split(',')])

        self.truth = truth      # {'image_name':[[x1,y1,x2,y2,cls_id],...]}
        self.imgs = list(self.truth.keys())

    def __len__(self):
        return len(self.truth.keys())

    def __getitem__(self, index):
        if not self.train:
            return self._get_val_item(index)
        img_path = self.imgs[index]
        bboxes = np.array(self.truth.get(img_path), dtype=np.float)     # nx5   [[x1,y1,x2,y2,cls_id],...]
        img_path = os.path.join(self.cfg.dataset_dir, img_path)
        use_mixup = self.cfg.mixup

        # if random.randint(0, 1):
            # use_mixup = 0

        if self.cfg.use_single_iamge:
            use_mixup = 0

        if use_mixup == 3:
            min_offset = 0.2
            cut_x = random.randint(int(self.cfg.w * min_offset), int(self.cfg.w * (1 - min_offset)))# 混合4张图时的分割线的x坐标，定义在输入网络图像坐标系下
            cut_y = random.randint(int(self.cfg.h * min_offset), int(self.cfg.h * (1 - min_offset)))# 混合4张图时的分割线的y坐标，定义在输入网络图像坐标系下

        r1, r2, r3, r4, r_scale = 0, 0, 0, 0, 0
        dhue, dsat, dexp, flip, blur = 0, 0, 0, 0, 0
        gaussian_noise = 0

        out_img = np.zeros([self.cfg.h, self.cfg.w, 3])
        out_bboxes = []

        for i in range(use_mixup + 1):
            if i != 0:
                img_path = random.choice(list(self.truth.keys()))
                bboxes = np.array(self.truth.get(img_path), dtype=np.float)     # nx5   [[x1,y1,x2,y2,cls_id],...]
                img_path = os.path.join(self.cfg.dataset_dir, img_path)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img is None:
                continue
            oh, ow, oc = img.shape
            dh, dw, dc = np.array(np.array([oh, ow, oc]) * self.cfg.jitter, dtype=np.int)

            dhue = rand_uniform_strong(-self.cfg.hue, self.cfg.hue)
            dsat = rand_scale(self.cfg.saturation)
            dexp = rand_scale(self.cfg.exposure)

            pleft = random.randint(-dw, dw)
            pright = random.randint(-dw, dw)
            ptop = random.randint(-dh, dh) # 裁剪后图像的边界在原始图像坐标系下的坐标
            pbot = random.randint(-dh, dh) # 负数代表往图像外偏移，正数代表往图像内偏移，这个偏移将用于裁剪图像和边界框
                                            # 定义在图像坐标系下

            flip = random.randint(0, 1) if self.cfg.flip else 0

            if (self.cfg.blur):
                tmp_blur = random.randint(0, 2)  # 0 - disable, 1 - blur background, 2 - blur the whole image
                if tmp_blur == 0:
                    blur = 0
                elif tmp_blur == 1:
                    blur = 1
                else:
                    blur = self.cfg.blur

            if self.cfg.gaussian and random.randint(0, 1):
                gaussian_noise = self.cfg.gaussian
            else:
                gaussian_noise = 0

            if self.cfg.letter_box:# 调整裁剪后的图片的宽高比和原始输入图像相同
                img_ar = ow / oh
                net_ar = self.cfg.w / self.cfg.h
                result_ar = img_ar / net_ar
                # print(" ow = %d, oh = %d, w = %d, h = %d, img_ar = %f, net_ar = %f, result_ar = %f \n", ow, oh, w, h, img_ar, net_ar, result_ar);
                if result_ar > 1:  # sheight - should be increased
                    oh_tmp = ow / net_ar
                    delta_h = (oh_tmp - oh) / 2
                    ptop = ptop - delta_h
                    pbot = pbot - delta_h
                    # print(" result_ar = %f, oh_tmp = %f, delta_h = %d, ptop = %f, pbot = %f \n", result_ar, oh_tmp, delta_h, ptop, pbot);
                else:  # swidth - should be increased
                    ow_tmp = oh * net_ar
                    delta_w = (ow_tmp - ow) / 2
                    pleft = pleft - delta_w
                    pright = pright - delta_w
                    # printf(" result_ar = %f, ow_tmp = %f, delta_w = %d, pleft = %f, pright = %f \n", result_ar, ow_tmp, delta_w, pleft, pright);

            swidth = ow - pleft - pright # 裁剪后的图像宽度
            sheight = oh - ptop - pbot   # 裁剪后的图像高度

            truth, min_w_h = fill_truth_detection(bboxes, self.cfg.boxes, self.cfg.classes, flip, pleft, ptop, swidth,
                                                  sheight, self.cfg.w, self.cfg.h)
            if (min_w_h / 8) < blur and blur > 1:  # disable blur if one of the objects is too small
                blur = min_w_h / 8

            ai = image_data_augmentation(img, self.cfg.w, self.cfg.h, pleft, ptop, swidth, sheight, flip,
                                         dhue, dsat, dexp, gaussian_noise, blur, truth)

            if use_mixup == 0:  # 如果不和其他图混合
                out_img = ai
                out_bboxes = truth
            if use_mixup == 1:  # 如果两张图混合
                if i == 0:
                    old_img = ai.copy()
                    old_truth = truth.copy()
                elif i == 1:
                    out_img = cv2.addWeighted(ai, 0.5, old_img, 0.5)
                    out_bboxes = np.concatenate([old_truth, truth], axis=0)
            elif use_mixup == 3:# 如果四张图混合
                if flip:# 因为随着裁剪图像的水平翻转，用于定义裁剪图像的左右边界也要交换，这样才能正确定义裁剪区域中的交集区域在裁剪图像坐标系下的位置
                    tmp = pleft
                    pleft = pright
                    pright = tmp

                # max(0, (-int(pleft) * self.cfg.w / swidth))代表交集在裁剪区域下的偏移，或者说是交集的上下左右边界和裁剪区域上下左右的边界间的间隔
                # left_shift、top_shift、right_shift、bot_shift分别表示交集区域左、上、右、下边界与裁剪区域左、上、右、下边界的间隔
                #### 下面的这个最外层min是多余的
                # left_shift = int(min(cut_x, max(0, (-int(pleft) * self.cfg.w / swidth))))   # min保证left_shift最大为cut_x，当间隔超过cut_x时，相当于一部分被设为原始图像均值的区域被划分进了交集区域
                # top_shift = int(min(cut_y, max(0, (-int(ptop) * self.cfg.h / sheight))))    # min保证top_shift最大为cut_y，当间隔超过cut_y时，相当于一部分被设为原始图像均值的区域被划分进了交集区域

                # right_shift = int(min((self.cfg.w - cut_x), max(0, (-int(pright) * self.cfg.w / swidth))))# min保证right_shift最大为self.cfg.w - cut_x，当间隔超过self.cfg.w - cut_x时，相当于一部分被设为原始图像均值的区域被划分进了交集区域
                # bot_shift = int(min(self.cfg.h - cut_y, max(0, (-int(pbot) * self.cfg.h / sheight))))     # min保证bot_shift最大为self.cfg.h - cut_y，当间隔超过self.cfg.h - cut_y时，相当于一部分被设为原始图像均值的区域被划分进了交集区域

                left_shift = int(max(0, (-int(pleft) * self.cfg.w / swidth)))   # min保证left_shift最大为cut_x，当间隔超过cut_x时，相当于一部分被设为原始图像均值的区域被划分进了交集区域
                top_shift = int(max(0, (-int(ptop) * self.cfg.h / sheight)))    # min保证top_shift最大为cut_y，当间隔超过cut_y时，相当于一部分被设为原始图像均值的区域被划分进了交集区域

                right_shift = int(max(0, (-int(pright) * self.cfg.w / swidth)))# min保证right_shift最大为self.cfg.w - cut_x，当间隔超过self.cfg.w - cut_x时，相当于一部分被设为原始图像均值的区域被划分进了交集区域
                bot_shift = int(max(0, (-int(pbot) * self.cfg.h / sheight))) 

                out_img, out_bbox = blend_truth_mosaic(out_img, ai, truth.copy(), self.cfg.w, self.cfg.h, cut_x,
                                                       cut_y, i, left_shift, right_shift, top_shift, bot_shift)
                out_bboxes.append(out_bbox)
                # print(img_path)
        if use_mixup == 3:
            out_bboxes = np.concatenate(out_bboxes, axis=0)     # 4nx5
        out_bboxes1 = np.zeros([self.cfg.boxes, 5])             # cfg.boxes为设定的最大Bbox数
        out_bboxes1[:min(out_bboxes.shape[0], self.cfg.boxes)] = out_bboxes[:min(out_bboxes.shape[0], self.cfg.boxes)]# 将单张图像的Bbox数限定在cfg.boxes内
        
        # out_img       ：[self.cfg.h, self.cfg.w, 3]，将要输入网络的图
        # out_bboxes1   ：nx5, [[x1,y1,x2,y2,cls_id],...]，定义在网络输入图像的坐标系下
        return out_img, out_bboxes1

    def _get_val_item(self, index):
        """
        """
        img_path = self.imgs[index]# train.txt中的图片名
        bboxes_with_cls_id = np.array(self.truth.get(img_path), dtype=np.float)
        img = cv2.imread(os.path.join(self.cfg.dataset_dir, img_path))
        # img_height, img_width = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, (self.cfg.w, self.cfg.h))
        # img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
        num_objs = len(bboxes_with_cls_id)
        target = {}
        # boxes to coco format
        boxes = bboxes_with_cls_id[...,:4]
        boxes[..., 2:] = boxes[..., 2:] - boxes[..., :2]  # box width, box height
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)   # nx4
        target['labels'] = torch.as_tensor(bboxes_with_cls_id[...,-1].flatten(), dtype=torch.int64) # n
        target['image_id'] = torch.tensor([get_image_id(img_path)])
        target['area'] = (target['boxes'][:,3])*(target['boxes'][:,2])  # nx1
        target['iscrowd'] = torch.zeros((num_objs,), dtype=torch.int64)
        return img, target


def get_image_id(filename:str) -> int:
    """
    Convert a string to a integer.
    Make sure that the images and the `image_id`s are in one-one correspondence.
    There are already `image_id`s in annotations of the COCO dataset,
    in which case this function is unnecessary.
    For creating one's own `get_image_id` function, one can refer to
    https://github.com/google/automl/blob/master/efficientdet/dataset/create_pascal_tfrecord.py#L86
    or refer to the following code (where the filenames are like 'level1_123.jpg')
    >>> lv, no = os.path.splitext(os.path.basename(filename))[0].split("_")
    >>> lv = lv.replace("level", "")
    >>> no = f"{int(no):04d}"
    >>> return int(lv+no)
    """
    raise NotImplementedError("Create your own 'get_image_id' function")
    lv, no = os.path.splitext(os.path.basename(filename))[0].split("_")
    lv = lv.replace("level", "")
    no = f"{int(no):04d}"
    return int(lv+no)


if __name__ == "__main__":
    from cfg import Cfg
    import matplotlib.pyplot as plt

    random.seed(2020)
    np.random.seed(2020)
    Cfg.dataset_dir = "F:\AAE_D2C_YOLO\YOLOv4-tiny\YOLO-tiny/training_data\mesh_01"
    dataset = Yolo_dataset(Cfg.train_label, Cfg)

    use_single_iamge = True
    for i in range(100):
        out_img, out_bboxes = dataset.__getitem__(i)
        a = draw_box(out_img.copy(), out_bboxes.astype(np.int32))
        # plt.imshow(a.astype(np.int32))
        # plt.show()
        cv2.imshow('1', a.astype(np.uint8))
        key = cv2.waitKey(0)
        if key == 27:
            break
