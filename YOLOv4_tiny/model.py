import torch
from torch import nn
import torch.nn.functional as F
from .yolo_layer import YoloLayer
from .utils import get_region_boxes
import sys

class Upsample_expand(nn.Module):
    def __init__(self, stride=2):
        super(Upsample_expand, self).__init__()
        self.stride = stride

    def forward(self, x):
        assert (x.data.dim() == 4)
        
        x = x.view(x.size(0), x.size(1), x.size(2), 1, x.size(3), 1).\
            expand(x.size(0), x.size(1), x.size(2), self.stride, x.size(3), self.stride).contiguous().\
            view(x.size(0), x.size(1), x.size(2) * self.stride, x.size(3) * self.stride)

        return x

class Conv_Bn_Activation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, bn=True, bias=False):
        super().__init__()
        pad = (kernel_size - 1) // 2

        self.conv = nn.ModuleList()
        if bias:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad))
        else:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False))
        if bn:
            self.conv.append(nn.BatchNorm2d(out_channels))
        if activation == "mish":
            self.conv.append(nn.Mish())
        elif activation == "relu":
            self.conv.append(nn.ReLU(inplace=True))
        elif activation == "leaky":
            self.conv.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == "linear":
            pass
        else:
            print("activate error !!! {} {} {}".format(sys._getframe().f_code.co_filename,
                                                       sys._getframe().f_code.co_name, sys._getframe().f_lineno))

    def forward(self, x):
        for l in self.conv:
            x = l(x)
        return x

class Block(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ch = ch
        self.conv1 = Conv_Bn_Activation(self.ch, self.ch, 3, 1, 'leaky')
        self.conv2 = Conv_Bn_Activation(self.ch//2, self.ch//2, 3, 1, 'leaky')
        self.conv3 = Conv_Bn_Activation(self.ch//2, self.ch//2, 3, 1, 'leaky')
        self.conv4 = Conv_Bn_Activation(self.ch, self.ch, 1, 1, 'leaky')
        self.maxpool = nn.MaxPool2d(2, 2, padding=0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = x1[:, self.ch//2:self.ch]
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        x5 = torch.cat((x4, x3), 1)# 这里一开始写成了（x3,x4），写反了
        x6 = self.conv4(x5)
        x7 = torch.cat((x1, x6), 1)
        x8 = self.maxpool(x7)
        return x8,x6

class Neck(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(512, 512, 3, 1, 'leaky')
        self.conv2 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv3 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        self.upsample = Upsample_expand(2)

    def forward(self,input1,input2):
        x1 = self.conv1(input1)
        x1 = self.conv2(x1)

        x2 = self.conv3(x1)
        x2 = self.upsample(x2)
        x2 = torch.cat((x2, input2), 1)# 这里一开始写成了（x3,x4），写反了

        return x1, x2

class Yolov4Head(nn.Module):
    def __init__(self, n_classes=1,inference=False):
        super().__init__()
        self.inference = inference
        self.conv1 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.out_ch = (4 + 1 + n_classes) * 3   
        self.conv2 = Conv_Bn_Activation(512, self.out_ch, 1, 1, 'linear', bn=False, bias=True)

        self.yolo1 = YoloLayer(
                        anchor_mask=[3, 4, 5], num_classes=n_classes,
                        anchors=[10,14,  23,27,  37,58,  81,82,  135,169,  344,319],
                        num_anchors=6, stride=32)

        self.conv3 = Conv_Bn_Activation(384, 256, 3, 1, 'leaky')
        self.conv4 = Conv_Bn_Activation(256, self.out_ch, 1, 1, 'linear', bn=False, bias=True)

        self.yolo2 = YoloLayer(
                        anchor_mask=[1,2,3], num_classes=n_classes,
                        anchors=[10,14,  23,27,  37,58,  81,82,  135,169,  344,319],
                        num_anchors=6, stride=16)

    def forward(self, input1, input2):
        x1 = self.conv1(input1)
        x1 = self.conv2(x1)

        x2 = self.conv3(input2)
        x2 = self.conv4(x2)

        if self.inference:
            y1 = self.yolo1(x1)         # y1   : [boxes,confs]
                                        # boxes: [batch, num_anchors1 * H1 * W1, 1, 4]
                                        # confs: [batch, num_anchors1 * H1 * W1, num_classes]
            y2 = self.yolo2(x2)         # y2   : [boxes,confs]
                                        # boxes: [batch, num_anchors2 * H2 * W2, 1, 4]
                                        # confs: [batch, num_anchors2 * H2 * W2, num_classes]

            return get_region_boxes([y1, y2])       # [boxes, confs]
                                                    # boxes: [batch, num_anchors1 * H1 * W1 + num_anchors2 * H2 * W2, 1, 4]
                                                    # confs: [batch, num_anchors1 * H1 * W1 + num_anchors2 * H2 * W2, num_classes]

        return [x1, x2]# [Bx255x13x13, Bx255x26x26]

class Yolov4(nn.Module):
    def __init__(self, n_classes=80, inference=False):
        super().__init__()
        self.conv_1 = Conv_Bn_Activation(3, 32, 3, 2, 'leaky')
        self.conv_2 = Conv_Bn_Activation(32, 64, 3, 2, 'leaky')

        self.down1 = Block(64)
        self.down2 = Block(128)
        self.down3 = Block(256)

        self.neek = Neck()

        self.head = Yolov4Head(n_classes,inference)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x,_ = self.down1(x)
        x,_ = self.down2(x)
        x,y = self.down3(x)

        x1, x2 = self.neek(x,y)

        output = self.head(x1,x2)

        return output

if __name__ == "__main__":


    model = Yolov4(inference=False)

    # pretrained_dict = torch.load('1.pth')
    # conv_0 = pretrained_dict['models.0.conv1.weight']
    # model.load_state_dict(pretrained_dict)
    # model.load_state_dict(torch.load('1.pth', map_location=lambda storage, loc: storage))

    # use_cuda = True
    # if use_cuda:
    #     model.cuda()
    # inputimg = torch.rand((1,3,416,416))
    # inputimg = inputimg.cuda()
    # output = model(inputimg)
    # print(output)
    # models = dict()
    # conv0 = model.conv_1.conv._modules['0']
    # models['conv0'] = conv0
    # print(conv0.weight.data-conv_0)
    # models['conv0'].weight.data.copy_(conv_0)
    # print(conv0.weight.data-conv_0)

    # print(output[0].size(),output[1].size())
    import load_weight
    from do_detect import do_detect
    import cv2
    import time
    from utils import plot_boxes_cv2 , load_class_names

    load_weight.load(model, 'yolo-tiny-coco.pth')
    

    ############################################################################################
    # use_cuda = True
    # if use_cuda:
    #     model.cuda()
    # img = cv2.imread('dog.jpg')
    # resized = cv2.resize(img, (416, 416))
    # resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # for i in range(2):  # This 'for' loop is for speed check
    #                 # Because the first iteration is usually longer
    #     boxes = do_detect(model, resized, 0.4, 0.6, use_cuda)  #  [B,bboxes]    bboxes:[[x1,y1,x2,y2,cof,cof,ID],...]


    # namesfile = 'coco.names'

    # class_names = load_class_names(namesfile)
    # plot_boxes_cv2(img, boxes[0], 'predictions.jpg', class_names)
    ############################################################################################


    ############################################################################################
    # model.eval()
    # model.cuda()

    # img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)

    # # img = img.expand(4,-1,-1,-1)

    # img = img.cuda()

    # for i in range(1):
    #     t0 = time.time()
    #     output = model(img)
    #     t1 = time.time()
    #     print(f'{(t1-t0)*1000} ms')
    # [boxes, confs] = output
    # # torch.save(boxes, 'boxes.pt')
    # # torch.save(confs, 'confs.pt')
    # a = 1
    ############################################################################################
    
    ############################################################################################
    # model = model
    # model.conv_1.requires_grad_(False)
    # model.conv_2.requires_grad_(False)
    # model.down1.requires_grad_(False)
    # model.down2.requires_grad_(False)
    # model.down3.requires_grad_(False)
    # model.neek.requires_grad_(False)
    ############################################################################################