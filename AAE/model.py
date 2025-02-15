import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json

class Encode(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.zeropad = nn.ZeroPad2d(padding=(1,2,1,2))
        self.conv1 = nn.Conv2d(3,128,5,2,bias=True)

        self.conv2 = nn.Conv2d(128,256,5,2,bias=True)
        self.conv3 = nn.Conv2d(256,512,5,2,bias=True)
        self.conv4 = nn.Conv2d(512,512,5,2,bias=True)
        self.fc = nn.Linear(32768,128,bias=True)
    def forward(self,x):
        x = nn.functional.relu(self.conv1(self.zeropad(x)))
        # x = x.cpu().permute((0,2,3,1))
        # print(x[0,0,1,:])
        
        x = nn.functional.relu(self.conv2(self.zeropad(x)))
        # x = x.cpu().permute((0,2,3,1))
        # print(x[0,0,0,:])

        x = nn.functional.relu(self.conv3(self.zeropad(x)))
        # x = x.cpu().permute((0,2,3,1))
        # print(x[0,0,0,:])

        x = nn.functional.relu(self.conv4(self.zeropad(x)))
        # x = x.cpu().permute((0,2,3,1))
        # print(x[0,0,1,:])

        x = x.permute((0,2,3,1))
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x

class Decode(nn.Module):
    def __init__(self) -> None:
        super().__init__() 
        self.fc = nn.Linear(128,32768,bias=True)
        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        # self.zeropad = nn.ZeroPad2d(padding=(1,2,1,2))
        self.conv1 = nn.Conv2d(512,512,5,1,2,bias=True)
        self.conv2 = nn.Conv2d(512,256,5,1,2,bias=True)
        self.conv3 = nn.Conv2d(256,128,5,1,2,bias=True)
        self.conv4 = nn.Conv2d(128,3,5,1,2,bias=True)
    def forward(self,x):
        x = nn.functional.relu(self.fc(x))
        # print(x[0,-20:])
        x = torch.reshape(x,(-1,8,8,512))
        ## x = torch.reshape(x,(-1,512,8,8))     # reshape成8x8x512   和   reshape成512x8x8再permute得到的结果是不同的
        ## x = x.cpu().permute((0,2,3,1))
        # x = x.cpu()
        # print(x[0,0,0,:])

        x = x.permute((0,3,1,2))
        x = nn.functional.relu(self.conv1(self.up(x)))
        # x = x.cpu().permute((0,2,3,1))
        # print(x[0,0,0,:])

        x = nn.functional.relu(self.conv2(self.up(x)))
        # x = x.cpu().permute((0,2,3,1))
        # print(x[0,0,0,:])

        x = nn.functional.relu(self.conv3(self.up(x)))
        x = torch.sigmoid(self.conv4(self.up(x)))
        return x

class AAE(nn.Module):
    def __init__(self,train=True) -> None:
        super().__init__()
        self._train = train
        self.encode = Encode()
        self.decode = Decode()
        self.embedding_normalized = torch.zeros([0])
        self.embed_obj_bbs_var = torch.zeros([0])

        self.embedding_normalized.float()
        if self._train:
            self.embedding_normalized.cuda()
    def forward(self,x):
        x = self.encode(x)
        # print(x)
        x = self.decode(x)
        return x
    def latent_code(self,x):
        encode = self.encode(x)
        return encode
    def similarity(self,_latent_code):
        cosine_similarity = torch.mm(self.embedding_normalized,_latent_code.T)
        idcs = torch.argmax(cosine_similarity,dim=0)
        if self._train:
            idcs = idcs.cpu()
        idcs = idcs.item()
        # max_ = cosine_similarity[idcs]
        # max__ = cosine_similarity.max()
        obj_bb = self.embed_obj_bbs_var[idcs].squeeze()
        return idcs, obj_bb