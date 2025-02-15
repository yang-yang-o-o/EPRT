import torch
import numpy as np
from . import model

def load(_model,_weightfile,use_cuda):
    '''
    model       :   pytorch model
    wightfile   :   .npz file
    '''

    weight = np.load(_weightfile)
    a = weight['w0']
    _model.encode.conv1.weight.data.copy_(torch.from_numpy(weight['w0'].transpose((3,2,0,1))))
    b = weight['b0']
    _model.encode.conv1.bias.data.copy_  (torch.from_numpy(weight['b0']))

    _model.encode.conv2.weight.data.copy_(torch.from_numpy(weight['w1'].transpose((3,2,0,1))))
    _model.encode.conv2.bias.data.copy_  (torch.from_numpy(weight['b1']))

    _model.encode.conv3.weight.data.copy_(torch.from_numpy(weight['w2'].transpose((3,2,0,1))))
    _model.encode.conv3.bias.data.copy_  (torch.from_numpy(weight['b2']))

    _model.encode.conv4.weight.data.copy_(torch.from_numpy(weight['w3'].transpose((3,2,0,1))))
    _model.encode.conv4.bias.data.copy_  (torch.from_numpy(weight['b3']))

    _model.encode.fc.weight.data.copy_(torch.from_numpy(weight['dense_w0'].transpose((1,0))))
    _model.encode.fc.bias.data.copy_  (torch.from_numpy(weight['dense_b0']))



    _model.decode.fc.weight.data.copy_(torch.from_numpy(weight['dense_w1'].transpose((1,0))))
    _model.decode.fc.bias.data.copy_  (torch.from_numpy(weight['dense_b1']))

    _model.decode.conv1.weight.data.copy_(torch.from_numpy(weight['w4'].transpose((3,2,0,1))))
    _model.decode.conv1.bias.data.copy_  (torch.from_numpy(weight['b4']))

    _model.decode.conv2.weight.data.copy_(torch.from_numpy(weight['w5'].transpose((3,2,0,1))))
    _model.decode.conv2.bias.data.copy_  (torch.from_numpy(weight['b5']))

    _model.decode.conv3.weight.data.copy_(torch.from_numpy(weight['w6'].transpose((3,2,0,1))))
    _model.decode.conv3.bias.data.copy_  (torch.from_numpy(weight['b6']))

    _model.decode.conv4.weight.data.copy_(torch.from_numpy(weight['w7'].transpose((3,2,0,1))))
    _model.decode.conv4.bias.data.copy_  (torch.from_numpy(weight['b7']))

    _model.embedding_normalized = torch.from_numpy(weight['embedding_normalized'])
    if use_cuda:
        _model.embedding_normalized = _model.embedding_normalized.float().cuda()
    _model.embed_obj_bbs_var = torch.from_numpy(weight['embed_obj_bbs_var'])#.float().cuda()

    return

if __name__ == "__main__":
    AAE = model.AAE()
    _weightfile = 'AAE/T_Less_obj_01.npz'
    load(AAE,_weightfile)   
