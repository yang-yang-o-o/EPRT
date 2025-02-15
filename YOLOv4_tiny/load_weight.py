import torch

def load(model,weightfile,coco=True):

    pretrained_dict = torch.load(weightfile)
    
    Models = dict()
    Models['0conv']  = model.conv_1.conv._modules['0']
    Models['0bn']    = model.conv_1.conv._modules['1']
    Models['1conv']  = model.conv_2.conv._modules['0']
    Models['1bn']    = model.conv_2.conv._modules['1']

    Models['2conv']  = model.down1.conv1.conv._modules['0']
    Models['2bn']    = model.down1.conv1.conv._modules['1']
    Models['4conv']  = model.down1.conv2.conv._modules['0']
    Models['4bn']    = model.down1.conv2.conv._modules['1']
    Models['5conv']  = model.down1.conv3.conv._modules['0']
    Models['5bn']    = model.down1.conv3.conv._modules['1']
    Models['7conv']  = model.down1.conv4.conv._modules['0']
    Models['7bn']    = model.down1.conv4.conv._modules['1']

    Models['10conv'] = model.down2.conv1.conv._modules['0']
    Models['10bn']   = model.down2.conv1.conv._modules['1']
    Models['12conv'] = model.down2.conv2.conv._modules['0']
    Models['12bn']   = model.down2.conv2.conv._modules['1']
    Models['13conv'] = model.down2.conv3.conv._modules['0']
    Models['13bn']   = model.down2.conv3.conv._modules['1']
    Models['15conv'] = model.down2.conv4.conv._modules['0']
    Models['15bn']   = model.down2.conv4.conv._modules['1']

    Models['18conv'] = model.down3.conv1.conv._modules['0']
    Models['18bn']   = model.down3.conv1.conv._modules['1']
    Models['20conv'] = model.down3.conv2.conv._modules['0']
    Models['20bn']   = model.down3.conv2.conv._modules['1']
    Models['21conv'] = model.down3.conv3.conv._modules['0']
    Models['21bn']   = model.down3.conv3.conv._modules['1']
    Models['23conv'] = model.down3.conv4.conv._modules['0']
    Models['23bn']   = model.down3.conv4.conv._modules['1']

    Models['26conv'] = model.neek.conv1.conv._modules['0']
    Models['26bn']   = model.neek.conv1.conv._modules['1']
    Models['27conv'] = model.neek.conv2.conv._modules['0']
    Models['27bn']   = model.neek.conv2.conv._modules['1']
    Models['32conv'] = model.neek.conv3.conv._modules['0']
    Models['32bn']   = model.neek.conv3.conv._modules['1']

    Models['28conv'] = model.head.conv1.conv._modules['0']
    Models['28bn']   = model.head.conv1.conv._modules['1']
    Models['29conv'] = model.head.conv2.conv._modules['0']
    Models['35conv'] = model.head.conv3.conv._modules['0']
    Models['35bn']   = model.head.conv3.conv._modules['1']
    Models['36conv'] = model.head.conv4.conv._modules['0']

        

    Models['0conv'].weight.data.copy_(pretrained_dict['models.0.conv1.weight'])
    Models['0bn'  ].weight.data.copy_                          (pretrained_dict['models.0.bn1.weight'])
    Models['0bn'  ].bias.data.copy_                              (pretrained_dict['models.0.bn1.bias'])
    Models['0bn'  ].running_mean.data.copy_              (pretrained_dict['models.0.bn1.running_mean'])
    Models['0bn'  ].running_var.data.copy_                (pretrained_dict['models.0.bn1.running_var'])
    Models['0bn'  ].num_batches_tracked.data.copy_(pretrained_dict['models.0.bn1.num_batches_tracked'])

    Models['1conv'].weight.data.copy_(pretrained_dict['models.1.conv2.weight'])
    Models['1bn'  ].weight.data.copy_                          (pretrained_dict['models.1.bn2.weight'])
    Models['1bn'  ].bias.data.copy_                              (pretrained_dict['models.1.bn2.bias'])
    Models['1bn'  ].running_mean.data.copy_              (pretrained_dict['models.1.bn2.running_mean'])
    Models['1bn'  ].running_var.data.copy_                (pretrained_dict['models.1.bn2.running_var'])
    Models['1bn'  ].num_batches_tracked.data.copy_(pretrained_dict['models.1.bn2.num_batches_tracked']) 

    Models['2conv'].weight.data.copy_(pretrained_dict['models.2.conv3.weight'])
    Models['2bn'  ].weight.data.copy_                          (pretrained_dict['models.2.bn3.weight']) 
    Models['2bn'  ].bias.data.copy_                              (pretrained_dict['models.2.bn3.bias'])              
    Models['2bn'  ].running_mean.data.copy_              (pretrained_dict['models.2.bn3.running_mean'])      
    Models['2bn'  ].running_var.data.copy_                (pretrained_dict['models.2.bn3.running_var'])        
    Models['2bn'  ].num_batches_tracked.data.copy_(pretrained_dict['models.2.bn3.num_batches_tracked'])

    Models['4conv'].weight.data.copy_(pretrained_dict['models.4.conv4.weight'])
    Models['4bn'  ].weight.data.copy_                          (pretrained_dict['models.4.bn4.weight']) 
    Models['4bn'  ].bias.data.copy_                              (pretrained_dict['models.4.bn4.bias'])
    Models['4bn'  ].running_mean.data.copy_              (pretrained_dict['models.4.bn4.running_mean'])
    Models['4bn'  ].running_var.data.copy_                (pretrained_dict['models.4.bn4.running_var'])
    Models['4bn'  ].num_batches_tracked.data.copy_(pretrained_dict['models.4.bn4.num_batches_tracked'])  

    Models['5conv'].weight.data.copy_(pretrained_dict['models.5.conv5.weight'])
    Models['5bn'  ].weight.data.copy_                          (pretrained_dict['models.5.bn5.weight']) 
    Models['5bn'  ].bias.data.copy_                              (pretrained_dict['models.5.bn5.bias'])
    Models['5bn'  ].running_mean.data.copy_              (pretrained_dict['models.5.bn5.running_mean'])
    Models['5bn'  ].running_var.data.copy_                (pretrained_dict['models.5.bn5.running_var'])
    Models['5bn'  ].num_batches_tracked.data.copy_(pretrained_dict['models.5.bn5.num_batches_tracked']) 

    Models['7conv'].weight.data.copy_(pretrained_dict['models.7.conv6.weight'])
    Models['7bn'  ].weight.data.copy_                          (pretrained_dict['models.7.bn6.weight']) 
    Models['7bn'  ].bias.data.copy_                              (pretrained_dict['models.7.bn6.bias'])
    Models['7bn'  ].running_mean.data.copy_              (pretrained_dict['models.7.bn6.running_mean'])
    Models['7bn'  ].running_var.data.copy_                (pretrained_dict['models.7.bn6.running_var'])
    Models['7bn'  ].num_batches_tracked.data.copy_(pretrained_dict['models.7.bn6.num_batches_tracked'])   


    Models['10conv'].weight.data.copy_(pretrained_dict['models.10.conv7.weight'])
    Models['10bn'  ].weight.data.copy_                          (pretrained_dict['models.10.bn7.weight']) 
    Models['10bn'  ].bias.data.copy_                              (pretrained_dict['models.10.bn7.bias'])
    Models['10bn'  ].running_mean.data.copy_              (pretrained_dict['models.10.bn7.running_mean'])
    Models['10bn'  ].running_var.data.copy_                (pretrained_dict['models.10.bn7.running_var'])
    Models['10bn'  ].num_batches_tracked.data.copy_(pretrained_dict['models.10.bn7.num_batches_tracked'])  

    Models['12conv'].weight.data.copy_(pretrained_dict['models.12.conv8.weight'])
    Models['12bn'  ].weight.data.copy_                          (pretrained_dict['models.12.bn8.weight']) 
    Models['12bn'  ].bias.data.copy_                              (pretrained_dict['models.12.bn8.bias'])
    Models['12bn'  ].running_mean.data.copy_              (pretrained_dict['models.12.bn8.running_mean'])
    Models['12bn'  ].running_var.data.copy_                (pretrained_dict['models.12.bn8.running_var'])
    Models['12bn'  ].num_batches_tracked.data.copy_(pretrained_dict['models.12.bn8.num_batches_tracked'])  

    Models['13conv'].weight.data.copy_(pretrained_dict['models.13.conv9.weight'])
    Models['13bn'  ].weight.data.copy_                          (pretrained_dict['models.13.bn9.weight']) 
    Models['13bn'  ].bias.data.copy_                              (pretrained_dict['models.13.bn9.bias'])
    Models['13bn'  ].running_mean.data.copy_              (pretrained_dict['models.13.bn9.running_mean'])
    Models['13bn'  ].running_var.data.copy_                (pretrained_dict['models.13.bn9.running_var'])
    Models['13bn'  ].num_batches_tracked.data.copy_(pretrained_dict['models.13.bn9.num_batches_tracked'])  

    Models['15conv'].weight.data.copy_(pretrained_dict['models.15.conv10.weight'])
    Models['15bn'  ].weight.data.copy_                          (pretrained_dict['models.15.bn10.weight']) 
    Models['15bn'  ].bias.data.copy_                              (pretrained_dict['models.15.bn10.bias'])
    Models['15bn'  ].running_mean.data.copy_              (pretrained_dict['models.15.bn10.running_mean'])
    Models['15bn'  ].running_var.data.copy_                (pretrained_dict['models.15.bn10.running_var'])
    Models['15bn'  ].num_batches_tracked.data.copy_(pretrained_dict['models.15.bn10.num_batches_tracked'])  


    Models['18conv'].weight.data.copy_(pretrained_dict['models.18.conv11.weight'])
    Models['18bn'  ].weight.data.copy_                          (pretrained_dict['models.18.bn11.weight']) 
    Models['18bn'  ].bias.data.copy_                              (pretrained_dict['models.18.bn11.bias'])
    Models['18bn'  ].running_mean.data.copy_              (pretrained_dict['models.18.bn11.running_mean'])
    Models['18bn'  ].running_var.data.copy_                (pretrained_dict['models.18.bn11.running_var'])
    Models['18bn'  ].num_batches_tracked.data.copy_(pretrained_dict['models.18.bn11.num_batches_tracked'])  

    Models['20conv'].weight.data.copy_(pretrained_dict['models.20.conv12.weight'])
    Models['20bn'  ].weight.data.copy_                          (pretrained_dict['models.20.bn12.weight']) 
    Models['20bn'  ].bias.data.copy_                              (pretrained_dict['models.20.bn12.bias'])
    Models['20bn'  ].running_mean.data.copy_              (pretrained_dict['models.20.bn12.running_mean'])
    Models['20bn'  ].running_var.data.copy_                (pretrained_dict['models.20.bn12.running_var'])
    Models['20bn'  ].num_batches_tracked.data.copy_(pretrained_dict['models.20.bn12.num_batches_tracked']) 

    Models['21conv'].weight.data.copy_(pretrained_dict['models.21.conv13.weight'])
    Models['21bn'  ].weight.data.copy_                          (pretrained_dict['models.21.bn13.weight']) 
    Models['21bn'  ].bias.data.copy_                              (pretrained_dict['models.21.bn13.bias'])
    Models['21bn'  ].running_mean.data.copy_              (pretrained_dict['models.21.bn13.running_mean'])
    Models['21bn'  ].running_var.data.copy_                (pretrained_dict['models.21.bn13.running_var'])
    Models['21bn'  ].num_batches_tracked.data.copy_(pretrained_dict['models.21.bn13.num_batches_tracked'])  

    Models['23conv'].weight.data.copy_(pretrained_dict['models.23.conv14.weight'])
    Models['23bn'  ].weight.data.copy_                          (pretrained_dict['models.23.bn14.weight']) 
    Models['23bn'  ].bias.data.copy_                              (pretrained_dict['models.23.bn14.bias'])
    Models['23bn'  ].running_mean.data.copy_              (pretrained_dict['models.23.bn14.running_mean'])
    Models['23bn'  ].running_var.data.copy_                (pretrained_dict['models.23.bn14.running_var'])
    Models['23bn'  ].num_batches_tracked.data.copy_(pretrained_dict['models.23.bn14.num_batches_tracked'])  


    Models['26conv'].weight.data.copy_(pretrained_dict['models.26.conv15.weight'])
    Models['26bn'  ].weight.data.copy_                          (pretrained_dict['models.26.bn15.weight']) 
    Models['26bn'  ].bias.data.copy_                              (pretrained_dict['models.26.bn15.bias'])
    Models['26bn'  ].running_mean.data.copy_              (pretrained_dict['models.26.bn15.running_mean'])
    Models['26bn'  ].running_var.data.copy_                (pretrained_dict['models.26.bn15.running_var'])
    Models['26bn'  ].num_batches_tracked.data.copy_(pretrained_dict['models.26.bn15.num_batches_tracked']) 

    Models['27conv'].weight.data.copy_(pretrained_dict['models.27.conv16.weight'])
    Models['27bn'  ].weight.data.copy_                          (pretrained_dict['models.27.bn16.weight']) 
    Models['27bn'  ].bias.data.copy_                              (pretrained_dict['models.27.bn16.bias'])
    Models['27bn'  ].running_mean.data.copy_              (pretrained_dict['models.27.bn16.running_mean'])
    Models['27bn'  ].running_var.data.copy_                (pretrained_dict['models.27.bn16.running_var'])
    Models['27bn'  ].num_batches_tracked.data.copy_(pretrained_dict['models.27.bn16.num_batches_tracked']) 

    Models['32conv'].weight.data.copy_(pretrained_dict['models.32.conv19.weight'])
    Models['32bn'  ].weight.data.copy_                          (pretrained_dict['models.32.bn19.weight']) 
    Models['32bn'  ].bias.data.copy_                              (pretrained_dict['models.32.bn19.bias'])
    Models['32bn'  ].running_mean.data.copy_              (pretrained_dict['models.32.bn19.running_mean'])
    Models['32bn'  ].running_var.data.copy_                (pretrained_dict['models.32.bn19.running_var'])
    Models['32bn'  ].num_batches_tracked.data.copy_(pretrained_dict['models.32.bn19.num_batches_tracked']) 


    Models['28conv'].weight.data.copy_(pretrained_dict['models.28.conv17.weight'])
    Models['28bn'  ].weight.data.copy_                          (pretrained_dict['models.28.bn17.weight']) 
    Models['28bn'  ].bias.data.copy_                              (pretrained_dict['models.28.bn17.bias'])
    Models['28bn'  ].running_mean.data.copy_              (pretrained_dict['models.28.bn17.running_mean'])
    Models['28bn'  ].running_var.data.copy_                (pretrained_dict['models.28.bn17.running_var'])
    Models['28bn'  ].num_batches_tracked.data.copy_(pretrained_dict['models.28.bn17.num_batches_tracked'])

    if coco:
        Models['29conv'].weight.data.copy_(pretrained_dict['models.29.conv18.weight'])
        Models['29conv'].bias.data.copy_(pretrained_dict['models.29.conv18.bias'])


    Models['35conv'].weight.data.copy_(pretrained_dict['models.35.conv20.weight'])
    Models['35bn'  ].weight.data.copy_                          (pretrained_dict['models.35.bn20.weight']) 
    Models['35bn'  ].bias.data.copy_                              (pretrained_dict['models.35.bn20.bias'])
    Models['35bn'  ].running_mean.data.copy_              (pretrained_dict['models.35.bn20.running_mean'])
    Models['35bn'  ].running_var.data.copy_                (pretrained_dict['models.35.bn20.running_var'])
    Models['35bn'  ].num_batches_tracked.data.copy_(pretrained_dict['models.35.bn20.num_batches_tracked'])  

    if coco:
        Models['36conv'].weight.data.copy_(pretrained_dict['models.36.conv21.weight'])
        Models['36conv'].bias.data.copy_(pretrained_dict['models.36.conv21.bias'])