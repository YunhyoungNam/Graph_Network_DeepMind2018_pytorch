# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 13:51:43 2022

@author: darkstar0983
"""
import torch
from src import modules
import torch.nn.init as init
from torch.optim import lr_scheduler


def dim_cal_GN(in_dim, out_dim):
    edge_dict=modules._make_default_edge_block_opt(None)
    edge_in_dim = 0
    for k, v in edge_dict.items():
        if v:
            edge_in_dim += in_dim
            
    
    node_dict=modules._make_default_node_block_opt(None, 'sum')
    node_in_dim = 0
    for k, v in node_dict.items():
        if 'use' in k and v:
            if 'edges' in k:
                node_in_dim += out_dim
            else:
                node_in_dim += in_dim
                
            
    global_dict=modules._make_default_global_block_opt(None, 'sum')
    global_in_dim = 0
    for k, v in global_dict.items():
        if 'use' in k and v:   
            if 'globals' in k:                    
                global_in_dim += in_dim 
            else:
                global_in_dim += out_dim 
        
            
    return {'edge_dim' : edge_in_dim,
            'node_dim' : node_in_dim,
            'global_dim' : global_in_dim}        


def dim_cal_GI(in_dim):
    edge_dict=modules._make_default_edge_block_opt(None)
    edge_in_dim = 0
    for k, v in edge_dict.items():
        if v:
            edge_in_dim += in_dim
            
    
    node_dict=modules._make_default_node_block_opt(None, 'sum')
    node_in_dim = 0
    for k, v in node_dict.items():
        if 'use' in k and v:
            node_in_dim += in_dim
                
            
    global_dict=modules._make_default_global_block_opt(None, 'sum')
    global_in_dim = 0
    for k, v in global_dict.items():
        if 'use' in k and v:
            global_in_dim += in_dim 
            
            
    return {'edge_dim' : edge_in_dim,
            'node_dim' : node_in_dim,
            'global_dim' : global_in_dim}      
    
     
def init_weights(net, init_type='normal', init_gain=0.02, debug=False):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if debug:
                print(classname)
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions??????
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.8)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler    
