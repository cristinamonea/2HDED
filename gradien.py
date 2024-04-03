# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 12:07:03 2022

@author: CEOSpace
"""

import os
import time
import torch
import torch.nn as nn
from collections import OrderedDict
from ipdb import set_trace as st

from math import sqrt
from tqdm import tqdm
# from .train_model import TrainModel
# from networks import networks

# import util.pytorch_ssim as pytorch_ssim

from sklearn.metrics import confusion_matrix
# import util.semseg.metrics.raster as metrics
import numpy as np

import pytorch_ssim
from skimage import measure
from pytorch import sobel


get_gradient = sobel.Sobel().cuda()
# output = torch.unsqueeze(outG[i_task][:,0,:,:], dim=1)
# depth = torch.unsqueeze(target, dim=1)
output = [[1, 2, 1,2], [3, 4, 3,4],[2, 1, 2,1],[3, 4, 3,4]] 
depth =  [[1, 2, 3,4], [4, 3, 2,1],[2, 3, 4,5],[5, 4, 3,2]]
                # print(output.size())
                # print(depth.size())
# output = torch.tensor(output) 
# output = torch.tensor(output)                
depth_grad = get_gradient(depth)
output_grad = get_gradient(output)
                
depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(depth)
depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(depth)
output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(output)
output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(output)
              
loss_dx = torch.log(torch.abs(output_grad_dx - depth_grad_dx) + 0.5).mean()
loss_dy = torch.log(torch.abs(output_grad_dy - depth_grad_dy) + 0.5).mean()
            