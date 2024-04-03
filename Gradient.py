# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 12:51:22 2022

@author: CEOSpace
"""

from PIL import Image
import torch.nn as nn
import torch
import numpy as np
from torchvision import transforms
from torch.autograd import Variable
#img = Image.open(’/home/soumya/Documents/cascaded_code_for_cluster/RGB256FullVal/frankfurt_000000_000294_leftImg8bit.png’).convert(‘LA’)
# img = Image.open(’/home/soumya/Downloads/PhotographicImageSynthesis_master/result_256p/final/frankfurt_000000_000294_gtFine_color.png.jpg’).convert(‘LA’)
#img.save(‘greyscale.png’)
# output = [[1, 2, 1,2], [3, 4, 3,4],[2, 1, 2,1],[3, 4, 3,4]] 
# depth =  [[1, 2, 3,4], [4, 3, 2,1],[2, 3, 4,5],[5, 4, 3,2]]
img = np.array([[1, 2, 1,2], [3, 4, 3,4],[2, 1, 2,1],[3, 4, 3,4]]) 
T=transforms.Compose([transforms.ToTensor()])
P=transforms.Compose([transforms.ToPILImage()])

ten=torch.unbind(T(img))
x=ten[0].unsqueeze(0).unsqueeze(0)

a=np.array([[1, 0, -1],[2,0,-2],[1,0,-1]])
conv1=nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
conv1.weight=nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0))

G_x=conv1(Variable(x)).data.view(1,256,512)

b=np.array([[1, 2, 1],[0,0,0],[-1,-2,-1]])
conv2=nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
conv2.weight=nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0))
G_y=conv2(Variable(x)).data.view(1,256,512)

G=torch.sqrt(torch.pow(G_x,2)+ torch.pow(G_y,2))
X=P(G)
# X.save(‘fake_grad.png’)