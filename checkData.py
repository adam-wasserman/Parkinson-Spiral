#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 18:05:18 2020

@author: adamwasserman
"""

import torch

Xc_train = torch.load('/projectnb/riseprac/GroupB/preprocessedData/Xc_train.pt')
Xc_test= torch.load('/projectnb/riseprac/GroupB/preprocessedData/Xc_test.pt')

for i in range(Xc_test.shape[0]):
    for j in range(Xc_train.shape[0]):
        if torch.eq(Xc_test[i],Xc_train[j]):
            print("Match found at test",i,'train',j)

print('If no messages above, the data is different!')