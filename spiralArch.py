#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 22:32:16 2020

@author: adamwasserman
"""

import torch
from torch import nn

class SpiralConv(nn.Module):
  def __init__(self, num_classes,size):
      super(SpiralConv, self).__init__()
      self.size = size
      self.num_classes = num_classes

      self.spiral_nn = nn.Sequential(
          nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=3, stride=2),
          nn.Conv2d(64, 192, kernel_size=5, padding=2),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=3, stride=2),
          nn.Conv2d(192, 384, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=3, stride=2),
          nn.Conv2d(384, 256, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=3, stride=2),
          nn.Conv2d(256, 256, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=3, stride=2),
      )
      
      self.fc_nn = nn.Sequential(
          nn.Dropout(),
          nn.Linear(4096, 4096),
          nn.ReLU(inplace=True),
          nn.Dropout(p = 0.35),
          nn.Linear(4096, 4096),
          nn.ReLU(inplace=True),
          nn.Linear(4096, self.num_classes),
          nn.Sigmoid()
      )

  def forward(self, spirals):
      spirals = self.spiral_nn(spirals)
      spirals = spirals.view(spirals.size(0), -1)

      # now we can concatenate them
      out = self.fc_nn(spirals)
      
      return out

# When we import as a module in another file, we can declare SimpleConv variable
# num_classes is the size of each input sample (how many samples are we breaking training data into?)
#   def simple_conv(pretrained=False, num_classes=2):
#       model = SimpleConv(num_classes, spiral_size=756*786, meander_size=744*822, circle_size=675*720)
#       return model

# dimensions = {"Meander": (744,822), "Spiral":(756,786),"Circle":(675,720)}