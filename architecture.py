"""
Created on Tue Aug  4 20:26:23 2020

@author: medha iyer and adamwasserman
"""


import torch
from torch import nn

class SimpleConv(nn.Module):
  def __init__(self, num_classes,size):
      super(SimpleConv, self).__init__()
      self.dim = 0 #dimensions of image after concatenation (set in forward())
      self.size = size
      self.num_classes = num_classes

      self.meander_nn = nn.Sequential(
          nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=3, stride=2),
          nn.Conv2d(64, 192, kernel_size=5, padding=2),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=3, stride=1),
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
      #self.meander_nn = nn.DataParallel(self.meander_nn)

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
      #self.spiral_nn = nn.DataParallel(self.spiral_nn)

      self.circle_nn = nn.Sequential(
          nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=3, stride=2),
          nn.Conv2d(64, 192, kernel_size=5, padding=2),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=3, stride=2),
          nn.Conv2d(192, 384, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=3, stride=1),
          nn.Conv2d(384, 256, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=3, stride=1),
          nn.Conv2d(256, 256, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=3, stride=2),
      )
      #self.circle_nn = nn.DataParallel(self.circle_nn)
      
      #self.concat_nn = nn.DataParallel(self.concat_nn)
      
      self.concat_nn = nn.Sequential(
          nn.Dropout(),
          nn.Linear(19712, 4096),
          nn.ReLU(inplace=True),
          nn.Dropout(p = 0.35),
          nn.Linear(4096, 4096),
          nn.ReLU(inplace=True),
          nn.Linear(4096, self.num_classes),
          nn.Sigmoid()
      )

    

  def forward(self, meanders, spirals, circles):
      meanders = self.meander_nn(meanders)
      meanders = meanders.view(meanders.size(0), -1)

      spirals = self.spiral_nn(spirals)
      spirals = spirals.view(spirals.size(0), -1)

      circles = self.circle_nn(circles)
      circles = circles.view(circles.size(0), -1)

      # now we can concatenate them
      combined = torch.cat((meanders, spirals, circles), dim=1)
      out = self.concat_nn(combined)

      return out

# When we import as a module in another file, we can declare SimpleConv variable
# num_classes is the size of each input sample (how many samples are we breaking training data into?)
#   def simple_conv(pretrained=False, num_classes=2):
#       model = SimpleConv(num_classes, spiral_size=756*786, meander_size=744*822, circle_size=675*720)
#       return model

# dimensions = {"Meander": (744,822), "Spiral":(756,786),"Circle":(675,720)}

if __name__ =='__main__':
    num_classes = 2
    model = SimpleConv(num_classes, meander_size=744*822, spiral_size=756*786, circle_size=675*720)