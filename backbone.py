import torch
from torch import nn

class BackboneNN(nn.Module):
  def __init__(self, num_classes,size):
      super(BackboneNN, self).__init__()
      self.size = size
      self.num_classes = num_classes
      # self.device1 = torch.device('cuda:0')
      self.alexnet = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
      self.modified_net = nn.Sequential(*list(self.alexnet.children())[:-1])

      self.meander_nn = self.modified_net
      self.spiral_nn = self.modified_net
      self.circle_nn = self.modified_net

      self.concat_nn = nn.Sequential(
          nn.Dropout(),
          nn.Linear(27648, 4096),
          nn.ReLU(inplace=True),
          nn.Dropout(p=0.35),
          nn.Linear(4096, 4096),
          nn.ReLU(inplace=True),
          nn.Linear(4096, self.num_classes),
          nn.Sigmoid()
      )

  # self.concat_nn = nn.DataParallel(self.concat_nn)

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