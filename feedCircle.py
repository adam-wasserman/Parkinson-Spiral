#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 22:31:40 2020

@author: adamwasserman
"""

from circleArch import CircleConv
import Dataset
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import os
from statistics import mean
import seaborn as sns

epochs =  1000#remember for circles it's practically multiplied by 4
batch_size = 10
threshold = 0.5
run_num = 4
losses = []
accs = []
precision = []
recall = []
f1 = []

conf_mat = torch.zeros(2,2)

filePath = '/projectnb/riseprac/GroupB/preprocessedData'

X_train = torch.load(os.path.join(filePath,"Xc_train.pt"))
X_test = torch.load(os.path.join(filePath,"Xc_test.pt"))
y_train = torch.load(os.path.join(filePath,"y_train.pt"))
y_test = torch.load(os.path.join(filePath,"y_test.pt"))

dataset = Dataset.Dataset(X_train, y_train)
data_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

# train_meander_loader = torch.DataLoader(train_meanders, batch_size)
# train_spiral_loader = torch.DataLoader(train_spirals, batch_size)
# train_circle_loader = torch.DataLoader(train_circles, batch_size)

device=torch.device('cuda:0')
NN = CircleConv(num_classes=1,size = (238,211)) #hardcoded for now
NN.to(device)

#TODO maybe set these as default values in constructor

optimizer = torch.optim.ASGD(params=NN.parameters(), lr=0.01) #TODO ask about lr
#torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma=0.1, last_epoch=-1) #commented out the learning rate decay // also dropped lr to 0.01
cost_func = nn.BCELoss()

for i in range(epochs):
    temp_accs = []
    temp_losses = []
    for j, (X,y) in enumerate(data_loader):
        current_batch = y.shape[0]
        X = X.to(device)
        y = y.to(device)
        yhat = NN.forward(X).reshape(current_batch) #reshaped to batchsize
        loss = cost_func(yhat, y)
        yhat = (yhat>threshold).float()
        acc = torch.eq(yhat.round(), y).float().mean()  # accuracy
        
        for pred,actual in zip(yhat.tolist(),y.tolist()):
            conf_mat[int(actual),int(pred)] += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        temp_losses.append(loss.data.item()) #was loss.data[0]
        temp_accs.append(acc.data.item()) #was acc.data[0]
        
        if j % 15 == 14:
            print("[{}/{}], loss: {} acc: {}".format(i,
                                                 epochs, np.round(loss.data.item(), 3), np.round(acc.data.item(), 3)))
    losses.append(mean(temp_losses))
    accs.append(mean(temp_accs))
    l_precision = (conf_mat[1,1])/((conf_mat[1,1]) + (conf_mat[0,1]))
    precision.append(l_precision)
    l_recall = (conf_mat[1,1])/((conf_mat[1,1]) + (conf_mat[1,0]))
    recall.append(l_recall)
    f1.append(2* ((l_precision*l_recall)/(l_precision+l_recall)))


x = list(range(len(losses)))

fig = plt.figure()
plt.plot(x,losses,color = 'r')
plt.xlabel('Minibatches')
plt.ylabel('Loss')
plt.savefig('/projectnb/riseprac/GroupB/Images/CircleLoss'+str(run_num)+'.png')

plt.plot(x,accs,color = 'g')
plt.xlabel('Minibatches')
plt.ylabel('Accuracy (dec)')
plt.savefig('/projectnb/riseprac/GroupB/Images/CircleAccuracy'+str(run_num)+'.png')

x = list(range(epochs))
plt.plot(x,precision,color='b',label = 'precision')
plt.plot(x,recall,color='r', label = 'recall')
plt.plot(x,f1,color='k',label = 'f1 score')
plt.legend()

plt.xlabel("Epoch")
plt.ylabel("Score (%)")
plt.savefig('/projectnb/riseprac/GroupB/Images/CircleScores'+str(run_num)+'.png')


sns_plot = sns.heatmap(conf_mat/torch.sum(conf_mat), annot=True,
            fmt='.2%', cmap='Blues')
conf_img = sns_plot.get_figure()    
conf_img.savefig('/projectnb/riseprac/GroupB/Images/CIRCLEconf_mat' + str(run_num)+ '.png')

print('Avg/final loss =',mean(losses),losses[-1])
print('Avg/final accuracy =',mean(accs),accs[-1])
print('Final precision =',precision[-1])
print('Final recall =',recall[-1])
print('Final f1 =',f1[-1])


torch.save(NN.state_dict(),'/projectnb/riseprac/GroupB/CircleState_dict'+str(run_num)+'.pt')
