#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 15:00:23 2020

@author: adamwasserman
"""

import os
import numpy as np #images come as numpy arrays; kept to be safe
import cv2
import torch
import random
"""
File set-up: Have the 6 image folders in a single directory
Pass the directory as the first argument to the preprocess function
All the healthy folders should begin with "Healthy"
and all the patient files with "Patient"
The shape drawn should follow the subject's condition
Naming should be in camel-case (no plural!)
EX: HealthySpiral
"""
#The data below represents the largest row and column size for each category
#dimensions = {"Meander": (744,822), "Spiral":(756,786),"Circle":(675,720)} # no longer used


dim= {"Meander": (561,580), "Spiral" : (678,686), "Circle" : (238,211)}


#def preprocess(inPath,outPath):
"""Uploads data into a numpy array
    parameter: filePath – the path to the Image_Data folder
    returns: a tuple containing a numpy array of the data and an vertical vector
    with the corresponding values
"""
outPath = '/projectnb/riseprac/GroupB/preprocessedData'

meanders = []
spirals = []
circles = []
values = [] # 1 for PD and 0 for Healthy

DATADIR = '/projectnb/riseprac/GroupB'
cat1 = ["Healthy","Patient"]
cat2 = ["Meander","Spiral"]
for health in cat1:
    tag = "H" if health == 'Healthy' else "P"
    size = 38 if health == "Healthy" else 32
    for subject in range(1,size+1):
        pat_meanders = []
        pat_spirals = []
        pat_circles = []
        pat_values = []
        for i in range (1,5):
            delete = False
            temp = []
            for shape in cat2:
                abrev = 'mea' if shape == 'Meander' else 'sp'
        
                path = os.path.join(DATADIR,health+shape)
                img_name = abrev + str(i) + '-' + tag+str(subject)+'.jpg'
                img_array = cv2.imread(os.path.join(path,img_name))
                
                if img_array is None: # look for missing data
                    delete = True
                else: #can only perform if img_array isn't None
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                    temp.append(torch.from_numpy(cv2.resize(img_array,dim[shape])))
                    
            path = os.path.join(DATADIR,health+"Circle")
            img_name = "circA-P"+str(subject)+".jpg"
            img_array = cv2.imread(os.path.join(path,img_name),cv2.COLOR_BGR2RGB)
            
            if img_array is None or delete == True: # datapoints with missing data
                temp.clear()
                continue
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            temp.append(torch.from_numpy(cv2.resize(img_array,dim["Circle"])))#hard-coded for now
            pat_meanders.append(temp[0])
            pat_spirals.append(temp[1])
            pat_circles.append(temp[2])
            pat_values.append(1.0 if health == "Patient" else 0.0)
        meanders.append(pat_meanders)
        spirals.append(pat_spirals)
        circles.append(pat_circles)
        values.append(pat_values)

#shuffle lists in unison

comb = list(zip(meanders,spirals,circles,values))
random.shuffle(comb)
meanders,spirals,circles,values, = zip(*comb)
# [(m1,sp1,ci1,y),(m2,sp2,ci2,y2),..]
# m1,sp1,ci1,y,m2,sp2,ci2,y2

#split the list along the patient dimension
Xm_train,Xs_train,Xc_train,y_train = meanders[13:], spirals[13:], circles[13:],values[13:]
Xm_test,Xs_test,Xc_test,y_test = meanders[:13], spirals[:13], circles[:13],values[:13]

#remove the extra 'dimension' in each list
Xm_train = [j for sublist in Xm_train for j in sublist]
Xs_train = [j for sublist in Xs_train for j in sublist]
Xc_train = [j for sublist in Xc_train for j in sublist]
Xm_test = [j for sublist in Xm_test for j in sublist]
Xs_test = [j for sublist in Xs_test for j in sublist]
Xc_test = [j for sublist in Xc_test for j in sublist]
y_train = [j for sublist in y_train for j in sublist]
y_test = [j for sublist in y_test for j in sublist]

#create tensors

Xm_train = torch.stack(Xm_train).type('torch.FloatTensor')
Xs_train = torch.stack(Xs_train).type('torch.FloatTensor')
Xc_train = torch.stack(Xc_train).type('torch.FloatTensor')
Xm_test = torch.stack(Xm_test).type('torch.FloatTensor')
Xs_test = torch.stack(Xs_test).type('torch.FloatTensor')
Xc_test = torch.stack(Xc_test).type('torch.FloatTensor')
y_train = torch.tensor(y_train).type('torch.FloatTensor')
y_test = torch.tensor(y_test).type('torch.FloatTensor')
#shape: NxRxCOLxC

#normalize data

Xm_train /= 255.0
Xs_train /= 255.0
Xc_train /= 255.0
Xm_test /= 255.0
Xs_test /= 255.0
Xc_test /= 255.0


#Rearrange dimensions so order = NxChannelxRowxCol
Xm_train,Xs_train,Xc_train = Xm_train.permute(0,3,1,2), Xs_train.permute(0,3,1,2), Xc_train.permute(0,3,1,2)
Xm_test,Xs_test,Xc_test = Xm_test.permute(0,3,1,2), Xs_test.permute(0,3,1,2), Xc_test.permute(0,3,1,2)

 
torch.save(Xm_train,os.path.join(outPath,"Xm_train.pt"))
torch.save(Xs_train,os.path.join(outPath,"Xs_train.pt"))
torch.save(Xc_train,os.path.join(outPath,"Xc_train.pt"))
torch.save(Xm_test,os.path.join(outPath,"Xm_test.pt"))
torch.save(Xs_test,os.path.join(outPath,"Xs_test.pt"))
torch.save(Xc_test,os.path.join(outPath,"Xc_test.pt"))
torch.save(y_train,os.path.join(outPath,"y_train.pt"))
torch.save(y_test,os.path.join(outPath,"y_test.pt"))



def getData(filePath):
    """Returns the X_train,X_test,y_train,y_test in that order"""
    
    X_train = torch.load(os.path.join(filePath,"X_train.pt"))
    X_test = torch.load(os.path.join(filePath,"X_test.pt"))
    y_train = torch.load(os.path.join(filePath,"y_train.pt"))
    y_test = torch.load(os.path.join(filePath,"y_test.pt"))
    return X_train,X_test,y_train,y_test
    
#preprocess('/projectnb/riseprac/GroupB','/projectnb/riseprac/GroupB')
