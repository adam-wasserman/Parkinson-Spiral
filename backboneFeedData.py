from backbone import BackboneNN
import Dataset
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

epochs = 200  # what value should we set this
batch_size = 5
threshold = 0.5
run_num = 1
losses = []
accs = []
precision = []
recall = []
f1 = []

conf_mat = torch.zeros(2, 2)

filePath = '/projectnb/riseprac/GroupB'

X_train = torch.load(os.path.join(filePath, "X_train.pt"))
X_test = torch.load(os.path.join(filePath, "X_test.pt"))
y_train = torch.load(os.path.join(filePath, "y_train.pt"))
y_test = torch.load(os.path.join(filePath, "y_test.pt"))

dataset = Dataset.Dataset(X_train, y_train)
data_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

# train_meander_loader = torch.DataLoader(train_meanders, batch_size)
# train_spiral_loader = torch.DataLoader(train_spirals, batch_size)
# train_circle_loader = torch.DataLoader(train_circles, batch_size)

device = torch.device(0)
print('Device available',device)

NN = BackboneNN(num_classes=1, size=(756, 822))  # hardcoded for now
NN.to(device)
print('Sucess!')

# TODO maybe set these as default values in constructor

optimizer = torch.optim.Adam(params=NN.parameters(), lr=0.05)  # TODO ask about lr
torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1, last_epoch=-1)
cost_func = nn.BCELoss()

for i in range(epochs):
    for j, (X, y) in enumerate(data_loader):
        curr_batch = X.shape[0]
        X = X.to(device)
        y = y.to(device)
        yhat = NN.forward(X[:, 0], X[:, 1], X[:, 2]).reshape(curr_batch)  # reshaped to batchsize
        loss = cost_func(yhat, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        yhat = (yhat > threshold).float()
        acc = torch.eq(yhat.round(), y).float().mean()  # accuracy
        
        for pred, actual in zip(yhat.tolist(), y.tolist()):
            conf_mat[int(actual), int(pred)] += 1
        
        losses.append(loss.data.item())  # was loss.data[0]
        accs.append(acc.data.item())  # was acc.data[0]
        if (j + 1) % 25 == 0:
            print("[{}/{}], loss: {} acc: {}".format(i,
                                                     epochs, loss.data.item(), 5, acc.data.item(), 5))
    temp_precision = (conf_mat[1, 1]) / ((conf_mat[1, 1]) + (conf_mat[0, 1]))
    precision.append(temp_precision)
    temp_recall = (conf_mat[1, 1]) / ((conf_mat[1, 1]) + (conf_mat[1, 0]))
    recall.append(temp_recall)
    f1.append(2 * ((temp_precision * temp_recall) / (temp_precision + temp_recall)))

x = list(range(len(losses)))

fig = plt.figure()
plt.plot(x, losses, color='r')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('./images/loss' + str(run_num) + '.png')

plt.plot(x, acc, color='g')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (dec)')
plt.savefig('./images/accuracy' + str(run_num) + '.png')

x = list(range(epochs))
plt.plot(x, precision, color='b', label='precision')
plt.plot(x, recall, color='r', label='recall')
plt.plot(x, f1, color='k', label='f1 score')

plt.xlabel("Epoch")
plt.ylabel("Score (%)")
plt.savefig('./images/scores' + str(run_num) + '.png')

sns_plot = sns.heatmap(conf_mat / np.sum(conf_mat), annot=True,
                       fmt='.2%', cmap='Blues')
sns_plot.savefig('./images/conf_mat' + str(run_num) + '.png')
torch.save('./images/scores.npy', conf_mat)

torch.save(NN.state_dict(), '/projectnb/riseprac/GroupB/state_dict' + str(run_num) + '.pt')


