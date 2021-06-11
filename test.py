import matplotlib.pyplot as plt
import pickle
from torchvision import models

with open("./check/origin.pkl", "rb") as f:
    train_list = pickle.load(f)


loss1 = []
loss2 = []
loss3 = []

for l in train_list:
    loss1.append(l[0])
    loss2.append(l[1])
    loss3.append(l[2])
begin = 0
epoch = range(0, 5 * len(train_list), 5)
fig = plt.subplot(111)
plt.xlabel('Iter')
plt.ylabel('Loss')

l1 = fig.plot(epoch, loss1, color='r', linestyle='-')
l2 = fig.plot(epoch, loss2, color='g', linestyle='-')
l3 = fig.plot(epoch, loss3, color='b', linestyle='-')


plt.legend(l1+l2+l3, ['Image Loss', 'Content Loss1', 'Content Loss2'])
plt.show()
