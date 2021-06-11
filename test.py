import matplotlib.pyplot as plt
import pickle
from torchvision import models

with open("./check/L1.pkl", "rb") as f:
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
end = 1000
l1 = fig.plot(epoch[:end], loss1[:end], color='r', linestyle='-')
#l2 = fig.plot(epoch[:end], loss2[:end], color='g', linestyle='-')
#l3 = fig.plot(epoch[:end], loss3[:end], color='b', linestyle='-')


plt.legend(l1, ['Image Loss', 'Content Loss1', 'Content Loss2'])
plt.show()
