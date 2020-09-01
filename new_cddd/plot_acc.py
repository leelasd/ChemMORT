import numpy as np
import matplotlib.pyplot as plt

train_acc = np.load('data/train_acc.npy',allow_pickle=True)

test_acc = np.load('data/test_acc.npy',allow_pickle=True)


train_step = []
train_accuracy = []
for step,acc in train_acc.item().items():
     train_step.append(step)
     train_accuracy.append(acc)




test_step = []
test_accuracy = []
for step, acc in test_acc.item().items():
    test_step.append(step)
    test_accuracy.append(acc)




plt.plot(train_step,train_accuracy,'r',test_step, test_accuracy,'g')
plt.legend(['train','test'])
plt.title('acc')
plt.show()