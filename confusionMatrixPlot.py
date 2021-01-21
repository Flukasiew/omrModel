# import for showing the confusion matrix
import itertools
import matplotlib.pyplot as plt
import numpy as np

#function found on https://colab.research.google.com/drive/1ISfhxFDntfOos7cOeT7swduSqzLEqyFn
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
   

   
classes_list = ["1", "2", "4", "8", "16", "0"]

confMatrix = np.array([[3802,20,24,18,4,56],
[37,15140,170,91,26,159],
[29,187,22693,258,108,209],
[44,162,336,24382,276,210],
[12,82,180,260,16613,109],
[6,10,6,9,4,0]])

plt.figure()
plot_confusion_matrix(confMatrix, classes=classes_list)

plt.show()

#metrics calculations

nsize = 18

tp = np.zeros(nsize)
fp = np.zeros(nsize)
fn = np.zeros(nsize)

matrix = np.array([[2269,9,16,6,13,10,7,10,2,1,1,0,1,0,0,0,0,32],
[13,3065,12,24,18,14,14,15,16,2,0,2,0,0,1,0,0,26],
[25,20,3778,22,24,15,21,12,19,3,1,1,0,0,0,0,0,30],
[25,28,30,4501,26,29,20,27,21,4,4,0,0,1,0,0,0,41],
[22,28,36,26,5161,37,32,32,14,3,3,0,0,0,0,0,0,52],
[8,25,30,36,38,5797,36,43,28,8,6,2,2,0,0,0,0,55],
[3,8,30,27,34,45,6244,43,67,28,10,3,4,1,1,0,0,60],
[2,6,5,40,45,54,43,6800,42,40,30,11,4,1,0,0,0,50],
[3,7,4,7,25,33,46,40,6627,37,43,29,5,3,1,0,0,77],
[2,0,5,10,9,26,32,41,29,5496,36,41,25,5,7,0,4,45],
[2,1,0,3,6,3,24,28,32,21,4736,27,32,26,6,0,1,38],
[0,0,0,0,2,1,5,17,22,30,20,3995,13,23,19,3,3,36],
[0,0,0,1,0,1,3,3,16,20,24,16,3215,19,23,11,1,27],
[0,0,0,0,0,2,1,0,3,14,10,15,17,2591,8,22,10,10],
[1,0,0,1,1,2,0,3,0,8,7,9,27,13,2072,9,18,16],
[0,0,0,0,0,0,3,1,4,5,5,2,9,15,5,1588,8,7],
[0,0,0,0,0,0,1,1,2,2,7,3,5,5,14,6,1252,3],
[0,0,3,2,3,1,3,7,5,3,2,2,2,2,0,1,0,0]])

for i in range(0, nsize):
  tp[i] = matrix[i][i]
  for j in range(0, nsize):
    if i != j:
      fp[i] += matrix[i][j]
  for k in range(0, nsize):
    if i != k:
      fn[i] += matrix[k][i]

print("Rhythm")
print(f"TP = {tp}")
print(f"FP = {fp}")
print(f"FN = {fn}")

ttp = np.sum(tp)
tfp = np.sum(fp)
tfn = np.sum(fn)

print(ttp)
print(tfp)
print(tfn)

#micro f1
micf1 = ttp/(ttp + tfp)
print(f"micro f1 = {micf1}")

#macro f1
f1s = np.zeros(nsize - 1)
for l in range(0, nsize - 1):
  f1s[l] = 2 * tp[l] / (2 * tp[l] + fp[l] + fn[l])
print(f1s)
macf1 = np.sum(f1s)/(nsize-1)
print(f"macro f1 = {macf1}")

#weighted f1
totals = np.zeros(nsize - 1)
for m in range(0, nsize - 1):
  for n in range(0, nsize - 1):
    totals[m] += matrix[n][m]
print(totals)
weif1 = 0
for o in range(0, nsize - 1):
  weif1 += f1s[o] * totals[o]
weif1 = weif1 / np.sum(totals)
print(f"weighted f1 = {weif1}")