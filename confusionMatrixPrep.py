import os
import random
import tensorflow as tf
import ctc_utils
import cv2
import numpy as np
import sys

# import for showing the confusion matrix
import itertools
import matplotlib.pyplot as plt
import numpy as np

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

voc = "drive/MyDrive/omrModel/Data/znaki.txt"
corpus = "drive/MyDrive/ValidSet/Data"
predict = "drive/MyDrive/omrModel/ctc_predict.py"

np.set_printoptions(threshold=sys.maxsize)

inputs = []
#7798
for i in range (6000, 7799):
    inputs.append(i)    
    
tf.reset_default_graph()
sess = tf.InteractiveSession()

# Read the dictionary
dict_file = open(voc,'r')
dict_list = dict_file.read().splitlines()
int2word = dict()
word2int = dict()
for word in dict_list:
    word_idx = len(int2word)
    int2word[word_idx] = word
    word2int[word] = word_idx
dict_file.close()
modelint = 45000

sym2r = dict()
for word in dict_list:
    if word.endswith("1"):
        sym2r[word] = 0
    elif word.endswith("2"):
        sym2r[word] = 1 
    elif word.endswith("4"):
        sym2r[word] = 2
    elif word.endswith("8"):
        sym2r[word] = 3
    elif word.endswith("16"):
        sym2r[word] = 4

rejects = ["|", "r1", "r2", "r4", "r8", "r16"]

sym2p = dict()
for word in dict_list:
    if word.find("c'''") != -1:
        sym2p[word] = 16
    elif word.find("b''") != -1:
        sym2p[word] = 15
    elif word.find("a''") != -1:
        sym2p[word] = 14
    elif word.find("g''") != -1:
        sym2p[word] = 13
    elif word.find("f''") != -1:
        sym2p[word] = 12
    elif word.find("e''") != -1:
        sym2p[word] = 11
    elif word.find("d''") != -1:
        sym2p[word] = 10
    elif word.find("c''") != -1:
        sym2p[word] = 9
    elif word.find("b'") != -1:
        sym2p[word] = 8
    elif word.find("a'") != -1:
        sym2p[word] = 7
    elif word.find("g'") != -1:
        sym2p[word] = 6
    elif word.find("f'") != -1:
        sym2p[word] = 5
    elif word.find("e'") != -1:
        sym2p[word] = 4
    elif word.find("d'") != -1:
        sym2p[word] = 3
    elif word.find("c'") != -1:
        sym2p[word] = 2
    elif word.find("b") != -1:
        sym2p[word] = 1
    elif word.find("a") != -1:
        sym2p[word] = 0
    

model = f'drive/MyDrive/ModeleFŁ/m{modelint}/saved_models-{modelint}.meta'

# Restore weights
saver = tf.train.import_meta_graph(model)
saver.restore(sess,model[:-5])

graph = tf.get_default_graph()

input = graph.get_tensor_by_name("model_input:0")
seq_len = graph.get_tensor_by_name("seq_lengths:0")
rnn_keep_prob = graph.get_tensor_by_name("keep_prob:0")
height_tensor = graph.get_tensor_by_name("input_height:0")
width_reduction_tensor = graph.get_tensor_by_name("width_reduction:0")
logits = tf.get_collection("logits")[0]

# Constants that are saved inside the model itself
WIDTH_REDUCTION, HEIGHT = sess.run([width_reduction_tensor, height_tensor])

decoded, _ = tf.nn.ctc_greedy_decoder(logits, seq_len)
confMatrixR = None
confMatrixP = None

for x_in in inputs:
    print(x_in)
    imgpath = f'{corpus}/{x_in}/{x_in}.jpg'

    image = cv2.imread(imgpath, 0)
    image = ctc_utils.resize(image, HEIGHT)
    image = ctc_utils.normalize(image)
    image = np.asarray(image).reshape(1,image.shape[0],image.shape[1],1)

    seq_lengths = [ image.shape[2] / WIDTH_REDUCTION ]

    prediction = sess.run(decoded,
                          feed_dict={
                              input: image,
                              seq_len: seq_lengths,
                              rnn_keep_prob: 1.0,
                          })

    str_predictions = ctc_utils.sparse_tensor_to_strs(prediction)
    predictions = []
    for w in str_predictions[0]:
        predictions.append(int2word[w])

    f = open(f'{corpus}/{x_in}/{x_in}.txt', "r")
    inputt = f.read()
    f.close

    labels = []
    inputtt = inputt.split(" ")
    for x in inputtt:
        if(x != ''):
            labels.append(x)

    rPredictions = []
    rLabels = []
    pPredictions = []
    pLabels = []

    for pred in predictions:
        if(pred != "|"):
            rPredictions.append(sym2r[pred])
        if(pred not in rejects):
            pPredictions.append(sym2p[pred])
    
    for lab in labels:
        if(lab != "|"):
            rLabels.append(sym2r[lab])
        if(lab not in rejects):
            pLabels.append(sym2p[lab])

    if (len(rPredictions) > len(rLabels) or len(rPredictions) < len(rLabels)):
        while(len(rPredictions) > len(rLabels)):
            rLabels.append(5)
        while(len(rLabels) > len(rPredictions)):
            rPredictions.append(5)

    if (len(pPredictions) > len(pLabels) or len(pPredictions) < len(pLabels)):
        while(len(pPredictions) > len(pLabels)):
            pLabels.append(17)
        while(len(pLabels) > len(pPredictions)):
            pPredictions.append(17)

    matrixR = tf.math.confusion_matrix(rLabels, rPredictions, num_classes = 6)

    matrixP = tf.math.confusion_matrix(pLabels, pPredictions, num_classes = 18)

    if(confMatrixR == None):
        confMatrixR = matrixR
    else:
        confMatrixR += matrixR

    if(confMatrixP == None):
        confMatrixP = matrixP
    else:
        confMatrixP += matrixP

print("Results: ")
print(confMatrixR.eval(session=sess))
print(confMatrixP.eval(session=sess))

