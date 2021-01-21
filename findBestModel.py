import os
from difflib import SequenceMatcher
import random
import tensorflow as tf
import ctc_utils
import cv2
import numpy as np

voc = "drive/MyDrive/omrModel/Data/znaki.txt"
corpus = "drive/MyDrive/ValidSet/Data"
predict = "drive/MyDrive/omrModel/ctc_predict.py"

inputs = []
#7798
for i in range (0, 7798):
    inputs.append(i)    
    
tf.reset_default_graph()
sess = tf.InteractiveSession()

# Read the dictionary
dict_file = open(voc,'r')
dict_list = dict_file.read().splitlines()
int2word = dict()
for word in dict_list:
    word_idx = len(int2word)
    int2word[word_idx] = word
dict_file.close()

for modelint in range(10000, 95001, 5000):

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

    results = []
    minres = 10
    maxres = -1

    for x_in in inputs:
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
        output = ""
        for w in str_predictions[0]:
            output += str(int2word[w])
            output += str('\t')

        output.rstrip()

        f = open(f'{corpus}/{x_in}/{x_in}.txt', "r")
        inputt = f.read()
        f.close

        res2 = " "
        for x in inputt:
            res2 += x.strip()

        result = " "
        for x in output:
            result += x.strip()

        finres = SequenceMatcher(None, res2, result).ratio()
        results.append(finres)
        if(finres > maxres):
          maxres = finres
        if(finres < minres):
          minres = finres

        if(x_in % 1000 == 0):
          print("Iter " + str(x_in))

    print("Results: ")
    fin = 0
    no = 0
    for i in results:
        fin+= i
        no+= 1
    print("Average: " + str(fin/no))
    print("Max: " + str(maxres))
    print("Min: " + str(minres))

    f = open("drive/MyDrive/ModeleFŁ/modelres.txt", "a")
    f.write(model + "\n")
    f.write("Average: " + str(fin/no) + "\n")
    f.write("Max: " + str(maxres) + "\n")
    f.write("Min: " + str(minres) + "\n")
    f.close()