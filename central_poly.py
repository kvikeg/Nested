import numpy as np
import matplotlib.pyplot as plt
import sys 
import os
import re
import random 

import tensorflow as tf
import keras


from settings import *


def read_weights(model):
    res = {}
    layer_num = 0
    for layer in model.layers:
        weights = layer.get_weights()
        if weights:
            res[layer_num] = {'weights': weights, 'name': layer.name, 'layer': layer}

        if layer.name == 'batch_normalization':
            res[layer_num]['epsilon'] = layer.epsilon

        layer_num += 1

    return res


def parse_file(filename):
    layers = []
    lines = []
    cur_layer = None
    with open(filename, 'r') as reader:
        line = reader.readline()
        while line != '':
            if line.startswith('Printing'):
                # start of a new layer
                if len(lines) > 0 and cur_layer != None:
                    layers.append(lines)
                    lines = []
                    cur_layer = None

                m = re.match("Printing layer: hidden_(\d+)", line)
                if m != None:
                    # new layer
                    cur_layer = m.group(1)
                else:
                    # this is an output layer
                    m = re.match("Printing layer: (.+)", line)
                    if m == None:
                        raise Exception("Cannot happen, as there was a match just a few lines before")
                    cur_layer = m.group(1)
            else:
                lines.append(line)

            line = reader.readline()

        if len(lines) > 0 and cur_layer != None:
            layers.append(lines)
            lines = []
            cur_layer = None

    return layers
    

def simple_max_approximation(x, y):
    return ((x**10 + y**10) / 2.0)**0.1

def get_max(val_list):
    max_index = -1
    index = 0
    max_val = 0.0

    for index in range(val_list):
        if index == 0:
            max_val = val_list[index]
        else:
            max_val = simple_max_approximation(val_list[index], max_val) 
        
    index = 0
    max_index = -1
    closest = max(val_list)
    for val in val_list:
        if abs(val - max_val) < closest:
            max_index = index
            closest = abs(val - max_val)
        index += 1

    return max_index


def run_poly(inputs, layers):
    for i in range(len(inputs)):
        exec("x{0}={1}".format(i, inputs[i]))

    for layeri in layers:
        for unit in layeri:
            exec(unit)

    exec("arr = [c{0}0, c{0}1, c{0}2, c{0}3, c{0}4, c{0}5, c{0}6, c{0}7, c{0}8, c{0}9]".format(5))
    mind = -1
    #mind = eval("arr.index(max(arr))")
    mind = eval("get_max(arr)")
    return mind

# def run_implicit(input, res):
    

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: central_poly.py <network filename in text format>")
        exit(0)

    layers = parse_file(sys.argv[1])

    # get the data
      # load data
    f_mnist = keras.datasets.fashion_mnist
    (X_train, Y_train), (X_test, Y_test) = f_mnist.load_data()
    class_labels = np.array(["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Bag", "Ankle Boot"])

    model = tf.keras.models.load_model("trained_model.h5")

    model_layers = read_weights(model)

    num_test_items = 1000
    diff_results = 0
    model_results = 0
    poly_results = 0
    print("Poly:\tModel:\tY:")
    for i in range(0,num_test_items):
        xind = random.randint(0, len(X_test))
        x = np.array(X_test[xind]).flatten()  
        ans = run_poly(x, layers)
        marr = model.predict(X_test[xind:xind+1])
        if ans == Y_test[xind]:
            poly_results += 1
        if np.argmax(marr) == Y_test[xind]:
            model_results += 1
        if np.argmax(marr) != ans:
            diff_results += 1
        print(ans, "\t", np.argmax(marr), "\t", Y_test[xind])

    print("Poly results: " + str(poly_results/num_test_items))
    print("Model results: " + str(model_results/num_test_items))
    print("Difference between poly and model: " + str(diff_results/num_test_items))
