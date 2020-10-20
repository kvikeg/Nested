import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import sys

def build_model():
    model = tf.keras.Sequential()
    # turn the shapes into a 1-dimensional array
    model.add(layers.Flatten(input_shape=[28, 28], name='input')) 
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(300, activation='relu', kernel_initializer="he_normal", name="hidden_1"))
    #model.add(layers.BatchNormalization())
    model.add(layers.Dense(100, activation='relu', kernel_initializer="he_normal", name="hidden_2"))
    #model.add(layers.BatchNormalization())
    model.add(layers.Dense(10, activation='softmax', name="output"))
    return model

if __name__ == "__main__":
    # load data
    f_mnist = keras.datasets.fashion_mnist
    (X_train, Y_train), (X_test, Y_test) = f_mnist.load_data()
    class_labels = np.array(["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Bag", "Ankle Boot"])

    if len(sys.argv) > 1:
        # model was passed here
        model = tf.keras.models.load_model(sys.argv[1])
    else:
        model = build_model()

        optimizer = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
        model.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5", save_best_only=True)
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        #history = model.fit(X_train, Y_train, epochs=40, validation_split=5/60, callbacks=[checkpoint_cb, early_stopping_cb])
        history = model.fit(X_train, Y_train, epochs=40, validation_split=5/60)
        model.save("trained_model.h5", overwrite=True)

    print("\n\nEvaluation:\n")
    results = model.evaluate(X_test, Y_test, verbose=1)
    print("test loss, test acc:", results)
    # the model is ready. 
