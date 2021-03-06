import numpy as np
import tensorflow as tf
import argparse

from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

def train():
    parser = argparse.ArgumentParser(
        "Trains the autoencoder"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        help="",
    )
    parser.add_argument(
        "-z",
        "--encoded_size",
        type=int
    )
    parser.add_argument(
        "-s",
        "--early-stop",
        type=int
    )
    args = parser.parse_args()

    # defaults
    epochs = args.epochs or 10
    encoded_size = args.encoded_size or 64
    early_stop = args.early_stop or 4

    (x_train, _), (x_test, _) = mnist.load_data()
    _, w, h = x_train.shape

    # autoencoder
    autoencoder = Sequential(name="Autoencoder")
    input_layer = layers.Input(shape=(w * h,))
    encoded_layer = layers.Dense(encoded_size, activation="relu")
    decoded_layer = layers.Dense(w * h, activation="sigmoid")
    autoencoder.add(input_layer)
    autoencoder.add(encoded_layer)
    autoencoder.add(decoded_layer)

    # encoder
    encoder = Sequential([input_layer, encoded_layer], "Encoder")


    # decoder
    decoder = Sequential([layers.Input(shape=(encoded_size,)),
                          decoded_layer],
                         "Decoder")

    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

    # train test reshape & rescale
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    early_stop = EarlyStopping(monitor='val_loss', patience=early_stop)

    # train time
    autoencoder.fit(x_train, x_train,
                    epochs=epochs,
                    shuffle=True,
                    validation_data=(x_test, x_test), callbacks=[early_stop])


# script mode
if __name__ == "__main__":
    train()
