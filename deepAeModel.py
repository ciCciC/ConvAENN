from keras import layers
from keras.datasets import mnist, fashion_mnist
import numpy as np
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
import matplotlib.pyplot as plt


class DeepAeModel(Model):
    """
    A Deep Convolutional AutoEncoder Model
    """

    def __init__(self, shape, activation):
        super(DeepAeModel, self).__init__(name='DeepAeModel')

        self.shape = shape
        self.channel = shape[-1]

        filters = 16
        padding = 'same'

        # Encoder block
        self.l1 = layers.Conv2D(filters, (3, 3), activation=activation, padding=padding, strides=2)
        self.p1 = layers.MaxPooling2D((2, 2), padding=padding)

        self.l2 = layers.Conv2D(filters / 2, (3, 3), activation=activation, padding=padding, strides=2)
        self.p2 = layers.MaxPooling2D((2, 2), padding=padding)

        self.l3 = layers.Conv2D(filters / 2, (3, 3), activation=activation, padding=padding, strides=2)
        self.output_encoder = layers.MaxPooling2D((2, 2), padding=padding)

        # Decoder block
        self.l4 = layers.Conv2D(filters / 2, (3, 3), activation=activation, padding=padding, strides=2)
        self.u4 = layers.UpSampling2D((2, 2))

        self.l5 = layers.Conv2D(filters / 2, (3, 3), activation=activation, padding=padding, strides=2)
        self.u5 = layers.UpSampling2D((2, 2))

        self.l6 = layers.Conv2D(filters, (3, 3), activation=activation, padding=padding, strides=2)
        self.u6 = layers.UpSampling2D((2, 2))

        self.output_decoder = layers.Conv2D(self.channel, (3, 3), activation='sigmoid', padding=padding)

    def call(self, inputs, training=None, mask=None):
        # Encoder block
        x = self.l1(inputs)
        x = self.p1(x)
        x = self.l2(x)
        x = self.p2(x)
        x = self.l3(x)
        output_encoded = self.output_encoder(x)

        # Decoder block
        x = self.l4(output_encoded)
        x = self.u4(x)
        x = self.l5(x)
        x = self.u5(x)
        x = self.l6(x)
        x = self.u6(x)
        output_decoded = self.output_decoder(x)

        return output_decoded

    def preprocess(self, array) -> np.array:
        # Normalise pixels
        array = array.astype("float32") / 255.0
        array = np.reshape(array, (len(array), self.shape[0], self.shape[1], self.channel))
        return array


if __name__ == '__main__':
    (train_data, _), (test_data, _) = fashion_mnist.load_data()

    model = DeepAeModel(shape=(28, 28, 1), activation='relu')

    train_data = model.preprocess(train_data)
    test_data = model.preprocess(test_data)

    # Define optimizer for Gradient Descent
    # Define loss for cost
    model.compile(optimizer=Adam(), loss=BinaryCrossentropy())

    history = model.fit(
        x=train_data,
        y=train_data,
        epochs=2,
        batch_size=128,
        shuffle=True
    )

    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.savefig('./data/dcnnae_loss_128.png')
