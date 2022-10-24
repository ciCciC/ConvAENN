from keras import Model, layers
from keras import Sequential


class AnomalyAeModel(Model):

    def __init__(self, units, activation='relu'):
        super(AnomalyAeModel, self).__init__(name='AnomalyAeModel')

        # Encoder block
        self.encoder = Sequential([
            layers.Dense(32, activation=activation),
            layers.Dense(16, activation=activation),
            layers.Dense(8, activation=activation)  # latent space 8x8
        ], name='model_encoder')

        # Decoder block
        self.decoder = Sequential([
            layers.Dense(16, activation=activation),
            layers.Dense(32, activation=activation),
            layers.Dense(units, activation="sigmoid")
        ], name='model_decoder')

    def call(self, inputs, training=None, mask=None):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

    @classmethod
    def from_config(cls, config):
        return cls(**config)
