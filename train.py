from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import L2
from tensorflow.keras.utils import plot_model


from dataset import GamesDataset



def build_model() -> keras.Model:
    base_inputs, base_outputs = build_base_block()
    res_outputs = build_residual_blocks(base_outputs)
    policy_out = build_policy_output(res_outputs)
    value_out = build_value_output(res_outputs)

    model = keras.Model(base_inputs, [policy_out, value_out], name='chess_model')
    model.compile(optimizer='adam', loss=['categorical_crossentropy', 'mean_squared_error'],
                  metrics=['accuracy'])
    return model

def build_base_block():
    # input_shape = (batch, channels, height, columns)
    inputs = layers.Input(shape=(18, 8, 8))
    x = layers.Conv2D(filters=256, kernel_size=5, padding='same',
        data_format='channels_first', use_bias=False, kernel_regularizer=L2(1e-4),
        name='input_Conv2d-256-5')(inputs)
    x = layers.BatchNormalization(axis=1, name='input_batchnorm')(x)
    x = layers.Activation('relu', name='input_relu')(x)
    return inputs, x

def build_residual_blocks(inputs):
    x = inputs
    for i in range(7):
        x = build_residual_block(x, i + 1)
    return x

def build_residual_block(inputs, index):
    res_name = "res"+str(index)
    x = layers.Conv2D(filters=256, kernel_size=3, padding="same",
               data_format="channels_first", use_bias=False, kernel_regularizer=L2(1e-4),
               name=res_name+"_conv1-256-3")(inputs)
    x = layers.BatchNormalization(axis=1, name=res_name+"_batchnorm1")(x)
    x = layers.Activation("relu", name=res_name+"_relu1")(x)
    x = layers.Conv2D(filters=256, kernel_size=3, padding="same",
               data_format="channels_first", use_bias=False, kernel_regularizer=L2(1e-4),
               name=res_name+"_conv2-256-3")(x)
    x = layers.BatchNormalization(axis=1, name="res"+str(index)+"_batchnorm2")(x)
    x = layers.add([x, inputs])
    x = layers.Activation("relu", name=res_name+"_relu2")(x)
    return x


def build_policy_output(inputs):
    x = layers.Conv2D(filters=2, kernel_size=1, data_format="channels_first", use_bias=False, kernel_regularizer=L2(1e-4),
                    name="policy_conv-1-2")(inputs)
    x = layers.BatchNormalization(axis=1, name="policy_batchnorm")(x)
    x = layers.Activation("relu", name="policy_relu")(x)
    x = layers.Flatten(name="policy_flatten")(x)
    x = layers.Dense(1968, kernel_regularizer=L2(1e-4),
                            activation="softmax", name="policy_out")(x)
    return x

def build_value_output(inputs):
    x = layers.Conv2D(filters=4, kernel_size=1, data_format="channels_first", use_bias=False, kernel_regularizer=L2(1e-4),
                      name="value_conv-1-4")(inputs)
    x = layers.BatchNormalization(axis=1, name="value_batchnorm")(x)
    x = layers.Activation("relu", name="value_relu")(x)
    x = layers.Flatten(name="value_flatten")(x)
    x = layers.Dense(256, kernel_regularizer=L2(1e-4),
                    activation="relu", name="value_dense")(x)
    x = layers.Dense(1, kernel_regularizer=L2(1e-4),
                            activation="tanh", name="value_out")(x)
    return x

def main():
    from datetime import datetime
    
    dataset = GamesDataset('feed/games.json', 256)
    model = build_model()

    model.summary()
    plot_model(model, to_file='assets/model.png', show_shapes=True)

    now = datetime.utcnow()
    model_filename = now.strftime(r"%Y-%m-%d--%H:%M")

    model.fit(dataset, epochs=15, batch_size=320, shuffle=True)
    model.save('models/' + model_filename)

if __name__ == '__main__':
    main()
