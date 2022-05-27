from tensorflow import keras
from tensorflow.keras import layers  # type: ignore
from tensorflow.keras.regularizers import L2  # type: ignore

# Ignore Pylance reportMissingImports


class ModelConfig:
    cnn_filter_num = 256
    cnn_first_filter_size = 5
    cnn_filter_size = 3
    res_layer_num = 7
    l2_reg = 1e-4


class LightConfig(ModelConfig):
    cnn_filter_num = 128
    cnn_first_filter_size = 3
    cnn_filter_size = 3
    res_layer_num = 10
    l2_reg = 1e-4


def build_model(mc: ModelConfig) -> keras.Model:
    base_inputs, base_outputs = build_base_block(mc)
    res_outputs = build_residual_blocks(base_outputs, mc)
    policy_out = build_policy_output(res_outputs, mc)
    value_out = build_value_output(res_outputs, mc)

    model = keras.Model(base_inputs, [policy_out, value_out], name="chess_model")
    model.compile(
        optimizer="adam",
        loss=["categorical_crossentropy", "mean_squared_error"],
        metrics=["accuracy"],
    )
    return model


def build_base_block(mc: ModelConfig):
    # input_shape = (batch, channels, height, columns)
    inputs = layers.Input(shape=(18, 8, 8))
    x = layers.Conv2D(
        filters=mc.cnn_filter_num,
        kernel_size=mc.cnn_first_filter_size,
        padding="same",
        data_format="channels_first",
        use_bias=False,
        kernel_regularizer=L2(mc.l2_reg),
        name=f"input_Conv2d-{mc.cnn_first_filter_size}-{mc.cnn_filter_num}",
    )(inputs)
    x = layers.BatchNormalization(axis=1, name="input_batchnorm")(x)
    x = layers.Activation("relu", name="input_relu")(x)
    return inputs, x


def build_residual_blocks(inputs, mc: ModelConfig):
    x = inputs
    for i in range(mc.res_layer_num):
        x = build_residual_block(x, i + 1, mc)
    return x


def build_residual_block(inputs, index, mc: ModelConfig):
    res_name = f"res-{index}"
    x = layers.Conv2D(
        filters=mc.cnn_filter_num,
        kernel_size=mc.cnn_filter_size,
        padding="same",
        data_format="channels_first",
        use_bias=False,
        kernel_regularizer=L2(mc.l2_reg),
        name=f"{res_name}_conv2-{mc.cnn_filter_size}-{mc.cnn_filter_num}",
    )(inputs)
    x = layers.BatchNormalization(axis=1, name=f"{res_name}_batchnorm1")(x)
    x = layers.Activation("relu", name=f"{res_name}_relu1")(x)
    x = layers.Conv2D(
        filters=mc.cnn_filter_num,
        kernel_size=mc.cnn_filter_size,
        padding="same",
        data_format="channels_first",
        use_bias=False,
        kernel_regularizer=L2(mc.l2_reg),
        name=f"{res_name}_conv2-256-3",
    )(x)
    x = layers.BatchNormalization(axis=1, name=f"res{index}_batchnorm2")(x)
    x = layers.add([x, inputs])
    x = layers.Activation("relu", name=f"{res_name}_relu2")(x)
    return x


def build_policy_output(inputs, mc: ModelConfig):
    x = layers.Conv2D(
        filters=2,
        kernel_size=1,
        data_format="channels_first",
        use_bias=False,
        kernel_regularizer=L2(mc.l2_reg),
        name="policy_conv-1-2",
    )(inputs)
    x = layers.BatchNormalization(axis=1, name="policy_batchnorm")(x)
    x = layers.Activation("relu", name="policy_relu")(x)
    x = layers.Flatten(name="policy_flatten")(x)
    x = layers.Dense(
        1968, kernel_regularizer=L2(mc.l2_reg), activation="softmax", name="policy_out"
    )(x)
    return x


def build_value_output(inputs, mc: ModelConfig):
    x = layers.Conv2D(
        filters=4,
        kernel_size=1,
        data_format="channels_first",
        use_bias=False,
        kernel_regularizer=L2(1e-4),
        name="value_conv-1-4",
    )(inputs)
    x = layers.BatchNormalization(axis=1, name="value_batchnorm")(x)
    x = layers.Activation("relu", name="value_relu")(x)
    x = layers.Flatten(name="value_flatten")(x)
    x = layers.Dense(
        256, kernel_regularizer=L2(mc.l2_reg), activation="relu", name="value_dense"
    )(x)
    x = layers.Dense(
        1, kernel_regularizer=L2(mc.l2_reg), activation="tanh", name="value_out"
    )(x)
    return x


if __name__ == "__main__":
    model = build_model(ModelConfig)
    model.summary()

    lmodel = build_model(LightConfig)
    lmodel.summary()
