import datetime
from tensorflow import keras


class TimeCallback(keras.callbacks.Callback):
    """Stop model training after <hours>"""

    def __init__(self, hours):
        self.endtime = datetime.datetime.now() + datetime.timedelta(hours=hours)

    def on_train_batch_begin(self, batch, logs=None):
        if datetime.datetime.now() > self.endtime:
            self.model.stop_training = True
            self.model.save("keras_model.h5")
            print("Model stopped early by TimeCallback")
            import os  # FIXME after model training is stopped, Python interpreter still running

            os._exit(
                0
            )  # Something wrong with dataset.prefetch, without it, the interpreter stops correctly
