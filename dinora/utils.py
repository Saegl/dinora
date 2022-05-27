def disable_tensorflow_log():
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf

    tf.get_logger().setLevel("INFO")
