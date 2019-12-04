from klapeyron_py_utils.types import types


def import_tensorflow(log_level=3):
    """
    Returns imported tensorflow package
    :param log_level: log level of tensorflow
    0     | DEBUG            | Print all messages
    1     | INFO             | Filter out INFO messages
    2     | WARNING          | Filter out INFO & WARNING messages
    3     | ERROR            | Filter out all messages
    :return: tensorflow package
    """
    types.assert_any_int(log_level)
    assert 0 <= log_level <= 3
    log_level_values = ['DEBUG','INFO','WARNING','ERROR']
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(log_level)
    import tensorflow as tf
    assert tf.__version__[0] == '2'
    tf.get_logger().setLevel(log_level_values[log_level])
    return tf
