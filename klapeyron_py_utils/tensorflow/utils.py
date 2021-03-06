from klapeyron_py_utils.tensorflow.imports import TfSetup
tf = TfSetup.import_tensorflow()


def map_arrays(fn, arrays, dtype=tf.float32):
    """
    Logic copied from https://stackoverflow.com/a/40409177/5030761
    :param fn: function (*args)
    :param arrays: union of arrays with the same first axis (like time axis)
    :param dtype: type to be returned from tf.map_fn for each time item
    :return: [fn(*args) for args in arrays.T] - its pseudocode
    """
    indices = tf.range(tf.shape(arrays[0])[0])
    out = tf.map_fn(lambda ii: fn(*[array[ii] for array in arrays]), indices, dtype=dtype)
    return out
