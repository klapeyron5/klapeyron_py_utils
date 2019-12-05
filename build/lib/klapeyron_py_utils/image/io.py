from klapeyron_py_utils.tensorflow.imports import import_tensorflow
tf = import_tensorflow()


def read_img_as_bytes_ndarray(img_path):
    """
    Return img as base64 byte string and as np.ndarray
    :param img_path:
    :return: (img_bytes, img_ndarray)
    """
    img_bytes = open(img_path,'rb').read()
    img_ndarray = tf.io.decode_image(img_bytes).numpy()
    return img_bytes, img_ndarray
