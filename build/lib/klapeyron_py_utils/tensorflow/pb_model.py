import numpy as np
from klapeyron_py_utils.tensorflow.imports import import_tensorflow
tf = import_tensorflow()


class PB_Model:
    def __init__(self, pb_path):
        self.pb_path = pb_path
        self.model = tf.saved_model.load(self.pb_path)

    def __call__(self, data):
        return self.predict(data)

    def predict(self, data, out_tensor='output'):
        assert isinstance(data, np.ndarray)
        out = self.model.signatures['serving_default'](tf.convert_to_tensor(data))[out_tensor]
        return out
