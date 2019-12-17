import numpy as np
from klapeyron_py_utils.types.common_types import is_any_int
from klapeyron_py_utils.tensorflow.imports import import_tensorflow
tf = import_tensorflow()


class PB_Model:
    def __init__(self, pb_path):
        self.pb_path = pb_path
        self.model = tf.saved_model.load(self.pb_path)

    def __call__(self, *args, **kwargs):
        return self.batch_predict(*args, **kwargs)

    def predict(self, data, out_tensor='output'):
        assert isinstance(data, np.ndarray)
        out = self.model.signatures['serving_default'](tf.convert_to_tensor(data))[out_tensor].numpy()
        return out

    def batch_predict(self, data, batch_size=128, out_tensor='output'):
        assert isinstance(data, np.ndarray)
        assert is_any_int(batch_size)
        assert batch_size > 0
        out = []
        i = 0
        while (i + 1) * batch_size < len(data):
            batch_out = self.predict(data[i * batch_size:(i + 1) * batch_size], out_tensor=out_tensor)
            out.extend(batch_out)
            i += 1
        batch_out = self.predict(data[i*batch_size:], out_tensor=out_tensor)
        out.extend(batch_out)
        out = np.array(out)
        return out
