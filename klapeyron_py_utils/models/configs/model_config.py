from klapeyron_py_utils.tensorflow.imports import TfSetup
tf = TfSetup.import_tensorflow(3)
from klapeyron_py_utils.custom_types.common_types import is_any_int


class Model_Config:
    INPUT_SHAPE = 'input_shape'

    def __init__(self, input_shape):
        assert len(input_shape)
        assert all([is_any_int(x) for x in input_shape])
        self.input_shape = tuple(input_shape)

    def get_config(self):
        config = {
            self.INPUT_SHAPE: self.input_shape
        }
        return config

    @staticmethod
    def get_config_from_tf(model_config: dict):
        model_config[Model_Config.INPUT_SHAPE] = [x.numpy() for x in model_config[Model_Config.INPUT_SHAPE]]
        return model_config
