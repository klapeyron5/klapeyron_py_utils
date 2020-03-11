from klapeyron_py_utils.tensorflow.imports import import_tensorflow
tf = import_tensorflow(3)
from lecorbusier.nets.configs.model_config import Model_Config
from lecorbusier.nets.configs.model_train_config import Model_Train_Config


class Base(tf.Module):
    bs = (None,)

    NAME = 'Base'

    def __init__(self, model_config: Model_Config, train_config: Model_Train_Config):
        super(Base, self).__init__()

        self.get_name = tf.function(
            lambda: self.NAME,
            input_signature=[])
        self.get_model_config = tf.function(
            model_config.get_config,
            input_signature=[])
        self.get_train_config = tf.function(
            train_config.get_config,
            input_signature=[])
