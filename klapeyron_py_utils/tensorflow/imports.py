import os
from klapeyron_py_utils.custom_types.common_types import is_any_int


class TfSetup:
    ALL_GPUS = 'all'

    @classmethod
    def set_visible_devices(cls, gpus=ALL_GPUS, only_cpu=False):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if only_cpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(-1)
        elif gpus == cls.ALL_GPUS:
            pass
        elif is_any_int(gpus):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus)
        else:
            assert all([is_any_int(x) and x >= 0 for x in gpus])
            var = ''
            for x in gpus: var += str(x) + ','
            os.environ["CUDA_VISIBLE_DEVICES"] = var

    @classmethod
    def import_tensorflow(cls, log_level=3):
        """
        Returns imported tensorflow package
        :param log_level: log level of tensorflow
        0     | DEBUG            | Print all messages
        1     | INFO             | Filter out INFO messages
        2     | WARNING          | Filter out INFO & WARNING messages
        3     | ERROR            | Filter out all messages
        :return: tensorflow package
        """
        is_any_int(log_level)
        assert 0 <= log_level <= 3
        log_level_values = ['DEBUG', 'INFO', 'WARNING', 'ERROR']
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(log_level)
        import tensorflow as tf
        assert tf.__version__[0] == '2'
        tf.get_logger().setLevel(log_level_values[log_level])
        return tf

    @classmethod
    def list_all_visible_devices(cls, tf):
        gpus = tf.config.get_visible_devices()  # this initializes gpus
        return gpus

    @classmethod
    def list_all_visible_gpus(cls, tf):
        gpus =  tf.config.get_visible_devices('GPU')  # this initializes gpus
        return gpus

    @classmethod
    def set_gpus_memory_growth_true(cls, tf):
        gpus = cls.list_all_visible_gpus(tf)
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)


def test():
    TfSetup.set_visible_devices(gpus=0, only_cpu=False)
    tf = TfSetup.import_tensorflow(3)
    print(TfSetup.list_all_visible_gpus(tf))
    TfSetup.set_gpus_memory_growth_true(tf)
    print(TfSetup.list_all_visible_gpus(tf))


if __name__ == '__main__':
    test()
