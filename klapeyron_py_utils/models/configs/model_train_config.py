from klapeyron_py_utils.tensorflow.imports import import_tensorflow
tf = import_tensorflow(3)


class Model_Train_Config:
    OPTIMIZER = 'optimizer'
    REG_L2_BETA = 'reg_l2_beta'
    DROPOUT_DROP_PROB = 'dropout_drop_prob'

    def __init__(self, optimizer, reg_l2_beta=0.0, dropout_drop_prob=0.0):
        assert hasattr(optimizer, 'get_config')
        reg_l2_beta = float(reg_l2_beta)
        assert isinstance(reg_l2_beta, float)
        dropout_drop_prob = float(dropout_drop_prob)
        assert isinstance(dropout_drop_prob, float)

        self.optimizer = optimizer
        self.reg_l2_beta = reg_l2_beta
        self.dropout_drop_prob = dropout_drop_prob

    def get_config(self):
        config = {
            self.OPTIMIZER: self.optimizer.get_config(),
            self.REG_L2_BETA: self.reg_l2_beta,
            self.DROPOUT_DROP_PROB: self.dropout_drop_prob,
        }
        return config

    @staticmethod
    def get_config_from_tf(model_train_config: dict):
        model_train_config[Model_Train_Config.OPTIMIZER] = {k:v.numpy() for k,v in model_train_config[Model_Train_Config.OPTIMIZER].items()}
        model_train_config[Model_Train_Config.REG_L2_BETA] = model_train_config[Model_Train_Config.REG_L2_BETA].numpy()
        model_train_config[Model_Train_Config.DROPOUT_DROP_PROB] = model_train_config[Model_Train_Config.DROPOUT_DROP_PROB].numpy()
        return model_train_config
