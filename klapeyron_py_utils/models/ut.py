import tensorflow as tf
import numpy as np
from klapeyron_py_utils.models.configs.model_train_config import Model_Train_Config
from klapeyron_py_utils.models.configs.model_config import Model_Config
from klapeyron_py_utils.models.SEResNet34_v2 import SEResNet34_v2
from klapeyron_py_utils.metrics.metrics import softmax_loss, softmax_weighted_loss


def ut_0():
    input_shape = (224, 224, 2)
    model_config = Model_Config(input_shape=input_shape)
    trn_config = Model_Train_Config(reg_l2_beta=0.001, dropout_drop_prob=0.2)

    m = SEResNet34_v2(model_config=model_config, train_config=trn_config)
    print(len(m.trainable_variables))
    tmp_dir = '.tmp/'
    tf.saved_model.save(m, tmp_dir)
    m = tf.saved_model.load(tmp_dir)
    import shutil
    shutil.rmtree(tmp_dir)

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001)

    bs = 32
    x = np.random.uniform(0, 1, (bs,) + input_shape).astype(np.float32)

    y = []
    one_hot = [
        [0, 1],
        [1, 0]
    ]
    for i in range(bs):
        y.append(one_hot[np.random.randint(2)])
    y = np.array(y).astype(np.float32)

    weights = np.array([1, 2], dtype=np.float32)

    with tf.GradientTape() as tape:
        out = m.get_logits(x, True)
        loss = softmax_weighted_loss(y, out, weights) + m.reg_loss() * m.reg_l2_beta
        loss_ = softmax_loss(y, out) + m.reg_loss() * m.reg_l2_beta
        predicts = tf.nn.softmax(out)
    vars = tape.watched_variables()
    print(len(vars))
    grads = tape.gradient(loss, vars)
    optimizer.apply_gradients(list(zip(grads, vars)))
    loss = loss.numpy()
    print(loss)
    print(loss_)


if __name__ == '__main__':
    ut_0()
