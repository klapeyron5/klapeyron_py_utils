# TODO use in train.py
from klapeyron_py_utils.tensorflow.imports import import_tensorflow
tf = import_tensorflow(3)

import numpy as np

from klapeyron_py_utils.metrics.metrics import metrics_EER
from matplotlib import pyplot as plt


class Eval_Pipeline:
    LOG_TAG_LAST = 'last'
    LOG_TAG_BEST_EER = 'best_eer'
    RESUME_TAGS = {LOG_TAG_LAST, LOG_TAG_BEST_EER}

    def __init__(self, data_manager, checkpoint_path=None):

        self.__init_model(checkpoint_path)

        self.dm = data_manager

    def eval_val(self, logs_print=True):
        labels = []
        logits = np.array([]).reshape((0, 2))
        for batch_data, batch_labels in self.dm.get_val_sample_label(batch_size=32):
            labels.extend(batch_labels)
            batch_logits = self.m.get_logits(batch_data, False).numpy()
            logits = np.concatenate([logits, batch_logits])
        labels = np.array(labels)
        predicts = tf.nn.softmax(logits).numpy()

        loss = self.m.softmax_loss(labels, logits).numpy()
        eer, thr, (thrs, FPRs, FNRs) = metrics_EER(predicts, labels)

        if logs_print:
            print('---val loss:', loss)
            print('---val eer:', eer)
            print('---val thr:', thr)

            fig, ax = plt.subplots()
            plt.plot(thrs, FPRs, label='FPR')
            plt.plot(thrs, FNRs, label='FNR')
            plt.legend()
            ax.grid()
            plt.show()

        return loss, eer

    def __init_model(self, start_checkpoint_path=None, model: tf.Module = None):
        if start_checkpoint_path is None:  # TODO options to init different models
            self.m = model
        else:
            self.m = tf.saved_model.load(start_checkpoint_path)  # TODO check the loaded model is the same as previous
