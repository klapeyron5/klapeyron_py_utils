import os
from klapeyron_py_utils.tensorflow.imports import import_tensorflow
tf = import_tensorflow(3)

from klapeyron_py_utils.types.common_types import is_any_int
import numpy as np
import json

from klapeyron_py_utils.metrics.eer import metrics_EER
from klapeyron_py_utils.models.configs.model_config import Model_Config
from klapeyron_py_utils.models.configs.model_train_config import Model_Train_Config


class Train_Pipeline:
    LOG_TAG_LAST = 'last'
    LOG_TAG_BEST_EER = 'best_eer'
    RESUME_TAGS = {LOG_TAG_LAST, LOG_TAG_BEST_EER}

    def __init__(self, logs_base_dir, csv_paths, batch_size, val_options,
                 preproc_trn_funcs_names, preproc_val_funcs_names, data_manager, model: tf.Module = None, resume_from_log=None,
                 start_checkpoint_path=None, start_ep=0, start_b_gl=0):

        self.init_logs(logs_base_dir)

        self.__resume(resume_from_log, start_checkpoint_path, start_ep, start_b_gl, model)

        self.dm = data_manager(batch_size, csv_paths, val_options, self.ep,
                               preproc_trn_funcs_names=preproc_trn_funcs_names,
                               preproc_val_funcs_names=preproc_val_funcs_names)
        self.batches_in_ep = self.dm.batches_in_ep

        # TODO
        run_info = {
            'csv_paths': csv_paths,
            'batch_size': batch_size,
            'start_checkpoint_path': start_checkpoint_path,
            'ep': self.ep,
            'b_gl': self.b_gl
        }
        self.dump_run_config(run_info)

    def train(self):
        while True:
            self.__deal_per_b()

    def eval_val(self, logs_print=True):
        labels = []
        logits = np.array([]).reshape((0, 2))
        for batch_data, batch_labels in self.dm.get_val_sample_label(batch_size=32):
            labels.extend(batch_labels)
            batch_logits = self.m.get_logits(batch_data, False).numpy()
            logits = np.concatenate([logits, batch_logits])
        labels = np.array(labels)
        predicts = tf.nn.softmax(logits).numpy()

        loss = self.m.loss(labels, logits).numpy()
        eer, thr, (thrs, FPRs, FNRs) = metrics_EER(predicts, labels)

        if logs_print:
            print('---val loss:', loss)
            print('---val eer:', eer)

        return loss, eer

    def __deal_per_ep(self):
        """
        Evals on val set
        Writes val metrics on tb
        Updates checkpoints
        :return:
        """
        if self.ep > self.ep_last:
            self.ep_last = self.ep

            val_loss, val_eer = self.eval_val(logs_print=True)

            self.upd_checkpoints(val_eer)

            with self.writers['ep']['val_eer'].as_default():
                tf.summary.scalar('ep', data=val_eer, step=self.ep)

            with self.writers['batch']['val_loss'].as_default():
                tf.summary.scalar('batch', data=val_loss, step=self.b_gl)
            with self.writers['batch']['val_eer'].as_default():
                tf.summary.scalar('batch', data=val_eer, step=self.b_gl)
        else:
            assert self.ep == self.ep_last

    def __deal_per_b(self):
        batch_data, batch_labels, self.b, self.ep = self.dm.get_next_batch_trn()

        self.__deal_per_ep()

        loss, acc = self.m.train_step(batch_data, batch_labels)
        loss = loss.numpy()
        acc = acc.numpy()

        self.b_gl += 1

        with self.writers['batch']['trn_loss'].as_default():
            tf.summary.scalar('batch', data=loss, step=self.b_gl)

        print(round(loss, 3), round(acc, 2), self.b, self.ep)

    def __resume(self, resume_from_log=None, start_checkpoint_path=None, start_ep=0, start_b_gl=0, model: tf.Module = None):
        if resume_from_log is None:
            self.__resume_custom(start_checkpoint_path, start_ep, start_b_gl, model=model)
        else:
            self.__resume_log(log_tag=resume_from_log, model=model)

    def __resume_log(self, log_tag, model):
        """
        Resumes from log_tag checkpoint
        :param log_tag:
        :return:
        """
        if log_tag == self.LOG_TAG_LAST:
            start_checkpoint_path = self.checkpoint_path_last
        elif log_tag == self.LOG_TAG_BEST_EER:
            start_checkpoint_path = self.checkpoint_path_best_eer
        else:
            raise Exception('wrong log_tag')
        errmsg = str(start_checkpoint_path) + ' should exists as dir'
        assert os.path.isdir(start_checkpoint_path), errmsg
        self.checkpoints_info_load()
        start_ep = self.checkpoints_info[log_tag]['ep']
        start_b_gl = self.checkpoints_info[log_tag]['batch']
        self.__resume_custom(start_checkpoint_path, start_ep, start_b_gl, model=model)

    def __resume_custom(self, start_checkpoint_path=None, start_ep=0, start_b_gl=0, eval_val_before_trn: bool = True, model: tf.Module = None):
        """
        Resumes from any checkpoint
        Replaces steps on log graphs
        :param start_checkpoint_path:
        :param start_ep:
        :param start_b_gl:
        :param eval_val_before_trn:
        :return:
        """
        self.__init_model(start_checkpoint_path, model)
        # TODO clear extra steps on plots
        assert is_any_int(start_ep)
        assert start_ep >= 0
        assert is_any_int(start_b_gl)
        assert start_b_gl >= 0
        self.ep = start_ep
        self.ep_last = self.ep - int(eval_val_before_trn)
        self.b_gl = start_b_gl

    def __init_model(self, start_checkpoint_path=None, model: tf.Module = None):
        if start_checkpoint_path is None:  # TODO options to init different models
            self.m = model
        else:
            self.m = tf.saved_model.load(start_checkpoint_path)  # TODO check the loaded model is the same as previous

    # logs wrap part --------------------
    LOG_DIR_TB = 'tb'
    LOG_DIR_CHECKPOINTS = 'checkpoints'

    def init_logs(self, logs_base_dir):
        assert os.path.isdir(logs_base_dir)
        self.logs_base_dir = logs_base_dir

        self.tb_dir = os.path.join(self.logs_base_dir, self.LOG_DIR_TB)
        if not os.path.isdir(self.tb_dir):
            os.mkdir(self.tb_dir)

        self.checkpoints_dir = os.path.join(self.logs_base_dir, self.LOG_DIR_CHECKPOINTS)
        if not os.path.isdir(self.checkpoints_dir):
            os.mkdir(self.checkpoints_dir)

        writers = {
            'batch': {
                'trn_loss': None,
                'val_loss': None,
                'val_eer': None,
            },
            'ep': {
                'val_eer': None,
            },
        }
        for type in writers:
            type_char = type[0] + '_'
            for name in writers[type]:
                dir = os.path.join(self.tb_dir, name)
                ext_name = type_char + name
                writer = tf.summary.create_file_writer(dir, name=ext_name, filename_suffix=ext_name)
                writers[type][name] = writer
        self.writers = writers

        self.checkpoint_path_last = os.path.join(self.checkpoints_dir, 'checkpoint_last')
        if not os.path.isdir(self.checkpoint_path_last):
            os.mkdir(self.checkpoint_path_last)

        self.checkpoint_path_best_eer = os.path.join(self.checkpoints_dir, 'checkpoint_opt_eer')
        if not os.path.isdir(self.checkpoint_path_best_eer):
            os.mkdir(self.checkpoint_path_best_eer)

        self.checkpoints_info_json = os.path.join(self.checkpoints_dir, 'checkpoints_info.json')
        self.runs_configs_txt = os.path.join(self.checkpoints_dir, 'runs_configs.txt')

    BEST_EER = 1.0
    checkpoints_info = {
        LOG_TAG_LAST: {'v': None, 'ep': 0, 'batch': 0},
        LOG_TAG_BEST_EER: {'v': BEST_EER, 'ep': None, 'batch': None},
    }

    def upd_checkpoints(self, eer):
        self.upd_checkpoint_LAST()
        self.upd_checkpoint_BEST_EER(eer)
        self.checkpoints_info_dump()

    def checkpoints_info_dump(self):
        with open(self.checkpoints_info_json, 'w') as f:
            ser = json.dumps(self.checkpoints_info, indent=2)
            f.write(ser)

    def checkpoints_info_load(self):
        errmsg = str(self.checkpoints_info_json) + ' should exists as file'
        assert os.path.isfile(self.checkpoints_info_json), errmsg
        with open(self.checkpoints_info_json) as f:
            self.checkpoints_info = json.load(f)

    def upd_checkpoint_LAST(self, eer=None):
        tf.saved_model.save(self.m, self.checkpoint_path_last)
        self.checkpoints_info[self.LOG_TAG_LAST]['ep'] = self.ep
        self.checkpoints_info[self.LOG_TAG_LAST]['batch'] = self.b_gl

        self.m = tf.saved_model.load(self.checkpoint_path_last)

    def upd_checkpoint_BEST_EER(self, eer):
        if eer < self.checkpoints_info[self.LOG_TAG_BEST_EER]['v']:
            tf.saved_model.save(self.m, self.checkpoint_path_best_eer)
            self.checkpoints_info[self.LOG_TAG_BEST_EER]['v'] = eer
            self.checkpoints_info[self.LOG_TAG_BEST_EER]['ep'] = self.ep
            self.checkpoints_info[self.LOG_TAG_BEST_EER]['batch'] = self.b_gl

    def dump_run_config(self, run_info):
        s = ''
        for k, v in run_info.items():
            s += str(k) + ': ' + str(v) + '\n'

        m_name = self.m.get_name().numpy().decode()
        s += 'model: ' + m_name + '\n'

        m_config = Model_Config.get_config_from_tf(self.m.get_model_config())
        for k, v in m_config.items():
            s += str(k) + ': ' + str(v) + '\n'
        m_train_config = Model_Train_Config.get_config_from_tf(self.m.get_train_config())
        for k, v in m_train_config.items():
            s += str(k) + ': ' + str(v) + '\n'

        preproc_trn_config = self.dm.preproc_trn.get_config()
        s += 'preproc trn: ' + str(preproc_trn_config) + '\n'

        preproc_val_config = self.dm.preproc_val.get_config()
        s += 'preproc val: ' + str(preproc_val_config) + '\n'

        self.add_in_file(self.runs_configs_txt, s)

    @staticmethod  # TODO
    def add_in_file(path, obj):
        with open(path, 'a') as file:
            if type(obj) != dict:
                file.write('\n' + str(obj) + '\n')
            else:
                file.write('\n')
                for key in obj:
                    file.write(str(key) + ':' + str(obj[key]) + '\n')
