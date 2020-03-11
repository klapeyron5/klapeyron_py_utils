import os
import numpy as np
from klapeyron_py_utils.dataset.csv import df_append
from klapeyron_py_utils.data_pipe.data_process_pipe import Data_process_pipe
from klapeyron_py_utils.types.common_types import is_any_int


class Data_manager:
    def __init__(self, batch_size, samples_csv_paths, preproc_trn, preproc_val, sample_type, csv_class, start_ep=None, select_from_dataset=None):
        self.__set_csv_class(csv_class)
        self.__get_dataset(samples_csv_paths, sample_type, select_from_dataset)
        self.set_batch_size(batch_size)
        self.__set_folds_files()
        self.__shuffle_trn()
        self.set_val_files()
        self.resume_ep(start_ep)
        self.set_preprocess_trn(preproc_trn)
        self.set_preprocess_val(preproc_val)

    def __set_csv_class(self, csv_class):
        from klapeyron_py_utils.dataset.csv import CSV
        assert isinstance(csv_class, CSV)
        self.CSV = csv_class

    def __get_dataset(self, samples_csv_path, sample_type, select_from_dataset=None):
        def f(csv_path):
            samples_csv = self.CSV.read_data_csv(csv_path, sample_type=sample_type, add_real_path_col=True, check_dataset_hash=False, get_stat=True)
            return samples_csv

        try:
            len(samples_csv_path)
        except Exception:
            self.samples_csv = f(samples_csv_path)
        else:
            if isinstance(samples_csv_path, str):
                self.samples_csv = f(samples_csv_path)
            else:
                self.samples_csv = None
                for csv_path in samples_csv_path:
                    samples_csv = f(csv_path)
                    if self.samples_csv is None:
                        self.samples_csv = samples_csv
                    else:
                        self.samples_csv = df_append(self.samples_csv, samples_csv)

        if select_from_dataset is not None:
            assert callable(select_from_dataset)
            self.samples_csv = select_from_dataset(self.samples_csv)
            self.CSV.check_csv_columns(self.samples_csv, sample_type)

    def __set_folds_files(self):
        self.folds_set = set(self.samples_csv[self.CSV.csv_col_fold].values)
        self.labels_set = set(self.samples_csv[self.CSV.csv_col_label].values)
        self.epoch_files_by_fold = {}
        for fold in self.folds_set:
            fold_csv = self.samples_csv.loc[self.samples_csv[self.CSV.csv_col_fold] == fold]
            self.epoch_files_by_fold[fold] = []
            for label in self.labels_set:
                label_csv = fold_csv.loc[fold_csv[self.CSV.csv_col_label] == label]
                files = label_csv[self.CSV.csv_col_realpath].values
                print('fold', fold, '; len(', label, '):', len(files))
                self.epoch_files_by_fold[fold].append(files)

    def __shuffle_trn(self):
        self.p = 0
        self.b = 0

        files = self.epoch_files_by_fold[self.CSV.FOLD_TRN]
        files = [np.random.permutation(x) for x in files]

        min_len = min([len(x) for x in files])
        self.epoch_files = [x[:min_len] for x in files]

        self.batches_in_ep = min_len // self.bs_h

    def set_batch_size(self, batch_size):
        assert batch_size % 2 == 0
        self.bs = batch_size
        self.bs_h = self.bs // 2  # TODO not binary

        self.batch_labels = np.array([[1, 0]for _ in range(self.bs_h)] + [[0, 1]for _ in range(self.bs_h)])  # TODO

    def get_batch_size(self):
        return self.bs

    def set_val_files(self):
        files_ = self.epoch_files_by_fold[self.CSV.FOLD_VAL]
        files = []
        labels = []
        for files__, label__ in zip(files_, [[1, 0], [0, 1]]):  # TODO labels
            files.append(files__)
            labels.append(np.array([label__ for _ in range(len(files__))]))
        files = np.concatenate(files)
        labels = np.concatenate(labels)


        assert len(files) == len(labels)

        self.val_files = files
        self.val_labels = labels

    def resume_ep(self, start_ep):
        if start_ep is None:
            self.ep = 0
        else:
            assert is_any_int(start_ep)
            assert 0 <= start_ep
            self.ep = start_ep

    def set_preprocess_trn(self, preproc_trn):
        isinstance(preproc_trn, Data_process_pipe)
        self.preproc_trn = preproc_trn

    def set_preprocess_val(self, preproc_val):
        isinstance(preproc_val, Data_process_pipe)
        self.preproc_val = preproc_val

    def get_next_batch_trn(self):
        """
        :return: batch data, batch labels, number of batch (in epoch), number of epoch
        """
        last_p = self.p+self.bs_h
        if last_p > len(self.epoch_files[0]):  # TODO not binary
            self.__shuffle_trn()
            self.ep += 1
            last_p = self.p+self.bs_h
        batch_files = np.concatenate([self.epoch_files[0][self.p:last_p],
                                      self.epoch_files[1][self.p:last_p]])
        batch = []
        for file in batch_files:
            x = self.preproc_trn(file)
            batch.append(x)
        batch = np.array(batch)
        self.b += 1
        self.p = last_p

        assert batch.shape[0] == self.batch_labels.shape[0]
        return batch, self.batch_labels, self.b, self.ep

    def get_val_sample_label(self, batch_size=-1):
        if batch_size == -1:
            batch_size = len(self.val_files)

        def get_data():
            batch_files = self.val_files[p:last_p]
            batch_labels = self.val_labels[p:last_p]
            batch = []
            for file in batch_files:
                x = self.preproc_val(file)
                batch.append(x)
            batch = np.array(batch)
            return batch, batch_labels

        p = 0
        last_p = p + batch_size
        while last_p < len(self.val_files):
            yield get_data()
            p = last_p
            last_p += batch_size

        for i in range(1):
            last_p = None
            yield get_data()


def ut_0():
    d = Data_manager(64, ['D:/labeler_07_Feb_2020/optflow_0_subset_stargazer_0__500ms_trn/data.csv', 'D:/labeler_07_Feb_2020/optflow_0_subset_stargazer_0__500ms_val/data.csv'])

    while True:
        batch, labels, b, ep = d.get_next_batch_trn()
