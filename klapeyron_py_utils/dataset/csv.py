import pandas as pd
import os
from klapeyron_py_utils.logs.filesystem import assert_filepath
from tqdm import tqdm
from klapeyron_py_utils.dataset.hash import get_md5_adler32_from_file, get_md5_adler32
from klapeyron_py_utils.logs.json import json_dump, json_load


class CSV:

    GENERAL_INFO_JSON_NAME = 'general_info.json'

    DATASET_DATA_FOLDER_NAME = 'data'

    SAMPLE_FILE = 'file'

    @staticmethod
    def read_data_csv(csv_path: str, sample_type: SAMPLE_FILE, add_real_path_col=True, check_dataset_hash=False, get_stat=False,
                      subset_name_for_general_info_json=None)-> pd.DataFrame:
        """
        Reads data_csv
        Checks dataset folder structure
        Checks all files from data_csv exist in dataset
        :param csv_path: path to data_csv
        :param get_stat: do u wish to get/see data statistics
        :return:
        """
        dataset_path, data_path = CSV.check_dataset_structure(csv_path)

        data_csv = pd.read_csv(csv_path, sep=',', index_col='Unnamed: 0')

        def assert_data_csv():
            CSV.check_csv_columns(data_csv, sample_type)

            # all paths start with 'data/'
            start_path = CSV.DATASET_DATA_FOLDER_NAME + '/'
            errmsg = 'each sample should start from ' + CSV.DATASET_DATA_FOLDER_NAME + '/'
            assert all([x[:5] == start_path for x in data_csv[CSV.csv_col_path]]), errmsg

            # no repeated paths
            assert len(set(data_csv[CSV.csv_col_path].values)) == len(data_csv[CSV.csv_col_path])

            # all paths exist
            realpath = [os.path.join(dataset_path, x) for x in data_csv[CSV.csv_col_path]]
            assert all([os.path.isfile(x) for x in realpath])
            if add_real_path_col:
                # add real paths as new column
                data_csv[CSV.csv_col_realpath] = realpath

        assert_data_csv()

        assert isinstance(check_dataset_hash, bool)
        if check_dataset_hash:
            errmsg = 'add_real_path_col should be True if u wish to check hashes'
            assert add_real_path_col, errmsg
            CSV.__check_dataset_markup_hash(data_csv, csv_path, dataset_path, subset_name_for_general_info_json)

        assert isinstance(get_stat, bool)
        if get_stat:
            print(csv_path+':')
            CSV.get_stat_from_data_csv(data_csv, sample_type)
        return data_csv

    @staticmethod
    def check_dataset_structure(csv_path: str):
        assert assert_filepath(csv_path)
        assert csv_path.endswith('.csv')
        dataset_path = os.path.dirname(csv_path)
        assert os.path.isdir(dataset_path)
        data_path = os.path.join(dataset_path, CSV.DATASET_DATA_FOLDER_NAME)
        assert os.path.isdir(data_path)
        return dataset_path, data_path

    @staticmethod
    def check_csv_columns(csv: pd.DataFrame, sample_type):
        if sample_type == CSV.SAMPLE_FILE:
            assert set(CSV.CSV_COLUMNS_FILE) - set(csv.columns) == set()
        else:
            errmsg = 'Wrong sample type: ' + str(sample_type)
            raise Exception(errmsg)

    @staticmethod
    def __check_dataset_markup_hash(data_csv: pd.DataFrame, origin_data_csv_path: str, dataset_path: str, subset_name: str):
        print('Checking hashes of files in dataset...')
        dataset_general_string = ''
        for ind, item in tqdm(data_csv.iterrows(), total=len(data_csv)):
            assert get_md5_adler32_from_file(item.realpath, onestr=True) == item.filehash
            dataset_general_string += item.path + item.filehash
        CSV.__set_dataset_and_markup_hash(dataset_path, origin_data_csv_path, dataset_general_string, subset_name)

    @staticmethod
    def __set_dataset_and_markup_hash(dataset_path, origin_data_csv_path, dataset_general_string, subset_name):
        # check if general_info_json is already exists
        if subset_name is None:
            general_info_json_path = os.path.join(dataset_path, CSV.GENERAL_INFO_JSON_NAME)
        else:
            general_info_json_path = os.path.join(dataset_path, subset_name + '_' + CSV.GENERAL_INFO_JSON_NAME)
        if os.path.isfile(general_info_json_path):
            CSV.dataset_general_info_json = json_load(general_info_json_path)
            errmsg = 'wrong json format at file:' + general_info_json_path
            assert set(CSV.dataset_general_info_json.keys()) == \
                   {CSV.general_info_key__dataset_hash,
                    CSV.general_info_key__dataset_and_markup_hash}, errmsg
            assert isinstance(
                CSV.dataset_general_info_json[CSV.general_info_key__dataset_hash],
                str), errmsg
            assert isinstance(CSV.dataset_general_info_json[
                                  CSV.general_info_key__dataset_and_markup_hash], str), errmsg

        # calculate dataset and markup hashes and compare with general_info_json
        dataset_hash = get_md5_adler32(bytes(dataset_general_string, 'utf8'), True)
        markup_hash = get_md5_adler32_from_file(bytes(origin_data_csv_path, 'utf8'), True)
        dataset_markup_hash = get_md5_adler32(bytes(dataset_general_string + markup_hash, 'utf8'), True)

        if os.path.isfile(general_info_json_path):
            errmsg = 'wrong hashes'
            assert CSV.dataset_general_info_json[
                       CSV.general_info_key__dataset_hash] == dataset_hash, errmsg
            assert CSV.dataset_general_info_json[
                       CSV.general_info_key__dataset_and_markup_hash] == dataset_markup_hash, errmsg
        else:
            CSV.dataset_general_info_json[
                CSV.general_info_key__dataset_hash] = dataset_hash
            CSV.dataset_general_info_json[
                CSV.general_info_key__dataset_and_markup_hash] = dataset_markup_hash
            json_dump(general_info_json_path, CSV.dataset_general_info_json)
        pass

    @staticmethod
    def get_stat_from_data_csv(data_csv, sample_type):
        print('sample_type: ', sample_type)
        # all stat by folds
        # labels stat
        folds = set(data_csv[CSV.csv_col_fold].values)
        labels = set(data_csv[CSV.csv_col_label].values)

        for fold in folds:
            print('fold:', fold, ':')
            data_csv_fold = data_csv.loc[data_csv.fold == fold]
            for label in labels:
                data_csv_fold_label = data_csv_fold.loc[data_csv_fold.label == label]
                print('label:', label, ', n:', len(data_csv_fold_label))
            print()
        print()

    csv_col_path = 'path'
    csv_col_filename = 'filename'
    csv_col_label = 'label'
    csv_col_fold = 'fold'
    CSV_COLUMNS_FILE = [
        csv_col_path,
        csv_col_filename,
        csv_col_label,
        csv_col_fold,
    ]

    csv_col_realpath = 'realpath'

    general_info_key__dataset_hash = 'dataset_hash'
    general_info_key__dataset_and_markup_hash = 'dataset_and_markup_hash'
    dataset_general_info_json = {general_info_key__dataset_hash: None, general_info_key__dataset_and_markup_hash: None}