import os


def assert_filepath(filepath):
    errmsg = str(filepath) + ' should exists as file'
    assert os.path.isfile(filepath), errmsg
