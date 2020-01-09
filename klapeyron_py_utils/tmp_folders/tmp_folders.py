import os
import shutil


def __create_tmp(tmp_dir):
    err_msg = 'ERROR: can\'t create tmp dir: '+tmp_dir
    try:
        os.mkdir(tmp_dir)
        assert os.path.isdir(tmp_dir)
    except Exception:
        raise Exception(err_msg)


class Tmp_erase_protection:
    def __init__(self, tmp_dir):
        self.tmp_dir = tmp_dir


def new_tmp(assert_tmp_content_only, tmp_dir='./tmp'):
    """
    Create new tmp folder (clear it if already exists)
    :param tmp_dir:
    :param assert_tmp_content_only: callable argumentless; should check if data in tmp_dir is only tmp data
    :return:
    """
    assert isinstance(tmp_dir, str)
    assert isinstance(assert_tmp_content_only, Tmp_erase_protection)
    tmp_dir = os.path.realpath(tmp_dir)
    if os.path.isdir(tmp_dir):
        err_msg = 'ERROR: erase protection: ' + tmp_dir + ' contains not only tmp data'
        if len(os.listdir(tmp_dir)) > 0:
            try:
                assert_tmp_content_only()
            except Exception:
                raise Exception(err_msg)
            try:
                shutil.rmtree(tmp_dir)
            except Exception:
                pass
            if os.path.isdir(tmp_dir):
                err_msg = 'ERROR: can\'t clear tmp dir: ' + tmp_dir
                assert len(os.listdir(tmp_dir)) == 0, err_msg
            else:
                __create_tmp(tmp_dir)
    else:
        __create_tmp(tmp_dir)
