import os
import shutil
import inspect


def __create_tmp(tmp_dir):
    err_msg = 'ERROR: can\'t create tmp dir: '+tmp_dir
    try:
        os.mkdir(tmp_dir)
        assert os.path.isdir(tmp_dir)
    except Exception:
        raise Exception(err_msg)


class Tmp_erase_protection:
    def __call__(self, tmp_dir):  # TODO make like interface
        """
        should check if data in tmp_dir is only tmp data
        """
        pass


def new_tmp(assert_tmp_content_only, tmp_dir='./tmp'):
    """
    Create new tmp folder (clear it if already exists)
    :param tmp_dir:
    :param assert_tmp_content_only: Tmp_erase_protection, __call__(tmp_dir) must be implemented
    :return:
    """
    assert isinstance(tmp_dir, str)
    assert isinstance(assert_tmp_content_only, Tmp_erase_protection)
    inspect.getfullargspec(assert_tmp_content_only.__call__)[0]  # TODO type check
    tmp_dir = os.path.realpath(tmp_dir)
    if os.path.isdir(tmp_dir):
        err_msg = 'ERROR: erase protection: ' + tmp_dir + ' contains not only tmp data'
        if len(os.listdir(tmp_dir)) > 0:
            try:
                assert_tmp_content_only(tmp_dir)
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
