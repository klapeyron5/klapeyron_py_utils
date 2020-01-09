import os
import glob
import numpy as np
from klapeyron_py_utils.tmp_folders.tmp_folders import new_tmp, Tmp_erase_protection


video_extensions = ('.mp4', '.avi', '.mov', '.MOV')


def ffmpeg_storyboard_from_video(video_path, storyboard_dir, storyboard_fps, storyboard_extension='.jpg'):
    """
    Write storyboard (frames of video) from video by ffmpeg with request:
    ffmpeg -hide_banner -loglevel panic -i movie_path -r fps_out -qscale:v 2 frames_dir/%04dextension
    :param video_path: path of video to storyboard
    :param storyboard_dir: folder to store frames of storyboard
    :param storyboard_fps: final storyboard fps
    :param storyboard_extension: extension of final frames of storyboard
    :return:
    """
    assert os.path.isfile(video_path)
    assert video_path.endswith(video_extensions)  # TODO
    assert os.path.isdir(storyboard_dir)
    assert storyboard_fps > 0
    assert storyboard_extension in {'.png', '.jpg'}

    request = 'ffmpeg -hide_banner -loglevel panic' \
              ' -i ' + video_path + \
              ' -r ' + str(storyboard_fps) + \
              ' -qscale:v 2' + \
              ' ' + storyboard_dir + '/%04d' + storyboard_extension
    os.system(request)


def ffmpeg_video_from_storyboard(storyboard_dir, video_path, storyboard_fps, storyboard_extension='.jpg'):
    # TODO
    assert os.path.isdir(storyboard_dir)
    assert isinstance(video_path, str)
    assert isinstance(storyboard_fps, int)

    request = 'ffmpeg -r ' + str(storyboard_fps) + ' -i ' + storyboard_dir + '/%04d' + storyboard_extension + ' ' + video_path
    os.system(request)


def ffmpeg_trim_video(video_path, start, period, dst_path='./tmp.mp4'):
    assert os.path.isfile(video_path)
    assert video_path.endswith(video_extensions)
    assert dst_path.endswith(video_extensions)
    assert start >= 0
    assert period > 0
    request = 'ffmpeg -hide_banner -loglevel panic' + \
              ' -i ' + video_path + \
              ' -ss ' + str(start) + \
              ' -t ' + str(period) + \
              ' ' + dst_path
    os.system(request)


class Tmp_erase_protection(Tmp_erase_protection):
    def __init__(self, tmp_dir, storyboard_extension, ):
        super().__init__(tmp_dir)
        self.storyboard_extension = storyboard_extension

    def __call__(self):
        # frames from storyboard
        # one video from trimming
        pardir, dirs, files = next(os.walk(self.tmp_dir))
        assert len(dirs) == 0
        extensions = [x.split('.')[-1] for x in files]
        s, c = np.unique(extensions, return_counts=True)
        s = np.array(['.'+x for x in s])
        assert len(s) <= 2
        assert self.storyboard_extension in s
        ind_stor = np.where(s == self.storyboard_extension)[0][0]
        ind_trim = ind_stor - 1
        if len(s) == 2:
            assert c[ind_trim] == 1
            assert s[ind_trim] in video_extensions
        # clear dir
        for file in files:
            os.remove(os.path.join(pardir,file))


def get_storyboard_paths_from_video(video_path, storyboard_fps, storyboard_dir='./tmp_ffmpeg', trim=None, storyboard_extension='.jpg'):
    """
    Returns paths of frames from storyboard from video
    :param video_path: path of video to storyboard
    :param storyboard_dir: empty or not existing folder to store frames of storyboard
    :param storyboard_fps: final storyboard fps
    :param trim: (start, finish) trim borders in seconds
    :param storyboard_extension: extension of final frames of storyboard
    :return:
    """
    new_tmp(storyboard_dir, Tmp_erase_protection())
    if trim is not None:
        assert len(trim) == 2
        trim_name = 'tmp.mp4'
        trim_path = os.path.join(storyboard_dir, trim_name)
        assert not trim_path.endswith(storyboard_extension)
        ffmpeg_trim_video(video_path, trim[0], trim[1], trim_path)
        assert os.path.isfile(trim_path), 'ERROR smth went wrong with trimming'
        video_path = trim_path
    ffmpeg_storyboard_from_video(video_path, storyboard_dir, storyboard_fps, storyboard_extension)
    storyboard_paths = glob.glob(storyboard_dir + '/*' + storyboard_extension)
    return storyboard_paths
