import os
import glob


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


def get_storyboard_paths_from_video(video_path, storyboard_dir, storyboard_fps, storyboard_extension='.jpg'):
    """
    Returns paths of frames from storyboard from video
    :param video_path: path of video to storyboard
    :param storyboard_dir: empty or not existing folder to store frames of storyboard
    :param storyboard_fps: final storyboard fps
    :param storyboard_extension: extension of final frames of storyboard
    :return:
    """
    if os.path.isdir(storyboard_dir):
        if len(os.listdir(storyboard_dir)) > 0:
            print('ERASE PROTECTION: ' + storyboard_dir + ' should be empty or does not exist')
            return []
    else:
        os.mkdir(storyboard_dir)
    ffmpeg_storyboard_from_video(video_path, storyboard_dir, storyboard_fps, storyboard_extension)
    storyboard_paths = glob.glob(storyboard_dir + '/*' + storyboard_extension)
    return storyboard_paths
