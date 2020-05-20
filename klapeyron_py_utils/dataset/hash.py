import os
import hashlib
import zlib


def calculate_dataset_hash(parent_dir: str) -> str:
    """
    get all files paths from parent_dir
    replace system path separator on '$' for each path
    calculate hash on file data + file name for each file
    sort all hashes
    concat into one string and calculate overall dataset hash
    :param parent_dir:
    :return:
    """
    parent_dir = os.path.realpath(parent_dir)
    assert os.path.isdir(parent_dir)
    print('calculating hash of dataset on path:', parent_dir, '...')
    files_hashes = []
    true_files_hashes = {}
    for pardir, _, files in os.walk(parent_dir):
        for file in files:
            file = os.path.join(pardir, file)
            hash = get_md5_adler32_from_file(file, True)
            true_files_hashes[file] = hash
            file = file.replace(os.path.sep, '$')
            hash = get_md5_adler32(bytes(file+hash, 'utf8'), True)
            files_hashes.append(hash)
    files_hashes.sort()
    general_string = ''
    for hash in files_hashes:
        general_string += hash
    dataset_hash = get_md5_adler32(bytes(general_string, 'utf8'), True)
    return dataset_hash, true_files_hashes


def get_md5_adler32(b, onestr=False):
    """
    calculates hashes md5 and adler32
    :param b: bytes to hash
    :param onestr: if True returns concatenated string of md5+adler32, if False returns tuple (md5,adler32)
    :return:
    """
    assert isinstance(b, bytes)
    md5 = hashlib.md5(b).hexdigest()
    adler32 = zlib.adler32(b)
    if onestr:
        return md5+str(adler32)
    return md5, adler32


def get_md5_adler32_from_file(path, onestr=False):
    """
    calculates hashes md5 and adler32
    :param path: path to file
    :param onestr: if True returns concatenated string of md5+adler32, if False returns tuple (md5,adler32)
    :return:
    """
    assert os.path.isfile(path), 'not a file: '+str(path)
    with open(path, 'rb') as f:
        f = f.read()
    return get_md5_adler32(f, onestr)


# a = calculate_dataset_hash('C:/Users\klapeyron-server-win/YandexDisk/rPPG/benchmark\src\data')
# print()