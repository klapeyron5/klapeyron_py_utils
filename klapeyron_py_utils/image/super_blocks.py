import numpy as np
from klapeyron_py_utils.types.common_types import is_any_int
import cv2


def get_super_blocks(img, block_y, block_x, input_type='blocks_number'):
    """
    Returns super_blocks_slices, super_blocks in 2D array
    Super blocks can be defined by enter number of blocks for each y/x axises
    or by enter size of each edge of block
    :param img: image (np.ndarray)
    :param block_y: number of blocks for y axis / block height
    :param block_x: number of blocks for x axis / block width
    :param input_type: blocks_number / block_size
    :return: super_blocks_slices shape==(h,w,4), where the last dim is [y_start, y_finish, x_start, x_finish]
             super_blocks shape==(h,w,block), where block is part of img
    """
    input_type_blocks_number = 'blocks_number'
    input_type_block_size = 'block_size'
    if input_type == input_type_blocks_number:
        return get_super_blocks_from_blocks_number(img, block_y, block_x)
    elif input_type == input_type_block_size:
        return get_super_blocks_from_block_size(img, block_y, block_x)
    else:
        raise ValueError


def get_super_blocks_from_blocks_number(img, blocks_N_y, blocks_N_x, accurately=False):
    h, w = img.shape[0], img.shape[1]
    if accurately:
        raise ValueError
    else:
        block_h = h // blocks_N_y
        block_w = w // blocks_N_x
        return get_super_blocks_from_block_size(img, block_h, block_w)


def get_super_blocks_from_block_size(img, block_h, block_w, blocks_fit=False):
    assert isinstance(img, np.ndarray)
    assert len(img.shape) > 1
    h, w = img.shape[0], img.shape[1]

    def get_blocks_size(img_size, block_size):
        assert is_any_int(img_size)
        assert is_any_int(block_size)

        full_blocks_n = img_size // block_size
        residue_size = img_size % block_size
        if blocks_fit:
            assert residue_size == 0
        first_block_size = residue_size // 2
        last_block_size = residue_size - residue_size // 2
        if first_block_size == 0:
            st = []
        else:
            st = [first_block_size]
        if last_block_size == 0:
            fin = []
        else:
            fin = [last_block_size]
        blocks_size = st + full_blocks_n * [block_size] + fin
        return blocks_size
    blocks_size_y = get_blocks_size(h, block_h)
    blocks_size_x = get_blocks_size(w, block_w)
    super_blocks_slices = []
    y_slice = 0
    for block_size_y in blocks_size_y:
        super_blocks_slices.append([])
        x_slice = 0
        for block_size_x in blocks_size_x:
            super_blocks_slices[-1].append([y_slice, y_slice+block_size_y, x_slice, x_slice+block_size_x])
            x_slice += block_size_x
        y_slice += block_size_y
    super_blocks_slices = np.array(super_blocks_slices)
    return_super_blocks = True
    if return_super_blocks:
        super_blocks = []
        y_slice = 0
        for block_size_y in blocks_size_y:
            super_blocks.append([])
            x_slice = 0
            for block_size_x in blocks_size_x:
                super_blocks[-1].append(img[y_slice:y_slice + block_size_y, x_slice:x_slice + block_size_x])
                x_slice += block_size_x
            y_slice += block_size_y
        super_blocks = np.array(super_blocks)
        return super_blocks_slices, super_blocks
    return super_blocks_slices


def visualise_super_blocks(img, super_blocks_slices):
    assert isinstance(img, np.ndarray)
    assert isinstance(super_blocks_slices, np.ndarray)
    assert super_blocks_slices.shape[-1] == 4
    color = (0, 255, 0)
    super_blocks_slices = super_blocks_slices.reshape((-1, 4))

    for slices in super_blocks_slices:
        contour = [[slices[2], slices[0]],
                   [slices[3], slices[0]],
                   [slices[3], slices[1]],
                   [slices[2], slices[1]]]
        contour = np.array(contour)
        cv2.drawContours(img, [contour], 0, color)


# img = cv2.imread('E:/unveildev_rPPG/rPPG\simple/tmp/0001.jpg')
# super_blocks_slices, super_blocks = get_super_blocks(img, 20, 20, 'block_size')
# visualise_super_blocks(img, super_blocks_slices)
# cv2.imwrite('./vis_blocks.png', img)
