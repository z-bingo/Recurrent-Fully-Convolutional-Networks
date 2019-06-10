import rawpy
import numpy as np
from mosaic_utils import *
import torch
from skimage.color import *
from PIL import Image

class RAW():
    def __init__(self, raw_image=None, pattern=None, black_level=None, white_balance=None, color_correction_matrix=None,
                 tone_curve=None, rgb_xyz_matrix=None):
        self.raw_image = raw_image
        self.pattern = pattern
        self.black_level = black_level
        self.white_balance = white_balance
        self.ccm = color_correction_matrix
        self.tone_curve = tone_curve
        self.rgb_xyz = rgb_xyz_matrix

    def fromFile(self, file_name):
        """
        从文件读取RAW文件
        :param file_name:
        :return:
        """
        raw = rawpy.imread(file_name)
        self.raw_image = raw.raw_image_visible.astype(np.float32)
        print('min:', np.min(raw.raw_image_visible))
        print('max:', np.max(raw.raw_image_visible))
        order = raw.raw_pattern.reshape(-1)
        pattern = str(raw.color_desc, encoding='gbk')
        self.pattern = ''.join([pattern[i] for i in order])
        self.black_level = [raw.black_level_per_channel[i] for i in order]
        self.white_balance = raw.camera_whitebalance[:3]
        self.ccm = None if np.sum(raw.color_matrix[:, :3] == 0) else raw.color_matrix[:, :3]
        self.tone_curve = raw.tone_curve
        self.rgb_xyz = raw.rgb_xyz_matrix[:3, :]
        return self

    def subtract_black(self):
        """
        仅仅减去黑电平  可以用作网络的输入
        :return:
        """
        # first, subtract the black level from raw image
        raw = np.zeros_like(self.raw_image)
        for i, (x, y) in zip(range(4), [(0, 0), (0, 1), (1, 0), (1, 1)]):
            raw[x::2, y::2] = self.raw_image[x::2, y::2] - self.black_level[i]
        return np.clip(raw, 0, 65535)

    def demosaicing(self, bilinear=False):
        raw_sub = self.subtract_black() / 65535
        print('max_raw_sum', np.max(raw_sub))
        if bilinear:
            rgb = demosaicing_bilinear(raw_sub, pattern=self.pattern)
        else:
            rgb = demosaicing_Malvar2004(raw_sub, pattern=self.pattern)
        return np.stack([rgb[:, :, 0]*self.white_balance[0], rgb[:, :, 1]*self.white_balance[1], rgb[:, :, 2]*self.white_balance[2]], axis=-1)

    def get_sRGB(self):
        rgb = self.demosaicing(bilinear=False)
        print('min:', np.min(rgb))
        print('max:', np.max(rgb))
        rgb = rgb / np.max(rgb)
        if not self.ccm is None:
            rgb_ccm = convertColor(rgb, self.ccm)
        else:
            rgb_ccm = rgb
        # camera RGB --> xyz
        if not self.rgb_xyz is None:
            rgb_xyz = convertColor(rgb_ccm, self.rgb_xyz)
        else:
            rgb_xyz = rgb_ccm
        # print('min_ccm',)
        # xyz --> sRGB
        srgb = xyz2rgb(rgb_xyz)
        print('min:', np.min(srgb))
        print('max:', np.max(srgb))
        srgb = np.clip(srgb, 0, 1)
        return (srgb*255).astype(np.uint8)


def convertColor(img, ccm):
    """
    The shape of image should be: C * H * W
    :param img:
    :param ccm:
    :return:
    """
    out = np.zeros_like(img)
    out[:, :, 0] = ccm[0, 0] * img[:, :, 0] + ccm[0, 1] * img[:, :, 1] + ccm[0, 2] * img[:, :, 2]
    out[:, :, 1] = ccm[1, 0] * img[:, :, 0] + ccm[1, 1] * img[:, :, 1] + ccm[1, 2] * img[:, :, 2]
    out[:, :, 2] = ccm[2, 0] * img[:, :, 0] + ccm[2, 1] * img[:, :, 1] + ccm[2, 2] * img[:, :, 2]
    return out

if __name__ == '__main__':
    raw = RAW()
    raw = raw.fromFile('00002_00_10s.ARW')
    srgb = raw.get_sRGB()
    Image.fromarray(srgb).show()

    raw = rawpy.imread('00002_00_10s.ARW')
    srgb = raw.postprocess(
        demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
        half_size=False,
        use_camera_wb=True,
        output_color=rawpy.ColorSpace.sRGB,
        output_bps=8,
        no_auto_bright=False,
        no_auto_scale=True
    )
    Image.fromarray(srgb).show()

    print('Over!')