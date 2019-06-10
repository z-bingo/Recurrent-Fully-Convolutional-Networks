# import torch
import numpy as np
from scipy.ndimage.filters import convolve, convolve1d

'''
从RAW图像中提取RGB各个分量对应的mask
@param pattern: 'rggb', 'grbg', 'bggr'
'''
def bayer_CFA_pattern(shape, pattern='rggb'):
    pattern = pattern.upper()
    bayer_cfa = dict({color: np.zeros(shape) for color in 'RGB'})
    raw = np.ones(shape)
    for color, (x, y) in zip(pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        bayer_cfa[color][x::2, y::2] = raw[x::2, y::2]
    return bayer_cfa['R'].astype(np.float32), bayer_cfa['G'].astype(np.float32), bayer_cfa['B'].astype(np.float32)

'''
Bilinear Demosaicing
'''
def demosaicing_bilinear(raw, pattern='rggb'):
    kernel_rb = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ]) / 4
    kernel_g = np.array([
        [0, 1, 0],
        [1, 4, 1],
        [0, 1, 0]
    ]) / 4
    # 获取每个颜色的分量
    mask_R, mask_G, mask_B = bayer_CFA_pattern(raw.shape, pattern)
    data_R, data_G, data_B = raw*mask_R, raw*mask_B, raw*mask_B
    # 双线性插值  demosaicing
    data_R = convolve(data_R, kernel_rb)
    data_G = convolve(data_G, kernel_g)
    data_B = convolve(data_B, kernel_rb)
    # if not wb:
    #     wb = white_balance_simple(data_R, data_G, data_B)
    # return the color image
    return np.stack([data_R, data_G, data_B], axis=-1)

"""
AHD demosaicing algorithm
"""
def demosaicing_AHD(raw, pattern='rggb'):
    pattern = pattern.upper()


"""
Malvar (2004) Bayer CFA Demosaicing
===================================
*Bayer* CFA (Colour Filter Array) *Malvar (2004)* demosaicing.
References
----------
-   :cite:`Malvar2004a` : Malvar, H. S., He, L.-W., Cutler, R., & Way, O. M.
    (2004). High-Quality Linear Interpolation for Demosaicing of
    Bayer-Patterned Color Images. In International Conference of Acoustic,
    Speech and Signal Processing (pp. 5-8). 
"""
def demosaicing_Malvar2004(CFA, pattern='RGGB'):
    """
    Returns the demosaiced *RGB* colourspace array from given *Bayer* CFA using
    *Malvar (2004)* demosaicing algorithm.
    Parameters
    ----------
    CFA : array_like
        *Bayer* CFA.
    pattern : unicode, optional
        **{'RGGB', 'BGGR', 'GRBG', 'GBRG'}**,
        Arrangement of the colour filters on the pixel array.
    Returns
    -------
    ndarray
        *RGB* colourspace array.
    """
    R_m, G_m, B_m = bayer_CFA_pattern(CFA.shape, pattern)

    GR_GB = np.array(
        [[0, 0, -1, 0, 0],
         [0, 0, 2, 0, 0],
         [-1, 2, 4, 2, -1],
         [0, 0, 2, 0, 0],
         [0, 0, -1, 0, 0]]) / 8  # yapf: disable

    Rg_RB_Bg_BR = np.array(
        [[0, 0, 0.5, 0, 0],
         [0, -1, 0, -1, 0],
         [-1, 4, 5, 4, - 1],
         [0, -1, 0, -1, 0],
         [0, 0, 0.5, 0, 0]]) / 8  # yapf: disable

    Rg_BR_Bg_RB = np.transpose(Rg_RB_Bg_BR)

    Rb_BB_Br_RR = np.array(
        [[0, 0, -1.5, 0,    0],
         [0, 2, 0,    2,    0],
         [-1.5, 0,    6,    0, -1.5],
         [0, 2, 0,    2,    0],
         [0, 0, -1.5, 0,    0]]) / 8  # yapf: disable

    R = CFA * R_m
    G = CFA * G_m
    B = CFA * B_m

    del G_m

    G = np.where(np.logical_or(R_m == 1, B_m == 1), convolve(CFA, GR_GB), G)

    RBg_RBBR = convolve(CFA, Rg_RB_Bg_BR)
    RBg_BRRB = convolve(CFA, Rg_BR_Bg_RB)
    RBgr_BBRR = convolve(CFA, Rb_BB_Br_RR)

    del GR_GB, Rg_RB_Bg_BR, Rg_BR_Bg_RB, Rb_BB_Br_RR

    # Red rows.
    R_r = np.transpose(np.any(R_m == 1, axis=1)[np.newaxis]) * np.ones(R.shape)
    # Red columns.
    R_c = np.any(R_m == 1, axis=0)[np.newaxis] * np.ones(R.shape)
    # Blue rows.
    B_r = np.transpose(np.any(B_m == 1, axis=1)[np.newaxis]) * np.ones(B.shape)
    # Blue columns
    B_c = np.any(B_m == 1, axis=0)[np.newaxis] * np.ones(B.shape)

    del R_m, B_m

    R = np.where(np.logical_and(R_r == 1, B_c == 1), RBg_RBBR, R)
    R = np.where(np.logical_and(B_r == 1, R_c == 1), RBg_BRRB, R)

    B = np.where(np.logical_and(B_r == 1, R_c == 1), RBg_RBBR, B)
    B = np.where(np.logical_and(R_r == 1, B_c == 1), RBg_BRRB, B)

    R = np.where(np.logical_and(B_r == 1, B_c == 1), RBgr_BBRR, R)
    B = np.where(np.logical_and(R_r == 1, R_c == 1), RBgr_BBRR, B)

    del RBg_RBBR, RBg_BRRB, RBgr_BBRR, R_r, R_c, B_r, B_c

    # if not wb:
    #     # 简单的白平衡算法
    #     wb = white_balance_simple(R, G, B)

    return np.stack([R, G, B], axis=-1)


"""
DDFAPD - Menon (2007) Bayer CFA Demosaicing
===========================================
*Bayer* CFA (Colour Filter Array) DDFAPD - *Menon (2007)* demosaicing.
References
----------
-   :cite:`Menon2007c` : Menon, D., Andriani, S., & Calvagno, G. (2007).
    Demosaicing With Directional Filtering and a posteriori Decision. IEEE
    Transactions on Image Processing, 16(1), 132-141.
    doi:10.1109/TIP.2006.884928
"""
def _cnv_h(x, y):
    """
    Helper function for horizontal convolution.
    """
    return convolve1d(x, y, mode='mirror')

def _cnv_v(x, y):
    """
    Helper function for vertical convolution.
    """
    return convolve1d(x, y, mode='mirror', axis=0)

def demosaicing_Menon2007(CFA, wb=(1.0, 1.0, 1.0), pattern='RGGB', refining_step=True):
    """
    Returns the demosaiced *RGB* colourspace array from given *Bayer* CFA using
    DDFAPD - *Menon (2007)* demosaicing algorithm.
    Parameters
    ----------
    CFA : array_like
        *Bayer* CFA.
    pattern : unicode, optional
        **{'RGGB', 'BGGR', 'GRBG', 'GBRG'}**,
        Arrangement of the colour filters on the pixel array.
    refining_step : bool
        Perform refining step.
    Returns
    -------
    ndarray
        *RGB* colourspace array.
    """

    # 测试  是不是没减去black_level
    # CFA -= np.min(CFA)

    R_m, G_m, B_m = bayer_CFA_pattern(CFA.shape, pattern)

    h_0 = np.array([0, 0.5, 0, 0.5, 0])
    h_1 = np.array([-0.25, 0, 0.5, 0, -0.25])

    R = CFA * R_m
    G = CFA * G_m
    B = CFA * B_m

    G_H = np.where(G_m == 0, _cnv_h(CFA, h_0) + _cnv_h(CFA, h_1), G)
    G_V = np.where(G_m == 0, _cnv_v(CFA, h_0) + _cnv_v(CFA, h_1), G)

    C_H = np.where(R_m == 1, R - G_H, 0)
    C_H = np.where(B_m == 1, B - G_H, C_H)

    C_V = np.where(R_m == 1, R - G_V, 0)
    C_V = np.where(B_m == 1, B - G_V, C_V)

    D_H = np.abs(C_H - np.pad(C_H, ((0, 0),
                                    (0, 2)), mode=str('reflect'))[:, 2:])
    D_V = np.abs(C_V - np.pad(C_V, ((0, 2),
                                    (0, 0)), mode=str('reflect'))[2:, :])

    del h_0, h_1, CFA, C_V, C_H

    k = np.array(
        [[0, 0, 1, 0, 1],
         [0, 0, 0, 1, 0],
         [0, 0, 3, 0, 3],
         [0, 0, 0, 1, 0],
         [0, 0, 1, 0, 1]])  # yapf: disable

    d_H = convolve(D_H, k, mode='constant')
    d_V = convolve(D_V, np.transpose(k), mode='constant')

    del D_H, D_V

    mask = d_V >= d_H
    G = np.where(mask, G_H, G_V)
    M = np.where(mask, 1, 0)

    del d_H, d_V, G_H, G_V

    # Red rows.
    R_r = np.transpose(np.any(R_m == 1, axis=1)[np.newaxis]) * np.ones(R.shape)
    # Blue rows.
    B_r = np.transpose(np.any(B_m == 1, axis=1)[np.newaxis]) * np.ones(B.shape)

    k_b = np.array([0.5, 0, 0.5])

    R = np.where(
        np.logical_and(G_m == 1, R_r == 1),
        G + _cnv_h(R, k_b) - _cnv_h(G, k_b),
        R,
    )

    R = np.where(
        np.logical_and(G_m == 1, B_r == 1) == 1,
        G + _cnv_v(R, k_b) - _cnv_v(G, k_b),
        R,
    )

    B = np.where(
        np.logical_and(G_m == 1, B_r == 1),
        G + _cnv_h(B, k_b) - _cnv_h(G, k_b),
        B,
    )

    B = np.where(
        np.logical_and(G_m == 1, R_r == 1) == 1,
        G + _cnv_v(B, k_b) - _cnv_v(G, k_b),
        B,
    )

    R = np.where(
        np.logical_and(B_r == 1, B_m == 1),
        np.where(
            M == 1,
            B + _cnv_h(R, k_b) - _cnv_h(B, k_b),
            B + _cnv_v(R, k_b) - _cnv_v(B, k_b),
        ),
        R,
    )

    B = np.where(
        np.logical_and(R_r == 1, R_m == 1),
        np.where(
            M == 1,
            R + _cnv_h(B, k_b) - _cnv_h(R, k_b),
            R + _cnv_v(B, k_b) - _cnv_v(R, k_b),
        ),
        B,
    )

    # RGB = np.stack([R, G, B])

    del k_b, R_r, B_r

    if refining_step:
        R, G, B = refining_step_Menon2007((R, G, B), (R_m, G_m, B_m), M)

    del M, R_m, G_m, B_m

    return np.stack([R*wb[0], G*wb[1], B*wb[2]], axis=-1)

demosaicing_DDFAPD = demosaicing_Menon2007

def refining_step_Menon2007(RGB, RGB_m, M):
    """
    Performs the refining step on given *RGB* colourspace array.
    Parameters
    ----------
    RGB : array_like
        *RGB* colourspace array.
    RGB_m : array_like
        *Bayer* CFA red, green and blue masks.
    M : array_like
        Estimation for the best directional reconstruction.
    Returns
    -------
    ndarray
        Refined *RGB* colourspace array.
    --------
    """

    R, G, B = RGB
    R_m, G_m, B_m = RGB_m
    # M = M.astype(np.float32)

    del RGB, RGB_m

    # Updating of the green component.
    R_G = R - G
    B_G = B - G

    FIR = np.ones(3) / 3

    B_G_m = np.where(
        B_m == 1,
        np.where(M == 1, _cnv_h(B_G, FIR), _cnv_v(B_G, FIR)),
        0,
    )
    R_G_m = np.where(
        R_m == 1,
        np.where(M == 1, _cnv_h(R_G, FIR), _cnv_v(R_G, FIR)),
        0,
    )

    del B_G, R_G

    G = np.where(R_m == 1, R - R_G_m, G)
    G = np.where(B_m == 1, B - B_G_m, G)

    # Updating of the red and blue components in the green locations.
    # Red rows.
    R_r = np.transpose(np.any(R_m == 1, axis=1)[np.newaxis]) * np.ones(R.shape)
    # Red columns.
    R_c = np.any(R_m == 1, axis=0)[np.newaxis] * np.ones(R.shape)
    # Blue rows.
    B_r = np.transpose(np.any(B_m == 1, axis=1)[np.newaxis]) * np.ones(B.shape)
    # Blue columns.
    B_c = np.any(B_m == 1, axis=0)[np.newaxis] * np.ones(B.shape)

    R_G = R - G
    B_G = B - G

    k_b = np.array([0.5, 0, 0.5])

    R_G_m = np.where(
        np.logical_and(G_m == 1, B_r == 1),
        _cnv_v(R_G, k_b),
        R_G_m,
    )
    R = np.where(np.logical_and(G_m == 1, B_r == 1), G + R_G_m, R)
    R_G_m = np.where(
        np.logical_and(G_m == 1, B_c == 1),
        _cnv_h(R_G, k_b),
        R_G_m,
    )
    R = np.where(np.logical_and(G_m == 1, B_c == 1), G + R_G_m, R)

    del B_r, R_G_m, B_c, R_G

    B_G_m = np.where(
        np.logical_and(G_m == 1, R_r == 1),
        _cnv_v(B_G, k_b),
        B_G_m,
    )
    B = np.where(np.logical_and(G_m == 1, R_r == 1), G + B_G_m, B)
    B_G_m = np.where(
        np.logical_and(G_m == 1, R_c == 1),
        _cnv_h(B_G, k_b),
        B_G_m,
    )
    B = np.where(np.logical_and(G_m == 1, R_c == 1), G + B_G_m, B)

    del B_G_m, R_r, R_c, G_m, B_G

    # Updating of the red (blue) component in the blue (red) locations.
    R_B = R - B
    R_B_m = np.where(
        B_m == 1,
        np.where(M == 1, _cnv_h(R_B, FIR), _cnv_v(R_B, FIR)),
        0,
    )
    R = np.where(B_m == 1, B + R_B_m, R)

    R_B_m = np.where(
        R_m == 1,
        np.where(M == 1, _cnv_h(R_B, FIR), _cnv_v(R_B, FIR)),
        0,
    )
    B = np.where(R_m == 1, R - R_B_m, B)

    del R_B, R_B_m, R_m

    return R, G, B

'''
recovery the RAW image from RGB image
@param pattern: the basis bayer pattern
'''
def mosaicing(rgb, pattern='rggb'):
    pattern = pattern.upper()
    dic = {'R': 0, 'G': 1, 'B': 2}
    raw = np.zeros(rgb.shape[:2])
    for color, (x, y) in zip(pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        raw[x::2, y::2] = rgb[x::2, y::2, dic[color]]
    return raw

if __name__ == '__main__':
    from PIL import Image
    img = np.array(Image.open('F:\\BinZhang\\Codes\\BurstDenosing\\burst-denoising-master\\dataset\\train\\0001c8fbfb30d3a6.jpg'))
    raw = np.zeros(img.shape[:2])
    raw[0::2, 0::2] = img[0::2, 0::2, 0]
    raw[0::2, 1::2] = img[0::2, 1::2, 1]
    raw[1::2, 0::2] = img[1::2, 0::2, 1]
    raw[1::2, 1::2] = img[1::2, 1::2, 2]
    rgb = demosaicing_bilinear(raw, 'rggb')
    Image.fromarray(raw).show(title='RAW')
    Image.fromarray(rgb).show(title='Original')
    Image.fromarray(img).show(title='Bilinear')

