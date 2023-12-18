"""
Main module - set of functions that simulate or correct colorblindness for images
as described in
https://www.researchgate.net/publication/326626897_Smartphone_Based_Image_Color_Correction_for_Color_Blindness
"""
import numpy as np
import cv2

## LMS Daltonization
def rgb_to_lms(img):
    """
    lms_matrix = np.array(
        [[17.8824, 43.5161, 4.11935],
        [3.45565, 27.1554, 3.86714],
        [0.0299566, 0.184309, 1.46709]
        ]
        )
    """
    lms_matrix = np.array(
        [[0.3904725 , 0.54990437, 0.00890159],
        [0.07092586, 0.96310739, 0.00135809],
        [0.02314268, 0.12801221, 0.93605194]]
        )
    return np.tensordot(img, lms_matrix, axes=([2], [1]))

def lms_to_rgb(img):
    """
    rgb_matrix = np.array(
        [[0.0809444479, -0.130504409, 0.116721066],
        [0.113614708, -0.0102485335, 0.0540193266],
        [-0.000365296938, -0.00412161469, 0.693511405]
        ]
        )
    """
    rgb_matrix = np.array(
        [[ 2.85831110e+00, -1.62870796e+00, -2.48186967e-02],
        [-2.10434776e-01,  1.15841493e+00,  3.20463334e-04],
        [-4.18895045e-02, -1.18154333e-01,  1.06888657e+00]]
        )
    return np.tensordot(img, rgb_matrix, axes=([2], [1]))

def simulate_colorblindness(img, colorblind_type):
    lms_img = rgb_to_lms(img)
    if colorblind_type.lower() in ['protanopia', 'p', 'pro']:
        sim_matrix = np.array([[0, 0.90822864, 0.008192], [0, 1, 0], [0, 0, 1]], dtype=np.float16)
    elif colorblind_type.lower() in ['deuteranopia', 'd', 'deut']:
        sim_matrix =  np.array([[1, 0, 0], [1.10104433,  0, -0.00901975], [0, 0, 1]], dtype=np.float16)
    elif colorblind_type.lower() in ['tritanopia', 't', 'tri']:
        sim_matrix = np.array([[1, 0, 0], [0, 1, 0], [-0.15773032,  1.19465634, 0]], dtype=np.float16)
    else:
        raise ValueError('{} is an unrecognized colorblindness type.'.format(colorblind_type))
    lms_img = np.tensordot(lms_img, sim_matrix, axes=([2], [1]))
    rgb_img = lms_to_rgb(lms_img)
    return rgb_img.astype(np.uint8)

def daltonize_correct(img, colorblind_type):
    colorblind_img = simulate_colorblindness(img, colorblind_type=colorblind_type)
    error_matrix = img - colorblind_img
    correction_matrix = np.array(
            [[0.0, 0.0, 0.0],
            [0.7, 1.0, 0.0],
            [0.7, 0.0, 1.0]]
            )
    corrected_error_matrix = np.tensordot(error_matrix, correction_matrix, axes=([2], [1]))
    return img + corrected_error_matrix

## color-blind filter service (CBFS) algorithm
# I couldn't find any implementations
# so this is my best guess
# see https://www.researchgate.net/publication/221023903_Efficient_edge-services_for_colorblind_users
def cbfs_correct(img, closeness=70):
    red_values = 0
    green_values = 0
    for y, x in zip(np.arange(img.shape[0]), np.arange(img.shape[1])):
        if (img[y, x, 0] > img[y, x, 1] + 45) and (img[y, x, 0] > img[y, x, 2] + 45):
            red_values += 1
        elif (img[y, x, 1] > img[y, x, 0] + 45) and (img[y, x, 1] > img[y, x, 2] + 45):
            green_values += 1
    if red_values > green_values:
        proximity_matrix = np.abs(img[..., 0] - np.array([[[255, ]]]))
    else:
        proximity_matrix = np.abs(img[..., 1] - np.array([[[255, ]]]))
    proximity_matrix = np.reshape(proximity_matrix, newshape=(img.shape[0], img.shape[1], 1))
    print(proximity_matrix.shape)
    hsl_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    hsl_img = np.where(
        proximity_matrix < closeness,
        hsl_img+np.array([[[-76.5, 63.75, -25.5]]]),
        hsl_img+np.array([[[0.0, -25.5, 25.5]]])
        )
    hsl_img = np.clip(hsl_img, a_min=0, a_max=255)
    hsl_img = cv2.cvtColor(hsl_img.astype(np.uint8), cv2.COLOR_HLS2RGB)
    return hsl_img

## LAB color correction
def lab_correct(img, l_shift=15, a_shift=15, b_shift=15):
    lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    for i, shift in enumerate([l_shift, a_shift, b_shift]):
        lab_img[..., i] = np.where(
            lab_img[..., i] > 127,
            lab_img[..., i] + shift,
            lab_img[..., i] - shift
            )
    lab_img = np.clip(lab_img, a_min=0, a_max=255)
    rgb_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB)
    return rgb_img

## HSV Color Shifting Algorithm

# hsv and rgb conversions from
# https://stackoverflow.com/questions/27041559/rgb-to-hsv-python-change-hue-continuously

def rgb_to_hsv(rgb):
    # Translated from source of colorsys.rgb_to_hsv
    # r,g,b should be a numpy arrays with values between 0 and 255
    # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
    rgb = rgb.astype('float')
    hsv = np.zeros_like(rgb)
    # in case an RGBA array was passed, just copy the A channel
    hsv[..., 3:] = rgb[..., 3:]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb[..., :3], axis=-1)
    minc = np.min(rgb[..., :3], axis=-1)
    hsv[..., 2] = maxc
    mask = maxc != minc
    hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
    rc = np.zeros_like(r)
    gc = np.zeros_like(g)
    bc = np.zeros_like(b)
    rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
    gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
    bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
    hsv[..., 0] = np.select(
        [r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
    hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
    return hsv

def hsv_to_rgb(hsv):
    # Translated from source of colorsys.hsv_to_rgb
    # h,s should be a numpy arrays with values between 0.0 and 1.0
    # v should be a numpy array with values between 0.0 and 255.0
    # hsv_to_rgb returns an array of uints between 0 and 255.
    rgb = np.empty_like(hsv)
    rgb[..., 3:] = hsv[..., 3:]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = (h * 6.0).astype('uint8')
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
    rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
    rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
    rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
    return rgb.astype('uint8')

def hsv_color_correct(img, colorblind_type='deuteranopia'):
    rgb_img = img/255
    hsv_img = rgb_to_hsv(img)
    if colorblind_type in ['deuteranopia ', 'protanopia', 'deut', 'pro', 'd', 'p']:
        green_ratio = (hsv_img[..., 0] - (60/360))/rgb_img[..., 1]
        blue_range = green_ratio*rgb_img[..., 2]
        hsv_img[..., 0] = 0.5 + blue_range
    elif colorblind_type in ['tritanopia', 't', 'tri']:
        blue_ratio = (hsv_img[..., 0] - (60/360))/rgb_img[..., 2]
        green_range = blue_ratio*rgb_img[..., 1]
        hsv_img[..., 0] = (120/360) + green_range
    hsv_img[..., 0] = np.where(
        hsv_img[..., 0] > 1.0,
        hsv_img[..., 0] - 1.0,
        hsv_img[..., 0]
        )
    hsv_img[..., 0] = np.where(
        hsv_img[..., 0] < 0.0,
        hsv_img[..., 0] + 1.0,
        hsv_img[..., 0]
        )
    hsv_img[..., 2] = hsv_img[..., 2] * 255
    rgb_img = hsv_to_rgb(hsv_img)
    return rgb_img
