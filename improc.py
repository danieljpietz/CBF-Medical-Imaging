import numpy as np
import scipy.ndimage as ndimage
import cv2

cimg = None
result_smoothed = None
result_smoothed2 = None
result2 = None

videoOut = None


def hessian(x):
    x_grad = np.gradient(x)
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype)
    for k, grad_k in enumerate(x_grad):
        tmp_grad = np.gradient(grad_k)
        for l, grad_kl in enumerate(tmp_grad):
            hessian[k, l, :, :] = grad_kl
    return hessian


def loadmap(map, thresh, smooth):
    global cimg, result_smoothed2, result_smoothed, result2, videoOut
    cimg = cv2.imread(map)
    grayimg = cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY)
    (thresh, img) = cv2.threshold(grayimg, thresh, 255, cv2.THRESH_BINARY)
    img2 = cv2.distanceTransform(img, cv2.DIST_L2, 0)
    result2 = cv2.normalize(
        img2, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    result_smoothed = ndimage.gaussian_filter(
        result2.astype(float), sigma=(smooth, smooth), order=0
    )

    gradient = np.array(np.gradient(result_smoothed)).transpose([1, 2, 0])
    h = hessian(result_smoothed).transpose([2, 3, 0, 1])

    result2 = cv2.cvtColor(result2, cv2.COLOR_GRAY2BGR)
    result_smoothed2 = cv2.normalize(
        result_smoothed,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )
    result_smoothed2 = cv2.cvtColor(result_smoothed2, cv2.COLOR_GRAY2BGR)

    return (result_smoothed, gradient, h)


def plot(s, save_video=False):
    global videoOut
    x = int(s[1])
    y = int(s[0])
    cv2.circle(cimg, [x, y], 2, (255, 0, 0), 2)
    cv2.imshow("image", cimg)
    cv2.waitKey(10)
