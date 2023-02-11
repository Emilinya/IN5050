import numpy as np
from PIL import Image
import scipy

RGB2YUV_MAT = np.array([
    [0.299, 0.587, 0.114],
    [-0.168736, -0.331264, 0.5],
    [0.5, -0.418688, -0.081312]
])
YUV2RGB_MAT = np.linalg.inv(RGB2YUV_MAT)

UV_SCALE = 2


class YUVImage():
    def __init__(self, width, height):
        self.width = width
        self.height = height

        N = self.width * self.height
        N_UV = (self.width // UV_SCALE) * (self.height // UV_SCALE)
        self.Y = np.zeros(N, dtype=np.uint8)
        self.U = np.zeros(N_UV, dtype=np.uint8)
        self.V = np.zeros(N_UV, dtype=np.uint8)


# code from https://stackoverflow.com/questions/48097941/strided-convolution-of-2d-in-numpy
def strided_convolve(array, kernel, s):
    cc = scipy.signal.fftconvolve(array, kernel[::-1, ::-1], mode='valid')
    idx = (np.arange(0, cc.shape[1], s), np.arange(0, cc.shape[0], s))
    xidx, yidx = np.meshgrid(*idx)
    return cc[yidx, xidx]


def readYUV(file_in, width, height):
    yuvImage = YUVImage(width, height)

    N = width * height
    N_UV = (width // UV_SCALE) * (height // UV_SCALE)
    with open(file_in, "rb") as infile:
        yuvImage.Y = np.array(list(infile.read(N)), dtype=np.uint8)
        yuvImage.U = np.array(list(infile.read(N_UV)), dtype=np.uint8)
        yuvImage.V = np.array(list(infile.read(N_UV)), dtype=np.uint8)

    return yuvImage


def yuv2rgb(yuvImage: YUVImage):
    w = yuvImage.width
    w_uv = w // UV_SCALE
    h = yuvImage.height
    h_uv = h // UV_SCALE
    rgbImage = np.zeros((h, w, 3), dtype=np.uint8)

    rgbImage[:, :, 0] = yuvImage.Y.reshape((h, w))
    if UV_SCALE > 1:
        # the U and V channels are downscaled
        # can you do this in a vectorized way? Probably, but idk
        for i in range(h_uv):
            for j in range(w_uv):
                li = i * UV_SCALE
                lj = j * UV_SCALE
                rgbImage[
                    li:li+UV_SCALE, lj:lj+UV_SCALE, 1
                ] = yuvImage.U[i * w_uv + j]
                rgbImage[
                    li:li + UV_SCALE, lj:lj + UV_SCALE, 2
                ] = yuvImage.V[i * w_uv + j]

        if h_uv * UV_SCALE != h:
            for j in range(w_uv):
                lj = j * UV_SCALE
                rgbImage[
                    h_uv*UV_SCALE:, lj:lj + UV_SCALE, 1
                ] = yuvImage.U[(h_uv-1) * w_uv + j]
                rgbImage[
                    h_uv*UV_SCALE:, lj:lj + UV_SCALE, 2
                ] = yuvImage.V[(h_uv-1) * w_uv + j]

        if w_uv * UV_SCALE != w:
            for i in range(h_uv):
                li = i * UV_SCALE
                rgbImage[
                    li:li + UV_SCALE, w_uv*UV_SCALE:, 1
                ] = yuvImage.U[i * w_uv + (w_uv-1)]
                rgbImage[
                    li:li + UV_SCALE, w_uv*UV_SCALE:, 2
                ] = yuvImage.V[i * w_uv + (w_uv-1)]

        if h_uv * UV_SCALE != h and w_uv * UV_SCALE != w:
            rgbImage[h_uv*UV_SCALE:, w_uv*UV_SCALE:, 1] = yuvImage.U[-1]
            rgbImage[h_uv*UV_SCALE:, w_uv*UV_SCALE:, 2] = yuvImage.V[-1]

    else:
        rgbImage[:, :, 1] = yuvImage.U.reshape((h, w))
        rgbImage[:, :, 2] = yuvImage.V.reshape((h, w))

    rgbImage = rgbImage - np.array([0., 128., 128.])
    rgbImage = np.clip(
        np.einsum("kl,ijl->ijk", YUV2RGB_MAT, rgbImage), 0, 255
    ).astype(np.uint8)

    return rgbImage


def rgb2yuv(rgbImage):
    h, w, _ = rgbImage.shape

    yuvImage = YUVImage(w, h)

    yuvRay = np.einsum("kl,ijl->ijk", RGB2YUV_MAT, rgbImage)
    yuvRay += np.array([0., 128., 128.])
    yuvRay = np.clip(yuvRay, 0, 255).astype(np.uint8)

    yuvImage.Y = yuvRay[:, :, 0].flatten()

    if UV_SCALE > 1:
        # we want to downsample the U and V channels
        kernel = np.full((UV_SCALE, UV_SCALE), 1/(UV_SCALE**2))
        yuvImage.U = strided_convolve(
            yuvRay[:, :, 1].astype(float), kernel, UV_SCALE
        ).astype(np.uint8)
        yuvImage.V = strided_convolve(
            yuvRay[:, :, 2].astype(float), kernel, UV_SCALE
        ).astype(np.uint8)
    else:
        yuvImage.U = yuvRay[:, :, 1]
        yuvImage.V = yuvRay[:, :, 2]

    yuvImage.U = yuvImage.U.flatten()
    yuvImage.V = yuvImage.V.flatten()

    return yuvImage


def saveYUV(yuvImage: YUVImage, file_out):
    assert yuvImage.Y.dtype == yuvImage.U.dtype == yuvImage.V.dtype == np.uint8
    with open(file_out, "wb") as outfile:
        outfile.write(yuvImage.Y)
        outfile.write(yuvImage.U)
        outfile.write(yuvImage.V)


def testConv():
    rgb2yuvConv = Image.open("test/test.jpg")
    rgbImage1 = np.asarray(rgb2yuvConv)

    yuvImage1 = rgb2yuv(rgbImage1)
    saveYUV(yuvImage1, "test/test.yuv")

    yuvImage2 = readYUV("test/test.yuv", yuvImage1.width, yuvImage1.height)
    rgbImage2 = yuv2rgb(yuvImage2)

    yuv2rgbConv = Image.fromarray(rgbImage2, mode="RGB")
    yuv2rgbConv.save("test/test.png")


def convert_forman():
    yuvImage = readYUV("foreman.yuv", 352, 288)
    rgbImage = yuv2rgb(yuvImage)
    image = Image.fromarray(rgbImage, mode="RGB")
    image.save("foreman.png")
