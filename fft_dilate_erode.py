import random
import cv2
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from collections import Counter


def q1():
    img = cv2.imread('rice.jpg', 0)
    plt.title('Rice')
    plt.imshow(img, cmap='gray')
    plt.show()
    img2 = cv2.dilate(img, np.ones((5, 5)), iterations=3)
    img3 = img2 - img
    img3[img3 < 50] = 0
    img3[img3 >= 50] = 255
    plt.imshow(img3, cmap='gray')
    plt.show()
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    img3_transpose = 255 - img3
    img35 = cv2.morphologyEx(img3, cv2.MORPH_OPEN, kernel)

    inverted = 255 - img35
    inverted[inverted < 50] = 0
    inverted[inverted >= 50] = 255

    plt.title('inverted')
    plt.imshow(inverted, cmap='gray')
    plt.show()

    im_copy = np.zeros(img3.shape, np.uint8)
    im_copy[0][1] = 255
    while True:
        dilate = cv2.dilate(im_copy, kernel, iterations=1)
        dilate = cv2.bitwise_and(dilate, inverted)
        if np.array_equal(im_copy, dilate):
            break
        im_copy = dilate

    fill_rice = np.bitwise_not(im_copy)
    plt.imshow(fill_rice, cmap='gray')
    plt.show()


def get_mask(typ, background, shape):
    """
    typ = 1 = horizontal line
    typ = 2 = allign line
    typ = 3 = center square
    typ = 4 = plus

    background = 0 = black
    background = 1 = white
    """
    im_mask = np.full((shape[0], shape[1], 2), background, np.uint8)
    typ_color = np.abs(background - 1)
    center = (shape[0] // 2, shape[1] // 2)
    if typ == 1:
        im_mask[center[0] - 10: center[0] + 10, :] = typ_color
    elif typ == 2:
        im_mask[:, center[1] - 10: center[1] + 10] = typ_color

    elif typ == 3:
        im_mask[center[0] - 10: center[0] + 10, center[1] - 10: center[1] + 10] = typ_color

    elif typ == 4:
        im_mask[center[0] - 10: center[0] + 10, :] = typ_color
        im_mask[:, center[1] - 10: center[1] + 10] = typ_color

    return im_mask


def q2():
    search_img = cv2.imread('fimage_.png', 0)
    plt.title('our search image')
    plt.imshow(search_img.astype(int), cmap='gray')
    plt.show()

    for i in range(1, 5):
        for j in [0, 1]:
            mask = get_mask(i, j, search_img.shape).astype(int)
            plt.title('mask')
            plt.imshow(mask[:, :, 0], cmap='gray')
            plt.show()

            rows, cols = search_img.shape
            crow, ccol = rows / 2, cols / 2  # center
            dft = cv2.dft(np.float32(search_img), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            fshift = dft_shift * mask
            f_ishift = np.fft.ifftshift(fshift)
            im_back = cv2.idft(f_ishift)
            im_back = cv2.magnitude(im_back[:, :, 0], im_back[:, :, 1])

            plt.title('idft - mag')
            plt.imshow(im_back, cmap='gray')
            plt.show()


def q3():
    four_img = cv2.imread('fft1.png', 0)
    reg4 = cv2.imread('4reg.png', 0)
    upside4 = cv2.imread('4upside.png', 0)

    plt.title(f'{four_img.shape} Search photo')
    plt.imshow(four_img, cmap='gray')
    plt.show()

    plt.title(f'{reg4.shape} regular 4 digit')
    plt.imshow(reg4, cmap='gray')
    plt.show()

    plt.title(f'{upside4.shape} Rotated 4 digit')
    plt.imshow(upside4, cmap='gray')
    plt.show()

    fourr = np.fft.fftshift(np.fft.fft2(four_img))
    regg = np.fft.fftshift(np.fft.fft2(reg4))
    upp = np.fft.fftshift(np.fft.fft2(upside4))

    test = (fourr * regg.conj()) / np.abs(fourr * regg.conj())
    test1 = (fourr * upp.conj()) / np.abs(fourr * upp.conj())

    r = np.abs(np.fft.ifftshift(np.fft.ifft2(test)))
    t = np.abs(np.fft.ifftshift(np.fft.ifft2(test1)))

    x, y = np.where(r > 0.025)
    z, u = np.where(t > 0.025)
    x = np.stack((x, z))
    y = np.stack((y, u))
    x, y = y, x

    plt.title(f'Found {len(x) + len(y)} occurrences ( GREEN )')
    plt.plot(x, y, 'go')
    plt.imshow(four_img, cmap='gray')
    plt.show()


def q4():
    img = cv2.imread('fimage.png', cv2.IMREAD_GRAYSCALE)
    plt.title('Original photo')
    plt.imshow(img, cmap='gray')
    plt.show()
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=30, param2=15, minRadius=250, maxRadius=300)
    updated = [tuple(trip) for trip in circles[0, :] if
               img.shape[0] // 2 - 15 < trip[0] < img.shape[0] // 2 + 15 and trip[1] > 300]
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i in updated:
        x, y, R = int(i[0]), int(i[1]), int(i[2])
        # draw the outer circle
        cv2.circle(img, (x, y), R, (124, 255, 0), 2)

    plt.title('Hough Circle Transform')
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    q1()
    q2()
    q3()
    q4()
