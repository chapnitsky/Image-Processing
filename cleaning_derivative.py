import random
import cv2
import numpy as np
import os
from copy import deepcopy
import matplotlib.pyplot as plt
from collections import Counter


def get_histogram(array):
    arr = np.array(array)
    vec = arr.flatten()
    count = Counter(vec)
    total_pixels = np.sum(list(count.values()))
    n_k = []
    for i in np.arange(256):  # RGB
        if not count[i]:
            n_k.append(0)
        else:
            n_k.append(count[i])

    n_k = n_k / total_pixels
    p_rk = np.array(n_k)

    dictgram = {}
    for i, val in enumerate(p_rk):
        dictgram[i] = val

    return p_rk, dictgram


def get_rhomb(M=200, N=200):
    black_img = np.zeros((M, N))  # Create a black image with Rhombus
    center = (round(M / 2.0), round(N / 2.0))
    R = N // 2
    start_y, end_y = -1 * (R // 2) + R, (R // 2) + R
    top_sizer = 1
    for i in range(start_y, R):  # Create the shape, need to find shape edges
        black_img[i][R - top_sizer:  R + top_sizer] = 120
        top_sizer += 1
    bottom_sizer = int(top_sizer)
    for j in range(R, end_y + 1):
        black_img[j][R - bottom_sizer: R + bottom_sizer] = 120
        bottom_sizer -= 1
    return black_img


def third(M=300, N=300):
    rhom = get_rhomb(M, N)

    kernel_blur = np.ones((5, 5)) / 25  # BLUR IT
    tmp = cv2.filter2D(rhom, -1, kernel_blur)
    plt.title('Rhombus with blur')
    plt.imshow(tmp, cmap='gray')
    plt.show()

    edges = tmp.copy()
    cutter = np.array([-1, 0, 1]) * 0.5  # Derivative

    edges = cv2.filter2D(edges, -1, cutter)
    edges = np.abs(edges)  # Abs for edges
    cv2.imshow('Edges', edges)
    plt.title('Rhombus Edges')
    plt.imshow(edges, cmap='gray')
    plt.show()


def second():
    path = os.getcwd()
    img = cv2.imread(path + '/bliss.png', 1)  # With color
    clouds = (180, 180, 180)
    blue = (140, 140, 140)
    clouds_sky_ground = [0] * 3

    for i, row in enumerate(img):
        for j, triplet in enumerate(row):
            b, g, r = triplet
            if b > clouds[0] and g > clouds[1] and r > clouds[2]:
                clouds_sky_ground[0] += 1  # Found cloud
            elif b > blue[0]:
                clouds_sky_ground[1] += 1
            else:
                clouds_sky_ground[2] += 1

    print(f'Clouds:\n\t{clouds_sky_ground[0]}\nSky:\n\t{clouds_sky_ground[1]}\nPlants:\n\t{clouds_sky_ground[2]}')


def cleaner():
    command = str(input('Enter a command by the letters: '
                        '\n\ta. Brightness\n\tb. Contrast\n\tc. cv2.threshold_TOZERO\n\td. Gamma\n\n'))
    number = float(input('Enter a number:'
                         '\n\ta. Brightness:\n\t\t-120 <= number <= 120\n\tb. Contrast:\n\t\t0.1 <= number <= 5\n\tc. '
                         'cv2.threshold_TOZERO:\n\t\t20 <= number <= 200\n\td. Gamma:\n\t\t0.1 <= number <= 5\n\n'))
    if not command or not number:
        print('Wrong syntax')
        exit(1)

    real_cmd = None
    real_num = None
    if command == 'a':
        if -120 <= number <= 120:
            real_num = number
            real_cmd = '+'
    elif command == 'b':
        if 0.1 <= number <= 5:
            real_num = number
            real_cmd = '*'
    elif command == 'c':
        if 20 <= number <= 200:
            real_num = number
            real_cmd = 'thresh'
    elif command == 'd':
        if 0.1 <= number <= 5:
            real_num = number
            real_cmd = '*'
    else:
        print('Wrong syntax')
        exit(1)

    if not real_cmd or not real_num:
        print('Wrong syntax')
        exit(1)

    cmd = real_cmd
    num = real_num
    path = os.getcwd()
    img = cv2.imread(path + '/bliss.png', cv2.IMREAD_GRAYSCALE)
    org = deepcopy(img)
    if cmd == '+':
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                img[i][j] += num

    elif cmd == '*':
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                img[i][j] *= num
    elif cmd == 'g':
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                img[i][j] = ((img[i][j] / 255.0) ** num) * 255

    elif cmd == 'thresh':
        dummy, img = cv2.threshold(img, num, 255, cv2.THRESH_TOZERO)

    else:
        print('Wrong syntax')
        exit(1)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i][j] = round(img[i][j])

    img = img.astype(int)
    cv2.imwrite(f'pic{str(cmd) + str(num)}.png', img)
    plt.title('Before cleaning')
    plt.imshow(img, cmap='gray')
    plt.show()
    cv2.imshow('Origin', org)

    bad_histo, dicter1 = get_histogram(img)
    entropy1 = 0
    for rk in bad_histo:
        if rk != 0:
            entropy1 += (rk * np.log(rk))

    cdf_before = np.cumsum(bad_histo)
    cdf_1 = np.ma.masked_equal(cdf_before, 0)
    cdf_2 = (cdf_1 - cdf_1.min()) * 255 / (cdf_1.max() - cdf_1.min())
    cdf_after = np.ma.filled(cdf_2, 0)
    for i in range(cdf_after.shape[0]):
        cdf_after[i] = round(cdf_after[i])

    cdf_after = cdf_after.astype(int)
    img_equal = cdf_after[img]
    good_histo, dicter2 = get_histogram(img_equal)
    entropy2 = 0
    for rk in good_histo:
        if rk != 0:
            entropy2 += (rk * np.log(rk))

    plt.title('After cleaning')
    plt.imshow(img_equal, cmap='gray')
    plt.show()

    dict1 = dict(zip(np.arange(0, 256), cdf_before * 256))

    dict2 = dict(zip(np.arange(0, 256), cdf_after))

    plt.bar(list(dicter1.keys()), dicter1.values(), label=f'Histogram before clean\nEntropy: {entropy1}')
    plt.legend(loc='best')
    plt.show()

    plt.plot(list(dict1.keys()), list(dict1.values()), label=f'CDF before cleaning')
    plt.legend(loc='best')
    plt.show()

    plt.bar(list(dicter2.keys()), dicter2.values(), label=f'Histogram after cleaning\nEntropy: {entropy2}')

    plt.legend(loc='best')
    plt.show()

    plt.plot(list(dict2.keys()), list(dict2.values()), label=f'CDF after cleaning')
    plt.legend(loc='best')
    plt.show()


def fourth(M=200, N=200):
    # add gaussian noise
    rhomb = get_rhomb(M, N)
    plt.title('Rhombus')
    plt.imshow(rhomb, cmap='gray')
    plt.show()
    PERECENT = 0.05
    length = round(PERECENT * rhomb.shape[0] * rhomb.shape[1])
    indexes = np.zeros((2, length))
    for i in range(2):
        for j in range(indexes.shape[1]):
            indexes[i][j] = random.randint(0, rhomb.shape[0] - 1)
    for i in range(indexes.shape[1]):
        rhomb[int(indexes[0][i])][int(indexes[1][i])] = np.random.normal(120, 20, 124)[0]
    rhomb = np.abs(rhomb)
    plt.title('Rhombus With noise')
    plt.imshow(rhomb, cmap='gray')
    plt.show()

    cut_brush = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]]) * 0.5  # Derivative and brushing, LAPLASIAN
    # edges_brushed = cv2.filter2D(rhomb, -1, cut_brush).
    edges = np.zeros((M, N))
    cut_m, cut_n = cut_brush.shape
    for j in range(0, N - cut_n + 1):
        for i in range(0, M - cut_m + 1):
            conv = cut_brush * rhomb[i: i + cut_m, j: j + cut_n]
            edges[i][j] = conv.sum()

    edges_brushed = np.abs(edges)  # Abs for edges

    plt.title('Rhombus Edges Brushed')
    plt.imshow(edges_brushed, cmap='gray')
    plt.show()


if __name__ == '__main__':
    cleaner()
    second()
    third()
    fourth()
