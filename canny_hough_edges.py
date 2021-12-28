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


def get_chess(M=150, N=150):
    white_img = np.ones((M, N)) * 255  # Create a black image with Rhombus
    center = (round(M / 2.0), round(N / 2.0))
    R = 50
    for k in range(0, (M // R) * R, R):
        for j in range(0, (N // R) * R, R):
            if (j // R) % 2 != 0 and (k // R) % 2 != 0 and (j + (2 * R)) <= N:
                white_img[k: k + R, j: j + R] = 0
            elif (j // R) % 2 == 0 and (k // R) % 2 == 0:
                white_img[k: k + R, j:  j + R] = 0

    return white_img


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


def conv(img, filt):
    M1, N1 = filt.shape
    M2, N2 = img.shape
    extra_left, extra_right = N1 - 1, N1 - 1
    extra_top, extra_bottom = M1 - 1, M1 - 1

    IMG = np.pad(img, ((extra_left, extra_right), (extra_top, extra_bottom)),
                 mode='constant', constant_values=0)

    IMG2 = np.zeros((M2, N2)).astype(int)
    IMG2[:] = 255

    for i in range(M2):
        for j in range(N2):
            IMG2[i][j] = IMG[i][j] * filt[0][0] + IMG[i][j + 1] * filt[0][1] + IMG[i][j + 2] * filt[0][2] + IMG[i + 1][
                j] * filt[1][0] + IMG[i + 1][j + 1] * filt[1][1] + IMG[i + 1][j + 2] * filt[1][2] + IMG[i + 2][j] * \
                         filt[2][0] + IMG[i + 2][j + 1] * filt[2][1] + IMG[i + 2][j + 2] * filt[2][2]

    return IMG2


def suppres(mix, th):
    M, N = mix.shape
    Z = np.zeros((M, N), dtype=np.uint8)
    angle = th * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255

                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = mix[i, j + 1]
                    r = mix[i, j - 1]
                # angle 45
                elif 22.5 <= angle[i, j] < 67.5:
                    q = mix[i + 1, j - 1]
                    r = mix[i - 1, j + 1]
                # angle 90
                elif 67.5 <= angle[i, j] < 112.5:
                    q = mix[i + 1, j]
                    r = mix[i - 1, j]
                # angle 135
                elif 112.5 <= angle[i, j] < 157.5:
                    q = mix[i - 1, j - 1]
                    r = mix[i + 1, j + 1]

                if (mix[i, j] >= q) and (mix[i, j] >= r):
                    Z[i, j] = mix[i, j]
                else:
                    Z[i, j] = 0

            except IndexError:
                pass

    return Z


def find_close_edges(M=150, N=150):
    chess = get_chess(M, N)
    picture = np.ones((M, N, 3)) * 255
    plt.title(f'Our chess of {M}x{N}')
    plt.imshow(chess, cmap='gray')
    plt.show()

    cut_brush = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]]) * 0.5  # Derivative and brushing, LoG
    img3 = conv(chess, cut_brush)
    # img3 = cv2.filter2D(chess, -1, cut_brush)
    img3 = np.abs(img3)
    maxi = img3.max()
    x_dots, y_dots = np.where(img3 == maxi)
    for i in range(M):
        for j in range(N):
            if chess[i][j] == 0:
                picture[i][j] = (0, 0, 0)

    for k in range(len(x_dots)):
        picture[x_dots[k]][y_dots[k]] = (255, 0, 0)
    cv2.imwrite('chess_dots.jpg', picture)
    plt.imshow(picture.astype(int))
    plt.title(f'Our chess collision points (in RED)')
    plt.show()


def get_weak_strong(mix, weak, strong):
    W = deepcopy(mix)
    S = deepcopy(mix)

    W[W < weak] = 0
    W[W >= strong] = 0
    W[(weak <= W) & (W < strong)] = weak

    S[S < strong] = 0
    S[S >= strong] = 255
    return W, S


def tracking(img, weak, strong=255):
    M, N = img.shape
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if img[i, j] == weak:
                try:
                    if ((img[i + 1, j - 1] == strong) or (img[i + 1, j] == strong) or (img[i + 1, j + 1] == strong)
                            or (img[i, j - 1] == strong) or (img[i, j + 1] == strong)
                            or (img[i - 1, j - 1] == strong) or (img[i - 1, j] == strong) or (
                                    img[i - 1, j + 1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img


def canny_edges(rhom, M=150, N=150):
    plt.title('Our rhombus')
    plt.imshow(rhom, cmap='gray')
    plt.show()

    gauss = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
    sobel_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobel_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    rhom = conv(rhom, gauss)
    rhomY = conv(rhom, sobel_Y)
    rhomX = conv(rhom, sobel_X)
    rhomb_MIX = np.hypot(rhomX, rhomY)
    rhomb_MIX = rhomb_MIX / rhomb_MIX.max() * 255
    rhomb_MIX = np.rint(rhomb_MIX)

    theta = np.arctan2(rhomY, rhomX)
    suppressed = suppres(rhomb_MIX, theta)
    weak = 38
    strong = 130
    best = get_weak_strong(suppressed, weak, strong)
    strong_cleaned = conv(best[1], gauss)
    weak_cleaned = conv(best[0], gauss)
    mixed = weak_cleaned + strong_cleaned
    mixed[mixed > 255] = 255
    final = tracking(deepcopy(mixed), 240, 250)
    final[final > 255] = 255
    final[final < 50] = 0
    plt.title('weak')
    plt.imshow(weak_cleaned, cmap='gray')
    plt.show()
    plt.title('strong')
    plt.imshow(strong_cleaned, cmap='gray')
    plt.show()
    # plt.title('Final result')
    # plt.imshow(final, cmap='gray')
    # plt.show()
    return final


def hough_line(img):
    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(-90.0, 90.0))
    width, height = img.shape
    diag_len = float(np.ceil(np.sqrt(width * width + height * height)))  # max_dist
    print(type(diag_len))
    rhos = np.linspace(start=-diag_len, stop=diag_len, num=int(diag_len * 2.0))
    # Cache some resuable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)
    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * int(diag_len), int(num_thetas)))
    y_idxs, x_idxs = np.nonzero(img)
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            rho = int(round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len)
            accumulator[rho, t_idx] += 1
    return accumulator, thetas, rhos


def hough(M=200, N=200):
    rhomb = get_rhomb(M, N)
    # plt.title('Rhombus')
    # plt.imshow(rhomb, cmap='gray')
    # plt.show()
    PERECENT = 0.05
    length = round(PERECENT * rhomb.shape[0] * rhomb.shape[1])
    indexes = np.zeros((2, length))
    for i in range(2):
        for j in range(indexes.shape[1]):
            indexes[i][j] = random.randint(0, rhomb.shape[0] - 1)
    for i in range(indexes.shape[1]):
        rhomb[int(indexes[0][i])][int(indexes[1][i])] = np.random.normal(120, 20, 124)[0]
    rhomb = np.abs(rhomb)
    canny = canny_edges(rhomb, rhomb.shape[0], rhomb.shape[1])
    plt.title('Canny')
    plt.imshow(canny, cmap='gray')
    plt.show()

    maxx = find_maxs(canny, 4)
    YS = []
    for key in list(maxx.keys()):
        ro, th = maxx[key]
        m = -np.cos(th) / np.sin(th)
        b = ro / np.sin(th)
        y = m * np.arange(0, canny.shape[0]) + b
        YS.append(y)

    cleaned = np.zeros((M, N)).astype(np.uint8)
    collision = []
    for j, y in enumerate(YS):
        y = [int(val) for val in y]
        cv2.line(cleaned, (0, y[0]), (canny.shape[0] - 1, y[-1]), (255, 255, 255), thickness=1)
        for k, y2 in enumerate(YS):
            if j == k:
                continue
            y2 = [int(val) for val in y2]

            for i in range(len(y) - 1):
                if y[i] == y2[i + 1] or y[i] == y2[i]:
                    collision.append((i, y[i]))
                    break

    collision = sorted(collision, key= lambda x: x[0])
    new = []
    for i, val in enumerate(collision):
        if not val:
            continue
        curx, cury = val

        for j, val2 in enumerate(collision):
            if not val2:
                continue
            nextx, nexty = val2
            if curx == nextx and 0 < abs(cury - nexty) < 4:
                collision[j] = None

    collision = [x for x in collision if x]
    print(collision)
    for dot in collision:
        x, y = dot
        
    plt.imshow(cleaned, cmap='gray')
    plt.show()


def find_maxs(canny, nums: int):
    accumulator, thetas, rhos = hough_line(canny)
    maxes = {}
    for i in range(nums):
        idx = np.argmax(accumulator)
        rind, tind = int(idx / accumulator.shape[1]), int(idx % accumulator.shape[1])
        rho = rhos[rind]
        theta = thetas[tind]
        maxes[idx] = (rho, theta)
        accumulator[rind, tind] = -1  # Reset
        print("rho={0:.2f}, theta={1:.0f}".format(rho, np.rad2deg(theta)))

    return maxes


if __name__ == '__main__':
    # find_close_edges()
    # canny_edges(get_rhomb())
    hough()
