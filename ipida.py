
# ------------------- Image Processing : Ex2_q3 :------------------------¶
# ------------------------------idan kelman -----------------------------¶
# -----------------------------  18.11.2021 -----------------------------¶


# ==============================================================================================
#                                   Timports
# ==============================================================================================


import cv2
import numpy as np
import matplotlib.pylab as plt
import math
from operator import itemgetter


# ==============================================================================================
#                                   Show Functions
# ==============================================================================================


def ShowImages(images ,names=''):

    newImages = []
    scale_percent = 100 # percent of original size
    width = int(images[0].shape[1] * scale_percent / 100)
    height = int(images[0].shape[0] * scale_percent / 100)
    dim = (width, height)

    for im in images:
        resized = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
        newImages.append(resized)
    numpy_horizontal = np.hstack(newImages)
    cv2.imshow('' ,numpy_horizontal)
    cv2.waitKey()
    cv2.destroyAllWindows()


def ShowImage(img ,name =''):

    '''a function that takes in an image , and shows it
    until key is pressed. if no name is spesified , then
    the name will be blank
    '''
    cv2.imshow(name ,img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def PutText(img ,txt):
    tmp = img.astype(np.uint8)
    y0, dy = 50, 40
    for i, line in enumerate(txt.split('\n')):
        y = y0 + i* dy
        cv2.putText(tmp, line, (50, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, [0, 0, 0], thickness=5)
        cv2.putText(tmp, line, (50, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, [255, 255, 255], thickness=1)

    return tmp


# ==============================================================================================
#                                   Generate function
# ==============================================================================================

def GenerateRhombus(ImSize, P):
    if P > ImSize or ImSize <= 0 or P <= 0:
        ImSize = 200
        P = 100

    startY = int((ImSize - P) / 2)
    startX = int((ImSize - P) / 2)
    Fill = np.zeros(ImSize).astype(np.uint8)
    newIm = []
    left = int(ImSize / 2)
    right = int(ImSize / 2)

    for i in range(0, startY):
        newIm.append(Fill.copy())

    for j in range(int(P / 2) - 1):
        Fill[left] = 255
        Fill[right] = 255
        left -= 1
        right += 1
        newIm.append(Fill.copy())

    for j in range(int(P / 2)):
        Fill[left] = 0
        Fill[right] = 0
        left += 1
        right -= 1
        newIm.append(Fill.copy())

    for i in range(0, startY):
        newIm.append(Fill.copy())

    return np.array(newIm)


# ==============================================================================================
#                                   Basic Operations
# ==============================================================================================

def GrayScale2(img):
    '''this function also converts the image to grayscale
    but a lot faster . doesnt itterate over all of the pixels
    , instead  , just creates a new array by the indecies of
    all of the colors . ofc opencv uses BGR and uint8'''
    B, G, R = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    imgGray = 0.2989 * R + 0.5870 * G + 0.1140 * B
    imgGray = imgGray.astype(np.uint8)
    return imgGray


def Derivative(img):
    newIm = img.copy()
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            newIm[i][j] = (img[i][j + 1] - img[i][j - 1]) * 100

    return newIm


def convolution2d(kernel, image, bias=0):
    m, n = kernel.shape
    if (m == n):
        y, x = image.shape
        y = y - m + 1
        x = x - m + 1
        new_image = np.zeros((y, x))
        for i in range(y):
            for j in range(x):
                new_image[i][j] = np.sum(image[i:i + m, j:j + m] * kernel) + bias

    return new_image


def AddNoise(amount, img):
    newIm = img.copy()
    n = img.shape[0] * img.shape[1]
    roof = int(n / amount) * 20
    adder = 100
    for t in range(roof):
        i = int(np.random.randint(img.shape[0]))
        j = int(np.random.randint(img.shape[1]))
        if i % 2 == 0:
            newIm[i][j] += adder
        else:
            newIm[i][j] -= adder

        if newIm[i][j] > 250:
            newIm[i][j] = 250

        if newIm[i][j] < 0:
            newIm[i][j] = 0

    return newIm


def changeThreshold(img, val, g=255):
    newIm = img.astype(np.uint8)
    newIm[newIm > val] = 255
    newIm[newIm <= val] = 0
    newIm[newIm >= 255] = g

    # print(newIm)
    return newIm


def changeThresholdKey(img):
    val = 90
    key = 0
    NewIm = img.astype(np.uint8)
    txt = "press 'a' to dec , and 'd' to inc"
    tmp = PutText(NewIm, txt)

    while (key != 27):
        cv2.imshow('', tmp)
        key = cv2.waitKey()
        if val > 20 and val < 200:
            if key == 100:  # normally -1 returned,so don't print it
                val += 1
                NewIm = changeThreshold(img, val)
            if key == 97:
                val -= 1
                NewIm = changeThreshold(img, val)

        tmp = PutText(NewIm, "your image with T : {}\n press 'a' to dec , and 'd' to inc".format(val))

    return NewIm


# ==============================================================================================
#                                   Canny Edge Detector
# ==============================================================================================


def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.uint8)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255

                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                # angle 45
                elif (22.5 <= angle[i, j] < 67.5):
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                # angle 90
                elif (67.5 <= angle[i, j] < 112.5):
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                # angle 135
                elif (112.5 <= angle[i, j] < 157.5):
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass

    return Z


def hysteresis(img, weak, strong=255):
    M, N = img.shape
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if (img[i, j] == weak):
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


def cannyEdge(img):
    # Images = []

    Rec = img
    # ShowImage(Rec)
    # Images.append(PutText(Rec,'OriginalIm'))

    kernel = np.ones((3, 3), np.float32) / 9
    Blur = convolution2d(kernel, Rec)

    # Canny Edge Detection :

    # Step 1 : Smooth the image
    gaussian = np.array(([1, 2, 1], [2, 4, 2], [1, 2, 1])) / 16
    Smoothed = convolution2d(gaussian, Blur)

    # Step 2: Calcualte Gradient and magnitude
    Sobelx = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]))
    Sobely = np.array(([-1, -2, -1], [0, 0, 0], [1, 2, 1]))
    Dx = convolution2d(Sobelx, Smoothed)
    Dy = convolution2d(Sobely, Smoothed)

    # print(Dx)
    # print(Dy)

    Mag = np.hypot(Dx, Dy)
    Mag = Mag / Mag.max() * 255
    Ang = np.arctan2(Dy, Dx)

    # Step2 = []

    # Images.append(PutText(Smoothed,''))
    # Step2.append(PutText(Dx,'Dx'))
    # Step2.append(PutText(Dy,'Dy'))

    # ShowImage(Mag)
    # Step2.append(PutText(Mag,'Mag'))

    # ShowImages(Step2)

    # Step 3: None maximum supression prependicular to edge

    Sup = non_max_suppression(np.copy(Mag), Ang)

    # ShowImage(PutText(Sup,'non_max_suppression'))
    # Images.append(PutText(Mag,''))
    # Images.append(PutText(Sup,'non_max_suppression'))

    # Step 4: Threshold using two thresholds . (strong , weak , no edge)

    # Threshold = []

    Weak = changeThreshold(Sup, 50, 240)
    Strong = changeThreshold(Sup, 100, 255)
    # Threshold.append(PutText(Weak,'Weak'))
    # Threshold.append(PutText(Strong,'Strong'))
    # ShowImages(Threshold)

    # Step 5 : Connect together componenets

    Comb = Weak + Strong
    Comb[Comb > 255] = 255
    # ShowImage(PutText(Weak,'Weak'))
    # ShowImage(PutText(Comb,'Comb'))
    F_Image = hysteresis(np.copy(Comb), 240, 255)
    # Images.append(PutText(Comb,'Combined'))
    # Images.append(PutText(F_Image,'F_Image'))

    # ShowImages(Images)

    return F_Image


# ==============================================================================================
#                                  Hough Transfrom
# ==============================================================================================


def HoughSpace(img):
    # Step 2: Hough Space

    img_shape = img.shape

    x_max = img_shape[0]
    y_max = img_shape[1]

    theta_max = 1.0 * math.pi
    theta_min = 0.0

    r_min = 0.0
    r_max = math.hypot(x_max, y_max)

    r_dim = 200
    theta_dim = 300

    hough_space = np.zeros((r_dim, theta_dim))

    for x in range(x_max):
        for y in range(y_max):
            if img[x, y] == 255: continue
            for itheta in range(theta_dim):
                theta = 1.0 * itheta * theta_max / theta_dim
                r = x * math.cos(theta) + y * math.sin(theta)
                ir = int(r_dim * (1.0 * r) / r_max)
                hough_space[ir, itheta] += 1

    plt.imshow(hough_space, origin='lower')
    plt.xlim(0, theta_dim)
    plt.ylim(0, r_dim)

    tick_locs = [i for i in range(0, theta_dim, 40)]
    tick_lbls = [round((1.0 * i * theta_max) / theta_dim, 1) for i in range(0, theta_dim, 40)]
    plt.xticks(tick_locs, tick_lbls)

    tick_locs = [i for i in range(0, r_dim, 20)]
    tick_lbls = [round((1.0 * i * r_max) / r_dim, 1) for i in range(0, r_dim, 20)]
    plt.yticks(tick_locs, tick_lbls)

    plt.xlabel(r'Theta')
    plt.ylabel(r'r')
    plt.title('Hough Space')

    plt.savefig("hough_space_r_theta.png", bbox_inches='tight')
    plt.show()

    plt.close()

    return hough_space


def DetectLines(hough_space, Picks):
    Cords = []
    amount = len(Picks)
    for i in range(0, hough_space.shape[0]):
        for j in range(0, hough_space.shape[1]):
            if (hough_space[i][j] in Picks):
                Cords.append([i, j])
                amount -= 1
                print(hough_space[i][j])
            if (amount <= 0):
                return Cords

    return Cords


def toCertisian(Cords):
    for i in range(0, len(Cords)):
        r = Cords[i][0]
        theta = Cords[i][1]
        Cords[i][0] = np.cos(theta) * r
        Cords[i][1] = np.sin(theta) * r

    return Cords


def N_max_elements(arr, N):
    list = np.concatenate(arr)

    list = np.sort(list)
    list = list[::-1]
    result_list = []
    result_list = list[0:N]

    return result_list


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


def AddLines(img, Cords, rhos, thetas):
    for cordinate in Cords:
        rho = rhos[cordinate[0]]
        theta = thetas[cordinate[1]]
        a = np.cos(theta)
        b = np.sin(theta)

        x0 = a * rho
        y0 = b * rho

        x1 = int(x0 + 1000 * (-b))

        y1 = int(y0 + 1000 * (a))

        x2 = int(x0 - 1000 * (-b))

        y2 = int(y0 - 1000 * (a))

        cv2.line(img, (x1, y1), (x2, y2), 255, 2)
        print("rho={0:.2f}, theta={1:.0f}".format(rho, np.rad2deg(theta)))

    return img


# ==============================================================================================
#                                   Main
# ==============================================================================================


def main():
    # Step 1 : Generate the image :
    # you can pick whatever size of the image you like ,
    # how many rows ,
    # and how many cols .

    Images = []
    Rec = GenerateRhombus(200, 150)
    ShowImage(Rec)
    w, h = Rec.shape
    plain = np.zeros((h, w), dtype=np.uint8)
    Images.append(PutText(Rec, 'OriginalIm'))

    # Step 2: add noise

    noiseAmount = 75
    noised = AddNoise(noiseAmount, Rec)
    ShowImage(noised)
    Images.append(PutText(noised, 'added noise with {}%'.format(noiseAmount)))

    # Step 3: canny Edge Detector

    CannyIm = cannyEdge(noised)
    Images.append(PutText(CannyIm, 'Canny'))

    # Step 4: Hough Space Calcualtion
    # itterate over all of the points that are lit
    # itterate over all of R , theta
    # hold a voting matrix for all of the points that those functions
    # went through.

    # Hspace = HoughSpace(Rec)
    accumulator, thetas, rhos = hough_line(CannyIm)
    ShowImage(accumulator / 255)

    # Step 4: find n most significant edges(or by a threshold)

    Maxs = N_max_elements(accumulator.copy(), 6)
    Cords = DetectLines(accumulator, Maxs.copy())

    # Step 5: draw the lines that build up the original image
    DrawLines = AddLines(plain, Cords, rhos, thetas)
    ShowImage(DrawLines)
    Images.append(PutText(DrawLines, 'DrawLines'))

    ShowImages(Images)


main()
