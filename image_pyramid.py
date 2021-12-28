# from google.colab import drive
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


# def get_face_mask(shape):
#     M, N, three = shape
#     mask = np.zeros((M, N))
#     center_head = (M//2, N//2 - 30)

#     mask[400:center_head[0]+10,  center_head[1]: center_head[1] + 100] = 255


#     return mask


def Laplacian_Pyramid_Blending_with_mask(A, B, m, num_levels=6):
    # assume mask is float32 [0,1]
    # generate Gaussian pyramid for A,B and mask
    GA = A.copy()
    GB = B.copy()
    GM = m.copy()
    gpA = [GA]
    gpB = [GB]
    gpM = [GM]
    for i in range(num_levels):
        GA = cv2.pyrDown(GA)
        GB = cv2.pyrDown(GB)
        GM = cv2.pyrDown(GM)
        gpA.append(np.float64(GA))
        gpB.append(np.float64(GB))
        gpM.append(np.float64(GM))

        # the bottom of the Lap-pyr holds the last (smallest) Gauss level
    lpA = [gpA[num_levels - 1]]
    lpB = [gpB[num_levels - 1]]
    gpMr = [gpM[num_levels - 1]]
    for i in range(num_levels - 1, 0, -1):
        print(i)
        # Laplacian: subtarct upscaled version of lower
        # level from current level
        # to get the high frequencies
        LA = np.subtract(gpA[i - 1], cv2.pyrUp(gpA[i]))
        LB = np.subtract(gpB[i - 1], cv2.pyrUp(gpB[i]))
        lpA.append(LA)
        lpB.append(LB)
        gpMr.append(gpM[i - 1])  # also reverse the masks

    LS = []
    for la, lb, gm in zip(lpA, lpB, gpMr):
        ls = la * gm + lb * (1.0 - gm)
        ls.dtype = np.float64
        LS.append(ls)
    # now reconstruct
    ls_ = LS[0]
    # ls_.dtype = np.float64
    for i in range(1, num_levels):
        print("LS" + str(i))
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])
    return ls_


if __name__ == "__main__":
    # drive.mount('content')

    doron1 = cv2.imread(f'd11.png', 1)
    doron1 = cv2.cvtColor(doron1, cv2.COLOR_BGR2RGB)
    _, ax = plt.subplots(1, 3)
    ax[0].imshow(doron1)
    ax[0].title.set_text('Doron')
    ax[0].axis('off')

    idan1 = cv2.imread(f'i11.png', 1)
    idan1 = cv2.cvtColor(idan1, cv2.COLOR_BGR2RGB)

    ax[1].imshow(idan1)
    ax[1].title.set_text('Idan')
    ax[1].axis('off')

    mask = cv2.imread(f'mask.png', 1).astype(float)
    mask[mask != 0.0] = 1.0
    ax[2].imshow(mask[:, :, 0], cmap='gray')
    ax[2].title.set_text('Mask')
    ax[2].axis('off')

    plt.show()

    res = Laplacian_Pyramid_Blending_with_mask(idan1, doron1, mask).astype(int)

    plt.title('Result')
    plt.axis('off')
    plt.imshow(res)
    plt.show()
