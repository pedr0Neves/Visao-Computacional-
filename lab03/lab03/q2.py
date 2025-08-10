import cv2
import numpy as np
from matplotlib import pyplot as plt

path = 'imagens/colorful-umbrellas.png'
im = cv2.imread(path)

#Y = (0.3 x R) + (0.59 x G) + (0.11 x B)
kernel_r = np.float32([[0, 0, 0],
                       [0, 0.3, 0],
                       [0, 0, 0]])
kernel_g = np.float32([[0, 0, 0],
                       [0, 0.59, 0],
                       [0, 0, 0]])
kernel_b = np.float32([[0, 0, 0],
                       [0, 0.11, 0],
                       [0, 0, 0]])

r = cv2.filter2D(im[:,:,2], -1, kernel_r)
g = cv2.filter2D(im[:,:,1], -1, kernel_g)
b = cv2.filter2D(im[:,:,0], -1, kernel_b)

im_gray = r + g + b
im_gray = im_gray.astype(np.uint8)

im_sepia = np.zeros_like(im)
im_sepia[:,:,0] = np.minimum(0.272 * im[:,:,0] + 0.534 * im[:,:,2] + 0.131 * im[:,:,1], 255)  # Canal Azul
im_sepia[:,:,1] = np.minimum(0.349 * im[:,:,0] + 0.686 * im[:,:,2] + 0.168 * im[:,:,1], 255)  # Canal Verde
im_sepia[:,:,2] = np.minimum(0.393 * im[:,:,0] + 0.769 * im[:,:,2] + 0.189 * im[:,:,1], 255)  # Canal Vermelho
im_sepia = im_sepia.astype(np.uint8)


plt.subplot(1, 3, 1), plt.imshow(im, cmap='gray')
plt.subplot(1, 3, 2), plt.imshow(im_gray, cmap='gray')
plt.subplot(1, 3, 3), plt.imshow(im_sepia, cmap='gray')
plt.show()