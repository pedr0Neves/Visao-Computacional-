import cv2
import numpy as np
import matplotlib.pyplot as plt

path = 'imagens/colorful-umbrellas.png'
im = cv2.imread(path)
im_sepia = np.zeros_like(im)

kernel_sepia = np.uint32([[0.393, 0.769, 0.189],
                          [0.349, 0.686, 0.168],
                          [0.272, 0.534, 0.131]])

im_sepia[:,:,0] = np.minimum(kernel_sepia[2, 0] * im[:,:,0] + kernel_sepia[2, 1] * im[:,:,2] + kernel_sepia[2, 2] * im[:,:,1], 255)
im_sepia[:,:,1] = np.minimum(kernel_sepia[1, 0] * im[:,:,0] + kernel_sepia[1, 1] * im[:,:,2] + kernel_sepia[1, 2] * im[:,:,1], 255)
im_sepia[:,:,2] = np.minimum(kernel_sepia[0, 0] * im[:,:,0] + kernel_sepia[0, 1] * im[:,:,2] + kernel_sepia[0, 2] * im[:,:,1], 255)

im_sepia = im_sepia.astype(np.uint8)

plt.subplot(1, 2, 1), plt.imshow(im, cmap='gray')
plt.subplot(1, 2, 2), plt.imshow(im_sepia, cmap='gray')
plt.show()