import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

filename = sys.argv[1]
im = cv2.imread(filename)
im_dst = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

hue = im_dst[:,:,0]
hist, _ = np.histogram(hue, bins=180)

h = 100 # azul
g = 50 # verde
t = 20 # tolerancia

min_nebula = np.array([h - t])
max_nebula = np.array([h + t])
mask_nebula = cv2.inRange(hue, min_nebula, max_nebula)

min_gamora = np.array([g - t])
max_gamora = np.array([g + t])
mask_gamora = cv2.inRange(hue, min_gamora, max_gamora)

hue[mask_nebula > 0] = (hue[mask_nebula > 0] - 50) % 180
hue[mask_gamora > 0] = (hue[mask_gamora > 0] + 50) % 180

im_dst[:,:,0] = hue
im_new = cv2.cvtColor(im_dst, cv2.COLOR_HSV2RGB)

#cv2.imshow('original', im)
#cv2.imshow('mascara gamora', mask_gamora)
#cv2.imshow('mascara nebula', mask_nebula)
#cv2.imshow('modificado', im_new)
#cv2.waitKey(0)
#cv2.destroyAllWindows

plt.figure(figsize=(20, 10))
plt.subplot(2,3,1), plt.imshow(im), plt.title('original')
plt.subplot(2,3,2), plt.imshow(hue), plt.title('hue channel')
plt.subplot(2,3,3), plt.plot(hist), plt.title('hue histogram')
plt.subplot(2,3,4), plt.imshow(mask_gamora, cmap='gray'), plt.title('mascara gamora')
plt.subplot(2,3,5), plt.imshow(mask_nebula,  cmap='gray'), plt.title('mascara nebulosa')
plt.subplot(2,3,6), plt.imshow(im_new), plt.title('modificada')
plt.tight_layout()
plt.show()