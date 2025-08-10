import cv2
import numpy as np
from matplotlib import pyplot as plt
from freq_filters import filter_image_freq

def load_image(paths):
    imgs = []
    for path in paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Error: Arquivo {path} não encontrado")
        imgs.append(img)

    return imgs

def print_image1(imgs):
    plt.figure()
    
    plt.subplot(331), plt.imshow(imgs[0], cmap='gray'), plt.title('halftone')
    plt.xticks([]), plt.yticks([])
    plt.subplot(332), plt.imshow(imgs[1], cmap='gray'), plt.title('pieces')
    plt.xticks([]), plt.yticks([])
    plt.subplot(333), plt.imshow(imgs[2], cmap='gray'), plt.title('salt & pepper')
    plt.xticks([]), plt.yticks([])


def spatial_processing(imgs):
    halftone = imgs[0]
    pieces = imgs[1]
    salt = imgs[2]

    hal_filt = cv2.GaussianBlur(halftone, ksize=(11, 11), sigmaX=1.7)
    sobel_x = cv2.Sobel(pieces,
                        cv2.CV_64F, 1, 0, 
                        ksize=5)
    sobel_y = cv2.Sobel(pieces, 
                        cv2.CV_64F, 0 ,1, 
                        ksize=5)
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    normalize = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    pieces_filt = cv2.addWeighted(pieces, 1, np.uint8(normalize), 0.25, 0)
    salt_filt = cv2.medianBlur(salt, 7)
    
    plt.subplot(334), plt.imshow(hal_filt, cmap='gray'), plt.title('Gaussiano')
    plt.xticks([]), plt.yticks([])
    plt.subplot(335), plt.imshow(normalize, cmap='gray'), plt.title('Sobel')
    plt.xticks([]), plt.yticks([])
    plt.subplot(336), plt.imshow(salt_filt, cmap='gray'), plt.title('Mediana')
    plt.xticks([]), plt.yticks([])

def freq_processing(imgs):
    hal_filt = filter_image_freq(imgs[0], fclass='lowpass', ftype='gaussian', d0=60)
    pieces_filt = filter_image_freq(imgs[1], fclass='highpass', ftype='ideal', d0=40)
    normalize = cv2.normalize(pieces_filt, None, 0, 255, cv2.NORM_MINMAX)
    pieces_filt = cv2.addWeighted(imgs[1], 1, np.uint8(normalize), 0.25, 0)
    salt_filt = filter_image_freq(imgs[2], fclass='lowpass', ftype='butterworth', d0=80, n=3)
    
    plt.subplot(337), plt.imshow(hal_filt, cmap='gray'), plt.title('lowpass gaussian')
    plt.xticks([]), plt.yticks([])
    plt.subplot(338), plt.imshow(normalize, cmap='gray'), plt.title('highpass ideal')
    plt.xticks([]), plt.yticks([])
    plt.subplot(339), plt.imshow(salt_filt, cmap='gray'), plt.title('lowpass butterworth')
    plt.xticks([]), plt.yticks([])

def main():
    paths = ['imagens/halftone.png', 'imagens/pieces.png', 'imagens/salt_noise.png']
    imgs = load_image(paths)
    print_image1(imgs)
    spatial_processing(imgs)
    freq_processing(imgs)
    plt.show()

if __name__ == "__main__":
    main()