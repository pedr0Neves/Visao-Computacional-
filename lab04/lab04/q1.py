import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import sqrt, atan2, pi
from collections import defaultdict

def pre_processing(img, show_steps=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    if show_steps:
        cv2.imshow('Blurred', blur)
        cv2.waitKey(0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(blur)

    if show_steps:
        cv2.imshow('Equalized', equalized)
        cv2.waitKey(0)

    edges = cv2.Canny(blur, 120, 240)

    if show_steps:
        cv2.imshow('Edges', edges)
        cv2.waitKey(0)
    
    return edges, equalized

def processing(blur, edges):
    circles = cv2.HoughCircles(blur, 
                               cv2.HOUGH_GRADIENT, 
                               dp=1.2, 
                               minDist=50, 
                               param1=180, 
                               param2=100,
                               minRadius=30, 
                               maxRadius=100)

    lines = cv2.HoughLinesP(edges, 
                            rho=1, 
                            theta=np.pi/180, 
                            threshold=65, 
                            minLineLength=30, 
                            maxLineGap=6)

    return circles, lines

def pos_processing(lines, circles, edges, img, show_steps=True):
    im_hough = np.copy(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(im_hough, (x1, y1), (x2, y2), (0, 0, 200), 3)

    im_circle = np.copy(img)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Desenhar círculo externo
            cv2.circle(im_hough, (i[0], i[1]), i[2], (0, 0, 200), 3)
            cv2.circle(im_circle, (i[0], i[1]), i[2], (0, 0, 200), 3)
            # Desenhar centro do círculo
            cv2.circle(im_hough, (i[0], i[1]), 2, (0, 0, 200), 3)
            cv2.circle(im_circle, (i[0], i[1]), 2, (0, 0, 200), 3)

    if show_steps:
        cv2.imshow('linhas', im_hough)
        cv2.waitKey(0)
    
    im_square = np.copy(img)
    im_oct = np.copy(img)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        approx_sq = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx_sq) == 4:
            cv2.drawContours(im_square, [approx_sq], 0, (0, 0, 200), 3)
        if len(approx) == 8:
            cv2.drawContours(im_oct, [approx], 0, (0, 0, 200), 3)

    return im_circle, im_square, im_oct

def print_img(im_circle, im_squared, im_oct):
    plt.subplot(311),plt.imshow(im_circle, cmap='gray'),plt.title('Circle')
    plt.xticks([]), plt.yticks([])
    plt.subplot(312),plt.imshow(im_squared,cmap = 'gray'),plt.title('Squared')
    plt.xticks([]), plt.yticks([])
    plt.subplot(313),plt.imshow(im_oct, cmap='gray'),plt.title('Octagon')
    plt.xticks([]), plt.yticks([])
    plt.show()


def main():
    filename = "imagens/placas-transito.jpg"
    im = cv2.imread(filename=filename)
    if im is None:
        raise ValueError(f"Não foi possível carregar a imagem: {filename}")
    img = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

    edges, blur = pre_processing(img)
    circles, lines = processing(blur, edges)
    im_circle, im_square, im_oct = pos_processing(lines, circles, edges, img)
    print_img(im_circle, im_square, im_oct)

if __name__ == "__main__":
    main()