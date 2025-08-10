import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randrange

image1 = cv2.imread("imagens/foto1.jpg")
image2 = cv2.imread("imagens/foto2.jpg")
image3 = cv2.imread("imagens/foto3.jpg")

h1 = image1.shape[0]
w1 = image1.shape[1]
image1 = cv2.resize(image1, (int(w1*0.2), int(h1*0.2)))

h2 = image2.shape[0]
w2 = image2.shape[1]
image2 = cv2.resize(image2, (int(w2*0.2), int(h2*0.2)))

h3 = image3.shape[0]
w3 = image3.shape[1]
image3 = cv2.resize(image3, (int(w3*0.2), int(h3*0.2)))

img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
img3 = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)

"""
def tecn1(img1, img2, transformation_matrix):
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]
    height = max(height1, height2)
    width = width1 + width2
    img1_transformed = cv2.warpPerspective(img1, transformation_matrix, (width, height))

    #img2 = cv2.resize(img2, (width, height))
    # Criar uma imagem para mostrar a combinação
    combined_img = cv2.addWeighted(img1_transformed, 0.5, img2, 0.5, 0)
    return combined_img
"""

def tecn2(im1, im2, transformation_matrix):
    height1, width1 = im1.shape[:2]
    height2, width2 = im2.shape[:2]
    transformation_matrix = transformation_matrix.astype(np.float32)

    # Transformar os cantos de img1 para ver onde eles acabam em img2
    corners_img1 = np.float32([[0, 0], [0, height1], [width1, height1], [width1, 0]]).reshape(-1, 1, 2)
    transformed_corners_img1 = cv2.perspectiveTransform(corners_img1, transformation_matrix)

    # Encontrar os limites da nova imagem combinada
    all_corners = np.concatenate((transformed_corners_img1, np.float32([[0, 0], [0, height2], [width2, height2], [width2, 0]]).reshape(-1, 1, 2)), axis=0)
    min_x = min(int(np.min(all_corners[:,:,0])), 0)
    min_y = min(int(np.min(all_corners[:,:,1])), 0)
    max_x = max(int(np.max(all_corners[:,:,0])), width2)
    max_y = max(int(np.max(all_corners[:,:,1])), height2)

    # Translação para compensar as coordenadas negativas
    # trans_mat = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
    trans_mat = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float32)

    # Aplicar homografia com a compensação de translação
    img_output_size = (max_x - min_x, max_y - min_y)
    img1_transformed = cv2.warpPerspective(im1, trans_mat.dot(transformation_matrix), img_output_size)
    img2_transformed = cv2.warpPerspective(im2, trans_mat, img_output_size)

    # Posicionar img2 na nova imagem de saída ajustada
    output_image = np.zeros((max_y - min_y, max_x - min_x, 3), dtype=image1.dtype)
    output_image[-min_y:height2-min_y, -min_x:width2-min_x] = im2

    # Mesclar img1_transformed sobre a nova área de img2
    output_image = np.where(img1_transformed.sum(axis=-1, keepdims=True)!=0, img1_transformed, output_image)
    return output_image

def homograpy(im1, im2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(im1,None)
    kp2, des2 = sift.detectAndCompute(im2,None)
    bf = cv2.BFMatcher()

    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    if len(good) >= 4:
        pts1 = []
        pts2 = []
        for m in good:
            pts1.append(kp1[m[0].queryIdx].pt)
            pts2.append(kp2[m[0].trainIdx].pt)

        points1 = np.float32(pts1).reshape(-1, 1, 2)
        points2 = np.float32(pts2).reshape(-1, 1, 2)

        transformation_matrix, inliers = cv2.findHomography(points1, points2, cv2.RANSAC)
        if transformation_matrix is None:
            raise ValueError("Falha ao calcular homografia - verifique seus pontos de correspondência")
    else:
        raise AssertionError("No enough keypoints.")
    
    return tecn2(im1, im2, transformation_matrix)

new_img1 = homograpy(image1, image2)
new_img2 = homograpy(image3, image2)
panorama = homograpy(image3, new_img1)

print(panorama.shape)
h, w = panorama.shape[:2]

# Definir as coordenadas do retângulo de corte (x1, y1, x2, y2)
x1, y1 = 0, int(h/2 - 150)  # Canto superior esquerdo (início do corte)
x2, y2 = int(w), int(h/2 + 150)  # Canto inferior direito (fim do corte)

# Cortar a imagem: imagem[y1:y2, x1:x2]
panorama = panorama[y1:y2, x1:x2]

print(panorama.shape)

#cv2.imshow("img1", image1)
#cv2.imshow("img2", image2)
#cv2.imshow("img3", image3)
#cv2.imshow("panorama 1", new_img1)
#cv2.imshow("panorama 2", new_img2)
cv2.imshow("panorama final", panorama)
cv2.waitKey(0)
cv2.destroyAllWindows()