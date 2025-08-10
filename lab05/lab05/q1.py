import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from cv_utils import waitKey

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
    output_image = np.zeros((max_y - min_y, max_x - min_x, 3), dtype=im1.dtype)
    output_image[-min_y:height2-min_y, -min_x:width2-min_x] = im2

    # Mesclar img1_transformed sobre a nova área de img2
    output_image = np.where(img1_transformed.sum(axis=-1, keepdims=True)!=0, img1_transformed, output_image)
    return output_image

def homograpy(im1, im2, detector, bf):
    kp1, des1 = detector.detectAndCompute(im1,None)
    kp2, des2 = detector.detectAndCompute(im2,None)

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

def stitch_img(im1, im2, descriptor="sift"):
    gry1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    gry2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    detector = None
    bf = None
    if descriptor == "sift":
        detector = cv2.SIFT_create()
        bf = cv2.BFMatcher()
    elif descriptor == "orb":
        detector = cv2.ORB_create(nfeatures=1000)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    elif descriptor == "surf":
        if 'xfeatures2d' in dir(cv2):
            detector = cv2.xfeatures2d.SURF_create()
            bf = cv2.BFMatcher()
        else:
            return None
    elif descriptor == "kaze":
        detector = cv2.KAZE_create()
        bf = cv2.BFMatcher()
    elif descriptor == "akaze":
        detector = cv2.AKAZE_create()
        bf = cv2.BFMatcher()
    elif descriptor == "brisk":
        detector = cv2.BRISK_create()
        bf = cv2.BFMatcher()
    else:
        print(f"Descritor {descriptor} não suportado.")
        return None
    
    return homograpy(im1, im2, detector, bf)

def main():
    try:
        img1 = cv2.imread('imagens/foto6.jpg')
        img2 = cv2.imread('imagens/foto7.jpg')

        if img1 is None:
            raise FileNotFoundError("Não foi possível carregar a imagem 1")
        if img2 is None:
            raise FileNotFoundError("Não foi possível carregar a imagem 2")

    except FileNotFoundError as e:
        print(e)
        sys.exit(1)


    width = int(img1.shape[1] * 0.3)
    height = int(img1.shape[0] * 0.3)
    dim = (width, height)
    img1 = cv2.resize(img1, dim, interpolation=cv2.INTER_AREA)
    width = int(img2.shape[1] * 0.3)
    height = int(img2.shape[0] * 0.3)
    dim = (width, height)
    img2 = cv2.resize(img2, dim, interpolation=cv2.INTER_AREA)
    

    descriptors = ["sift", "orb", "surf", "kaze", "akaze", "brisk"]

    for descriptor in descriptors:
        panorama = stitch_img(img1, img2, descriptor=descriptor)
        if panorama is not None:
            cv2.imshow(f"Panorama {descriptor.upper()}", panorama)
            waitKey(f"Panorama {descriptor.upper()}", 27)
        else:
            print("Error: não foi possível gerar o panorama.")
            print(f"Error: não foi possível usar {descriptor.upper()}")
    
    cv2.destroyAllWindows()
    print("\nProcesso concluído.")

if __name__ == "__main__":
    main()