import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

im_line = cv2.imread('imagens/line.jpg')
im_circle = cv2.imread('imagens/circle.jpg')

#criando uma imagem 300x300 full branca
blank_image = np.ones((300, 300, 3), np.uint8)
blank_image = cv2.bitwise_not(blank_image)
blank_h, blank_w = blank_image.shape[:2]

# pegando caracteristicas das imagens
# height = blank_image.shape[0]
# width = blank_image.shape[1]
height, width = blank_image.shape[:2]
line_height, line_width = im_line.shape[:2]
circle_height, circle_width = im_circle.shape[:2]

# modelando o corpo
M_rotation_body = cv2.getRotationMatrix2D((line_height/2, line_width/2), 90, 1)
im_body = cv2.warpAffine(im_line, M_rotation_body, (line_height, line_width))

# criando mascara de 100x100 pixels
mask = np.zeros((100, 100, 3), np.uint8)
mask = cv2.rectangle(mask, (0, 0), (100, 100), (255, 255, 255), -1)
M_rotation = cv2.getRotationMatrix2D((mask.shape[1]/2, mask.shape[0]/2),90, 1)
mask = cv2.warpAffine(mask, M_rotation, (mask.shape[1], mask.shape[0]))

# aplicando a mascara no corpo
im_body = cv2.bitwise_xor(im_body, mask)
im_body = cv2.bitwise_not(im_body)

# pegando cordenadas do corpo
y = int((height - im_body.shape[1])/2 - 30)
x = int((width - im_body.shape[0])/2)

# adicionando o corpo
roi = blank_image[y:y+im_body.shape[0], x:x+im_body.shape[1]]
roi[:] = cv2.bitwise_and(roi, im_body)
blank_image[y:y+im_body.shape[0], x:x+im_body.shape[1]] = roi

# modelando a cabeça
#pegando as cordenadas do cabeça
y = int((height - circle_height - im_body.shape[0])/2 - 30)
x = int((width - circle_width)/2)

# adicionando a cabeça ao corpo
roi = blank_image[y:y+circle_height, x:x+circle_width]
roi[:] = cv2.bitwise_and(roi, im_circle)
blank_image[y:y+circle_height, x:x+circle_width] = roi

# modelando os braços
arm_width = line_width * 0.75
im_arm = cv2.resize(im_line, (int(arm_width), int(line_height)), interpolation=cv2.INTER_LINEAR)

# pegando as coordenadas do braço
y = int((height - line_height)/2 - 30)
x = int(width/2 - 10)

# adicionando o braço ao corpo
roi = blank_image[y:y+int(line_height), x:x+int(arm_width)]
roi[:] = cv2.bitwise_and(roi, im_arm)
blank_image[y:y+int(line_height), x:x+int(arm_width)] = roi

# modelando as pernas
leg_width = arm_width * 2
im_leg = cv2.resize(im_line,(int(leg_width), int(line_height)), interpolation=cv2.INTER_LINEAR)
M_rotation_leg = cv2.getRotationMatrix2D((leg_width/2, line_height/2), -45, 1)
im_leg = cv2.warpAffine(im_leg, M_rotation_leg, (int(leg_width), int(line_height)))

# criando a mascara para as pernas
mask = np.zeros((100, 100, 3), np.uint8)
mask = cv2.rectangle(mask, (0, 0), (100, 100), (255, 255, 255), -1)
mask = cv2.resize(mask, (int(mask.shape[1] * 1.5), mask.shape[0]), interpolation=cv2.INTER_LINEAR)
M_rotation = cv2.getRotationMatrix2D((mask.shape[1]/2, mask.shape[0]/2), -45, 1)
mask = cv2.warpAffine(mask, M_rotation, (mask.shape[1], mask.shape[0]))

# aplicando a mascara nas pernas
im_leg = cv2.bitwise_xor(im_leg, mask)
im_leg = cv2.bitwise_not(im_leg)

# pegando as coordenadas das pernas
y = int(height/2 + line_height/2 - 44)
x = int(width/2 - line_width/4 - 6)

# adicionando a perna ao corpo
roi = blank_image[y:y+im_leg.shape[0], x:x+im_leg.shape[1]]
roi[:] = cv2.bitwise_and(roi, im_leg)
blank_image[y:y+im_leg.shape[0], x:x+im_leg.shape[1]] = roi

blank_image = cv2.bitwise_and(np.flip(blank_image, 1), blank_image)

cv2.imshow("boneco", blank_image)
cv2.waitKey(0)
cv2.destroyAllWindows()