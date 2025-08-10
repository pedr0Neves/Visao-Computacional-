import cv2
import numpy as np
import matplotlib.pyplot as plt
from pyzbar.pyzbar import decode
from itertools import groupby

filename = "imagens/barcode-code-128.png"
show_staps = False
img = cv2.imread(filename=filename)

if img is None:
    raise ValueError(f"Não foi possível carregar a imagem: {filename}")
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
img = cv2.resize(img, (img.shape[1]*3, img.shape[0]*3), interpolation=cv2.INTER_CUBIC)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#blur = cv2.GaussianBlur(gray, (3, 3), 2)
edge = cv2.Canny(gray, 30, 100, apertureSize=3, L2gradient=True)
if show_staps:
    plt.subplot(211),
    plt.imshow(gray, cmap='gray'), plt.title("original")
    plt.subplot(212),
    plt.imshow(edge, cmap='gray'), plt.title("canny")
    plt.show()

lines = cv2.HoughLines(image=edge,
                       rho=1.0,
                       theta=np.pi/180,
                       threshold=125)

if lines is not None:
    horizontal_lines = []
    for r_theta in lines:
        arr = np.array(r_theta[0], dtype=np.float64)
        r, theta = arr
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*r
        y0 = b*r
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        if abs(theta) == 0:
            horizontal_lines.append((x1, y1, x2, y2))

        for line in horizontal_lines:
            x1, y1, x2, y2 = line
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255))

plt.imshow(img, cmap='gray'), plt.title("hough"), plt.show()


codes = decode(img)
for code in codes:
    data = code.data.decode('utf-8')
    type = code.type
    points = code.polygon

    if type == 'CODE128':
        if len(points) == 4:
            pts = np.array(points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], True, (255, 0, 0), 2)

            cv2.putText(img, 
                        f"{type}: {data}",
                        (points[0].x, points[0].y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2)

    cv2.imshow('Detecção de Código de Barras', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
