import sys
import cv2 

def gamma_correction(img, gamma,c=1.0):
   i = img.copy()
   i[:,:] = 255*(c*(img[:,:]/255.0)**(1.0 / gamma))
   return i

filename = sys.argv[1]
im = cv2.imread(filename)

b, g, r = cv2.split(im)

b_gamma = gamma_correction(b, 0.5)
g_gamma = gamma_correction(g, 1.5)
img_gamma = cv2.merge([b_gamma, g_gamma, r])


cv2.imshow("original", im)
cv2.imshow("gamma", img_gamma)
cv2.waitKey(0)
cv2.destroyAllWindows