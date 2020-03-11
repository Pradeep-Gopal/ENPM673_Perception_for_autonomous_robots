import numpy as np
import cv2
from matplotlib import pyplot as plt

video = cv2.VideoCapture('Night Drive - 2689.mp4')

#trying with gamma correction
#creating LUT table:
gamma = 2.5
inv_gamma = 1.0/gamma
Value_red = 0.8
table=np.zeros(256, dtype='uint8')
for i in range(256):
    table[i] = Value_red*255* np.power((i/255), inv_gamma)
while (True):
    opened, frame = video.read()
    if opened:
        image = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
        adjusted = cv2.LUT(image, table)
                
# =============================================================================
#   Code for Using CLAHE for Hue, Saturation and Value channels of BGR->HSV image
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10,10))
#         adjusted_blur = cv2.medianBlur(adjusted, 7)
#         image_hsv = cv2.cvtColor(adjusted_blur, cv2.COLOR_BGR2HSV)
#         h,s,v = cv2.split(image_hsv.copy())
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10,10))
#         output_h = clahe.apply(h)
#         output_s = clahe.apply(s)
#         output_v = clahe.apply(v)
#         output = cv2.merge((output_h, output_s, output_v))
#         fin_output = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)
# 
# =============================================================================
        cv2.imshow('image', image)
        cv2.imshow('corrected', adjusted)
        # cv2.imshow('fin',fin_output)
        if cv2.waitKey(2) & 0xff == ord('q'):
            break
cv2.destroyAllWindows()
video.release()