import cv2
import numpy as np
from matplotlib import pyplot as plt
import math


x=list(range(0, 256))
# mean_blue=np.array([170.1123])
# std_blue=np.array([36.1331436])
# mean_green=np.array([239.95395])
# std_green=np.array([7.3541856])
# mean_orange=np.array([252.3011604])
# std_orange=np.array([2.373163])

mean_blue=np.array([153.6105954477455])
std_blue=np.array([35.645345579156796])
mean_green=np.array([237.72562582031875])
std_green=np.array([8.415651120364487])
mean_orange=np.array([251.9544956696381])
std_orange=np.array([2.5698329315125505])


def gaussian(x, mu, sig):
    gauss = ((1/(sig*math.sqrt(2*math.pi)))*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))))
    return gauss


def plot_bellcurve():
    gauss_b = gaussian(x, mean_blue, std_blue)
    gauss_g = gaussian(x, mean_green, std_green)
    gauss_r = gaussian(x, mean_orange, std_orange)
    plt.plot(gauss_b, 'b')
    plt.plot(gauss_g, 'g')
    plt.plot(gauss_r, 'r')
    plt.show()
    return gauss_b, gauss_g, gauss_r


def image_process(frame, gauss_b, gauss_g, gauss_r):
    frame_r = frame[:,:,2]
    frame_b = frame[:,:,0]
    out = np.zeros(frame_r.shape, dtype = np.uint8)

    for i in range(0, frame_r.shape[0]):
        for j in range(0, frame_r.shape[1]):
            y = frame_r[i][j]

            if gauss_r[y] > 0.050 and frame_b[i][j] < 150:
                out[i][j] = 255

            if gauss_g[y] > 0.02 and frame_b[i][j] < 150:
                out[i][j] = 0

            if gauss_b[y] > 0.001 and frame_b[i][j] < 150:
                out[i][j] = 0

    kernel1 = np.ones((2, 2), np.uint8)

    dilation1 = cv2.dilate(out, kernel1, iterations=6)

    _, contours1, _ = cv2.findContours(dilation1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    frame = draw_circle(frame, contours1)
    cv2.imshow('Orange detection', frame)
    return


def draw_circle(frame, contours1):
    for contour in contours1:
        if cv2.contourArea(contour) > 20:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            if radius > 13:
                cv2.circle(frame, center, radius, (0, 0, 255), 2)
    return frame


def main():
    gauss_b, gauss_g, gauss_r = plot_bellcurve()
    video = cv2.VideoCapture("detectbuoy.avi")
    while video.isOpened():
        opened, frame = video.read()
        if opened:
            image_process(frame, gauss_b, gauss_g, gauss_r)
            cv2.waitKey(1)
        else:
            break
        k = cv2.waitKey(15) & 0xff
        if k == 27:
            break

    cv2.destroyAllWindows()
    video.release()


if __name__ == '__main__':
    main()
