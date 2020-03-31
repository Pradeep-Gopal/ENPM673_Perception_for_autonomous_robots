import cv2
import numpy as np
from matplotlib import pyplot as plt
import math


x=list(range(0, 256))
# mean_blue=np.array([230.92855834531522])
# std_blue=np.array([11.825125082753653])
# mean_green=np.array([200.98927408174694])
# std_green=np.array([12.952317147474373])
# mean_orange=np.array([244.4299727613101])
# std_orange=np.array([5.067551652277579])

mean_blue=np.array([190.98365470748638])
std_blue=np.array([20.76008569347966])
mean_green=np.array([155.0722972243808])
std_green=np.array([15.859964345501147])
mean_orange=np.array([239.87056150841687])
std_orange=np.array([9.134057716050851])


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
    frame_g=frame[:,:,1]
    frame_r=frame[:,:,2]

    out=np.zeros(frame_g.shape, dtype = np.uint8)

    for i in range(0, frame_g.shape[0]):
        for j in range(0, frame_g.shape[1]):
            z = frame_g[i][j]

            if gauss_r[z] > 0.03 and gauss_g[z] < 0.025 and gauss_b[z] < 0.02 and frame_r[i][j] < 180:
                #                     print(z)
                out[i][j] = 255
            else:
                out[i][j] = 0
    ret, threshold3 = cv2.threshold(out, 240, 255, cv2.THRESH_BINARY)
    kernel3 = np.ones((2, 2), np.uint8)

    dilation3 = cv2.dilate(threshold3, kernel3, iterations=9)
    _, contours1, _ = cv2.findContours(dilation3, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    frame = draw_circle(frame, contours1)
    cv2.imshow('Green detection', frame)
    return


def draw_circle(frame, contours1):
    for contour in contours1:
        if cv2.contourArea(contour) > 31:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            if 13 < radius < 16:
                cv2.circle(frame, center, radius, (0, 255, 0), 2)
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
