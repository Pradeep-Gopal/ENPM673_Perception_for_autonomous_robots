import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

x = list(range(0, 256))

# o_mean_blue=np.array([153.6105954477455])
# o_std_blue=np.array([35.645345579156796])
# o_mean_green=np.array([237.72562582031875])
# o_std_green=np.array([8.415651120364487])
# o_mean_orange=np.array([251.9544956696381])
# o_std_orange=np.array([2.5698329315125505])

# y_mean_green = np.array([228.36676848886714])
# y_std_green = np.array([11.534612731323769])
# y_mean_blue = np.array([168.69932624883327])
# y_std_blue = np.array([37.736126383805704])
# y_mean_orange = np.array([235.21833286773568])
# y_std_orange = np.array([5.662585713799174])

# g_mean_blue=np.array([190.98365470748638])
# g_std_blue=np.array([20.76008569347966])
# g_mean_green=np.array([155.0722972243808])
# g_std_green=np.array([15.859964345501147])
# g_mean_orange=np.array([239.87056150841687])
# g_std_orange=np.array([9.134057716050851])

o_mean_blue=np.array([154.72662419580544])
o_std_blue=np.array([36.30795123353113])
o_mean_green=np.array([238.14753766499953])
o_std_green=np.array([8.417587391439978])
o_mean_orange=np.array([252.24937318452012])
o_std_orange=np.array([2.346205081137928])

y_mean_green = np.array([230.99625753104914])
y_std_green = np.array([9.384343859959012])
y_mean_blue = np.array([170.86092610188882])
y_std_blue = np.array([38.29599853584383])
y_mean_orange = np.array([235.26442800066383])
y_std_orange = np.array([5.645941025982294])

g_mean_blue=np.array([184.90079776755584])
g_std_blue=np.array([20.577573078086086])
g_mean_green=np.array([152.06534796946667])
g_std_green=np.array([15.898153337093653])
g_mean_orange=np.array([236.81438517865615])
g_std_orange=np.array([11.714241624592178])


def gaussian(x, mu, sig):
    gauss = ((1 / (sig * math.sqrt(2 * math.pi))) * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))))
    return gauss


def yellow_plot_bellcurve():
    y_gauss_b = gaussian(x, y_mean_blue, y_std_blue)
    y_gauss_g = gaussian(x, y_mean_green, y_std_green)
    y_gauss_r = gaussian(x, y_mean_orange, y_std_orange)
    plt.plot(y_gauss_b, 'b')
    plt.plot(y_gauss_g, 'g')
    plt.plot(y_gauss_r, 'r')
    plt.show()
    return y_gauss_b, y_gauss_g, y_gauss_r


def orange_plot_bellcurve():
    o_gauss_b = gaussian(x, o_mean_blue, o_std_blue)
    o_gauss_g = gaussian(x, o_mean_green, o_std_green)
    o_gauss_r = gaussian(x, o_mean_orange, o_std_orange)
    plt.plot(o_gauss_b, 'b')
    plt.plot(o_gauss_g, 'g')
    plt.plot(o_gauss_r, 'r')
    plt.show()
    return o_gauss_b, o_gauss_g, o_gauss_r


def green_plot_bellcurve():
    g_gauss_b = gaussian(x, g_mean_blue, g_std_blue)
    g_gauss_g = gaussian(x, g_mean_green, g_std_green)
    g_gauss_r = gaussian(x, g_mean_orange, g_std_orange)
    plt.plot(g_gauss_b, 'b')
    plt.plot(g_gauss_g, 'g')
    plt.plot(g_gauss_r, 'r')
    plt.show()
    return g_gauss_b, g_gauss_g, g_gauss_r


def image_process(frame, y_gauss_b, y_gauss_g, y_gauss_r, g_gauss_b, g_gauss_g, g_gauss_r, o_gauss_b, o_gauss_g,
                  o_gauss_r):
    frame_r = frame[:, :, 2]
    frame_g = frame[:, :, 1]
    frame_b = frame[:, :, 0]

    out1 = np.zeros(frame_g.shape, dtype=np.uint8)
    out3 = np.zeros(frame_r.shape, dtype = np.uint8)
    out2 = np.zeros(frame_r.shape, dtype = np.uint8)

    for i in range(0, frame_r.shape[0]):
        for j in range(0, frame_r.shape[1]):
            y = frame_r[i][j]
            z = frame_g[i][j]
            h = frame_r[i][j]

            if ((y_gauss_r[y] + y_gauss_r[z]) / 2) > 0.05 and ((y_gauss_b[y] + y_gauss_b[z]) / 2) < 0.015 and frame_b[i][j] < 130:
                out2[i][j] = 255
            else:
                out2[i][j] = 0

            if g_gauss_r[z] > 0.03 and g_gauss_g[z] < 0.025 and g_gauss_b[z] < 0.02 and frame_r[i][j] < 180:
                out1[i][j] = 255
            else:
                out1[i][j] = 0

            if o_gauss_r[h] > 0.050 and frame_b[i][j] < 150:
                out3[i][j] = 255
            if o_gauss_g[h] > 0.02 and frame_b[i][j] < 150:
                out3[i][j] = 0
            if o_gauss_b[h] > 0.001 and frame_b[i][j] < 150:
                out3[i][j] = 0

    ret, threshold2 = cv2.threshold(out2, 240, 255, cv2.THRESH_BINARY)
    kernel2 = np.ones((3, 3), np.uint8)

    dilation2 = cv2.dilate(threshold2, kernel2, iterations=6)
    contours2, _ = cv2.findContours(dilation2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    ret, threshold3 = cv2.threshold(out1, 240, 255, cv2.THRESH_BINARY)
    kernel3 = np.ones((2, 2), np.uint8)

    dilation3 = cv2.dilate(threshold3, kernel3, iterations=9)
    contours1, _ = cv2.findContours(dilation3, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    kernel1 = np.ones((2, 2), np.uint8)
    dilation1 = cv2.dilate(out3, kernel1, iterations=6)
    contours3, _ = cv2.findContours(dilation1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    frame = draw_circle(frame, contours1, contours2, contours3)

    cv2.imshow('yellow detection', frame)
    return


def draw_circle(frame, contours1, contours2, contours3):
    for contour in contours2:
        if cv2.contourArea(contour) > 20:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y) - 1)
            radius = int(radius) - 1
            if radius > 12:
                cv2.circle(frame, center, radius, (0, 255, 255), 2)

    for contour in contours1:
        if cv2.contourArea(contour) > 31:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            if 13 < radius < 16:
                cv2.circle(frame, center, radius, (0, 255, 0), 2)

    for contour in contours3:
        if cv2.contourArea(contour) > 18:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            if radius > 13:
                cv2.circle(frame, center, radius, (0, 0, 255), 2)
    return frame


def main():
    y_gauss_b, y_gauss_g, y_gauss_r = yellow_plot_bellcurve()
    g_gauss_b, g_gauss_g, g_gauss_r = green_plot_bellcurve()
    o_gauss_b, o_gauss_g, o_gauss_r = orange_plot_bellcurve()

    video = cv2.VideoCapture("detectbuoy.avi")
    while video.isOpened():
        opened, frame = video.read()
        if opened:
            image_process(frame, y_gauss_b, y_gauss_g, y_gauss_r, g_gauss_b, g_gauss_g, g_gauss_r, o_gauss_b, o_gauss_g,
                          o_gauss_r)
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
