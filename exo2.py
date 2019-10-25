import cv2
import numpy as np


def rgb(image, colors):
    img = cv2.imread(image)
    cv2.imshow('base', img)

    img_copy = img.copy()

    for c in 'bgr':
        if c not in colors:
            if c == 'r':
                img_copy[:, :, 2] = 0

            if c == 'g':
                img_copy[:, :, 1] = 0

            if c == 'b':
                img_copy[:, :, 0] = 0

    cv2.imshow('BGR', img_copy)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


def main():
    rgb('images/bgr.png', 'br')


if __name__ == '__main__':
    main()
