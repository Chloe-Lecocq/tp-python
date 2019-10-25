import cv2
import numpy as np
from matplotlib import pyplot as plt


def template_matching(taux):
    # reading in the image
    img_base = 'images/baboon1.jpg'
    img_rgb = cv2.imread(img_base, 0)
    img_copy = cv2.imread(img_base)

    img_rgb_grey = cv2.imread(img_base, 0)

    # converting to grayscale
    # reading the template to be matched.
    template = cv2.imread('images/baboon1_tete.jpg', 0)
    hist_template_img = cv2.calcHist([template], [0], None, [256], [0, 256])
    cv2.normalize(hist_template_img, hist_template_img)
    h, w = template.shape[::-1]
    h_i, w_i = img_rgb_grey.shape[::-1]
    mask = np.zeros(img_rgb.shape[:2], np.uint8)

    for i in range(0, h_i-w, 10):
        for j in range(0, w_i-h, 10):
            mask_step = mask.copy()
            mask_step[j:h + j, i:w + i] = 255
            masked_img = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_step)
            cv2.imshow('Detected', masked_img)
            # cv2.waitKey(1)

            hist_masked_img = cv2.calcHist([masked_img], [0], mask_step, [256], [0, 256])

            cv2.normalize(hist_masked_img, hist_masked_img)
            if cv2.compareHist(hist_template_img, hist_masked_img, cv2.HISTCMP_CORREL) > taux/100:
                cv2.rectangle(img_copy, (i, j), (i + h, j + w), (0, 255, 255), 2)
    cv2.imshow('Detected', img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    template_matching(97.7)


if __name__ == '__main__':
    main()
