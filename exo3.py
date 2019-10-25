import cv2
import numpy as np


def get_similar_image(image, liste):
    image = cv2.imread(image)
    more_similar = -1
    img = None
    hist_image = cv2.calcHist([image], [0], None, [256], [0, 256])
    for img_list in liste:
        img_list = cv2.imread(img_list)
        hist_img_list = cv2.calcHist([img_list], [0], None, [256], [0, 256])
        similarity = cv2.compareHist(hist_image, hist_img_list, cv2.HISTCMP_CORREL)
        if more_similar < similarity:
            more_similar = similarity
            img = img_list

    cv2.imshow('similar', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    images = ['images/beach.jpg', 'images/dog.jpg', 'images/polar.jpg', 'images/bear.jpg', 'images/lake.jpg',
              'images/moose.jpg']
    get_similar_image('images/waves.jpg', images)


if __name__ == '__main__':
    main()
