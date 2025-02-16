#Download images from https://drive.google.com/file/d/1KqllafwQiJR-Ronos3N-AHNfnoBb8I7H/view?usp=sharing

#Bonus from this challenge https://docs.google.com/document/d/1q96VgmpJXlC95h9we-jiuxonEoYrZoqgTD2wjGk5TlI/edit?usp=sharing



import cv2
import numpy as np

expected_outcome = [
    [5, 8],
    [6, 3],
    [2, 4],
    [2, 4],
    [1, 7],
    [3, 5],
    [4, 3],
    [5, 5],
    [2, 6],
    [4, 2] ]

def coinCounting(filename):
    
    im = cv2.imread(filename)
    target_size = (int(704/2),int(960/2))  # (int(im.shape[1]/2),int(im.shape[0]/2))

    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    

    # define threshold
    # blue
    lower_yellow = np.array([22,50,50])
    upper_yellow = np.array([35,255,255])
    lower_blue = np.array([94,50,50])
    upper_blue = np.array([150,255,255])
    
    # thresholding
    # b, g, r
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # median
    mask_yellow = cv2.medianBlur(mask_yellow, 5)

    mask_blue = cv2.medianBlur(mask_blue, 5)

    # morphology
    def morph (img, kernel_size, morph_type, structure_type):
        kernel = cv2.getStructuringElement(structure_type,(kernel_size,kernel_size))
        return cv2.morphologyEx(img, morph_type, kernel)
    
    morph_yellow = morph(mask_yellow, 20, cv2.MORPH_OPEN, cv2.MORPH_ELLIPSE)
    morph_yellow = morph(morph_yellow, 10, cv2.MORPH_ERODE, cv2.MORPH_ELLIPSE)

    kernel_size = 7
    kernel = np.zeros((kernel_size,kernel_size), np.uint8)
    kernel = cv2.circle(kernel, (int(kernel_size/2),int(kernel_size/2)), int(kernel_size/2), (255,255,255), 1)
    morph_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
    
    kernel_size = 5
    kernel = np.zeros((kernel_size,kernel_size), np.uint8)
    kernel = cv2.circle(kernel, (int(kernel_size/2),int(kernel_size/2)), int(kernel_size/2), (255,255,255), 1)
    morph_blue = cv2.morphologyEx(morph_blue, cv2.MORPH_OPEN, kernel)

    kernel_size = 7
    kernel = np.zeros((kernel_size,kernel_size), np.uint8)
    kernel = cv2.circle(kernel, (int(kernel_size/2),int(kernel_size/2)), int(kernel_size/2), (255,255,255), 1)
    morph_blue = cv2.morphologyEx(morph_blue, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('morph_blue 0', morph_blue)

    kernel = np.array([
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0],
                    [1, 0, 0, 0, 0]],  np.uint8)
    morph_blue = cv2.morphologyEx(morph_blue, cv2.MORPH_OPEN, kernel)
    cv2.imshow('morph_blue 1', morph_blue)

    kernel = np.array([
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0]],  np.uint8)
    morph_blue = cv2.morphologyEx(morph_blue, cv2.MORPH_OPEN, kernel)
    morph_blue = cv2.morphologyEx(morph_blue, cv2.MORPH_OPEN, kernel)
    morph_blue = cv2.morphologyEx(morph_blue, cv2.MORPH_OPEN, kernel)
    cv2.imshow('morph_blue 2', morph_blue)

    # kernel = np.array([
    #                 [1, 0, 0, 0, 0],
    #                 [0, 1, 0, 0, 0],
    #                 [0, 0, 1, 0, 0],
    #                 [0, 0, 0, 1, 0],
    #                 [0, 0, 0, 0, 1]],  np.uint8)
    # morph_blue = cv2.morphologyEx(morph_blue, cv2.MORPH_OPEN, kernel)
    # cv2.imshow('morph_blue 3', morph_blue)

    # kernel = np.array([
    #             [1, 0, 0, 0, 0, 0, 0, 0, 1],
    #             [1, 0, 0, 0, 0, 0, 0, 0, 1],
    #             [1, 1, 0, 0, 0, 0, 0, 1, 1],
    #             [1, 1, 0, 0, 0, 0, 0, 1, 1],
    #             [1, 1, 1, 1, 1, 1, 1, 1, 1],
    #             [0, 1, 1, 1, 1, 1, 1, 1, 0],
    #             [0, 1, 1, 1, 1, 1, 1, 1, 0],
    #             [0, 0, 1, 1, 1, 1, 1, 0, 0],
    #             [0, 0, 0, 1, 1, 1, 0, 0, 0]],  np.uint8)
    # morph_blue = cv2.morphologyEx(morph_blue, cv2.MORPH_OPEN, kernel)
    # cv2.imshow('morph_blue 4', morph_blue)

    # morph_blue = morph(morph_blue, 15, cv2.MORPH_OPEN, cv2.MORPH_ELLIPSE)
    # cv2.imshow('morph_blue 5', morph_blue)

    # morph_blue = morph(morph_blue, 12, cv2.MORPH_ERODE, cv2.MORPH_ELLIPSE)
    # cv2.imshow('morph_blue 6', morph_blue)
   
    # kernel = np.array([
    #             [0, 0, 0, 0, 0, 0, 0, 0, 1],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 1],
    #             [0, 0, 0, 0, 0, 0, 0, 1, 1],
    #             [0, 0, 0, 0, 0, 0, 0, 1, 1],
    #             [1, 1, 1, 1, 1, 1, 1, 1, 1],
    #             [1, 1, 1, 1, 1, 1, 1, 1, 0],
    #             [0, 1, 1, 1, 1, 1, 1, 1, 0],
    #             [0, 0, 1, 1, 1, 1, 1, 0, 0],
    #             [0, 0, 0, 1, 1, 1, 0, 0, 0]],  np.uint8)
    # morph_blue = cv2.morphologyEx(morph_blue, cv2.MORPH_OPEN, kernel)
    # cv2.imshow('morph_blue 7', morph_blue)

    # kernel = np.array([
    #             [0, 0, 0, 0, 0, 0, 0, 0, 1],
    #             [0, 0, 0, 0, 0, 0, 0, 1, 0],
    #             [0, 0, 0, 0, 0, 0, 1, 0, 0],
    #             [0, 0, 0, 0, 0, 1, 0, 0, 0],
    #             [1, 1, 1, 1, 1, 1, 1, 1, 1],
    #             [0, 0, 0, 1, 0, 0, 0, 0, 0],
    #             [0, 0, 1, 0, 0, 0, 0, 0, 0],
    #             [0, 1, 0, 0, 0, 0, 0, 0, 0],
    #             [1, 0, 0, 0, 0, 0, 0, 0, 0]],  np.uint8)
    # morph_blue = cv2.morphologyEx(morph_blue, cv2.MORPH_ERODE, kernel)
    # cv2.imshow('morph_blue 8', morph_blue)

    #

    blur_amount = 15
 
    # contours
    contours_yellow, hierarchy_yellow = cv2.findContours(morph_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours_blue, hierarchy_blue = cv2.findContours(morph_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    yellow = len(contours_yellow)
    blue = len(contours_blue)

    im_blue_contour = im.copy()
    cv2.drawContours(im_blue_contour, contours_blue, -1, (0, 255, 0), 2)

    # print('Yellow = ',yellow)
    # print('Blue = ', blue)
    cv2.imshow('Original Image',im)

    
   
    # cv2.imshow('HSV Image',hsv)
    # cv2.imshow('Yellow Coin', mask_yellow)
    cv2.imshow('Blue Coin', mask_blue)
    # cv2.imshow('Morph Yellow Coin', morph_yellow)
    cv2.imshow('Morph Blue Coin', morph_blue)
    # cv2.imshow('Contour Blue Coin', im_blue_contour)

     

    return [yellow,blue]

# process
for i in range(1,11):
    result = coinCounting('.\CoinCounting\coin'+str(i)+'.jpg')
    diff = np.subtract(result, expected_outcome[i-1])
    correct = ''
    if list(diff) == [0,0]:
        correct = 'PASS'
    print(i,":", expected_outcome[i-1], result, diff, correct)
    cv2.waitKey()

