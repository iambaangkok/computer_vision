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

def morph (img, kernel_size, morph_type, structure_type):
    kernel = cv2.getStructuringElement(structure_type,(kernel_size,kernel_size))
    # kernel = np.zeros((kernel_size,kernel_size), np.uint8)
    # kernel = cv2.circle(kernel, (int(kernel_size/2),int(kernel_size/2)), int(kernel_size/2), (255,255,255), -1)
    return cv2.morphologyEx(img, morph_type, kernel)

def coinCounting(filename):
    
    im = cv2.imread(filename)
    target_size = (int(704/2),int(960/2))  # (int(im.shape[1]/2),int(im.shape[0]/2))
    im = cv2.resize(im,target_size)
    
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    blue_hsv = hsv # cv2.cvtColor(im+[10,0,0], cv2.COLOR_BGR2HSV)

    # im = BrightnessContrast(im, 0)

    # contrast
    alpha = 0.5 # Simple contrast control
    beta = 0 # Simple brightness control

    # im = cv2.convertScaleAbs(im, alpha=alpha, beta=beta)

    # define threshold
    # blue
    lower_yellow = np.array([22,50,50])
    upper_yellow = np.array([35,255,255])
    lower_blue = np.array([94,15,50])
    upper_blue = np.array([150,255,255])
    lower_bg = np.array([0,0,0])
    upper_bg = np.array([180,70,200])

    lower_blue2 = np.array([80,8,50])
    upper_blue2 = np.array([160,255,255])
    
    # thresholding
    # b, g, r
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_blue = cv2.inRange(blue_hsv, lower_blue, upper_blue)
    mask_bg = cv2.inRange(hsv, lower_bg, upper_bg)

    mask_bg = 255-mask_bg

    # median
    mask_bg = cv2.medianBlur(mask_bg, 9)
    cv2.imshow('BG', mask_bg)
    mask_yellow = cv2.medianBlur(mask_yellow, 5)
    mask_blue = cv2.medianBlur(mask_blue, 9)

    cv2.imshow('mask_blue pre bitwise', mask_blue)
    mask_blue = np.bitwise_or(mask_blue, mask_bg)
    # mask_blue = cv2.medianBlur(mask_blue, 9)
    
    # cv2.imshow('mask_blue pre inrange2', mask_blue)
    # im_masked_blue = cv2.bitwise_and(im, im, mask_blue)

    # hsv_masked_blue = cv2.cvtColor(im_masked_blue, cv2.COLOR_BGR2HSV)

    # mask_blue = cv2.inRange(hsv_masked_blue, lower_blue2, upper_blue2)
    # mask_blue = cv2.medianBlur(mask_blue, 9)

    # morphology

    
    morph_yellow = morph(mask_yellow, 20, cv2.MORPH_OPEN, cv2.MORPH_ELLIPSE)
    morph_yellow = morph(morph_yellow, 10, cv2.MORPH_ERODE, cv2.MORPH_ELLIPSE)

    # morph_blue = morph(mask_blue, 12, cv2.MORPH_DILATE, cv2.MORPH_ELLIPSE)
    morph_blue = morph(mask_blue, 15, cv2.MORPH_CLOSE, cv2.MORPH_ELLIPSE)
    # morph_blue = morph(mask_blue, 20, cv2.MORPH_ERODE, cv2.MORPH_ELLIPSE)
    cv2.imshow('Morph Blue Coin 1', morph_blue)
    # morph_blue = morph(morph_blue, 12, cv2.MORPH_ERODE, cv2.MORPH_ELLIPSE)
    # morph_blue = morph(morph_blue, 10, cv2.MORPH_OPEN, cv2.MORPH_ELLIPSE)
    # morph_blue = morph(morph_blue, 20, cv2.MORPH_OPEN, cv2.MORPH_ELLIPSE)
    # morph_blue = morph(morph_blue, 20, cv2.MORPH_OPEN, cv2.MORPH_ELLIPSE)
    # morph_blue = morph(morph_blue,5, cv2.MORPH_CLOSE, cv2.MORPH_ELLIPSE)
    # morph_blue = morph(morph_blue, 20, cv2.MORPH_DILATE, cv2.MORPH_CROSS)

    #

    blur_amount = 15
    # morph_blue = cv2.GaussianBlur(morph_blue, (blur_amount, blur_amount), 0)
    # morph_blue = cv2.medianBlur(morph_blue, blur_amount)
    # outline = cv2.Canny(blurred, 30, 150)

    # kernel = np.array([[0, -1, 0],
    #                [-1, 5,-1],
    #                [0, -1, 0]])
    # morph_blue = cv2.filter2D(src=morph_blue, ddepth=-1, kernel=kernel)
    # morph_blue = cv2.filter2D(src=morph_blue, ddepth=-1, kernel=kernel)
    # morph_blue = cv2.filter2D(src=morph_blue, ddepth=-1, kernel=kernel)
 
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
    cv2.imshow('Yellow Coin', mask_yellow)
    cv2.imshow('Blue Coin', mask_blue)
    cv2.imshow('Morph Yellow Coin', morph_yellow)
    cv2.imshow('Morph Blue Coin', morph_blue)
    cv2.imshow('Contour Blue Coin', im_blue_contour)

     

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

