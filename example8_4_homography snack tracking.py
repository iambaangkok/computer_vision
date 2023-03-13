import numpy as np
import cv2

cap = cv2.VideoCapture(0)
detector = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
# detector = cv2.SIFT_create()
matcher = cv2.BFMatcher()


refs = [
    cv2.imread('SnackTracking/BigSheet.jpg', cv2.COLOR_BGR2GRAY),
    cv2.imread('SnackTracking/Haitai.jpg', cv2.COLOR_BGR2GRAY),
    cv2.imread('SnackTracking/Oreo.jpg', cv2.COLOR_BGR2GRAY)
    ]
scales = [0.5,0.5,0.5]
colors = [(255,0,0), (0,255,0), (0,0,255)]
hs = []
ws = []

kp1s = []
des1s = []
for i in range(len(refs)):
    h,w,_ = refs[i].shape
    hs.append(h)
    ws.append(w)

    refs[i] = cv2.resize(refs[i],(int(w*scales[i]),int(h*scales[i])))
    
    kp1, des1 = detector.detectAndCompute(refs[i], None)
    kp1s.append(kp1)
    des1s.append(des1)

while (True):
    _, target = cap.read()
    # result = []
    kp2, des2 = detector.detectAndCompute(target, None)

    

    for i in range(len(refs)):
        matches = matcher.knnMatch(des1s[i], des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.5 * n.distance:
                good.append(m)
                
        if len(good) > 8:
            print(len(good))
            ref_pts = np.float32([kp1s[i][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            target_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(ref_pts, target_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            h, w, _ = refs[i].shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            target = cv2.polylines(target, [np.int32(dst)], True, colors[i], 3, cv2.LINE_AA)

    # result = cv2.drawMatches(refs[i], kp1s[i], target, kp2, good, None, flags=2)

    cv2.imshow('result', target)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
