import cv2

caption = cv2.VideoCapture('../videos/test.mp4')
# caption = cv2.VideoCapture(0)

if not caption.isOpened():
    print("Cannot open camera")
    exit()

ret, frame1 = caption.read()
ret, frame2 = caption.read()

while True:
    frame1 = cv2.resize(frame1, (1024, 640), interpolation=cv2.INTER_AREA)
    frame2 = cv2.resize(frame2, (1024, 640), interpolation=cv2.INTER_AREA)
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html
    ret, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
    dilation = cv2.dilate(thresh, None, iterations=10)

    # https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(type(contours))

    # Draw contours 
    #cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)

    for contour in contours:
        hull = cv2.convexHull(contour)
        if cv2.contourArea(hull) < 2000:
            continue
        x, y, w, h = cv2.boundingRect(hull)
        cv2.drawContours(frame1, [hull], -1, (0, 255, 0), 2)

        # cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 3)
    
    cv2.imshow("frame", frame1)
    frame1 = frame2
    ret, frame2 = caption.read()

    if cv2.waitKey(1) == 27: # pressing ESC
        break

caption.release()
cv2.destroyAllWindows()