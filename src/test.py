import numpy as np
import cv2

capture = cv2.VideoCapture(0)
idx = 0

while True:
    ret, frame = capture.read()

    frame = cv2.flip(frame, 1)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord('c'):
        cv2.imwrite(f'LIL{idx}.png', frame)
        idx += 1

capture.release()
cv2.destroyAllWindows()