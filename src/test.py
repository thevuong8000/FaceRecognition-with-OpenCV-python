import cv2
import detective

capture = cv2.VideoCapture(0)
smile_cascade = cv2.CascadeClassifier("./cascade/haarcascade_smile.xml")


while True:
    ret, frame = capture.read()
    frame = cv2.flip(frame, 1)

    
    faces = detective.face_detect(frame)

    for face in faces:
        (x, y, w, h) = face
        roi = frame[y: y + h, x: x + w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        face_area = w * h

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        smiles = smile_cascade.detectMultiScale(gray)
        for smile in smiles:
            (x1, y1, w1, h1) = smile
            smile_area = w1 * h1
            if smile_area >= face_area / 8 and y + y1 > (y + y + h) / 2:
                cv2.rectangle(frame, (x + x1, y + y1), (x + x1 + w1, y + y1 + h1), (0, 255, 0), 2)
    
    cv2.imshow('frame', frame)
    k = cv2.waitKey(1) & 0xFF

    if k == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()